"""S3: upgraded SNOMED LLM step for low-confidence rows (local Ollama, Qwen3-8B q4, Apache-2.0).

X's current step (termmap/llm.py) uses Qwen2.5-1.5B-Instruct: one forward pass scoring letters A-E over the
stacker's top-5 (excerpt + term + option code/description). The upgrade adds semantic tags, synonyms and the
6 nearest diagnosis-train examples (term -> gold concept), asks for strict JSON {"choice": i}, temperature 0,
fixed seed, cached by prompt hash; any failure falls back to the stacker's pick (never raises)."""
from __future__ import annotations

import hashlib
import json
import time
import urllib.request

import numpy as np
import pandas as pd

from termmap import config as C
from termmap.normalize import clean
from termmap_plus import core as K
from termmap_plus.features_plus import window

OLLAMA = "http://localhost:11434/api/generate"
MODEL = "qwen3:8b"
CACHE = K.ART / "v2b" / "llm8b_cache.jsonl"


def neighbours(frames, k=6):
    """6 nearest diagnosis-train SNOMED examples per row (char TF-IDF on the entity)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    t = K.diag_train()
    tr = t[t.coding_system == C.SNOMED].reset_index(drop=True)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True).fit(tr.entity.map(clean))
    M = vec.transform(tr.entity.map(clean))
    d = frames[C.SNOMED]
    S = (vec.transform(d.entity.map(clean)) @ M.T).toarray()
    idx = np.argsort(-S, axis=1, kind="stable")[:, :k]
    return {m: [(tr.entity.iat[j], tr.code.iat[j]) for j in idx[m]] for m in range(len(d))}


def build_prompt(entity, chunk, options, nn, con, syn):
    lines = []
    for i, c in enumerate(options, 1):
        s = [x for x in syn.get(c, []) if x.lower() != con.preferred.get(c, "").lower()][:2]
        lines.append(f"{i}. {con.preferred.get(c, '')} [{con.tag.get(c, '')}]" + (f" (synonyms: {'; '.join(s)})" if s else ""))
    ex = "\n".join(f"- \"{e}\" -> {con.preferred.get(g, '')} [{con.tag.get(g, '')}]" for e, g in nn)
    return (f"You are a clinical terminology coder mapping chart terms to SNOMED CT.\n"
            f"How our coders mapped similar terms (follow their conventions):\n{ex}\n\n"
            f"Term: \"{entity}\"\nContext: \"{window(entity, chunk)}\"\n\nCandidates:\n" + "\n".join(lines) +
            "\n\nPick the single best candidate for the term as used in the context. Answer with JSON only: {\"choice\": <number>}")


class LLM:
    def __init__(self):
        self.cache = {}
        self.times = []
        if CACHE.exists():
            for line in CACHE.read_text(encoding="utf8").splitlines():
                d = json.loads(line)
                self.cache[d["k"]] = d["v"]

    def choose(self, prompt: str, n: int) -> int | None:
        key = hashlib.sha1((MODEL + prompt).encode("utf8")).hexdigest()
        if key in self.cache:
            return self.cache[key]
        body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False, "format": "json", "think": False,
                           "options": {"temperature": 0, "seed": 13, "num_predict": 16, "num_ctx": 2048}}).encode()
        t0 = time.perf_counter()
        v = None
        try:
            req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
            out = json.loads(urllib.request.urlopen(req, timeout=120).read())
            ch = int(json.loads(out.get("response", "{}")).get("choice", 0))
            v = ch - 1 if 1 <= ch <= n else None
        except Exception:
            v = None  # never raise: fall back to the stacker's pick
        self.times.append(time.perf_counter() - t0)
        self.cache[key] = v
        with CACHE.open("a", encoding="utf8") as f:
            f.write(json.dumps({"k": key, "v": v}) + "\n")
        return v


def run_llm(X: pd.DataFrame, tau_max: float = 0.8):
    """LLM choice for diagnosis SNOMED rows whose top stacker probability < tau_max."""
    from termmap.resources import load_aliases
    con = K.concepts()[C.SNOMED]
    a = load_aliases(C.SNOMED)
    syn = a[a.code.isin(set(X[X.system == C.SNOMED].code))].groupby("code").alias.agg(lambda v: list(dict.fromkeys(v))[:5]).to_dict()
    fr = K.dev_frames()
    nn = neighbours(fr)
    llm = LLM()
    out = {}
    sel = X[(X.system == C.SNOMED) & X.is_diag]
    for q, g in sel.groupby("qid", sort=False):
        g = g.sort_values(["p", "fused_rank"], ascending=[False, True], kind="stable").head(5)
        if g.p.iat[0] >= tau_max:
            continue
        m = int(q.split(":")[1])
        d = fr[C.SNOMED]
        opts = list(g.code)
        ch = llm.choose(build_prompt(d.entity.iat[m], d.chunk_text.iat[m], opts, nn[m], con, syn), len(opts))
        out[q] = (opts[ch] if ch is not None else opts[0], float(g.p.iat[0]))
        if len(out) % 25 == 0:
            print(f"  llm8b {len(out)} rows, median {np.median(llm.times) if llm.times else 0:.1f}s/call", flush=True)
    return out, llm.times

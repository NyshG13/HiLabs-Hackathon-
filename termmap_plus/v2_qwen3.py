"""v2.7: Qwen3-Reranker-0.6B (Apache-2.0) zero-shot yes/no score on the top-10 candidates per row per
system (official prompt format; score = P(yes)). Cached by (row_id, system, concept_id). Local only.
  python -m termmap_plus.v2_qwen3"""
import termmap_plus  # noqa: F401
import time

import numpy as np
import pandas as pd
import torch

from termmap import config as C
from termmap_plus import core as K
from termmap_plus import v2 as V
from termmap_plus.features_plus import window

MODEL = "Qwen/Qwen3-Reranker-0.6B"
CACHE = V.V2 / "qwen3_cache.parquet"
INSTR = {C.ICD: "Given a clinical term and its chart context, judge whether the ICD-10-CM code description is the correct code for the term.",
         C.SNOMED: "Given a clinical term and its chart context, judge whether the SNOMED CT concept is the correct concept for the term."}
PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
          "provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def doc_text(system, code, tree, syn):
    con = K.concepts()[system]
    pref = con.preferred.get(code, "")
    extra = tree.get(code, "") if system == C.ICD else f"semantic tag: {con.tag.get(code, '')}"
    s = [x for x in syn.get(code, []) if x.lower() != pref.lower()][:3]
    return f"{pref} | {extra}" + (f" | synonyms: {'; '.join(s)}" if s else "")


def score_pairs(pairs, bs=16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = MODEL
    tok = AutoTokenizer.from_pretrained(path, padding_side="left", cache_dir=str(K.ART / "hf_cache"))
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16, cache_dir=str(K.ART / "hf_cache")).to(C.device()).eval()
    yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
    out = np.zeros(len(pairs), np.float32)
    order = np.argsort([len(p) for p in pairs], kind="stable")
    with torch.inference_mode():
        for i in range(0, len(pairs), bs):
            idx = order[i:i + bs]
            enc = tok([PREFIX + pairs[j] + SUFFIX for j in idx], padding=True, truncation=True, max_length=512,
                      return_tensors="pt").to(C.device())
            logits = model(**enc, logits_to_keep=1).logits[:, -1, [no, yes]].float()
            out[idx] = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    del model
    torch.cuda.empty_cache()
    return out


def qwen3_features(X: pd.DataFrame, s_col: str = "s_ref", k: int = 10) -> pd.DataFrame:
    """X must carry a reference score column (X's stacker) to choose the top-k per qid."""
    from termmap.resources import load_aliases
    from termmap_plus.pools import build_icd_tree
    fr = K.dev_frames()
    tree = build_icd_tree().set_index("code").path_text.to_dict()
    syn = {}
    for s in K.SYSTEMS:
        a = load_aliases(s)
        syn.update(a.groupby("code").alias.agg(lambda v: list(dict.fromkeys(v))[:6]).to_dict())
    X = X.copy()
    top = X.sort_values(["qid", s_col], ascending=[True, False]).groupby("qid").head(k)
    keys, texts = [], []
    for r in top.itertuples():
        s, m = r.qid.split(":")
        d = fr[s]
        rid = d.row_id.iat[int(m)]
        query = f"term: {d.entity.iat[int(m)]} | context: {window(d.entity.iat[int(m)], d.chunk_text.iat[int(m)])}"
        keys.append((rid, s, r.code))
        texts.append(f"<Instruct>: {INSTR[s]}\n<Query>: {query}\n<Document>: {doc_text(s, r.code, tree, syn)}")
    cache = pd.read_parquet(CACHE) if CACHE.exists() else pd.DataFrame(columns=["row_id", "system", "code", "q3"])
    lut = {(a, b, c): v for a, b, c, v in zip(cache.row_id, cache.system, cache.code, cache.q3)}
    todo = [i for i, kk in enumerate(keys) if kk not in lut]
    if todo:
        t0 = time.perf_counter()
        sc = score_pairs([texts[i] for i in todo])
        print(f"qwen3 scored {len(todo)} pairs in {time.perf_counter() - t0:.0f}s", flush=True)
        new = pd.DataFrame([(*keys[i], float(v)) for i, v in zip(todo, sc)], columns=["row_id", "system", "code", "q3"])
        pd.concat([cache, new], ignore_index=True).to_parquet(CACHE)
        for i, v in zip(todo, sc):
            lut[keys[i]] = float(v)
    top = top.assign(q3=[lut[kk] for kk in keys])
    X["q3_p"] = -1.0
    X.loc[top.index, "q3_p"] = top.q3.to_numpy()
    X["q3_rank"] = np.where(X.q3_p >= 0, X.groupby("qid").q3_p.rank(ascending=False, method="first"), k + 1)
    X["q3_margin"] = np.where(X.q3_p >= 0, X.q3_p - X.groupby("qid").q3_p.transform("max"), -1.0)
    return X


Q3_FEATS = ["q3_p", "q3_rank", "q3_margin"]

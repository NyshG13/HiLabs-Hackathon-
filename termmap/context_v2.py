"""context_v2 experiment: section heading, clause, span extension and co-entities.

Protocol (Step 0 fix): train is split 80/20 by chunk. SapBERT, the abbreviation table and every
cross-encoder are fitted on train80 only; cross-encoder scores for slice20 therefore come from models that
never saw those rows, and the stacker is fitted on slice20 only (code-frequency priors from train80, so a
row never counts its own label). Every decision is made on dev. Test is never read.

Runs (all compared on dev):
  current             span off, cross-encoder ctx80, base features
  +span_extension     span on,  ctx80, base + via_span_ext
  +stacker_context    span off, ctx80, base + context features            (Round A without span)
  +RoundA             span on,  ctx80, base + via_span_ext + context features
  +RoundB             span on,  cross-encoder v2 (enriched query), base + via_span_ext
  +RoundA+RoundB      span on,  v2, base + via_span_ext + context features

  python -m termmap.context_v2 --stages split,dense,cands,cross,score,evalA,crossv2,scorev2,eval
Outputs only under artifacts/context_v2/ and outputs/context_v2/.
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import time

import numpy as np
import pandas as pd

from . import config as C
from .context import SECTIONS, annotate
from .data import load_split

ART, OUT = C.ARTIFACTS_V2, C.OUTPUTS_V2
SPL, MOD, CND = ART / "splits", ART / "models", ART / "candidates"
ABBR = ART / "abbreviations.json"
SAP = MOD / "sapbert80"
V2_MAX_LEN = 256
V2_QUERY_BUDGET = V2_MAX_LEN - 3 - 64  # candidate side keeps ~64 tokens; context is trimmed to fit the rest
for _p in (SPL, MOD, CND / "off", CND / "on", OUT):
    _p.mkdir(parents=True, exist_ok=True)


def _free():
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------------------------
def stage_split():
    tr = load_split("train")
    groups = tr.group.drop_duplicates().to_numpy()
    order = np.random.default_rng(C.SEED).permutation(len(groups))
    cut = int(0.8 * len(groups))
    part = {groups[g]: ("train80" if i < cut else "slice20") for i, g in enumerate(order)}
    for name in ("train80", "slice20"):
        tr[tr.group.map(part) == name].to_csv(SPL / f"{name}.csv", index=False)
    log("split", {n: int((tr.group.map(part) == n).sum()) for n in ("train80", "slice20")})


def frame(name: str) -> pd.DataFrame:
    from .data import load_input
    df = load_input(SPL / f"{name}.csv") if name in ("train80", "slice20") else load_split(name)
    return annotate(df)  # co-entities come from the same split (a chunk never straddles splits)


def stage_dense():
    from .dense import Encoder, build_dense_index, fine_tune
    from .normalize import AbbreviationTable
    from .resources import load_aliases, snomed_raw
    t80 = frame("train80")
    ab = AbbreviationTable()
    ab.mine_descriptions(snomed_raw().description)
    ab.mine_descriptions(load_aliases(C.ICD).alias)
    ab.mine_gold(t80.entity, t80.display)
    ABBR.write_text(json.dumps(ab.to_json()), encoding="utf8")
    fine_tune(t80, out_dir=SAP, abbr_path=ABBR)
    _free()
    enc = Encoder(str(SAP))
    for s in C.SYSTEMS:
        build_dense_index(s, load_aliases(s), enc, ART / f"dense_{s}.npy")
    log("dense done")


def stage_cands(tags=("off", "on")):
    from .candidates import synthetic_queries
    from .dense import Encoder
    from .retrieval import GoldMemory, Mention, SystemRetriever, load_frontend
    C.seed_everything()
    ab, tg = load_frontend(ABBR)
    enc = Encoder(str(SAP))
    mem = GoldMemory(load_split("train"))  # labels of the whole train split; same-chunk rows are excluded below
    mem.attach_dense(enc)
    frames = {n: frame(n) for n in ("train80", "slice20", "dev")}
    synth = synthetic_queries(int(len(frames["train80"]) * C.SYNTH_RATIO), frames["train80"], ab)
    synth.to_parquet(CND / "synthetic_rows.parquet")
    for system in C.SYSTEMS:
        R = SystemRetriever(system, dense_path=ART / f"dense_{system}.npy")
        for tag in tags:
            for name, df in frames.items():
                d = df[df.coding_system == system].reset_index(drop=True)
                ms = [Mention(e, c, g, cl) for e, c, g, cl in zip(d.entity, d.chunk_text, d.group, d.clause)]
                cand = R.retrieve(ms, ab, tg, encoder=enc, memory=mem, use_memory_groups=(name != "dev"),
                                  span_extension=(tag == "on"))
                cand.to_parquet(CND / tag / f"{name}_{system}.parquet")
                log("cands", tag, name, system, len(d), len(cand))
        s = synth[synth.coding_system == system].reset_index(drop=True)
        ms = [Mention(e, "", g, e) for e, g in zip(s.entity, s.group)]
        R.retrieve(ms, ab, tg, encoder=enc).to_parquet(CND / "off" / f"synth_{system}.parquet")
        del R
        _free()


# ----------------------------------------------------------------------------------------------
def _concepts():
    from .resources import load_concepts
    return {s: load_concepts(s).set_index("code") for s in C.SYSTEMS}


def _v2_tokenizer():
    from transformers import AutoTokenizer
    from .reranker import V2_TOKENS
    tok = AutoTokenizer.from_pretrained(C.CROSS_BASE)
    tok.add_special_tokens({"additional_special_tokens": V2_TOKENS})
    return tok


def stage_cross(kind: str):
    """kind: ctx80 | noctx80 | v2"""
    from .reranker import V2_TOKENS, build_groups, query_text_v2, train_cross
    rng = random.Random(C.SEED)
    con = _concepts()
    t80 = frame("train80")
    synth = pd.read_parquet(CND / "synthetic_rows.parquet")
    tag = "on" if kind == "v2" else "off"
    tok = _v2_tokenizer() if kind == "v2" else None
    groups = []
    for system in C.SYSTEMS:
        parts = [(t80[t80.coding_system == system].reset_index(drop=True), CND / tag / f"train80_{system}.parquet")]
        if kind != "noctx80":
            s = synth[synth.coding_system == system].reset_index(drop=True)
            s["section"], s["clause"], s["coents"] = "none", s.entity, [[] for _ in range(len(s))]
            parts.append((s, CND / "off" / f"synth_{system}.parquet"))
        for df, path in parts:
            cand = pd.read_parquet(path)
            q = [dict(system=system, entity=r.entity, chunk=r.chunk_text, subconcept=r.subconcept, code=r.code,
                      section=r.section, clause=r.clause, coents=r.coents) for r in df.itertuples()]
            qf = None
            if kind == "v2":
                qf = lambda q_, sub: query_text_v2(q_["system"], q_["entity"], q_["chunk"], q_["section"], q_["clause"],
                                                   q_["coents"], tok, V2_QUERY_BUDGET)
            groups += build_groups(q, cand, con, rng, context=(kind != "noctx80"), query_fn=qf)
    log(f"groups {kind}: {len(groups)}")
    if kind == "v2":
        train_cross(groups, MOD / "cross-v2", max_len=V2_MAX_LEN, special_tokens=V2_TOKENS)
    elif kind == "ctx80":
        train_cross(groups, MOD / "cross-ctx80", max_len=C.CROSS_MAX_LEN)
    else:
        train_cross(groups, MOD / "cross-noctx80", max_len=C.COMPARATOR_MAX_LEN)
    _free()


def stage_score(kind: str, tags=("off", "on")):
    """Score slice20 + dev candidates with a train80 cross-encoder; identical (query, candidate) pairs across
    the span-off and span-on candidate sets are scored once."""
    from .reranker import CrossEncoder, cand_text, query_text, query_text_v2
    con = _concepts()
    col = {"ctx80": "ce_ctx80", "noctx80": "ce_noctx80", "v2": "ce_v2"}[kind]
    path = MOD / {"ctx80": "cross-ctx80", "noctx80": "cross-noctx80", "v2": "cross-v2"}[kind]
    max_len = {"ctx80": C.CROSS_MAX_LEN, "noctx80": C.COMPARATOR_MAX_LEN, "v2": V2_MAX_LEN}[kind]
    tok = _v2_tokenizer() if kind == "v2" else None
    frames = {n: frame(n) for n in ("slice20", "dev")}
    jobs, pairs = [], {}
    for tag in tags:
        for name, df in frames.items():
            for system in C.SYSTEMS:
                d = df[df.coding_system == system].reset_index(drop=True)
                p = CND / tag / f"{name}_{system}.parquet"
                cand = pd.read_parquet(p)
                if kind == "v2":
                    qcache = {m: query_text_v2(system, d.entity.iat[m], d.chunk_text.iat[m], d.section.iat[m],
                                               d.clause.iat[m], d.coents.iat[m], tok, V2_QUERY_BUDGET)
                              for m in cand.mention.unique()}
                else:
                    qcache = {m: query_text(system, d.entity.iat[m], d.chunk_text.iat[m], d.subconcept.iat[m],
                                            kind == "ctx80") for m in cand.mention.unique()}
                keys = [(qcache[m], cand_text(system, con[system].loc[c], a)) for m, c, a in
                        zip(cand.mention, cand.code, cand.best_alias)]
                for k in keys:
                    pairs.setdefault(k, None)
                jobs.append((p, cand, keys))
    uniq = list(pairs)
    log(f"score {kind}: {sum(len(j[2]) for j in jobs)} pairs, {len(uniq)} unique")
    ce = CrossEncoder(str(path), max_len=max_len)
    sc = ce.score([a for a, _ in uniq], [b for _, b in uniq])
    del ce
    _free()
    lut = dict(zip(uniq, sc))
    for p, cand, keys in jobs:
        cand[col] = [lut[k] for k in keys]
        cand.to_parquet(p)


# ----------------------------------------------------------------------------------------------
CTX_FEATURES = ["section", "hist_ix", "fam_ix", "ros_ix", "clause_char_sim", "clause_dense_sim", "coentity_sim"]


def add_context_features(X: pd.DataFrame, d: pd.DataFrame, system: str, con: dict, emb) -> pd.DataFrame:
    """Section (categorical), section x candidate interaction flags, clause similarity (char + SapBERT) and
    max SapBERT similarity to co-entities. No fitted parameters beyond the train80 SapBERT."""
    from sklearn.feature_extraction.text import HashingVectorizer
    from .normalize import clean
    X = X.copy()
    m = X.mention.to_numpy()
    sec = d.section.to_numpy()[m]
    X["section"] = pd.Categorical(sec, categories=SECTIONS)
    pref = con[system].preferred.reindex(X.code).fillna("").to_numpy()
    pl = np.char.lower(pref.astype(str))
    codes = X.code.to_numpy()
    if system == C.ICD:
        hist_c = np.array([c[:3] in ("Z86", "Z87") for c in codes])
        fam_c = np.array([c[:3] in ("Z80", "Z81", "Z82", "Z83", "Z84") for c in codes])
        ros_c = np.array([c[:1] == "R" for c in codes])
    else:
        hist_c = np.array(["history of" in p and "family" not in p for p in pl])
        fam_c = np.array(["family history" in p or "fh:" in p for p in pl])
        ros_c = np.zeros(len(X), bool)
    X["hist_ix"] = ((sec == "history") & hist_c).astype(int)
    X["fam_ix"] = ((sec == "family") & fam_c).astype(int)
    X["ros_ix"] = ((sec == "ros") & ros_c).astype(int)
    hv = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 4), n_features=2 ** 20, alternate_sign=False, norm="l2")
    clauses = [clean(c) for c in d.clause]
    A = hv.transform([clauses[i] for i in m])
    B = hv.transform([clean(p) for p in pref])
    X["clause_char_sim"] = np.asarray(A.multiply(B).sum(1)).ravel()
    ce = emb([d.clause.iat[i] for i in range(len(d))])
    pe = emb(list(pref))
    X["clause_dense_sim"] = np.einsum("ij,ij->i", ce[m], pe)
    co = np.zeros(len(X), np.float32)
    co_lists = d.coents.tolist()
    for i in np.unique(m):
        if co_lists[i]:
            E = emb(co_lists[i][:20])
            rows = np.where(m == i)[0]
            co[rows] = (pe[rows] @ E.T).max(1)
    X["coentity_sim"] = co
    return X


class Embedder:
    """Cached SapBERT embeddings of arbitrary strings (train80 SapBERT by default)."""

    def __init__(self, path=None):
        from .dense import Encoder
        from .normalize import clean
        self.enc, self.clean, self.cache = Encoder(str(path or SAP)), clean, {}

    def __call__(self, texts):
        todo = sorted({t for t in texts if t not in self.cache})
        if todo:
            for t, v in zip(todo, self.enc.encode([self.clean(t) or t.lower() for t in todo]).astype(np.float32)):
                self.cache[t] = v
        return np.stack([self.cache[t] for t in texts]) if texts else np.zeros((0, 768), np.float32)


RUNS = [
    ("current", "off", "ce_ctx80", False, False),
    ("+span_extension", "on", "ce_ctx80", True, False),
    ("+stacker_context", "off", "ce_ctx80", False, True),
    ("+RoundA (span+stacker_context)", "on", "ce_ctx80", True, True),
    ("+RoundB (span+ce_context_v2)", "on", "ce_v2", True, False),
    ("+RoundA+RoundB", "on", "ce_v2", True, True),
]


def stage_eval(only=None):
    from collections import Counter
    from .experiment import top1
    from .features import FEATURES, FeatureBuilder
    from .stack import Stacker
    con = _concepts()
    full_train = load_split("train")
    freq = {s: Counter(full_train.loc[full_train.coding_system == s, "code"]) for s in C.SYSTEMS}
    fb = FeatureBuilder(con, frame("train80"))  # priors from train80 only
    emb = Embedder()
    frames = {n: frame(n) for n in ("slice20", "dev")}
    cache = {}

    def assemble(name, tag, ce_col, ctx):
        key = (name, tag, ce_col, ctx)
        if key in cache:
            return cache[key]
        Xs, Rs = [], []
        for system in C.SYSTEMS:
            d = frames[name][frames[name].coding_system == system].reset_index(drop=True)
            cand = pd.read_parquet(CND / tag / f"{name}_{system}.parquet")
            if ce_col not in cand:
                return None
            cand = cand.assign(ce_ctx=cand[ce_col], ce_noctx=cand["ce_noctx80"])
            X = fb.build(cand, d, system)
            if ctx:
                X = add_context_features(X, d, system, con, emb)
            X["qid"] = [f"{system}:{i}" for i in X.mention]
            X["label"] = (X.code.to_numpy() == d.code.to_numpy()[X.mention.to_numpy()]).astype(int)
            d = d.assign(qid=[f"{system}:{i}" for i in range(len(d))])
            Xs.append(X)
            Rs.append(d)
        cache[key] = (pd.concat(Xs, ignore_index=True), pd.concat(Rs, ignore_index=True))
        return cache[key]

    rows, subrows = [], []
    for name, tag, ce_col, span, ctx in RUNS:
        if only and name not in only:
            continue
        tr, dv = assemble("slice20", tag, ce_col, ctx), assemble("dev", tag, ce_col, ctx)
        if tr is None or dv is None:
            log("skip", name, "(scores missing)")
            continue
        feats = list(FEATURES) + (["via_span_ext"] if span else []) + (CTX_FEATURES if ctx else [])
        Xt, _ = tr
        Xd, Rd = dv
        st = Stacker(feats).fit(Xt)
        Xd = Xd.assign(s=st.score(Xd))
        best = top1(Xd, "s")
        R = Rd.assign(pred=Rd.qid.map(best.code).fillna(""))
        R["ok"] = R.pred == R.code
        R["unseen"] = [freq[s].get(c, 0) == 0 for s, c in zip(R.coding_system, R.code)]
        hit = Xd[Xd.label == 1].groupby("qid").fused_rank.min()
        R["r50"] = R.qid.map(hit).fillna(999) < 50
        acc = {s: float(R[R.coding_system == s].ok.mean()) for s in C.SYSTEMS}
        row = {"run": name, "span_extension": span, "stacker_context": ctx, "ce_context_v2": ce_col == "ce_v2",
               "dev_ICD": acc[C.ICD], "dev_SNOMED": acc[C.SNOMED], "dev_min": min(acc.values()),
               "unseen_acc": float(R[R.unseen].ok.mean()),
               "unseen_ICD": float(R[R.unseen & (R.coding_system == C.ICD)].ok.mean()),
               "unseen_SNOMED": float(R[R.unseen & (R.coding_system == C.SNOMED)].ok.mean()),
               "recall50_ICD": float(R[R.coding_system == C.ICD].r50.mean()),
               "recall50_SNOMED": float(R[R.coding_system == C.SNOMED].r50.mean()),
               "n_slice20_queries": int(Xt.qid.nunique()), "n_dev": len(R)}
        if ctx or span:
            g = dict(zip(feats, st.model.booster_.feature_importance("gain").round(1)))
            row["new_feature_gain"] = json.dumps({f: g[f] for f in feats if f not in FEATURES})
        rows.append(row)
        for sub, g in R[R.coding_system == C.SNOMED].groupby("subconcept"):
            subrows.append({"run": name, "subconcept": sub, "n": len(g), "acc": float(g.ok.mean())})
        log(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k.startswith(("dev", "unseen", "recall"))})
    ab = pd.DataFrame(rows)
    old = OUT / "ablation.csv"
    if old.exists() and only:
        prev = pd.read_csv(old)
        ab = pd.concat([prev[~prev.run.isin(ab.run)], ab], ignore_index=True)
    ab.to_csv(old, index=False)
    sb = pd.DataFrame(subrows)
    if (OUT / "snomed_by_subconcept.csv").exists() and only:
        prev = pd.read_csv(OUT / "snomed_by_subconcept.csv")
        sb = pd.concat([prev[~prev.run.isin(sb.run)], sb], ignore_index=True)
    sb.pivot_table(index="subconcept", columns="run", values="acc").round(4).to_csv(OUT / "snomed_by_subconcept.csv")
    return ab


# ================================================================================================
# VANILLA SETUP (the comparison the user chose): the existing vanilla models are reused read-only
# (models/sapbert-ft, models/cross-ctx, models/cross-noctx, data/processed indexes) and the stacker is
# cross-fitted on dev exactly as in termmap/experiment.py, so `current (vanilla)` reproduces vanilla.
# Only train and dev are touched; test is never read.
# ================================================================================================
VAN = ART / "vanilla"
VCND, VMOD = VAN / "candidates", VAN / "models"
for _p in (VCND / "off", VCND / "on", VMOD):
    _p.mkdir(parents=True, exist_ok=True)


def van_frame(name: str) -> pd.DataFrame:
    return annotate(load_split(name))


def van_cands():
    """span-off dev candidates = copies of the vanilla files (already scored); span-on candidates for train
    (cross-encoder v2 training groups) and dev are generated with the vanilla resources."""
    import shutil
    from .dense import Encoder
    from .retrieval import GoldMemory, Mention, SystemRetriever, load_frontend
    from .train import CAND as VANILLA_CAND
    for system in C.SYSTEMS:
        for split in ("dev", "test"):
            shutil.copyfile(VANILLA_CAND / f"{split}_{system}.parquet", VCND / "off" / f"{split}_{system}.parquet")
    C.seed_everything()
    ab, tg = load_frontend()
    enc = Encoder(str(C.MODELS / "sapbert-ft"))
    mem = GoldMemory(load_split("train"))
    mem.attach_dense(enc)
    frames = {n: van_frame(n) for n in ("dev", "test", "train")}
    for system in C.SYSTEMS:
        R = SystemRetriever(system)
        for name, df in frames.items():
            d = df[df.coding_system == system].reset_index(drop=True)
            ms = [Mention(e, c, g, cl) for e, c, g, cl in zip(d.entity, d.chunk_text, d.group, d.clause)]
            cand = R.retrieve(ms, ab, tg, encoder=enc, memory=mem, use_memory_groups=(name == "train"),
                              span_extension=True)
            cand.to_parquet(VCND / "on" / f"{name}_{system}.parquet")
            log("vanilla cands on", name, system, len(d), len(cand))
        del R
        _free()


def _score_files(kind: str, files: list, model_path, col: str, max_len: int):
    from .reranker import CrossEncoder, cand_text, query_text, query_text_v2
    con = _concepts()
    tok = _v2_tokenizer() if kind == "v2" else None
    jobs, pairs = [], {}
    for p, d, system in files:
        cand = pd.read_parquet(p)
        if kind == "v2":
            qc = {m: query_text_v2(system, d.entity.iat[m], d.chunk_text.iat[m], d.section.iat[m], d.clause.iat[m],
                                   d.coents.iat[m], tok, V2_QUERY_BUDGET) for m in cand.mention.unique()}
        else:
            qc = {m: query_text(system, d.entity.iat[m], d.chunk_text.iat[m], d.subconcept.iat[m], kind == "ctx")
                  for m in cand.mention.unique()}
        keys = [(qc[m], cand_text(system, con[system].loc[c], a)) for m, c, a in zip(cand.mention, cand.code, cand.best_alias)]
        for k in keys:
            pairs.setdefault(k, None)
        jobs.append((p, cand, keys))
    uniq = list(pairs)
    log(f"vanilla score {kind}: {sum(len(j[2]) for j in jobs)} pairs, {len(uniq)} unique")
    ce = CrossEncoder(str(model_path), max_len=max_len)
    lut = dict(zip(uniq, ce.score([a for a, _ in uniq], [b for _, b in uniq])))
    del ce
    _free()
    for p, cand, keys in jobs:
        cand[col] = [lut[k] for k in keys]
        cand.to_parquet(p)


def van_score(kinds=("ctx", "noctx")):
    fr = {n: van_frame(n) for n in ("dev", "test")}
    for kind in kinds:
        tags = ("on",) if kind != "v2" else ("off", "on")  # span-off copies already carry vanilla ce scores
        files = [(VCND / t / f"{n}_{s}.parquet", fr[n][fr[n].coding_system == s].reset_index(drop=True), s)
                 for n in fr for t in tags for s in C.SYSTEMS]
        model, col, ml = {"ctx": (C.MODELS / "cross-ctx", "ce_ctx", C.CROSS_MAX_LEN),
                          "noctx": (C.MODELS / "cross-noctx", "ce_noctx", C.COMPARATOR_MAX_LEN),
                          "v2": (VMOD / "cross-v2", "ce_v2", V2_MAX_LEN)}[kind]
        _score_files(kind, files, model, col, ml)


def van_cross_v2():
    """Cross-encoder with the enriched (context_v2) query, trained on ALL of train exactly like the vanilla
    cross-encoder (same negatives scheme, hyper-parameters, seed and synthetic rows), with span-on
    candidates. Saved to a new folder."""
    from .reranker import V2_TOKENS, build_groups, query_text_v2, train_cross
    from .train import CAND as VANILLA_CAND
    rng = random.Random(C.SEED)
    con = _concepts()
    tr = van_frame("train")
    synth = pd.read_parquet(VANILLA_CAND / "synthetic_rows.parquet")
    tok = _v2_tokenizer()
    qf = lambda q_, sub: query_text_v2(q_["system"], q_["entity"], q_["chunk"], q_["section"], q_["clause"],
                                       q_["coents"], tok, V2_QUERY_BUDGET)
    groups = []
    for system in C.SYSTEMS:
        s = synth[synth.coding_system == system].reset_index(drop=True)
        s["section"], s["clause"], s["coents"] = "none", s.entity, [[] for _ in range(len(s))]
        for df, path in ((tr[tr.coding_system == system].reset_index(drop=True), VCND / "on" / f"train_{system}.parquet"),
                         (s, VANILLA_CAND / f"synth_{system}.parquet")):
            cand = pd.read_parquet(path)
            q = [dict(system=system, entity=r.entity, chunk=r.chunk_text, subconcept=r.subconcept, code=r.code,
                      section=r.section, clause=r.clause, coents=r.coents) for r in df.itertuples()]
            groups += build_groups(q, cand, con, rng, context=True, query_fn=qf)
    log(f"vanilla groups v2: {len(groups)}")
    train_cross(groups, VMOD / "cross-v2", max_len=V2_MAX_LEN, special_tokens=V2_TOKENS)
    _free()


VAN_RUNS = [
    ("current (vanilla)", "off", "ce_ctx", False, False),
    ("+span_extension", "on", "ce_ctx", True, False),
    ("+stacker_context", "off", "ce_ctx", False, True),
    ("+RoundA (span+stacker_context)", "on", "ce_ctx", True, True),
    ("+RoundB (span+ce_context_v2)", "on", "ce_v2", True, False),
    ("+RoundA+RoundB", "on", "ce_v2", True, True),
]


def _van_assemble(split, df, tag, ce_col, ctx, fb, con, emb):
    Xs, Rs = [], []
    for system in C.SYSTEMS:
        d = df[df.coding_system == system].reset_index(drop=True)
        cand = pd.read_parquet(VCND / tag / f"{split}_{system}.parquet")
        if ce_col not in cand or "ce_noctx" not in cand:
            return None
        if "via_span_ext" not in cand:
            cand["via_span_ext"] = 0
        X = fb.build(cand.assign(ce_ctx=cand[ce_col]), d, system)
        if ctx:
            X = add_context_features(X, d, system, con, emb)
        X["qid"] = [f"{system}:{i}" for i in X.mention]
        X["label"] = (X.code.to_numpy() == d.code.to_numpy()[X.mention.to_numpy()]).astype(int)
        X["group"] = d.group.to_numpy()[X.mention.to_numpy()]
        Xs.append(X)
        Rs.append(d.assign(qid=[f"{system}:{i}" for i in range(len(d))]))
    return pd.concat(Xs, ignore_index=True), pd.concat(Rs, ignore_index=True)


def _van_metrics(name, R, X, score_col, freq, span, ctx, ce_col):
    from .experiment import top1
    best = top1(X, score_col)
    R = R.assign(pred=R.qid.map(best.code).fillna(""))
    R["ok"] = R.pred == R.code
    R["unseen"] = [freq[s].get(c, 0) == 0 for s, c in zip(R.coding_system, R.code)]
    hit = X[X.label == 1].groupby("qid").fused_rank.min()
    R["r50"] = R.qid.map(hit).fillna(999) < 50
    acc = {s: float(R[R.coding_system == s].ok.mean()) for s in C.SYSTEMS}
    row = {"run": name, "span_extension": span, "stacker_context": ctx, "ce_context_v2": ce_col == "ce_v2",
           "ICD": acc[C.ICD], "SNOMED": acc[C.SNOMED], "min": min(acc.values()),
           "unseen_acc": float(R[R.unseen].ok.mean()),
           "unseen_ICD": float(R[R.unseen & (R.coding_system == C.ICD)].ok.mean()),
           "unseen_SNOMED": float(R[R.unseen & (R.coding_system == C.SNOMED)].ok.mean()),
           "recall50_ICD": float(R[R.coding_system == C.ICD].r50.mean()),
           "recall50_SNOMED": float(R[R.coding_system == C.SNOMED].r50.mean()), "n": len(R)}
    # same metrics restricted to subconcept == DiagnosisCode rows (training still uses every row)
    D = R[R.subconcept == "DiagnosisCode"]
    dacc = {s: float(D[D.coding_system == s].ok.mean()) for s in C.SYSTEMS}
    row.update({"ICD_diag": dacc[C.ICD], "SNOMED_diag": dacc[C.SNOMED], "min_diag": min(dacc.values()),
                "unseen_acc_diag": float(D[D.unseen].ok.mean()), "n_diag": len(D)})
    return row, R


def van_test(names=None):
    """Final, once-only test evaluation: each configuration's stacker is fitted on ALL of dev and applied to
    test. Test labels are only used to compute these numbers, never to choose anything."""
    from collections import Counter
    from .features import FEATURES, FeatureBuilder
    from .stack import Stacker
    con = _concepts()
    train = load_split("train")
    freq = {s: Counter(train.loc[train.coding_system == s, "code"]) for s in C.SYSTEMS}
    fb = FeatureBuilder(con, train)
    emb = Embedder(C.MODELS / "sapbert-ft")
    dev, test = van_frame("dev"), van_frame("test")
    rows = []
    for k, (name, tag, ce_col, span, ctx) in enumerate(VAN_RUNS):
        if names and name not in names:
            continue
        a = _van_assemble("dev", dev, tag, ce_col, ctx, fb, con, emb)
        b = _van_assemble("test", test, tag, ce_col, ctx, fb, con, emb)
        if a is None or b is None:
            log("skip test", name)
            continue
        feats = list(FEATURES) + (["via_span_ext"] if span else []) + (CTX_FEATURES if ctx else [])
        st = Stacker(feats).fit(a[0])
        Xt, Rt = b
        row, Rt = _van_metrics(name, Rt, Xt.assign(s=st.score(Xt)), "s", freq, span, ctx, ce_col)
        rows.append(row)
        Rt[["entity", "coding_system", "subconcept", "code", "pred", "ok", "unseen"]].to_csv(
            OUT / f"test_predictions_run{k}.csv", index=False)
        log("TEST", name, {kk: round(v, 4) for kk, v in row.items() if isinstance(v, float)})
    pd.DataFrame(rows).to_csv(OUT / "test_results.csv", index=False)


def van_eval(only=None):
    """Stacker cross-fitted on dev (5-fold GroupKFold by chunk); out-of-fold dev accuracy, as in vanilla."""
    from collections import Counter
    from .experiment import top1
    from .features import FEATURES, FeatureBuilder
    from .stack import crossfit
    con = _concepts()
    train = load_split("train")
    freq = {s: Counter(train.loc[train.coding_system == s, "code"]) for s in C.SYSTEMS}
    fb = FeatureBuilder(con, train)
    emb = Embedder(C.MODELS / "sapbert-ft")
    dev = van_frame("dev")
    cache = {}

    def assemble(tag, ce_col, ctx):
        if (tag, ce_col, ctx) not in cache:
            cache[(tag, ce_col, ctx)] = _van_assemble("dev", dev, tag, ce_col, ctx, fb, con, emb)
        return cache[(tag, ce_col, ctx)]

    rows, subrows = [], []
    for name, tag, ce_col, span, ctx in VAN_RUNS:
        if only and name not in only:
            continue
        got = assemble(tag, ce_col, ctx)
        if got is None:
            log("skip", name, "(scores missing)")
            continue
        X, R = got
        feats = list(FEATURES) + (["via_span_ext"] if span else []) + (CTX_FEATURES if ctx else [])
        row, R = _van_metrics(name, R, X.assign(s=crossfit(X, feats)), "s", freq, span, ctx, ce_col)
        row = {("dev_" + k if k in ("ICD", "SNOMED", "min", "ICD_diag", "SNOMED_diag", "min_diag") else k): v
               for k, v in row.items()}
        rows.append(row)
        for sub, g in R[R.coding_system == C.SNOMED].groupby("subconcept"):
            subrows.append({"run": name, "subconcept": sub, "n": len(g), "acc": float(g.ok.mean())})
        log(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
                   if k.startswith(("dev", "unseen", "recall"))})
    ab = pd.DataFrame(rows)
    path = OUT / "ablation.csv"
    if path.exists() and only is None:
        pass
    elif path.exists():
        prev = pd.read_csv(path)
        ab = pd.concat([prev[~prev.run.isin(ab.run)], ab], ignore_index=True)
    order = {r[0]: i for i, r in enumerate(VAN_RUNS)}
    ab = ab.sort_values("run", key=lambda s: s.map(order)).reset_index(drop=True)
    ab.to_csv(path, index=False)
    sp = OUT / "snomed_by_subconcept_long.csv"
    sb = pd.DataFrame(subrows)
    if sp.exists() and only is not None:
        prev = pd.read_csv(sp)
        sb = pd.concat([prev[~prev.run.isin(sb.run)], sb], ignore_index=True)
    sb.to_csv(sp, index=False)
    piv = sb.pivot_table(index="subconcept", columns="run", values="acc").round(4)
    piv.insert(0, "n", sb.groupby("subconcept").n.first())
    piv.to_csv(OUT / "snomed_by_subconcept.csv")
    return ab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="van_cands,van_score,van_evalA,van_crossv2,van_scorev2,van_eval")
    a = ap.parse_args()
    C.seed_everything()
    for st in a.stages.split(","):
        t = time.time()
        log("=== stage", st)
        if st == "split":
            stage_split()
        elif st == "dense":
            stage_dense()
        elif st == "cands":
            stage_cands()
        elif st == "cross":
            stage_cross("ctx80")
            stage_cross("noctx80")
        elif st == "score":
            stage_score("ctx80")
            stage_score("noctx80")
        elif st == "evalA":
            stage_eval(only=[r[0] for r in RUNS if r[2] == "ce_ctx80"])
        elif st == "crossv2":
            stage_cross("v2")
        elif st == "scorev2":
            stage_score("v2", tags=("on",))
        elif st == "eval":
            stage_eval()
        elif st == "van_cands":
            van_cands()
        elif st == "van_score":
            van_score(("ctx", "noctx"))
        elif st == "van_evalA":
            van_eval(only=[r[0] for r in VAN_RUNS if r[2] == "ce_ctx"])
        elif st == "van_crossv2":
            van_cross_v2()
        elif st == "van_scorev2":
            van_score(("v2",))
        elif st == "van_eval":
            van_eval()
        elif st == "van_test":
            van_test()
        elif st == "van_testA":  # Round A configurations on test, before Round B training
            van_test(names=[r[0] for r in VAN_RUNS if r[2] == "ce_ctx"])
        log(f"=== stage {st} done in {(time.time() - t) / 60:.1f} min")


if __name__ == "__main__":
    main()

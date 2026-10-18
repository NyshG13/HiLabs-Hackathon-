"""Production inference: file in -> both codes out, for every row.

Stages are loaded one at a time (retrieval -> cross-encoders -> stacker/joint -> LLM) so the whole
engine fits a 4 GB GPU; on CPU it runs unchanged, only slower.
"""
from __future__ import annotations

import gc
import pickle
import time

import numpy as np
import pandas as pd

from . import config as C
from .data import load_input, load_split


def _free():
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass


# ----------------------------------------------------------------------------------------------
def train_subconcept_model():
    """Small TF-IDF + logistic regression that predicts the CodEx subconcept when the input lacks it."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    tr = load_split("train").drop_duplicates(["entity", "chunk_text"])
    text = tr.entity + " || " + tr.chunk_text.str[:300]
    m = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True),
                      LogisticRegression(max_iter=2000, C=4.0))
    m.fit(text, tr.subconcept)
    (C.MODELS / "subconcept.pkl").write_bytes(pickle.dumps(m))
    return m


def memory_frame(splits: tuple[str, ...]) -> pd.DataFrame:
    return pd.concat([load_split(s) for s in splits], ignore_index=True)


def predict_frame(df: pd.DataFrame, memory_splits=("train",), use_llm: bool | None = None,
                  topk_alternatives: int = 3, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    from .candidates import generate
    from .dense import Encoder
    from .features import FeatureBuilder
    from .reranker import CrossEncoder, cand_text, query_text
    from .resources import load_concepts
    from .retrieval import GoldMemory, SystemRetriever, load_frontend
    from .stack import Bridge, Stacker, apply_newp, joint_marginals, mention_probs

    use_llm = C.USE_LLM if use_llm is None else use_llm
    C.seed_everything()
    timing = {}
    t0 = time.perf_counter()
    settings = pickle.loads((C.MODELS / "stack_settings.pkl").read_bytes())
    con = {s: load_concepts(s).set_index("code") for s in C.SYSTEMS}
    df = df.copy().reset_index(drop=True)
    for col in ("chunk_text", "subconcept", "coding_system", "group"):
        if col not in df:
            df[col] = ""
    df["group"] = np.where(df.group.astype(str).str.len() > 0, df.group, df.chunk_text)

    # one mention per (entity, chunk)
    df["mkey"] = df.entity + "\x1f" + df.chunk_text
    M = df.drop_duplicates("mkey")[["mkey", "entity", "chunk_text", "group"]].reset_index(drop=True)
    M["subconcept"] = ""
    if C.USE_SUBCONCEPT:  # legacy behaviour: subconcept from input, else predicted
        sub_in = df[df.subconcept.str.len() > 0].groupby("mkey").subconcept.agg(
            lambda s: "DiagnosisCode" if "DiagnosisCode" in set(s) else s.iat[0])
        M["subconcept"] = M.mkey.map(sub_in).fillna("")
        if (M.subconcept == "").any():
            p = C.MODELS / "subconcept.pkl"
            sm = pickle.loads(p.read_bytes()) if p.exists() else train_subconcept_model()
            miss = M.subconcept == ""
            M.loc[miss, "subconcept"] = sm.predict(M.entity[miss] + " || " + M.chunk_text[miss].str[:300])
            M["subconcept_predicted"] = miss
    # which mentions get the joint ICD<->SNOMED check: rows requested in both systems, else every mention
    want_pair = set(df[df.coding_system == C.ICD].mkey) & set(df[df.coding_system == C.SNOMED].mkey)
    if not want_pair or (df.coding_system == "").all():
        want_pair = set(M.mkey[M.subconcept == "DiagnosisCode"]) if C.USE_SUBCONCEPT else set(M.mkey)

    ab, tg = load_frontend()
    mem_df = memory_frame(memory_splits)
    enc = Encoder()
    mem = GoldMemory(mem_df)
    mem.attach_dense(enc)
    cands = {}
    for system in C.SYSTEMS:
        R = SystemRetriever(system)
        cands[system] = generate(M, system, R, enc, mem, ab, tg)
        del R
        _free()
    timing["retrieval_s"] = time.perf_counter() - t0
    del enc, mem
    _free()

    t1 = time.perf_counter()
    for name, ctx, colname in (("cross-ctx", True, "ce_ctx"), ("cross-noctx", False, "ce_noctx")):
        ce = CrossEncoder(str(C.MODELS / name), max_len=C.CROSS_MAX_LEN if ctx else C.COMPARATOR_MAX_LEN)
        for system in C.SYSTEMS:
            cand = cands[system]
            qa = [query_text(system, M.entity.iat[m], M.chunk_text.iat[m], M.subconcept.iat[m], ctx) for m in cand.mention]
            cb = [cand_text(system, con[system].loc[c], a) for c, a in zip(cand.code, cand.best_alias)]
            cand[colname] = ce.score(qa, cb)
        del ce
        _free()
    timing["rerank_s"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    fb = FeatureBuilder(con, mem_df)
    stacker = Stacker.load(C.MODELS / "stacker.pkl")
    Xs = []
    for system in C.SYSTEMS:
        X = fb.build(cands[system], M, system)
        X["qid"] = [f"{system}:{m}" for m in X.mention]
        X["s"] = stacker.score(X)
        Xs.append(X)
    X = pd.concat(Xs, ignore_index=True)
    X["p"] = mention_probs(X, "s", settings["T"])
    pairs = [(f"{C.ICD}:{i}", f"{C.SNOMED}:{i}") for i, k in enumerate(M.mkey) if k in want_pair]
    bridge = Bridge(load_split("train"), con)
    newp, agree = joint_marginals(X, pairs, bridge, settings["w"])
    if settings["joint_adopted"]:
        apply_newp(X, newp)
    else:
        X["pj"] = X.p
    del bridge
    _free()
    timing["stack_joint_s"] = time.perf_counter() - t2

    choice = {}
    llm_cfg = settings["llm"]
    t3 = time.perf_counter()
    if use_llm and llm_cfg.get("used"):
        from .llm import Adjudicator
        adj = Adjudicator()
        for q, g in X.groupby("qid", sort=False):
            if g.pj.max() >= llm_cfg["tau"]:
                continue
            system, m = q.split(":")
            g = g.sort_values(["pj", "fused_rank"], ascending=[False, True], kind="stable").head(5)
            opts = [(c, con[system].preferred.get(c, "")) for c in g.code]
            lp = adj.choose(system, M.entity.iat[int(m)], M.chunk_text.iat[int(m)], opts)
            s = np.log(np.clip(g.pj.to_numpy(), 1e-9, 1)) + llm_cfg["alpha"] * lp
            choice[q] = g.code.iat[int(np.argmax(s))]
        del adj
        _free()
    timing["llm_s"] = time.perf_counter() - t3

    # assemble per mention
    out = {}
    for q, g in X.groupby("qid", sort=False):
        g = g.sort_values(["pj", "fused_rank"], ascending=[False, True], kind="stable")
        best = choice.get(q, g.code.iat[0])
        pmax = float(g.pj.max())
        alts = "; ".join(f"{c} ({p:.2f})" for c, p in zip(g.code.head(topk_alternatives), g.pj.head(topk_alternatives)))
        out[q] = (best, float(settings["iso"].predict([pmax])[0]), alts, q in choice)
    for system, tag in ((C.ICD, "icd10cm"), (C.SNOMED, "snomed")):
        res = [out.get(f"{system}:{i}", ("", 0.0, "", False)) for i in range(len(M))]
        M[f"{tag}_code"] = [r[0] for r in res]
        M[f"{tag}_desc"] = [con[system].preferred.get(r[0], "") for r in res]
        M[f"{tag}_confidence"] = [round(r[1], 4) for r in res]
        M[f"{tag}_alternatives"] = [r[2] for r in res]
        M[f"{tag}_llm_adjudicated"] = [r[3] for r in res]
        M[f"{tag}_abstain"] = M[f"{tag}_confidence"] < C.ABSTAIN_THRESHOLD
    M["joint_agree"] = [agree.get(f"{C.ICD}:{i}") for i in range(len(M))]
    keep = [c for c in M.columns if c not in ("entity", "chunk_text", "group", "subconcept")]
    res = df.drop(columns=[c for c in ("subconcept_predicted",) if c in df]).merge(M[keep], on="mkey", how="left")
    if (res.coding_system != "").any():
        res["code"] = np.where(res.coding_system == C.ICD, res.icd10cm_code,
                               np.where(res.coding_system == C.SNOMED, res.snomed_code, ""))
        res["confidence"] = np.where(res.coding_system == C.ICD, res.icd10cm_confidence, res.snomed_confidence)
    timing["total_s"] = time.perf_counter() - t0
    timing["terms"] = len(M)
    timing["ms_per_term"] = 1000 * timing["total_s"] / max(1, len(M))
    if verbose:
        print({k: round(v, 2) if isinstance(v, float) else v for k, v in timing.items()})
    return res.drop(columns=["mkey", "row_id"], errors="ignore"), timing


def predict_file(inp, out, **kw) -> dict:
    df = load_input(inp)
    gold = df["code"].copy() if "code" in df else None
    res, timing = predict_frame(df.drop(columns=["code"], errors="ignore"), **kw)
    if gold is not None:
        res["gold_code"] = gold.values
    res = res.rename(columns={"entity": "clinical_entity"})
    lead = [c for c in ("chunk_id", "clinical_entity", "chunk_text", "coding_system", "subconcept", "icd10cm_code", "snomed_code", "code", "confidence") if c in res]
    res = res[lead + [c for c in res.columns if c not in lead and c != "group"]]
    res.to_csv(out, index=False)
    print(f"wrote {len(res)} rows -> {out}")
    return timing

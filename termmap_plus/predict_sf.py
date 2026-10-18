"""Subconcept-free X+ inference (Round 3 reference state, tag xplus-sf).

  python -m termmap_plus.predict_sf --input eval.csv --output preds.csv [--no-llm]

Input guard: the schema is entity (or clinical_entity) + chunk_text (or chunk_id). Every other column is dropped
in read_input, before anything else runs. The frames the pipeline sees have no subconcept / sub_in column at all
(so any read would raise), X's subconcept switch is off, and the cross-encoder query is round3.sf.query_sf.
Pipeline (recipe artifacts/plus/sf/sf_recipe.json):
  ICD    : X retriever + ICD siblings + convention kNN additions -> X features (SF CE) + v1 traps + Qwen3-Reranker
           top-10 (chosen by the shared stacker) + kNN votes -> ICD-only LightGBM stacker (20 seeds), no gate
  SNOMED : X retriever -> shared LightGBM stacker (20 seeds) [-> diagnosis-tag filter] -> X's Qwen2.5 override
"""
from __future__ import annotations

import termmap_plus  # noqa: F401
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from termmap import config as C
from termmap_plus import core as K

OUT = K.ART / "sf"
RECIPE = OUT / "sf_recipe.json"
SCHEMA = ("entity", "chunk_text", "chunk_id")


# ---- input guard ------------------------------------------------------------------------------
def read_input(path) -> pd.DataFrame:
    path = Path(path)
    raw = (pd.read_excel(path, dtype=str) if path.suffix.lower() in (".xlsx", ".xls")
           else pd.read_csv(path, dtype=str, keep_default_na=False))
    return guard(raw)


def guard(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep only the input schema; everything else (subconcept, coding_system, gold, ...) is dropped unread."""
    ent = "entity" if "entity" in raw else ("clinical_entity" if "clinical_entity" in raw else None)
    if ent is None:
        raise SystemExit("input needs an 'entity' or 'clinical_entity' column")
    df = pd.DataFrame({"entity": raw[ent]})
    for c in ("chunk_text", "chunk_id"):
        df[c] = raw[c] if c in raw else ""
    df = df.fillna("").astype(str).reset_index(drop=True)
    assert list(df.columns) == list(SCHEMA)
    return df


# ---- pieces -----------------------------------------------------------------------------------
def _frames(M: pd.DataFrame) -> dict:
    f = M[["entity", "chunk_text", "chunk_id", "group", "row_id"]].assign(is_diag=True, code="")
    assert not {"subconcept", "sub_in", "coding_system"} & set(f.columns)
    return {s: f.copy() for s in K.SYSTEMS}


def sf_score(pools: dict, tpl: str, fresh: bool = False) -> float:
    """ce_ctx with the SF query; ce_noctx with X's no-context query (never had a subconcept field). Cached."""
    from round3 import sf as SF
    from termmap.reranker import CrossEncoder, cand_text, query_text
    from termmap_plus.pools import CE_CACHE as BASE
    t0 = time.perf_counter()
    con, fr = K.concepts(), K.dev_frames()
    lut = {}
    if not fresh:  # score caches (exact query/candidate text); fresh=True scores every pair
        ctx = pd.read_parquet(SF.CE_CACHE) if SF.CE_CACHE.exists() else pd.DataFrame(columns=["tpl", "q", "c", "score"])
        lut = {("ctx", q, c): v for t, q, c, v in zip(ctx.tpl, ctx.q, ctx.c, ctx.score) if t == tpl}
        if BASE.exists():
            b = pd.read_parquet(BASE)
            b = b[b.kind == "noctx"]
            lut.update({("noctx", q, c): v for q, c, v in zip(b.q, b.c, b.score)})
    keys = {}
    for s, cand in pools.items():
        d = fr[s]
        ct = [cand_text(s, con[s].loc[k], a) for k, a in zip(cand.code, cand.best_alias)]
        keys[(s, "ctx")] = [("ctx", SF.query_sf(s, d.entity.iat[m], d.chunk_text.iat[m], SF.PHRASE[tpl]), c)
                            for m, c in zip(cand.mention, ct)]
        keys[(s, "noctx")] = [("noctx", query_text(s, d.entity.iat[m], "", "", False), c) for m, c in zip(cand.mention, ct)]
    for kind in ("ctx", "noctx"):
        todo = sorted({k for (s, kk), ks in keys.items() if kk == kind for k in ks} - set(lut))
        if todo:
            ce = CrossEncoder(str(C.MODELS / ("cross-ctx" if kind == "ctx" else "cross-noctx")),
                              max_len=C.CROSS_MAX_LEN if kind == "ctx" else C.COMPARATOR_MAX_LEN)
            sc = ce.score([k[1] for k in todo], [k[2] for k in todo])
            del ce
            lut.update({k: float(v) for k, v in zip(todo, sc)})
    for s, cand in pools.items():
        cand["ce_ctx"] = [lut[k] for k in keys[(s, "ctx")]]
        cand["ce_noctx"] = [lut[k] for k in keys[(s, "noctx")]]
    return time.perf_counter() - t0


def x_features(pools: dict) -> pd.DataFrame:
    from termmap.data import load_split
    from termmap.features import FeatureBuilder
    from termmap.retrieval import load_frontend
    from termmap_plus.features_plus import add_features
    if "fb" not in K._cache:
        K._cache["fb"] = FeatureBuilder(K.concepts(), load_split("train"))  # training memory
    Xs = []
    for s, cand in pools.items():
        d = K.dev_frames()[s]
        X = K._cache["fb"].build(cand, d, s)  # d has no subconcept column: tag_compat = 0 (and is not a feature)
        X["qid"] = [f"{s}:{m}" for m in X.mention]
        X["system"], X["label"], X["is_diag"] = s, 0, True
        X["group"] = d.group.to_numpy()[X.mention.to_numpy()]
        Xs.append(X)
    X = pd.concat(Xs, ignore_index=True)
    ab, _ = load_frontend()
    return add_features(X, K.dev_frames(), K.concepts(), ab)


def stackers(rec: dict) -> dict:
    """Shared stacker (both systems' pooled dev+test rows) and ICD-only stacker, 20 seeds each, on the SF tables."""
    p = OUT / f"stackers_{rec['code']}.pkl"
    if p.exists():
        return pickle.loads(p.read_bytes())
    from round3 import sf as SF
    from termmap import stack as S
    XA, _ = SF.tables(rec["tpl"])
    XA, _ = SF.with_tag(XA, rec["tag"], rec["featsA_base"])
    XI = pd.read_parquet(OUT / f"XIq_{rec['tpl']}_{rec['tag']}.parquet")
    if rec["tag"] == "dtag":
        XI, _ = SF.with_tag(XI, "dtag", rec["featsI_base"])
    out = {"A": [], "I": []}
    base = S.LGB_PARAMS["random_state"]
    try:
        for seed in rec["seeds"]:
            S.LGB_PARAMS["random_state"] = seed
            out["A"].append(S.Stacker(rec["featsA"]).fit(XA))
            out["I"].append(S.Stacker(rec["featsI"]).fit(XI))
    finally:
        S.LGB_PARAMS["random_state"] = base
    p.write_bytes(pickle.dumps(out))
    return out


def _tag_col(X: pd.DataFrame, tags: list[str]) -> pd.DataFrame:
    """cand_is_diag_tag from the candidate only; tag list frozen in the recipe (diagnosis-TRAIN gold, 99.5%)."""
    tag = K.concepts()[C.SNOMED].tag
    return X.assign(cand_is_diag_tag=((X.system == C.SNOMED) & X.code.map(tag).isin(set(tags))).astype(int))


class _Gate:
    def __init__(self, cache_file: Path):
        from termmap import llm as L
        L.CACHE_FILE = cache_file  # deterministic outputs, so a cache only saves time
        self.adj = L.Adjudicator()

    def __call__(self, X, frames, tau=0.6, alpha=0.25) -> dict:
        con, out = K.concepts(), {}
        for q, g in X[X.system == C.SNOMED].groupby("qid", sort=False):
            if g.p.max() >= tau:
                continue
            m = int(q.split(":")[1])
            g = g.sort_values(["p", "fused_rank"], ascending=[False, True], kind="stable").head(5)
            lp = self.adj.choose(C.SNOMED, frames[C.SNOMED].entity.iat[m], frames[C.SNOMED].chunk_text.iat[m],
                                 [(c, con[C.SNOMED].preferred.get(c, "")) for c in g.code])
            sc = np.log(np.clip(g.p.to_numpy(), 1e-9, 1)) + alpha * lp
            out[q] = g.code.iat[int(np.argmax(sc))]
        return out


def _memory(splits: list[str]) -> None:
    """Memory refit: diagnosis rows of `splits` become the kNN / fallback memory (retrieval memory: build_pools)."""
    from termmap.data import load_split
    d = pd.concat([load_split(s) for s in splits], ignore_index=True)
    K._cache["dtrain"] = d[d.subconcept.isin(K.flags()["scope_values"])].reset_index(drop=True)  # training rows only


def predict_frame(df: pd.DataFrame, use_llm: bool = True, fresh_dir: Path | None = None,
                  memory: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """fresh_dir: every cache (CE, Qwen3, gate) starts empty there. memory: labelled splits used as memory."""
    import torch
    from round3 import sf as SF
    from termmap_plus import pools as PL
    from termmap_plus import v2_qwen3 as Q
    from termmap_plus import v2b_knn as N
    from termmap_plus.predict import _plug
    C.USE_SUBCONCEPT = False  # X's switch: FeatureBuilder never looks for a subconcept column
    t0 = time.perf_counter()
    rec = json.loads(RECIPE.read_text(encoding="utf8"))
    models = stackers(rec)
    memory = memory or rec["memory_splits"]
    _memory(memory)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    fresh = fresh_dir is not None
    if fresh:
        Path(fresh_dir).mkdir(parents=True, exist_ok=True)
    df = guard(df)
    df["group"] = np.where(df.chunk_id.str.len() > 0, df.chunk_id, df.chunk_text)
    df["row_id"] = [K.row_id(e, c) for e, c in zip(df.entity, df.chunk_text)]
    M = df.drop_duplicates("row_id").reset_index(drop=True)
    frames = _frames(M)
    _plug(frames)
    opts = json.loads((K.ART / "state.json").read_text(encoding="utf8"))["opts"]
    PL.score_pools = lambda pools: sf_score(pools, rec["tpl"], fresh)  # build_pools' scorer -> SF query
    pools, lat = PL.build_pools({**opts, "memory_splits": memory})
    votes = N.knn_votes(frames)
    picd = N.add_knn_candidates({C.ICD: pools[C.ICD].copy()}, votes)[C.ICD]
    sf_score({C.ICD: picd}, rec["tpl"], fresh)
    # shared stacker on the v1 pools (both systems)
    XA = x_features(pools)
    if rec["tag"] == "dtag":
        XA = _tag_col(XA, rec["allowed_tags"])
    XA["s"] = np.mean([m.score(XA) for m in models["A"]], axis=0)
    # ICD: kNN pool, Qwen3 top-10 chosen by the shared score, ICD stacker
    Q.CACHE = (Path(fresh_dir) / "qwen3_cache.parquet") if fresh else OUT / "qwen3_cache_sf.parquet"
    XI = N.knn_features(x_features({C.ICD: picd}), votes)
    if rec["tag"] == "dtag":
        XI = _tag_col(XI, rec["allowed_tags"])
    XI = XI.merge(XA[["qid", "code", "s"]].rename(columns={"s": "s_ref"}), on=["qid", "code"], how="left")
    XI = Q.qwen3_features(XI, "s_ref").drop(columns=["s_ref"])
    XI["s"] = np.mean([m.score(XI) for m in models["I"]], axis=0)
    XI["p"] = K.x_probs(XI)
    # SNOMED: shared score [-> tag filter] -> gate
    XS = XA[XA.system == C.SNOMED].copy()
    if rec["filter"]:
        XS = SF.tag_filter(XS)
    XS["p"] = K.x_probs(XS)
    if fresh:
        (Path(fresh_dir) / "llm_cache.jsonl").unlink(missing_ok=True)
    gfile = (Path(fresh_dir) / "llm_cache.jsonl") if fresh else OUT / "llm_cache_sf_predict.jsonl"
    choice = _Gate(gfile)(XS, frames) if use_llm else {}
    con = K.concepts()
    fallback = {s: K.diag_train()[K.diag_train().coding_system == s].code.value_counts().index[0] for s in K.SYSTEMS}
    best = {}
    for X in (XI, XS):
        for q, g in X.groupby("qid", sort=False):
            g = g.sort_values(["p", "fused_rank"], ascending=[False, True], kind="stable")
            best[q] = (choice.get(q, g.code.iat[0]), float(g.p.iat[0]))
    res = []
    for i in range(len(M)):
        r = {"row_id": M.row_id.iat[i]}
        for s, tag, ct in ((C.ICD, "icd10cm_code", "confidence_icd"), (C.SNOMED, "snomed_code", "confidence_snomed")):
            code, p = best.get(f"{s}:{i}", (fallback[s], 0.0))
            if code not in con[s].index:
                code, p = fallback[s], 0.0
            r[tag], r[ct] = code, round(p, 4)
        res.append(r)
    res = pd.DataFrame(res)
    res["confidence"] = res[["confidence_icd", "confidence_snomed"]].min(axis=1)
    final = df.merge(res, on="row_id", how="left").rename(columns={"entity": "clinical_entity"})
    lead = [c for c in ("chunk_id", "chunk_text") if final[c].str.len().gt(0).any()] or ["chunk_text"]
    final = final[lead + ["clinical_entity", "icd10cm_code", "snomed_code", "confidence_icd", "confidence_snomed", "confidence"]]
    sec = time.perf_counter() - t0
    timing = {"terms": len(M), "seconds": round(sec, 1), "ms_per_term": round(1000 * sec / max(1, len(M)), 1),
              "retrieval_p50_p95_ms": [round(x, 1) for x in lat], "device": str(C.device()), "recipe": rec["code"],
              "llm": use_llm, "memory": memory, "routed_to_gate": len(choice),
              "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else 0}
    return final, timing


def main(argv=None):
    ap = argparse.ArgumentParser(prog="termmap_plus.predict_sf")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--fresh-cache", default=None, help="directory where every cache starts empty")
    ap.add_argument("--memory", default=None, help="labelled splits used as memory (default: recipe, train,dev,test)")
    a = ap.parse_args(argv)
    df = read_input(a.input)
    res, timing = predict_frame(df, use_llm=not a.no_llm, fresh_dir=a.fresh_cache,
                                memory=a.memory.split(",") if a.memory else None)
    con = K.concepts()
    assert res.icd10cm_code.isin(con[C.ICD].index).all() and res.snomed_code.isin(con[C.SNOMED].index).all()
    assert len(res) == len(df)
    res.to_csv(a.output, index=False, lineterminator="\n")
    print(f"wrote {len(res)} rows -> {a.output}  {json.dumps(timing)}")
    return timing


if __name__ == "__main__":
    main()

"""X+ inference on a judges'-format file.

  python -m termmap_plus predict --input eval.csv --output preds.csv [--memory train,dev,test] [--no-llm]

Input: entity (or clinical_entity) + optional chunk_text / chunk_id. Any other column (subconcept,
coding_system, code, display, ...) is dropped unread. Output: one row per input row with both codes.
"""
from __future__ import annotations

import termmap_plus  # noqa: F401
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from termmap import config as C
from termmap_plus import core as K
from termmap_plus.run_part1 import SEEDS, STATE

STACKERS = K.ART / "stackers_plus.pkl"
FORBIDDEN = {"subconcept", "coding_system", "code", "display", "icd10cm_code", "snomed_code"}


def read_input(path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_excel(path, dtype=str) if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(path, dtype=str, keep_default_na=False)
    df = df.rename(columns={"clinical_entity": "entity"}).fillna("")
    if "entity" not in df:
        raise SystemExit("input needs an 'entity' or 'clinical_entity' column")
    keep = ["entity"] + [c for c in ("chunk_text", "chunk_id") if c in df]
    df = df[keep].copy()  # everything else (incl. subconcept / coding_system / gold) is never read
    for c in ("chunk_text", "chunk_id"):
        if c not in df:
            df[c] = ""
    return df


def _plug(frames: dict, empty_pools: bool = True):
    K._cache["dev"] = frames
    if empty_pools:
        K.vanilla_pools = lambda: {s: pd.DataFrame({"mention": pd.Series([], dtype=int)}) for s in K.SYSTEMS}


def stackers(st: dict):
    """X's LightGBM stacker (5 seeds) fitted on ALL dev rows with the kept X+ options and features."""
    if STACKERS.exists():
        return pickle.loads(STACKERS.read_bytes())
    from termmap import stack as S
    from termmap.features import FEATURES
    from termmap.retrieval import load_frontend
    from termmap_plus.features_plus import add_features
    from termmap_plus.pools import build_pools
    from termmap_plus.test_once import use_split
    use_split("dev")
    pools, _ = build_pools(st["opts"])
    X = K.x_features(pools)
    if st.get("part2_features"):
        ab, _ = load_frontend()
        X = add_features(X, K.dev_frames(), K.concepts(), ab)
    feats = list(FEATURES) + st.get("part2_features", [])
    models = []
    base = S.LGB_PARAMS["random_state"]
    for seed in SEEDS:
        S.LGB_PARAMS["random_state"] = seed
        models.append(S.Stacker(feats).fit(X))
    S.LGB_PARAMS["random_state"] = base
    STACKERS.write_bytes(pickle.dumps(models))
    return models


def predict_frame(df: pd.DataFrame, memory_splits=("train",), use_llm: bool = True) -> tuple[pd.DataFrame, dict]:
    from termmap.retrieval import load_frontend
    from termmap_plus.features_plus import add_features
    from termmap_plus.pools import build_pools
    t0 = time.perf_counter()
    st = json.loads(STATE.read_text(encoding="utf8"))
    models = stackers(st)
    df = df.reset_index(drop=True)
    df["group"] = np.where(df.chunk_id.str.len() > 0, df.chunk_id, df.chunk_text)
    df["row_id"] = [K.row_id(e, c) for e, c in zip(df.entity, df.chunk_text)]
    M = df.drop_duplicates("row_id").reset_index(drop=True)
    frames = {s: M.assign(is_diag=True, sub_in=K.CONST_SUB, subconcept=K.CONST_SUB, code="") for s in K.SYSTEMS}
    _plug(frames)
    pools, _ = build_pools({**st["opts"], "memory_splits": list(memory_splits)})
    X = K.x_features(pools)
    if st.get("part2_features"):
        ab, _ = load_frontend()
        X = add_features(X, K.dev_frames(), K.concepts(), ab)
    X["s"] = np.mean([m.score(X) for m in models], axis=0)
    X["p"] = K.x_probs(X)
    gate = st["llm_gate"] if use_llm and _qwen_available() else {s: False for s in K.SYSTEMS}
    choice = K.llm_gate(X, gate)
    con = K.concepts()
    fallback = {s: K.diag_train()[K.diag_train().coding_system == s].code.value_counts().index[0] for s in K.SYSTEMS}
    out = {}
    for q, g in X.groupby("qid", sort=False):
        g = g.sort_values(["p", "fused_rank"], ascending=[False, True], kind="stable")
        out[q] = (choice.get(q, g.code.iat[0]), float(g.p.iat[0]))
    res = []
    for i in range(len(M)):
        r = {"row_id": M.row_id.iat[i]}
        for s, tag in ((C.ICD, "icd10cm"), (C.SNOMED, "snomed")):
            code, p = out.get(f"{s}:{i}", (fallback[s], 0.0))
            if code not in con[s].index:  # every code must exist in the supplied dictionary
                code, p = fallback[s], 0.0
            r[f"{tag}_code"], r[f"confidence_{'icd' if tag == 'icd10cm' else 'snomed'}"] = code, round(p, 4)
        res.append(r)
    res = pd.DataFrame(res)
    res["confidence"] = res[["confidence_icd", "confidence_snomed"]].min(axis=1)
    final = df.merge(res, on="row_id", how="left").rename(columns={"entity": "clinical_entity"})
    lead = [c for c in ("chunk_id", "chunk_text") if c in final and final[c].str.len().gt(0).any()] or ["chunk_text"]
    final = final[lead + ["clinical_entity", "icd10cm_code", "snomed_code", "confidence_icd", "confidence_snomed", "confidence"]]
    timing = {"terms": len(M), "seconds": round(time.perf_counter() - t0, 1),
              "ms_per_term": round(1000 * (time.perf_counter() - t0) / max(1, len(M)), 1), "llm_gate": gate}
    return final, timing


def _qwen_available() -> bool:
    from termmap import config as TC
    return any((Path(TC.MODELS) / "hf_cache" / "hub").glob("models--" + TC.LLM.replace("/", "--")))


def main(args):
    df = read_input(args.input)
    res, timing = predict_frame(df, memory_splits=tuple(args.memory.split(",")), use_llm=not args.no_llm)
    con = K.concepts()
    assert res.icd10cm_code.isin(con[C.ICD].index).all() and res.snomed_code.isin(con[C.SNOMED].index).all()
    assert len(res) == len(df)
    res.to_csv(args.output, index=False)
    print(f"wrote {len(res)} rows -> {args.output}  {timing}")

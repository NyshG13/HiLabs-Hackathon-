"""X+ v2 harness: pooled dev+test evaluation ("pooled dev+test (used for selection)").

The stacker is fitted on dev/test rows, so everything is grouped 5-fold CV over the pool (group = chunk,
fixed folds), with X's LightGBM stacker averaged over fixed seeds. All rows train; diagnosis rows score.
"""
from __future__ import annotations

import termmap_plus  # noqa: F401
import json
import math

import numpy as np
import pandas as pd

from termmap import config as C
from termmap.features import FEATURES
from termmap_plus import core as K

V2 = K.ART / "v2"
V2.mkdir(parents=True, exist_ok=True)
SEEDS20 = tuple(range(13, 33))
SEEDS20_B = tuple(range(101, 121))
LEDGER2_CSV, LEDGER2_MD = K.RES / "ledger_v2.csv", K.RES / "ledger_v2.md"
V1_FEATS = ["extra_words", "extra_frac", "missing_words", "missing_frac", "head_match", "abbr_match"]


def pooled_frames():
    from termmap_plus.test_once import split_frames
    dv, te = split_frames("dev"), split_frames("test")
    out = {}
    for s in K.SYSTEMS:
        a, b = dv[s].assign(split="dev"), te[s].assign(split="test")
        out[s] = pd.concat([a, b], ignore_index=True)
    return out, {s: len(dv[s]) for s in K.SYSTEMS}


def plug_pool():
    from termmap.train import CAND
    frames, off = pooled_frames()
    K._cache["dev"] = frames

    def vp():
        out = {}
        for s in K.SYSTEMS:
            a = pd.read_parquet(CAND / f"dev_{s}.parquet")
            b = pd.read_parquet(CAND / f"test_{s}.parquet").assign(mention=lambda d: d.mention + off[s])
            out[s] = pd.concat([a, b], ignore_index=True)
        return out
    K.vanilla_pools = vp
    return frames


def base_X(rebuild=False) -> pd.DataFrame:
    """v1 final state on the pool: ICD tree/sibling pools + X features + v1 lexical-trap features."""
    p = V2 / "X_pool_v1.parquet"
    if p.exists() and not rebuild:
        plug_pool()
        return pd.read_parquet(p)
    from termmap.retrieval import load_frontend
    from termmap_plus.features_plus import add_features
    from termmap_plus.pools import build_pools
    plug_pool()
    st = json.loads((K.ART / "state.json").read_text(encoding="utf8"))
    pools, lat = build_pools(st["opts"])
    X = K.x_features(pools)
    ab, _ = load_frontend()
    X = add_features(X, K.dev_frames(), K.concepts(), ab)
    X.to_parquet(p)
    (V2 / "latency.json").write_text(json.dumps(lat), encoding="utf8")
    return X


def folds(X: pd.DataFrame) -> np.ndarray:
    p = K.SHARED / "folds_pool.json"
    if p.exists():
        fm = json.loads(p.read_text(encoding="utf8"))
    else:
        from sklearn.model_selection import GroupKFold
        fm = {}
        for k, (_, te) in enumerate(GroupKFold(n_splits=5).split(X, groups=X.group)):
            for g in X.group.iloc[te].unique():
                fm[g] = k
        p.write_text(json.dumps(fm), encoding="utf8")
    return X.group.map(fm).to_numpy()


def oof(X: pd.DataFrame, feats, seeds) -> np.ndarray:
    from termmap import stack as S
    f = folds(X)
    s = np.zeros(len(X))
    base = S.LGB_PARAMS["random_state"]
    try:
        for seed in seeds:
            S.LGB_PARAMS["random_state"] = seed
            for k in range(5):
                tr, te = f != k, f == k
                s[te] += S.Stacker(feats).fit(X[tr]).score(X[te]) / len(seeds)
    finally:
        S.LGB_PARAMS["random_state"] = base
    return s


def evaluate(X: pd.DataFrame, feats, seeds=SEEDS20, gate=True, extra_score=None):
    X = X.copy()
    X["s"] = oof(X, feats, seeds)
    if extra_score is not None:
        X["s"] = X["s"] + extra_score
    X["p"] = K.x_probs(X)
    choice = K.llm_gate(X, {C.ICD: False, C.SNOMED: bool(gate)})
    P = K.predictions(X, override=choice)
    P["split"] = [K.dev_frames()[s].split.iat[int(q.split(":")[1])] for s, q in zip(P.system, P.qid)]
    return P, X


def split_metrics(P):
    return {"pool": K.metrics(P), "dev": K.metrics(P[P.split == "dev"]), "test": K.metrics(P[P.split == "test"])}


def ledger2(step, name, P, prev, status, targets, note=""):
    m = split_metrics(P)
    fixed = broken = 0
    pm = None
    if prev is not None:
        j = P.merge(prev, on="qid", suffixes=("", "_prev"))
        fixed = int(((j.pred == j.gold) & (j.pred_prev != j.gold_prev)).sum())
        broken = int(((j.pred != j.gold) & (j.pred_prev == j.gold_prev)).sum())
        pm = split_metrics(prev)
    keep = None
    if pm is not None and status == "?":
        keep = (m["pool"]["min"] > pm["pool"]["min"]) and (fixed - broken >= math.sqrt(fixed + broken)) and \
               (m["pool"]["unseen"] - pm["pool"]["unseen"] >= -0.015)
        status = "KEPT" if keep else "DROPPED"
    d = lambda k, sp="pool": (m[sp][k] - pm[sp][k]) if pm else 0.0
    row = {"step": step, "name": name, "status": status, "targets": targets,
           "pool_ICD": m["pool"]["ICD"], "pool_SNOMED": m["pool"]["SNOMED"], "pool_min": m["pool"]["min"],
           "pool_unseen": m["pool"]["unseen"], "ceiling_ICD": m["pool"]["ceiling_ICD"], "ceiling_SNOMED": m["pool"]["ceiling_SNOMED"],
           "dev_min": m["dev"]["min"], "test_min": m["test"]["min"], "dev_ICD": m["dev"]["ICD"], "dev_SNOMED": m["dev"]["SNOMED"],
           "test_ICD": m["test"]["ICD"], "test_SNOMED": m["test"]["SNOMED"],
           "d_pool_min": d("min"), "d_pool_ICD": d("ICD"), "d_pool_SNOMED": d("SNOMED"), "d_pool_unseen": d("unseen"),
           "fixed": fixed, "broken": broken, "note": note}
    df = pd.read_csv(LEDGER2_CSV) if LEDGER2_CSV.exists() else pd.DataFrame()
    df = pd.concat([df[df.step.astype(str) != str(step)] if len(df) else df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(LEDGER2_CSV, index=False)
    with LEDGER2_MD.open("w", encoding="utf8") as f:
        f.write("# X+ v2 ledger — pooled dev+test (used for selection); v1 test headline stays the honest number\n\n"
                "| step | name | status | pool ICD | pool SNOMED | pool min | Δmin | pool unseen | dev min | test min | fixed | broken | note |\n"
                "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in df.itertuples():
            f.write(f"| {r.step} | {r.name} | {r.status} | {r.pool_ICD:.3f} | {r.pool_SNOMED:.3f} | {r.pool_min:.3f} | {r.d_pool_min:+.3f} | "
                    f"{r.pool_unseen:.3f} | {r.dev_min:.3f} | {r.test_min:.3f} | {r.fixed} | {r.broken} | {'' if pd.isna(r.note) else r.note} |\n")
    print(f"STEP {step} — {name} — {status} — targets: {targets}\n"
          f"pooled dev+test (used for selection): ICD {m['pool']['ICD']:.3f} (Δ {d('ICD'):+.3f}) | SNOMED {m['pool']['SNOMED']:.3f} "
          f"(Δ {d('SNOMED'):+.3f}) | min {m['pool']['min']:.3f} (Δ {d('min'):+.3f}) | unseen {m['pool']['unseen']:.3f} (Δ {d('unseen'):+.3f}) | "
          f"fixed {fixed} broken {broken}\n  dev: ICD {m['dev']['ICD']:.3f} SNOMED {m['dev']['SNOMED']:.3f} min {m['dev']['min']:.3f} | "
          f"test: ICD {m['test']['ICD']:.3f} SNOMED {m['test']['SNOMED']:.3f} min {m['test']['min']:.3f}" + (f" | {note}" if note else ""), flush=True)
    K.log_decision(f"v2 step {step} {name}: {status} (pool min {m['pool']['min']:.3f}, Δ {d('min'):+.3f}, fixed {fixed} broken {broken}) {note}")
    return keep, m

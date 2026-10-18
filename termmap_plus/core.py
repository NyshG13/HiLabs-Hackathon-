"""Shared plumbing for X+: flags, scope, eval-format dev, fixed CV folds, X evaluator, ledger."""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from termmap import config as C
from termmap.data import load_split

ROOT = C.ROOT
ART = ROOT / "artifacts" / "plus"
SHARED = ROOT / "shared"
RES = ROOT / "results"
for _p in (ART, SHARED, RES):
    _p.mkdir(parents=True, exist_ok=True)
CONST_SUB = "DiagnosisCode"
SYSTEMS = C.SYSTEMS


def flags() -> dict:
    return yaml.safe_load((ROOT / "configs" / "plus.yaml").read_text(encoding="utf8"))


def norm(s: str) -> str:
    return " ".join(str(s).lower().split())


def row_id(entity: str, chunk: str) -> str:
    return hashlib.sha1((norm(entity) + "||" + norm(chunk)).encode("utf8")).hexdigest()[:16]


def log_decision(line: str) -> None:
    with (ROOT / "DECISIONS.md").open("a", encoding="utf8") as f:
        f.write(f"- {time.strftime('%Y-%m-%d %H:%M')} {line}\n")


# ----------------------------------------------------------------------------------------------
# Data views
# ----------------------------------------------------------------------------------------------
_cache: dict = {}


def dev_frames() -> dict[str, pd.DataFrame]:
    """Per system: ALL dev rows (as in X's cached candidates; mention = row position), with is_diag,
    row_id and the subconcept X sees at predict time (constant for diagnosis rows, gold for the rest,
    which are only ever stacker-training rows)."""
    if "dev" not in _cache:
        dev = load_split("dev")
        out = {}
        for s in SYSTEMS:
            d = dev[dev.coding_system == s].reset_index(drop=True)
            d["is_diag"] = d.subconcept == CONST_SUB
            d["row_id"] = [row_id(e, c) for e, c in zip(d.entity, d.chunk_text)]
            d["sub_in"] = np.where(d.is_diag, CONST_SUB, d.subconcept)
            out[s] = d
        _cache["dev"] = out
    return _cache["dev"]


def diag_train() -> pd.DataFrame:
    if "dtrain" not in _cache:
        tr = load_split("train")
        _cache["dtrain"] = tr[tr.subconcept.isin(flags()["scope_values"])].reset_index(drop=True)
    return _cache["dtrain"]


def seen_codes() -> dict[str, set]:
    t = diag_train()
    return {s: set(t.loc[t.coding_system == s, "code"]) for s in SYSTEMS}


def build_step0_files() -> dict:
    tr, dev, te = load_split("train"), load_split("dev"), load_split("test")
    full = pd.concat([tr, dev, te], ignore_index=True)
    ct = pd.crosstab(full.subconcept, full.coding_system)
    scope = flags()["scope_values"]
    n_in = int(full.subconcept.isin(scope).sum())
    icd_out = int(((full.coding_system == C.ICD) & ~full.subconcept.isin(scope)).sum())
    info = {"scope_values": scope, "rows_in_scope": n_in, "icd_rows_outside_scope": icd_out,
            "note": "1 ICD row (train, ProcedureCode) falls outside; dev/test ICD rows are 100% in scope"}
    (SHARED / "scope.json").write_text(json.dumps(info, indent=1), encoding="utf8")
    diag = dev[dev.subconcept.isin(scope)].copy()
    diag["row_id"] = [row_id(e, c) for e, c in zip(diag.entity, diag.chunk_text)]
    ev = diag.drop_duplicates("row_id")[["row_id", "entity", "chunk_text"]]
    ev.to_csv(SHARED / "dev_evalformat.csv", index=False)
    g = diag.assign(system=diag.coding_system.map({C.ICD: "ICD", C.SNOMED: "SNOMED"}))
    g[["row_id", "system", "code"]].rename(columns={"code": "gold"}).to_csv(SHARED / "dev_gold.csv", index=False)
    hashes = {}
    for name, d in (("train", tr), ("dev", dev), ("test", te)):
        hashes[name] = hashlib.sha256("\n".join(sorted(set(d.chunk_text.map(norm)))).encode()).hexdigest()[:12]
    return {"crosstab": ct, "scope": info, "n_eval_rows": len(ev), "gold": g.system.value_counts().to_dict(),
            "split_sha256_12": hashes}


# ----------------------------------------------------------------------------------------------
# Folds (fixed for every step): chunk-group -> fold, derived once from X's vanilla dev feature table
# ----------------------------------------------------------------------------------------------
def fold_map(X_base: pd.DataFrame | None = None) -> dict:
    p = SHARED / "folds.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf8"))
    from sklearn.model_selection import GroupKFold
    fm = {}
    for k, (_, te) in enumerate(GroupKFold(n_splits=5).split(X_base, groups=X_base.group)):
        for g in X_base.group.iloc[te].unique():
            fm[g] = k
    p.write_text(json.dumps(fm), encoding="utf8")
    return fm


# ----------------------------------------------------------------------------------------------
# X evaluator: X's own features + X's own LightGBM stacker, cross-fitted on dev with the fixed folds,
# then (optionally per system) X's Qwen gate. Scored on diagnosis rows.
# ----------------------------------------------------------------------------------------------
def concepts():
    if "con" not in _cache:
        from termmap.resources import load_concepts
        _cache["con"] = {s: load_concepts(s).set_index("code") for s in SYSTEMS}
    return _cache["con"]


def vanilla_pools() -> dict[str, pd.DataFrame]:
    from termmap.train import CAND
    return {s: pd.read_parquet(CAND / f"dev_{s}.parquet") for s in SYSTEMS}


def x_features(pools: dict[str, pd.DataFrame]) -> pd.DataFrame:
    from termmap.features import FeatureBuilder
    if "fb" not in _cache:
        _cache["fb"] = FeatureBuilder(concepts(), load_split("train"))
    fb = _cache["fb"]
    Xs = []
    for s in SYSTEMS:
        d = dev_frames()[s]
        X = fb.build(pools[s], d.assign(subconcept=d.sub_in), s)
        X["qid"] = [f"{s}:{m}" for m in X.mention]
        X["system"] = s
        X["label"] = (X.code.to_numpy() == d.code.to_numpy()[X.mention.to_numpy()]).astype(int)
        X["group"] = d.group.to_numpy()[X.mention.to_numpy()]
        X["is_diag"] = d.is_diag.to_numpy()[X.mention.to_numpy()]
        Xs.append(X)
    return pd.concat(Xs, ignore_index=True)


def x_stacker_oof(X: pd.DataFrame, feats=None, seeds=(13,)) -> np.ndarray:
    """X's LightGBM stacker, out-of-fold with the fixed folds; scores averaged over `seeds` (X's own
    hyper-parameters, only random_state varies) to remove ~+-1pt seed noise from step comparisons."""
    from termmap import stack as S
    from termmap.features import FEATURES
    feats = feats or FEATURES
    fm = fold_map(X)
    folds = X.group.map(fm).to_numpy()
    s = np.zeros(len(X))
    base_seed = S.LGB_PARAMS["random_state"]
    try:
        for seed in seeds:
            S.LGB_PARAMS["random_state"] = seed
            for k in range(5):
                tr, te = folds != k, folds == k
                s[te] += S.Stacker(feats).fit(X[tr]).score(X[te]) / len(seeds)
    finally:
        S.LGB_PARAMS["random_state"] = base_seed
    return s


def x_probs(X: pd.DataFrame, col: str = "s") -> np.ndarray:
    import pickle
    from termmap.stack import mention_probs
    T = pickle.loads((C.MODELS / "stack_settings.pkl").read_bytes())["T"]  # read-only
    return mention_probs(X, col, T)


def llm_gate(X: pd.DataFrame, systems_on: dict, tau: float = 0.6, alpha: float = 0.25) -> dict:
    """X's step-8 gate, reading/writing a private copy of the LLM cache (never the vanilla one)."""
    from termmap import llm as L
    priv = ART / "llm_cache.jsonl"
    if not priv.exists():
        priv.write_bytes((C.CACHE / "llm_cache.jsonl").read_bytes())
    L.CACHE_FILE = priv
    on = [s for s, v in systems_on.items() if v]
    if not on:
        return {}
    if "adj" not in _cache:
        _cache["adj"] = L.Adjudicator()
    adj, con, choice = _cache["adj"], concepts(), {}
    for q, g in X[X.is_diag & X.system.isin(on)].groupby("qid", sort=False):
        if g.p.max() >= tau:
            continue
        s, m = q.split(":")
        d = dev_frames()[s]
        g = g.sort_values(["p", "fused_rank"], ascending=[False, True], kind="stable").head(5)
        lp = adj.choose(s, d.entity.iat[int(m)], d.chunk_text.iat[int(m)],
                        [(c, con[s].preferred.get(c, "")) for c in g.code])
        sc = np.log(np.clip(g.p.to_numpy(), 1e-9, 1)) + alpha * lp
        choice[q] = g.code.iat[int(np.argmax(sc))]
    return choice


def predictions(X: pd.DataFrame, score_col: str = "s", override: dict | None = None) -> pd.DataFrame:
    """One row per diagnosis gold row: system, row_id, gold, pred, conf, in_pool."""
    from termmap.experiment import top1
    best = top1(X, score_col)
    pool_hit = X.groupby("qid").label.max()
    rows = []
    for s in SYSTEMS:
        d = dev_frames()[s]
        for m in np.where(d.is_diag)[0]:
            q = f"{s}:{m}"
            pred = (override or {}).get(q, best.code.get(q, ""))
            rows.append(dict(system=s, qid=q, row_id=d.row_id.iat[m], gold=d.code.iat[m], pred=pred,
                             conf=float(best[score_col].get(q, 0.0)), in_pool=bool(pool_hit.get(q, 0))))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# Ledger
# ----------------------------------------------------------------------------------------------
LEDGER_CSV, LEDGER_MD = RES / "ledger.csv", RES / "ledger.md"


def metrics(P: pd.DataFrame) -> dict:
    seen = seen_codes()
    P = P.assign(ok=P.pred == P.gold, unseen=[g not in seen[s] for s, g in zip(P.system, P.gold)])
    out = {}
    for s, tag in ((C.ICD, "ICD"), (C.SNOMED, "SNOMED")):
        g = P[P.system == s]
        out[tag] = float(g.ok.mean())
        out[f"unseen_{tag}"] = float(g[g.unseen].ok.mean())
        out[f"seen_{tag}"] = float(g[~g.unseen].ok.mean())
        out[f"ceiling_{tag}"] = float(g.in_pool.mean())
    out["min"] = min(out["ICD"], out["SNOMED"])
    out["unseen"] = float(P[P.unseen].ok.mean())
    return out


def ledger(step: str, name: str, P: pd.DataFrame, prev: pd.DataFrame | None, prev_m: dict | None,
           status: str, targets: str, latency: tuple | None = None, note: str = "", cv_std: float | None = None) -> dict:
    m = metrics(P)
    fixed = broken = 0
    if prev is not None:
        j = P.merge(prev, on="qid", suffixes=("", "_prev"))
        fixed = int(((j.pred == j.gold) & (j.pred_prev != j.gold_prev)).sum())
        broken = int(((j.pred != j.gold) & (j.pred_prev == j.gold_prev)).sum())
    d = lambda k: (m[k] - prev_m[k]) if prev_m else 0.0
    row = {"step": step, "name": name, "status": status, "targets": targets, **{k: round(v, 4) for k, v in m.items()},
           "d_ICD": round(d("ICD"), 4), "d_SNOMED": round(d("SNOMED"), 4), "d_min": round(d("min"), 4),
           "d_unseen": round(d("unseen"), 4), "fixed": fixed, "broken": broken,
           "p50_ms": round(latency[0], 1) if latency else None, "p95_ms": round(latency[1], 1) if latency else None,
           "cv_std_min": round(cv_std, 4) if cv_std is not None else None, "note": note}
    df = pd.read_csv(LEDGER_CSV) if LEDGER_CSV.exists() else pd.DataFrame()
    df = pd.concat([df[df.step.astype(str) != str(step)] if len(df) else df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(LEDGER_CSV, index=False)
    with LEDGER_MD.open("w", encoding="utf8") as f:
        f.write("# X+ ledger (diagnosis dev, judges' format)\n\n| step | name | status | ICD | SNOMED | min | Δmin | unseen I/S | ceiling I/S | fixed | broken | p95 ms | note |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in df.itertuples():
            f.write(f"| {r.step} | {r.name} | {r.status} | {r.ICD:.3f} | {r.SNOMED:.3f} | {r.min:.3f} | {r.d_min:+.3f} | "
                    f"{r.unseen_ICD:.3f}/{r.unseen_SNOMED:.3f} | {r.ceiling_ICD:.3f}/{r.ceiling_SNOMED:.3f} | {r.fixed} | {r.broken} | "
                    f"{'' if pd.isna(r.p95_ms) else r.p95_ms} | {'' if pd.isna(r.note) else r.note} |\n")
    print(f"STEP {step} — {name} — {status} — targets: {targets}\n"
          f"ICD {m['ICD']:.3f} (Δ {d('ICD'):+.3f}) | SNOMED {m['SNOMED']:.3f} (Δ {d('SNOMED'):+.3f}) | min {m['min']:.3f} (Δ {d('min'):+.3f}) | "
          f"unseen ICD/SNOMED {m['unseen_ICD']:.3f}/{m['unseen_SNOMED']:.3f} | ceiling ICD/SNOMED {m['ceiling_ICD']:.3f}/{m['ceiling_SNOMED']:.3f} | "
          f"fixed {fixed} broken {broken} | p95 latency {row['p95_ms']} ms" + (f" | {note}" if note else ""), flush=True)
    return m


def keep_rule(m: dict, prev_m: dict, fixed: int, broken: int) -> bool:
    return (m["min"] - prev_m["min"] >= 0.005) and (m["unseen"] - prev_m["unseen"] >= -0.01) and fixed > broken

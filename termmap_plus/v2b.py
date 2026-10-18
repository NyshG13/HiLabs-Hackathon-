"""X+ v2b: per-system decisions. Separate X-style LightGBM stackers per system (20 fixed seeds, pooled
grouped 5-fold CV); one ledger row per system; keep a component for a system only if, on that system's pooled
rows: Δacc > 0, fixed − broken ≥ √(fixed + broken), unseen drop ≤ 1.5 pt."""
from __future__ import annotations

import termmap_plus  # noqa: F401
import json
import math

import numpy as np
import pandas as pd

from termmap import config as C
from termmap.features import FEATURES
from termmap_plus import core as K
from termmap_plus import v2 as V

VB = K.ART / "v2b"
VB.mkdir(parents=True, exist_ok=True)
LED_CSV, LED_MD = K.RES / "ledger_v2b.csv", K.RES / "ledger_v2b.md"
STATE = VB / "state_v2b.json"
TAG = {C.ICD: "ICD", C.SNOMED: "SNOMED"}


def state():
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf8"))
    base = list(FEATURES) + V.V1_FEATS
    return {"feats": {C.ICD: base, C.SNOMED: base}, "prev": None, "components": {C.ICD: [], C.SNOMED: []}}


def save(st):
    STATE.write_text(json.dumps(st), encoding="utf8")


def oof_sys(X: pd.DataFrame, feats: dict, seeds=V.SEEDS20) -> np.ndarray:
    from termmap import stack as S
    f = V.folds(X)
    s = np.zeros(len(X))
    base = S.LGB_PARAMS["random_state"]
    try:
        for sysname, fs in feats.items():
            m = (X.system == sysname).to_numpy()
            Xs, fsys = X[m], f[m]
            out = np.zeros(m.sum())
            for seed in seeds:
                S.LGB_PARAMS["random_state"] = seed
                for k in range(5):
                    tr, te = fsys != k, fsys == k
                    out[te] += S.Stacker(fs).fit(Xs[tr]).score(Xs[te]) / len(seeds)
            s[m] = out
    finally:
        S.LGB_PARAMS["random_state"] = base
    return s


def evaluate(X, feats: dict, seeds=V.SEEDS20, gate_snomed=True):
    X = X.copy()
    X["s"] = oof_sys(X, feats, seeds)
    X["p"] = K.x_probs(X)
    choice = K.llm_gate(X, {C.ICD: False, C.SNOMED: bool(gate_snomed)})
    P = K.predictions(X, override=choice)
    P["split"] = [K.dev_frames()[s].split.iat[int(q.split(":")[1])] for s, q in zip(P.system, P.qid)]
    return P, X


def decide(P, prev, system):
    a, b = P[P.system == system], prev[prev.system == system]
    j = a.merge(b, on="qid", suffixes=("", "_p"))
    fixed = int(((j.pred == j.gold) & (j.pred_p != j.gold_p)).sum())
    broken = int(((j.pred != j.gold) & (j.pred_p == j.gold_p)).sum())
    ma, mb = K.metrics(P), K.metrics(prev)
    t = TAG[system]
    dacc, dun = ma[t] - mb[t], ma[f"unseen_{t}"] - mb[f"unseen_{t}"]
    keep = dacc > 0 and (fixed - broken) >= math.sqrt(fixed + broken) and dun >= -0.015
    return keep, dict(fixed=fixed, broken=broken, dacc=dacc, dunseen=dun)


def ledger(step, name, P, prev, targets, forced=None, note=""):
    """Writes one row per system; returns {system: keep}."""
    ms = V.split_metrics(P)
    df = pd.read_csv(LED_CSV) if LED_CSV.exists() else pd.DataFrame()
    keeps = {}
    for s in K.SYSTEMS:
        t = TAG[s]
        if prev is None:
            keep, d = None, dict(fixed=0, broken=0, dacc=0.0, dunseen=0.0)
        else:
            keep, d = decide(P, prev, s)
        status = forced or ("KEPT" if keep else "DROPPED")
        keeps[s] = keep
        row = {"step": step, "system": t, "name": name, "status": status, "targets": targets,
               "pool_acc": ms["pool"][t], "d_acc": d["dacc"], "pool_unseen": ms["pool"][f"unseen_{t}"], "d_unseen": d["dunseen"],
               "ceiling": ms["pool"][f"ceiling_{t}"], "dev_acc": ms["dev"][t], "test_acc": ms["test"][t],
               "pool_min": ms["pool"]["min"], "fixed": d["fixed"], "broken": d["broken"], "note": note}
        df = pd.concat([df[~((df.step.astype(str) == str(step)) & (df.system == t))] if len(df) else df, pd.DataFrame([row])],
                       ignore_index=True)
        print(f"STEP {step} [{t}] — {name} — {status} — targets: {targets}\n  pooled dev+test (used for selection): {t} "
              f"{ms['pool'][t]:.3f} (Δ {d['dacc']:+.3f}) | unseen {ms['pool'][f'unseen_{t}']:.3f} (Δ {d['dunseen']:+.3f}) | "
              f"ceiling {ms['pool'][f'ceiling_{t}']:.3f} | fixed {d['fixed']} broken {d['broken']} | dev {ms['dev'][t]:.3f} "
              f"test {ms['test'][t]:.3f} | pool min {ms['pool']['min']:.3f}" + (f" | {note}" if note else ""), flush=True)
        K.log_decision(f"v2b {step} [{t}] {name}: {status} (pool {t} {ms['pool'][t]:.3f}, Δ {d['dacc']:+.3f}, "
                       f"fixed {d['fixed']} broken {d['broken']}) {note}")
    df.to_csv(LED_CSV, index=False)
    with LED_MD.open("w", encoding="utf8") as f:
        f.write("# X+ v2b ledger — per system; pooled dev+test (used for selection)\n\n"
                "| step | system | name | status | pool acc | Δacc | unseen | Δunseen | ceiling | dev | test | pool min | fixed | broken | note |\n"
                "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in df.itertuples():
            f.write(f"| {r.step} | {r.system} | {r.name} | {r.status} | {r.pool_acc:.3f} | {r.d_acc:+.3f} | {r.pool_unseen:.3f} | "
                    f"{r.d_unseen:+.3f} | {r.ceiling:.3f} | {r.dev_acc:.3f} | {r.test_acc:.3f} | {r.pool_min:.3f} | {r.fixed} | {r.broken} | "
                    f"{'' if pd.isna(r.note) else r.note} |\n")
    return keeps

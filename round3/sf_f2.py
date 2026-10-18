"""FINAL BUILD F2: CE-2 against the subconcept-free reference (SF: X's context CE without the subconcept field,
tag_compat -> cand_is_diag_tag; round3.sf variant none_dtag). CE-2's query is "{entity} || context: {window}"
(no subconcept, no system), scores from artifacts/plus/ce2/full_pool_scores.parquet (read-only).

  python -m round3.sf_f2 A_replace A_add [B_replace B_add]
    X_replace : X's context CE score (ce_ctx and its ce_* rank/margin/prob features) replaced by CE-2 X
    X_add     : CE-2 X added as features (score, rank, margin, prob, gap) next to the SF context CE
Per-system keep rule vs the SF reference (Round 3 rule): dacc > 0, fixed - broken >= sqrt(fixed + broken),
unseen drop <= 1.5 pt. Both stackers refitted (20 seeds, fixed folds); gate from an EMPTY cache per variant.
"""
from __future__ import annotations

import termmap_plus  # noqa: F401
import json
import math
import sys
import time

import numpy as np
import pandas as pd

from termmap import config as C
from termmap.features import FEATURES
from termmap_plus import core as K
from termmap_plus import v2 as V
from round3 import sf as SF

OUT = SF.OUT
LED_CSV, LED_MD = K.RES / "ledger_sf_f2.csv", K.RES / "ledger_sf_f2.md"
REF = "none_dtag"
FULL = K.ART / "ce2" / "full_pool_scores.parquet"


def attach(X: pd.DataFrame, col: str) -> np.ndarray:
    f = pd.read_parquet(FULL)
    lut = dict(zip(zip(f.row_id, f.system, f.concept_id), f[col]))
    fr = K.dev_frames()
    rid = {q: fr[q.split(":")[0]].row_id.iat[int(q.split(":")[1])] for q in X.qid.unique()}
    v = np.array([lut[k] for k in zip(X.qid.map(rid), X.system, X.code)], dtype=float)  # 100% coverage checked
    return v


def add_feats(X: pd.DataFrame, v: np.ndarray, pre: str) -> tuple[pd.DataFrame, list[str]]:
    X = X.copy()
    X[pre] = v
    g = X.groupby("qid", sort=False)[pre]
    mx = g.transform("max")
    X[pre + "_margin"] = X[pre] - mx
    X[pre + "_rank"] = g.rank(ascending=False, method="first")
    e = np.exp(X[pre] - mx)
    X[pre + "_prob"] = e / e.groupby(X.qid).transform("sum")
    second = g.transform(lambda s: s.nlargest(2).iloc[-1] if len(s) > 1 else s.iloc[0])
    X[pre + "_gap"] = np.where(X[pre] >= mx, X[pre] - second, X[pre] - mx)
    return X, [pre, pre + "_margin", pre + "_rank", pre + "_prob", pre + "_gap"]


def decide(P, ref, s):
    t = "ICD" if s == C.ICD else "SNOMED"
    j = P[P.system == s].merge(ref[ref.system == s], on="qid", suffixes=("", "_r"))
    fixed = int(((j.pred == j.gold) & (j.pred_r != j.gold)).sum())
    broken = int(((j.pred != j.gold) & (j.pred_r == j.gold)).sum())
    m, mr = K.metrics(P), K.metrics(ref)
    a = P[P.system == s]
    r = {"acc": m[t], "d_acc": m[t] - mr[t], "unseen": m[f"unseen_{t}"], "d_unseen": m[f"unseen_{t}"] - mr[f"unseen_{t}"],
         "fixed": fixed, "broken": broken, "min": m["min"], "d_min": m["min"] - mr["min"],
         "dev": float((a[a.split == "dev"].pred == a[a.split == "dev"].gold).mean()),
         "test": float((a[a.split == "test"].pred == a[a.split == "test"].gold).mean())}
    r["keep"] = bool(r["d_acc"] > 0 and fixed - broken >= math.sqrt(fixed + broken) and r["d_unseen"] >= -0.015)
    return r


def ledger(code, name, res, note):
    df = pd.read_csv(LED_CSV) if LED_CSV.exists() else pd.DataFrame()
    rows = [{"step": code, "system": t, "name": name, "status": "KEPT" if r["keep"] else "DROPPED", **{k: r[k] for k in
             ("acc", "d_acc", "unseen", "d_unseen", "dev", "test", "min", "d_min", "fixed", "broken")}, "note": note}
            for t, r in res.items()]
    df = pd.concat([df[df.step.astype(str) != code] if len(df) else df, pd.DataFrame(rows)], ignore_index=True)
    df.to_csv(LED_CSV, index=False)
    with LED_MD.open("w", encoding="utf8") as f:
        f.write("# F2 — CE-2 vs the subconcept-free reference (SF none_dtag) — pooled dev+test, 20 seeds, fixed folds\n\n"
                "Keep rule per system: Δacc > 0, fixed − broken ≥ √(fixed + broken), unseen drop ≤ 1.5 pt.\n\n"
                "| step | system | name | status | pool acc | Δacc | unseen | Δunseen | dev | test | pool min | Δmin | fixed | broken | note |\n"
                "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in df.itertuples():
            f.write(f"| {r.step} | {r.system} | {r.name} | {r.status} | {r.acc:.3f} | {r.d_acc:+.3f} | {r.unseen:.3f} | {r.d_unseen:+.3f} | "
                    f"{r.dev:.3f} | {r.test:.3f} | {r.min:.3f} | {r.d_min:+.3f} | {r.fixed} | {r.broken} | {'' if pd.isna(r.note) else r.note} |\n")


def variant(code: str):
    t0 = time.perf_counter()
    which, how = code.split("_")
    ref = pd.read_parquet(OUT / f"P_{REF}.parquet")
    XA, XI = SF.tables("none")
    st = json.loads((K.ART / "v2c" / "state_v2c.json").read_text(encoding="utf8"))
    fA = list(FEATURES) + st["featsA"]
    fI = list(FEATURES) + V.V1_FEATS + st["featsI_extra"]
    XA, fA = SF.with_tag(XA, "dtag", fA)
    XI, fI = SF.with_tag(XI, "dtag", fI)
    col = "ce2" + which
    if how == "replace":
        XA = SF.recompute_ce(XA.assign(ce_ctx=attach(XA, col)))
        XI = SF.recompute_ce(XI.assign(ce_ctx=attach(XI, col)))
    else:
        XA, f = add_feats(XA, attach(XA, col), "c2" + which.lower())
        XI, _ = add_feats(XI, attach(XI, col), "c2" + which.lower())
        fA, fI = fA + f, fI + f
    gate = SF.Gate(f"f2_{code}")
    Xs = SF.snomed_fit(XA, fA)
    Pi = SF.icd_eval(XI, fI, Xs[["qid", "code", "s"]].rename(columns={"s": "s_ref"}), save=f"f2_{code}")
    Ps, calls = SF.snomed_eval(Xs, gate, False)
    P = SF.add_split(pd.concat([Pi, Ps], ignore_index=True))
    P.to_parquet(OUT / f"P_f2_{code}.parquet")
    res = {t: decide(P, ref, s) for s, t in ((C.ICD, "ICD"), (C.SNOMED, "SNOMED"))}
    name = f"{'replace X context CE with' if how == 'replace' else 'add as features'} CE-2 {which}"
    note = f"gate calls {calls} (empty cache); non-diag top SNOMED {SF.non_diag_rate(P):.3f}; {time.perf_counter() - t0:.0f}s"
    ledger(code, name, res, note)
    print(f"F2 {code}: " + " | ".join(f"{t} {r['acc']:.3f} ({r['d_acc']:+.3f}, {r['fixed']}/{r['broken']}, unseen {r['d_unseen']:+.3f}) "
                                     f"{'KEPT' if r['keep'] else 'DROPPED'}" for t, r in res.items())
          + f" | min {res['ICD']['min']:.3f} | {note}", flush=True)
    (OUT / f"result_f2_{code}.json").write_text(json.dumps({"code": code, "res": res, "feats": {"A": fA, "I": fI}}, indent=1),
                                                encoding="utf8")


if __name__ == "__main__":
    for c in sys.argv[1:]:
        variant(c)

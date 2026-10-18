"""Part 2 (light): 2.2 lexical traps, 2.3 clinical checklist (+ veto), added to X's stacker (5-seed OOF,
fixed folds), then X's per-system LLM gate.  python -m termmap_plus.run_part2"""
import termmap_plus  # noqa: F401
import json

import numpy as np
import pandas as pd

from termmap.features import FEATURES
from termmap.retrieval import load_frontend
from termmap_plus import core as K
from termmap_plus.features_plus import CHECK_FEATURES, TRAP_FEATURES, add_features
from termmap_plus.pools import build_pools
from termmap_plus.run_part1 import SEEDS, STATE


def score(X, feats, gate, veto=False):
    X = X.copy()
    X["s"] = K.x_stacker_oof(X, feats=feats, seeds=SEEDS)
    if veto:
        X.loc[X.veto == 1, "s"] -= 100.0  # hard veto: never the top pick
    X["p"] = K.x_probs(X)
    return K.predictions(X, override=K.llm_gate(X, gate)), X


def step(name, code, P, prev, prev_m, targets, st, lat, note=""):
    j = P.merge(prev, on="qid", suffixes=("", "_prev"))
    fixed = int(((j.pred == j.gold) & (j.pred_prev != j.gold_prev)).sum())
    broken = int(((j.pred != j.gold) & (j.pred_prev == j.gold_prev)).sum())
    m = K.metrics(P)
    keep = K.keep_rule(m, prev_m, fixed, broken)
    K.ledger(code, name, P, prev, prev_m, "KEPT" if keep else "DROPPED", targets, latency=lat, note=note)
    K.log_decision(f"Step {code} {name}: {'KEPT' if keep else 'DROPPED'} (min {prev_m['min']:.3f}->{m['min']:.3f}, "
                   f"unseen {prev_m['unseen']:.3f}->{m['unseen']:.3f}, fixed {fixed} broken {broken}) {note}")
    P.to_parquet(K.ART / f"P_step{code}.parquet")
    return keep, m


def main():
    st = json.loads(STATE.read_text(encoding="utf8"))
    prev = pd.read_parquet(K.ART / st["prev_P"])
    prev_m = K.metrics(prev)
    pools, lat = build_pools(st["opts"])
    X = K.x_features(pools)
    ab, _ = load_frontend()
    X = add_features(X, K.dev_frames(), K.concepts(), ab)
    # sanity: the kept state reproduces with base features
    P0, _ = score(X, FEATURES, st["llm_gate"])
    print("kept-state reproduction:", round(float((P0.pred.values == prev.pred.values).mean()), 4))

    feats = list(FEATURES)
    # 2.2 lexical traps
    P, _ = score(X, feats + TRAP_FEATURES, st["llm_gate"])
    keep, m = step("lexical-trap features (extra/missing words, head noun, abbreviation match)", "2.2", P, prev, prev_m,
                   "nearest-string-is-wrong (e.g. 'pulmonary' for 'hypertension')", st, lat)
    if keep:
        feats += TRAP_FEATURES
        prev, prev_m = P, m
    # 2.3 checklist as features, then hard veto if it removes gold in <=0.5% of diagnosis rows
    P, Xs = score(X, feats + CHECK_FEATURES, st["llm_gate"])
    diag = Xs[Xs.is_diag]
    gold_vetoed = float(diag[(diag.label == 1)].groupby("qid").veto.max().reindex(diag.qid.unique()).fillna(0).mean())
    use_veto = gold_vetoed <= 0.005
    if use_veto:
        P, _ = score(X, feats + CHECK_FEATURES, st["llm_gate"], veto=True)
    keep, m = step(f"clinical checklist slots ({'hard veto' if use_veto else 'soft: veto demoted'})", "2.3", P, prev, prev_m,
                   "laterality / acuity / status / encounter / dm type / stage / with-without mismatches", st, lat,
                   note=f"veto would remove gold in {gold_vetoed:.2%} of diag rows")
    if keep:
        feats += CHECK_FEATURES
        prev, prev_m = P, m
    st["part2_features"] = [f for f in feats if f not in FEATURES]
    st["veto"] = bool(use_veto and "ck_lat" in feats)
    st["prev_P"] = "P_final_dev.parquet"
    prev.to_parquet(K.ART / "P_final_dev.parquet")
    STATE.write_text(json.dumps(st), encoding="utf8")
    print("final dev state:", {k: round(v, 4) for k, v in prev_m.items() if k in ("ICD", "SNOMED", "min", "unseen")})


if __name__ == "__main__":
    main()

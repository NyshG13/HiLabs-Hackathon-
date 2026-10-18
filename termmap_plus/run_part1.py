"""Part 1 steps of the cut-down plan, one at a time, each measured on diagnosis dev (judges' format).

  python -m termmap_plus.run_part1 check      # rebuild pools with every option off == X (sanity + latency)
  python -m termmap_plus.run_part1 1.3        # SNOMED semantic-tag scope
  python -m termmap_plus.run_part1 1.9        # ICD tree + sibling expansion
"""
import termmap_plus  # noqa: F401
import json
import sys

import pandas as pd

from termmap_plus import core as K
from termmap_plus.pools import build_pools

STATE = K.ART / "state.json"
SEEDS = (13, 1, 2, 3, 4)
STEPS = {
    "1.3": ("SNOMED semantic-tag scope (99.5% of diag-train gold) + diag-only memory", "snomed_tag_scope", "filtered_out / wrong-hierarchy distractors"),
    "1.9": ("ICD tree + sibling expansion (top-20: unspecified, leaves<=15, laterality, 7th char)", "icd_siblings", "sibling_gap"),
}


def state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf8"))
    gate = json.loads((K.ART / "state_llm_gate.json").read_text(encoding="utf8"))
    return {"opts": {"pool_size": 64}, "llm_gate": gate, "prev_P": "P_step1_1.parquet"}


def evaluate(opts: dict, gate: dict):
    pools, lat = build_pools(opts)
    X = K.x_features(pools)
    X["s"] = K.x_stacker_oof(X, seeds=SEEDS)
    X["p"] = K.x_probs(X)
    choice = K.llm_gate(X, gate)
    return K.predictions(X, override=choice), lat


def main(step: str):
    st = state()
    prev = pd.read_parquet(K.ART / st["prev_P"])
    prev_m = K.metrics(prev)
    if step == "check":  # rebuilt reference: same retrieval path, options off, 5-seed stacker
        P, lat = evaluate(st["opts"], st["llm_gate"])
        same = (P.pred.values == prev.pred.values).mean()
        print(f"rebuild check: {same:.4f} of predictions identical to step 1.1 (single-seed, cached pools)")
        K.ledger("1.R", "reference for Part 1: rebuilt pools, options off, X stacker 5-seed mean", P, prev, prev_m,
                 "REFERENCE", "removes stacker seed noise from step deltas", latency=lat)
        K.log_decision(f"Part 1 reference 1.R: rebuilt pools + X stacker averaged over seeds {SEEDS}; single-seed refits flip "
                       f"{1 - same:.1%} of predictions, so every Part 1 step is compared against 1.R with the same 5 seeds")
        P.to_parquet(K.ART / "P_step1.R.parquet")
        st["latency"], st["prev_P"] = lat, "P_step1.R.parquet"
        STATE.write_text(json.dumps(st), encoding="utf8")
        return
    name, key, targets = STEPS[step]
    opts = {**st["opts"], key: True}
    P, lat = evaluate(opts, st["llm_gate"])
    j = P.merge(prev, on="qid", suffixes=("", "_prev"))
    fixed = int(((j.pred == j.gold) & (j.pred_prev != j.gold_prev)).sum())
    broken = int(((j.pred != j.gold) & (j.pred_prev == j.gold_prev)).sum())
    m = K.metrics(P)
    keep = K.keep_rule(m, prev_m, fixed, broken)
    K.ledger(step, name, P, prev, prev_m, "KEPT" if keep else "DROPPED", targets, latency=lat)
    K.log_decision(f"Step {step} {name}: {'KEPT' if keep else 'DROPPED'} (min {prev_m['min']:.3f}->{m['min']:.3f}, "
                   f"unseen {prev_m['unseen']:.3f}->{m['unseen']:.3f}, fixed {fixed} broken {broken})")
    P.to_parquet(K.ART / f"P_step{step}.parquet")
    if keep:
        st["opts"], st["prev_P"] = opts, f"P_step{step}.parquet"
    STATE.write_text(json.dumps(st), encoding="utf8")


if __name__ == "__main__":
    main(sys.argv[1])

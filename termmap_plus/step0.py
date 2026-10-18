"""Step 0: scope, eval-format dev, split check, leak fix, X baseline (ledger row 0)."""
import termmap_plus  # noqa: F401  (sets the legacy subconcept path before termmap loads)
import time

import pandas as pd

from termmap_plus import core as K


def main():
    info = K.build_step0_files()
    print(info["crosstab"].to_string())
    print("scope:", info["scope"])
    print("eval-format dev rows:", info["n_eval_rows"], "gold rows:", info["gold"])
    print("split sha256[:12]:", info["split_sha256_12"])
    K.log_decision(f"Step 0 scope = {info['scope']['scope_values']} ({info['scope']['rows_in_scope']} rows; "
                   f"{info['scope']['icd_rows_outside_scope']} ICD row outside, in train only)")
    K.log_decision(f"Step 0 split sha256[:12] train/dev/test = {info['split_sha256_12']} (existing split reused, never regenerated)")
    K.log_decision("Step 0 leak fix: X reads subconcept in the cross-encoder query and stacker feature tag_compat; "
                   "every diagnosis row has the single value DiagnosisCode, so X+ feeds that constant at predict time "
                   "(coding_system is never read: both systems are always predicted)")

    t = time.perf_counter()
    X = K.x_features(K.vanilla_pools())
    X["s"] = K.x_stacker_oof(X)
    X["p"] = K.x_probs(X)
    P_nogate = K.predictions(X)
    choice = K.llm_gate(X, K.flags()["part1"]["llm_gate"])
    P = K.predictions(X, override=choice)
    el = time.perf_counter() - t
    X.to_parquet(K.ART / "X_step0.parquet")
    P.to_parquet(K.ART / "P_step0.parquet")
    P_nogate.to_parquet(K.ART / "P_step0_nogate.parquet")
    K.ledger("0", "X baseline (eval format, constant subconcept, LLM gate on)", P, None, None, "BASELINE",
             "honest starting point", note=f"no-gate min={K.metrics(P_nogate)['min']:.3f}; eval {el:.0f}s on cached X scores")


if __name__ == "__main__":
    main()

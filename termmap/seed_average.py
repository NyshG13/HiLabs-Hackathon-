"""Seed-averaged context_v2 ablation: the LightGBM stacker has ~+-1 point seed noise, so every
configuration is refitted with 5 seeds (cached candidate scores, no retraining) and averaged.
Dev = 5-fold cross-fitted out-of-fold accuracy; test = stacker fitted on all of dev, scored once.
Writes outputs/context_v2/ablation_5seed.csv and test_results_5seed.csv."""
from __future__ import annotations

from collections import Counter

import pandas as pd

from . import config as C
from . import context_v2 as V
from . import stack as S
from .data import load_split
from .features import FEATURES, FeatureBuilder

SEEDS = (13, 1, 2, 3, 4)
KEYS = ["ICD", "SNOMED", "min", "unseen_acc", "ICD_diag", "SNOMED_diag", "min_diag", "unseen_acc_diag"]


def main():
    con = V._concepts()
    train = load_split("train")
    freq = {s: Counter(train.loc[train.coding_system == s, "code"]) for s in C.SYSTEMS}
    fb = FeatureBuilder(con, train)
    emb = V.Embedder(C.MODELS / "sapbert-ft")
    dev, test = V.van_frame("dev"), V.van_frame("test")
    dev_rows, test_rows = [], []
    for name, tag, ce_col, span, ctx in V.VAN_RUNS:
        a = V._van_assemble("dev", dev, tag, ce_col, ctx, fb, con, emb)
        b = V._van_assemble("test", test, tag, ce_col, ctx, fb, con, emb)
        feats = list(FEATURES) + (["via_span_ext"] if span else []) + (V.CTX_FEATURES if ctx else [])
        for seed in SEEDS:
            S.LGB_PARAMS["random_state"] = seed
            Xd, Rd = a
            r, _ = V._van_metrics(name, Rd, Xd.assign(s=S.crossfit(Xd, feats)), "s", freq, span, ctx, ce_col)
            dev_rows.append({**r, "seed": seed})
            st = S.Stacker(feats).fit(Xd)
            Xt, Rt = b
            r, _ = V._van_metrics(name, Rt, Xt.assign(s=st.score(Xt)), "s", freq, span, ctx, ce_col)
            test_rows.append({**r, "seed": seed})
        print(name, "done", flush=True)
    S.LGB_PARAMS["random_state"] = C.SEED
    order = {r[0]: i for i, r in enumerate(V.VAN_RUNS)}
    for rows, fname in ((dev_rows, "ablation_5seed.csv"), (test_rows, "test_results_5seed.csv")):
        df = pd.DataFrame(rows)
        g = df.groupby("run")[KEYS]
        out = g.mean().round(4).join(g.std().round(4).add_suffix("_sd")[["min_sd", "min_diag_sd"]])
        out = out.sort_index(key=lambda s: s.map(order)).reset_index()
        out.to_csv(V.OUT / fname, index=False)
        df.to_csv(V.OUT / fname.replace(".csv", "_perseed.csv"), index=False)
        print(fname)
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()

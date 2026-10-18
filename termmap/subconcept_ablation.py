"""How much does the vanilla pipeline depend on the gold `subconcept` column (absent at evaluation time)?

Dev only, vanilla models read-only, stacker cross-fitted on dev exactly as in termmap/experiment.py:
  gold       subconcept from the labelled file (what every reported number used so far)
  predicted  subconcept predicted from entity + chunk by a TF-IDF/logistic model fitted on train only
  none       subconcept blanked everywhere ("unknown" in the cross-encoder query, tag_compat = 0)
Only the context cross-encoder reads subconcept, so only its scores are recomputed.
Writes outputs/context_v2/subconcept/ablation.csv. Never reads test.

  python -m termmap.subconcept_ablation
"""
from __future__ import annotations

import gc
from collections import Counter

import numpy as np
import pandas as pd

from . import config as C
from .data import load_split

OUT = C.OUTPUTS_V2 / "subconcept"
OUT.mkdir(parents=True, exist_ok=True)


def fit_subconcept_model(train: pd.DataFrame):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    tr = train.drop_duplicates(["entity", "chunk_text"])
    m = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True),
                      LogisticRegression(max_iter=2000, C=4.0))
    m.fit(tr.entity + " || " + tr.chunk_text.str[:300], tr.subconcept)
    return m


def main():
    from .experiment import top1
    from .features import FEATURES, FeatureBuilder
    from .reranker import CrossEncoder, cand_text, query_text
    from .resources import load_concepts
    from .stack import crossfit
    from .train import CAND

    C.seed_everything()
    con = {s: load_concepts(s).set_index("code") for s in C.SYSTEMS}
    train, dev = load_split("train"), load_split("dev")
    sm = fit_subconcept_model(train)
    dev_pred = dev.assign(subconcept=sm.predict(dev.entity + " || " + dev.chunk_text.str[:300]))
    sub_acc = float((dev_pred.subconcept == dev.subconcept).mean())
    print("predicted-subconcept accuracy on dev:", round(sub_acc, 4))
    print(pd.crosstab(dev.subconcept, dev_pred.subconcept, rownames=["gold"], colnames=["pred"]))
    modes = {"gold": dev, "predicted": dev_pred, "none": dev.assign(subconcept="")}

    cands = {s: pd.read_parquet(CAND / f"dev_{s}.parquet") for s in C.SYSTEMS}  # read-only
    ce = CrossEncoder(str(C.MODELS / "cross-ctx"), max_len=C.CROSS_MAX_LEN)
    scores = {}
    for mode, df in modes.items():
        for s in C.SYSTEMS:
            if mode == "gold":
                scores[(mode, s)] = cands[s].ce_ctx.to_numpy()
                continue
            d = df[df.coding_system == s].reset_index(drop=True)
            cand = cands[s]
            q = {m: query_text(s, d.entity.iat[m], d.chunk_text.iat[m], d.subconcept.iat[m], True)
                 for m in cand.mention.unique()}
            scores[(mode, s)] = ce.score([q[m] for m in cand.mention],
                                         [cand_text(s, con[s].loc[c], a) for c, a in zip(cand.code, cand.best_alias)])
            print("rescored", mode, s, flush=True)
    del ce
    gc.collect()

    fb = FeatureBuilder(con, train)
    freq = {s: Counter(train.loc[train.coding_system == s, "code"]) for s in C.SYSTEMS}
    rows = []
    for mode, df in modes.items():
        Xs, Rs = [], []
        for s in C.SYSTEMS:
            d = df[df.coding_system == s].reset_index(drop=True)
            gold_sub = dev[dev.coding_system == s].subconcept.to_numpy()
            X = fb.build(cands[s].assign(ce_ctx=scores[(mode, s)]), d, s)
            X["qid"] = [f"{s}:{i}" for i in X.mention]
            X["label"] = (X.code.to_numpy() == d.code.to_numpy()[X.mention.to_numpy()]).astype(int)
            X["group"] = d.group.to_numpy()[X.mention.to_numpy()]
            Xs.append(X)
            Rs.append(d.assign(qid=[f"{s}:{i}" for i in range(len(d))], gold_sub=gold_sub))
        X, R = pd.concat(Xs, ignore_index=True), pd.concat(Rs, ignore_index=True)
        X["s"] = crossfit(X, FEATURES)
        R["pred"] = R.qid.map(top1(X, "s").code).fillna("")
        R["ok"] = R.pred == R.code
        R["unseen"] = [freq[s].get(c, 0) == 0 for s, c in zip(R.coding_system, R.code)]
        acc = {s: float(R[R.coding_system == s].ok.mean()) for s in C.SYSTEMS}
        D = R[R.gold_sub == "DiagnosisCode"]
        dacc = {s: float(D[D.coding_system == s].ok.mean()) for s in C.SYSTEMS}
        row = {"subconcept_mode": mode, "dev_ICD": acc[C.ICD], "dev_SNOMED": acc[C.SNOMED], "dev_min": min(acc.values()),
               "unseen_acc": float(R[R.unseen].ok.mean()),
               "diag_only_ICD": dacc[C.ICD], "diag_only_SNOMED": dacc[C.SNOMED], "diag_only_min": min(dacc.values()),
               "predicted_subconcept_acc": sub_acc if mode == "predicted" else None}
        for sub, g in R[R.coding_system == C.SNOMED].groupby("gold_sub"):
            row[f"SNOMED_{sub}"] = float(g.ok.mean())
        rows.append(row)
        print(mode, {k: round(v, 4) for k, v in row.items() if isinstance(v, float)}, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "ablation.csv", index=False)
    print("wrote", OUT / "ablation.csv")


if __name__ == "__main__":
    main()

"""X+ v2 steps.  python -m termmap_plus.v2_steps <step>   (base | 2.1 | 2.2 | 2.3 | 2.4 | 2.5)"""
import termmap_plus  # noqa: F401
import json
import sys

import numpy as np
import pandas as pd

from termmap import config as C
from termmap.features import FEATURES
from termmap_plus import core as K
from termmap_plus import v2 as V

STATE2 = V.V2 / "state_v2.json"


def st2():
    if STATE2.exists():
        return json.loads(STATE2.read_text(encoding="utf8"))
    return {"feats": list(FEATURES) + V.V1_FEATS, "seeds": list(V.SEEDS20), "prev": None}


def save(st):
    STATE2.write_text(json.dumps(st), encoding="utf8")


def run_base():
    """v1 state on the pool with the 5-seed stacker (v1 protocol), then v2.1 = 20 seeds + flip test."""
    X = V.base_X()
    st = st2()
    P5, _ = V.evaluate(X, st["feats"], seeds=(13, 1, 2, 3, 4))
    P5.to_parquet(V.V2 / "P_v1_5seed.parquet")
    V.ledger2("v1", "X+ v1 final state, 5-seed stacker (pooled reference)", P5, None, "REFERENCE", "starting point")
    P20, _ = V.evaluate(X, st["feats"], seeds=V.SEEDS20)
    P20b, _ = V.evaluate(X, st["feats"], seeds=V.SEEDS20_B)
    flip20 = float((P20.pred.values != P20b.pred.values).mean())
    flip5 = None
    P5b, _ = V.evaluate(X, st["feats"], seeds=(201, 202, 203, 204, 205))
    flip5 = float((P5.pred.values != P5b.pred.values).mean())
    P20.to_parquet(V.V2 / "P_v2.1.parquet")
    V.ledger2("v2.1", "stacker averaged over 20 fixed seeds", P20, P5, "KEPT (protocol)", "stacker seed noise",
              note=f"flips between two 20-seed sets {flip20:.2%} (two 5-seed sets {flip5:.2%})")
    st["prev"] = "P_v2.1.parquet"
    save(st)


def apply_kept(X, st):
    from termmap.retrieval import load_frontend
    from termmap_plus import v2_features as F
    ab, _ = load_frontend()
    fr = K.dev_frames()
    fns = {"2.3": lambda X: F.tree_features(X, fr, ab), "2.4": lambda X: F.policy_features(X, fr, F.mine_policy()),
           "2.5": lambda X: F.tag_features(X, fr)}
    for code in st.get("kept", []):
        X = fns[code](X)
    return X, fns


def run_22():
    """Where picking fails: gold in pool but not picked (pooled rows, 20-seed stacker)."""
    from termmap_plus.v2_features import bucket
    st = st2()
    X = V.base_X()
    X, _ = apply_kept(X, st)
    P, Xs = V.evaluate(X, st["feats"], seeds=V.SEEDS20)
    Xs[["qid", "code", "s", "p", "label"]].to_parquet(V.V2 / "X_scored_v2.1.parquet")
    miss = P[(P.pred != P.gold) & P.in_pool].copy()
    rank = Xs.assign(r=Xs.groupby("qid").s.rank(ascending=False, method="first"))
    gr = rank[rank.label == 1].set_index("qid").r
    miss["gold_rank"] = miss.qid.map(gr)
    miss["rank_bucket"] = pd.cut(miss.gold_rank, [1, 2, 5, 20, 1e9], labels=["2", "3-5", "6-20", ">20"])
    fr = K.dev_frames()
    miss["entity"] = [fr[s].entity.iat[int(q.split(":")[1])] for s, q in zip(miss.system, miss.qid)]
    miss["bucket"] = [bucket(s, e, g, p) for s, e, g, p in zip(miss.system, miss.entity, miss.gold, miss.pred)]
    con = K.concepts()
    miss["gold_desc"] = [con[s].preferred.get(g, "") for s, g in zip(miss.system, miss.gold)]
    miss["pred_desc"] = [con[s].preferred.get(p, "") for s, p in zip(miss.system, miss.pred)]
    lines = ["# v2.2 picking failures — pooled dev+test (gold in pool, not picked)", ""]
    for s, tag in ((C.ICD, "ICD"), (C.SNOMED, "SNOMED")):
        g = miss[miss.system == s]
        tot = int((P.system == s).sum())
        lines += [f"## {tag}: {len(g)} picking misses of {tot} rows ({len(g) / tot:.1%}); "
                  f"retrieval misses {int(((P.system == s) & ~P.in_pool).sum())}", "",
                  "gold rank: " + ", ".join(f"{k}: {v}" for k, v in g.rank_bucket.value_counts().sort_index().items()), "",
                  "| bucket | n | share |", "|---|---|---|"]
        vc = g.bucket.value_counts()
        for b, n in vc.items():
            lines.append(f"| {b} | {n} | {n / len(g):.1%} |")
        lines.append("")
        for b in vc.index:
            lines.append(f"**{b}** examples:")
            for r in g[g.bucket == b].head(5).itertuples():
                lines.append(f"- `{r.entity[:60]}` gold {r.gold} ({r.gold_desc[:55]}) | pred {r.pred} ({r.pred_desc[:55]}) | gold rank {int(r.gold_rank)}")
            lines.append("")
        print(tag, "top-3 buckets:", vc.head(3).to_dict(), "| gold rank:", g.rank_bucket.value_counts().sort_index().to_dict(), flush=True)
    (K.RES / "picking_failures_v2.md").write_text(chr(10).join(lines), encoding="utf8")
    miss.to_csv(V.V2 / "picking_misses.csv", index=False)


def run_feature_step(code):
    from termmap_plus import v2_features as F
    names = {"2.3": ("ICD tree evidence features", F.TREE_FEATS, "ICD siblings (over/under-specific)"),
             "2.4": ("specificity policy + coding defaults (features)", F.POLICY_FEATS, "over/under-specific, 7th character"),
             "2.5": ("SNOMED semantic-tag classifier P(tag)", F.TAG_FEATS, "SNOMED wrong semantic tag")}
    name, new, targets = names[code]
    st = st2()
    prev = pd.read_parquet(V.V2 / st["prev"])
    X = V.base_X()
    X, fns = apply_kept(X, st)
    X = fns[code](X)
    P, _ = V.evaluate(X, st["feats"] + new, seeds=V.SEEDS20)
    keep, m = V.ledger2(f"v{code}", name, P, prev, "?", targets)
    P.to_parquet(V.V2 / f"P_v{code}.parquet")
    if keep:
        st["feats"] += new
        st.setdefault("kept", []).append(code)
        st["prev"] = f"P_v{code}.parquet"
    save(st)


def run_27():
    from termmap_plus.v2_qwen3 import Q3_FEATS, qwen3_features
    st = st2()
    prev = pd.read_parquet(V.V2 / st["prev"])
    X = V.base_X()
    X, _ = apply_kept(X, st)
    ref = pd.read_parquet(V.V2 / "X_scored_v2.1.parquet")[["qid", "code", "s"]].rename(columns={"s": "s_ref"})
    X = X.merge(ref, on=["qid", "code"], how="left")
    X = qwen3_features(X)
    # Qwen3 alone: argmax P(yes) among the top-10, diagnosis rows
    alone = X[X.is_diag & (X.q3_p >= 0)].sort_values(["qid", "q3_p"], ascending=[True, False]).groupby("qid").head(1)
    acc = {s: float((alone[alone.system == s].label == 1).mean()) for s in K.SYSTEMS}
    P, _ = V.evaluate(X.drop(columns=["s_ref"]), st["feats"] + Q3_FEATS, seeds=V.SEEDS20)
    keep, m = V.ledger2("v2.7", "Qwen3-Reranker-0.6B score on top-10 (zero-shot yes/no)", P, prev, "?",
                        "picking errors (gold at rank 2-5)", note=f"Qwen3 alone top-1 ICD {acc[C.ICD]:.3f} SNOMED {acc[C.SNOMED]:.3f}")
    P.to_parquet(V.V2 / "P_v2.7.parquet")
    if keep:
        st["feats"] += Q3_FEATS
        st.setdefault("kept", []).append("2.7")
        st["prev"] = "P_v2.7.parquet"
    save(st)


if __name__ == "__main__":
    step = sys.argv[1]
    if step in ("base", "2.1"):
        run_base()
    elif step == "2.2":
        run_22()
    elif step == "2.7":
        run_27()
    else:
        run_feature_step(step)

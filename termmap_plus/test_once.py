"""Part 3: score the internal TEST split once (diagnosis rows, judges' format). Approved by the user.
Stackers are fitted on ALL of dev; test labels are only used to compute the numbers below.
  python -m termmap_plus.test_once"""
import termmap_plus  # noqa: F401
import json
import pickle

import numpy as np
import pandas as pd

from termmap import config as C
from termmap.data import load_split
from termmap.features import FEATURES
from termmap.retrieval import load_frontend
from termmap_plus import core as K
from termmap_plus.features_plus import add_features
from termmap_plus.pools import build_pools
from termmap_plus.run_part1 import SEEDS, STATE


def split_frames(name):
    df = load_split(name)
    out = {}
    for s in K.SYSTEMS:
        d = df[df.coding_system == s].reset_index(drop=True)
        d["is_diag"] = d.subconcept == K.CONST_SUB
        d["row_id"] = [K.row_id(e, c) for e, c in zip(d.entity, d.chunk_text)]
        d["sub_in"] = np.where(d.is_diag, K.CONST_SUB, d.subconcept)
        out[s] = d
    return out


def use_split(name):
    """Point the X+ plumbing at another split (dev frames / vanilla pools)."""
    from termmap.train import CAND
    frames = split_frames(name)
    K._cache["dev"] = frames
    K.vanilla_pools = lambda: {s: pd.read_parquet(CAND / f"{name}_{s}.parquet") for s in K.SYSTEMS}


def fit_score(Xd, Xt, feats, seeds):
    from termmap import stack as S
    s = np.zeros(len(Xt))
    base = S.LGB_PARAMS["random_state"]
    for seed in seeds:
        S.LGB_PARAMS["random_state"] = seed
        s += S.Stacker(feats).fit(Xd).score(Xt) / len(seeds)
    S.LGB_PARAMS["random_state"] = base
    return s


def build(name, opts, plus_feats):
    use_split(name)
    pools, lat = build_pools(opts)
    X = K.x_features(pools)
    if plus_feats:
        ab, _ = load_frontend()
        X = add_features(X, K.dev_frames(), K.concepts(), ab)
    return X, lat


def main():
    st = json.loads(STATE.read_text(encoding="utf8"))
    gate_plus = st["llm_gate"]
    plus_feats = st.get("part2_features", [])
    rows, out = [], {}

    # (A) X as-is: vanilla pools, X's saved stacker (fit on all dev, seed 13), LLM gate on both systems
    use_split("test")
    Xt = K.x_features(K.vanilla_pools())
    stk = pickle.loads((C.MODELS / "stacker.pkl").read_bytes())  # read-only
    Xt["s"] = stk.score(Xt)
    Xt["p"] = K.x_probs(Xt)
    out["X as-is (saved stacker, LLM gate both)"] = K.predictions(Xt, override=K.llm_gate(Xt, {C.ICD: True, C.SNOMED: True}))

    # (B) X reference: rebuilt pools, options off, 5-seed stacker fit on all dev, per-system gate (1.1)
    ref_opts = {"pool_size": 64}
    Xd, _ = build("dev", ref_opts, False)
    Xt, _ = build("test", ref_opts, False)
    Xt["s"] = fit_score(Xd, Xt, FEATURES, SEEDS)
    Xt["p"] = K.x_probs(Xt)
    out["X reference (5-seed, gate ICD off / SNOMED on)"] = K.predictions(Xt, override=K.llm_gate(Xt, gate_plus))

    # (C) X+ final: kept Part 1 options + kept Part 2 features
    Xd, _ = build("dev", st["opts"], bool(plus_feats))
    Xt, lat = build("test", st["opts"], bool(plus_feats))
    feats = list(FEATURES) + plus_feats
    Xt["s"] = fit_score(Xd, Xt, feats, SEEDS)
    Xt["p"] = K.x_probs(Xt)
    out["X+ final"] = K.predictions(Xt, override=K.llm_gate(Xt, gate_plus))

    base = None
    for name, P in out.items():
        m = K.metrics(P)
        rows.append({"run": name, **{k: round(v, 4) for k, v in m.items()}})
        P.to_parquet(K.ART / f"P_TEST_{len(rows)}.parquet")
    df = pd.DataFrame(rows)
    a, c = out["X as-is (saved stacker, LLM gate both)"], out["X+ final"]
    j = c.merge(a, on="qid", suffixes=("", "_a"))
    fixed = int(((j.pred == j.gold) & (j.pred_a != j.gold_a)).sum())
    broken = int(((j.pred != j.gold) & (j.pred_a == j.gold_a)).sum())
    txt = ["TermMap-X+ — internal TEST, scored once (diagnosis rows, judges' format: entity + chunk_text only)",
           f"rows: ICD {int((a.system == C.ICD).sum())} | SNOMED {int((a.system == C.SNOMED).sum())}",
           f"kept steps: Part 1 {[k for k, v in st['opts'].items() if v is True]} | LLM gate {gate_plus} | Part 2 features {plus_feats}",
           "", df.to_string(index=False), "",
           f"X+ final vs X as-is: fixed {fixed}, broken {broken}; latency p50/p95 {lat[0]:.0f}/{lat[1]:.0f} ms/term (includes scoring uncached pairs)"]
    (K.RES / "TEST_RUN.txt").write_text("\n".join(txt), encoding="utf8")
    print("\n".join(txt))
    K.log_decision(f"TEST scored once: " + "; ".join(f"{r['run']}: ICD {r['ICD']:.3f} SNOMED {r['SNOMED']:.3f} min {r['min']:.3f}" for r in rows))


if __name__ == "__main__":
    main()

"""X+ v2b steps.  python -m termmap_plus.v2b_steps <ref|s0|s2>"""
import termmap_plus  # noqa: F401
import sys
import time

import numpy as np
import pandas as pd

from termmap import config as C
from termmap_plus import core as K
from termmap_plus import v2 as V
from termmap_plus import v2b as B


def with_q3(X):
    from termmap_plus.v2_qwen3 import qwen3_features
    ref = pd.read_parquet(V.V2 / "X_scored_v2.1.parquet")[["qid", "code", "s"]].rename(columns={"s": "s_ref"})
    X = X.merge(ref, on=["qid", "code"], how="left")
    return qwen3_features(X).drop(columns=["s_ref"])


def run_ref():
    st = B.state()
    X = V.base_X()
    P, _ = B.evaluate(X, st["feats"])
    P.to_parquet(B.VB / "P_ref.parquet")
    prev21 = pd.read_parquet(V.V2 / "P_v2.1.parquet")
    B.ledger("ref", "v2 state with SEPARATE per-system stackers (20 seeds)", P, None, "per-system baseline", forced="REFERENCE",
             note=f"shared-stacker v2.1 pool ICD {K.metrics(prev21)['ICD']:.3f} SNOMED {K.metrics(prev21)['SNOMED']:.3f}")
    st["prev"] = "P_ref.parquet"
    B.save(st)


def run_s0():
    from termmap_plus.v2_qwen3 import Q3_FEATS
    st = B.state()
    prev = pd.read_parquet(B.VB / st["prev"])
    X = with_q3(V.base_X())
    feats = {s: st["feats"][s] + Q3_FEATS for s in K.SYSTEMS}
    P, _ = B.evaluate(X, feats)
    keeps = B.ledger("S0", "Qwen3-Reranker-0.6B zero-shot score on top-10 (per-system decision)", P, prev, "picking errors")
    # combined state: per system, take the variant that was kept
    newP = pd.concat([P[P.system == s] if keeps[s] else prev[prev.system == s] for s in K.SYSTEMS], ignore_index=True)
    for s in K.SYSTEMS:
        if keeps[s]:
            st["feats"][s] = feats[s]
            st["components"][s].append("qwen3_rerank")
    newP.to_parquet(B.VB / "P_S0.parquet")
    st["prev"] = "P_S0.parquet"
    B.save(st)
    # CPU latency of ICD-only Qwen3 scoring (10 candidates per term), measured on 5 terms
    import torch
    from termmap_plus import v2_qwen3 as Q
    pairs = ["<Instruct>: " + Q.INSTR[C.ICD] + "\n<Query>: term: chronic kidney disease stage 3 | context: assessment ckd stage 3 egfr 45\n"
             f"<Document>: Chronic kidney disease, stage {i} | Chronic kidney disease (CKD) > CKD stage {i}" for i in range(50)]
    dev0 = C.DEVICE
    C.DEVICE = "cpu"
    t0 = time.perf_counter()
    Q.score_pairs(pairs, bs=10)
    cpu = (time.perf_counter() - t0) / 5
    C.DEVICE = dev0
    print(f"ICD-only Qwen3 CPU latency: {cpu:.2f} s/term (10 candidates/term, incl. model load amortised over 5 terms)", flush=True)
    K.log_decision(f"v2b S0 ICD-only Qwen3-Reranker CPU latency ~{cpu:.2f} s/term (10 pairs/term; includes one-time model load / 5)")


def current_X(st, pools=None):
    """v2b current state features on the pool (base X + kept components)."""
    X = V.base_X() if pools is None else None
    if pools is not None:
        from termmap.retrieval import load_frontend
        from termmap_plus.features_plus import add_features
        X = K.x_features(pools)
        ab, _ = load_frontend()
        X = add_features(X, K.dev_frames(), K.concepts(), ab)
    if any("qwen3_rerank" in st["components"][s] for s in K.SYSTEMS):
        X = with_q3(X)
    return X


def run_s0b():
    """X's Qwen2.5 LLM gate for SNOMED, re-tested with the 20-seed stacker (it overrode gold in 17 misses)."""
    st = B.state()
    prev = pd.read_parquet(B.VB / st["prev"])
    X = current_X(st)
    P, _ = B.evaluate(X, st["feats"], gate_snomed=False)
    P = pd.concat([prev[prev.system == C.ICD], P[P.system == C.SNOMED]], ignore_index=True)  # SNOMED-only change
    keeps = B.ledger("S0b", "X's Qwen2.5 LLM gate OFF for SNOMED (20-seed stacker)", P, prev,
                     "gate overriding correct stacker picks (17 of 119 SNOMED misses)")
    if keeps[C.SNOMED]:
        st["gate_snomed"] = False
        st["components"][C.SNOMED].append("llm_gate_off")
        P.to_parquet(B.VB / "P_S0b.parquet")
        st["prev"] = "P_S0b.parquet"
    B.save(st)


def run_s2():
    from termmap_plus import v2b_knn as N
    from termmap_plus.pools import build_pools, score_pools
    st = B.state()
    prev = pd.read_parquet(B.VB / st["prev"])
    V.plug_pool()
    votes = N.knn_votes(K.dev_frames())
    import json as _j
    opts = _j.loads((K.ART / "state.json").read_text(encoding="utf8"))["opts"]
    pp = B.VB / "pools_v1.pkl"
    import pickle
    if pp.exists():
        pools = pickle.loads(pp.read_bytes())
    else:
        pools, _ = build_pools(opts)
        pp.write_bytes(pickle.dumps(pools))
    pools2 = N.add_knn_candidates({s: p.copy() for s, p in pools.items()}, votes)
    score_pools(pools2)
    X = current_X(st, pools2)
    X = N.knn_features(X, votes)
    feats = {s: st["feats"][s] + N.KNN_FEATS for s in K.SYSTEMS}
    P, _ = B.evaluate(X, feats, gate_snomed=st.get("gate_snomed", True))
    keeps = B.ledger("S2", "convention kNN (votes + top-3 voted codes added to pool)", P, prev,
                     "learnable annotator conventions (43% SNOMED / 60% ICD of audited misses)")
    newP = pd.concat([P[P.system == s] if keeps[s] else prev[prev.system == s] for s in K.SYSTEMS], ignore_index=True)
    for s in K.SYSTEMS:
        if keeps[s]:
            st["feats"][s] = feats[s]
            st["components"][s].append("knn")
    newP.to_parquet(B.VB / "P_S2.parquet")
    st["prev"] = "P_S2.parquet"
    B.save(st)


def current_state_X(st):
    """Rebuild the full current v2b feature table (incl. kNN pools/features if kept for any system)."""
    import pickle
    if any("knn" in st["components"][s] for s in K.SYSTEMS):
        from termmap_plus import v2b_knn as N
        from termmap_plus.pools import score_pools
        V.plug_pool()
        votes = N.knn_votes(K.dev_frames())
        pools = pickle.loads((B.VB / "pools_v1.pkl").read_bytes())
        added = N.add_knn_candidates({s: p.copy() for s, p in pools.items()}, votes)
        pools2 = {s: (added[s] if "knn" in st["components"][s] else pools[s]) for s in K.SYSTEMS}  # only where kept
        score_pools(pools2)
        return N.knn_features(current_X(st, pools2), votes)
    return current_X(st)


def run_s3():
    from termmap_plus.v2b_llm import run_llm
    st = B.state()
    prev = pd.read_parquet(B.VB / st["prev"])
    X = current_state_X(st)
    P0, Xs = B.evaluate(X, {C.SNOMED: st["feats"][C.SNOMED]}, gate_snomed=False)  # SNOMED stacker only, old gate off
    llm, times = run_llm(Xs, tau_max=0.8)
    f = V.folds(Xs)
    fold_of = dict(zip(Xs.qid, f))
    taus = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    Ps = P0[P0.system == C.SNOMED].copy()
    Ps["fold"] = Ps.qid.map(fold_of)

    def apply(df, tau):
        return [llm[q][0] if (q in llm and llm[q][1] < tau) else p for q, p in zip(df.qid, df.pred)]
    chosen = {}
    for k in range(5):  # tau tuned inside the folds
        tr = Ps[Ps.fold != k]
        chosen[k] = max(taus, key=lambda t: (float((np.array(apply(tr, t)) == tr.gold.to_numpy()).mean()), -t))
    Ps["pred"] = [llm[q][0] if (q in llm and llm[q][1] < chosen[fk]) else p for q, p, fk in zip(Ps.qid, Ps.pred, Ps.fold)]
    P = pd.concat([prev[prev.system == C.ICD], Ps.drop(columns=["fold"])], ignore_index=True)
    routed = sum(1 for q, fk in zip(Ps.qid, Ps.fold) if q in llm and llm[q][1] < chosen[fk])
    t = np.array(times) if times else np.array([0.0])
    note = (f"Qwen3-8B q4 via Ollama; tau per fold {sorted(set(chosen.values()))}; routed {routed}/{len(Ps)} diag SNOMED rows "
            f"({1000 * routed / len(Ps):.0f} calls per 1,000 rows); new calls {len(times)}, s/call p50 {np.percentile(t, 50):.1f} "
            f"p95 {np.percentile(t, 95):.1f} (GPU partial offload)")
    keeps = B.ledger("S3", "upgraded SNOMED LLM override (Qwen3-8B, kNN examples, JSON choice)", P, prev,
                     "SNOMED picking misses (different concept)", note=note)
    if keeps[C.SNOMED]:
        st["components"][C.SNOMED].append("llm8b_override")
        P.to_parquet(B.VB / "P_S3.parquet")
        st["prev"] = "P_S3.parquet"
    B.save(st)


if __name__ == "__main__":
    {"ref": run_ref, "s0": run_s0, "s0b": run_s0b, "s2": run_s2, "s3": run_s3}[sys.argv[1]]()

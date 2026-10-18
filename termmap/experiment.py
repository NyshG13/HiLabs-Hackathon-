"""End-to-end experiment on the internal split: ablations, comparator, joint three-arm test, LLM gate,
calibration and error analysis. Writes outputs/results.json, outputs/test_predictions.csv and
outputs/errors_test.csv, and saves the fitted stacker / settings used by the production pipeline."""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from . import config as C
from .data import load_split
from .evaluate import error_category, label_ceiling, summarize
from .features import FEATURES, FeatureBuilder
from .resources import load_concepts
from .stack import (Bridge, Stacker, apply_newp, crossfit, fit_temperature, joint_marginals, mention_probs,
                    pair_index, top1)
from .train import CAND

SETTINGS = C.MODELS / "stack_settings.pkl"


def concepts():
    return {s: load_concepts(s).set_index("code") for s in C.SYSTEMS}


def assemble(split: str, fb: FeatureBuilder) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_split(split)
    Xs, Rs = [], []
    for system in C.SYSTEMS:
        d = df[df.coding_system == system].reset_index(drop=True)
        cand = pd.read_parquet(CAND / f"{split}_{system}.parquet")
        X = fb.build(cand, d, system)
        X["qid"] = [f"{split}:{system}:{m}" for m in X.mention]
        X["label"] = (X.code.to_numpy() == d.code.to_numpy()[X.mention.to_numpy()]).astype(int)
        X["group"] = d.group.to_numpy()[X.mention.to_numpy()]
        d["qid"] = [f"{split}:{system}:{m}" for m in range(len(d))]
        Xs.append(X)
        Rs.append(d)
    return pd.concat(Xs, ignore_index=True), pd.concat(Rs, ignore_index=True)


def predict_rows(R: pd.DataFrame, X: pd.DataFrame, col: str, name: str = "pred") -> pd.DataFrame:
    t = top1(X, col)
    R = R.copy()
    R[name] = R.qid.map(t.code).fillna("")
    R[name + "_score"] = R.qid.map(t[col]).fillna(0.0)
    return R


def acc_min(R: pd.DataFrame, col: str = "pred") -> tuple[float, float, float]:
    a = {s: float((g[col] == g.code).mean()) for s, g in R.groupby("coding_system")}
    return a.get(C.ICD, 0.0), a.get(C.SNOMED, 0.0), min(a.values())


def llm_scores(X: pd.DataFrame, R: pd.DataFrame, col: str, con: dict, k: int = 5) -> dict:
    from .llm import Adjudicator
    adj = Adjudicator()
    out = {}
    rows = R.set_index("qid")
    for i, (q, g) in enumerate(X.groupby("qid", sort=False)):
        g = g.sort_values([col, "fused_rank"], ascending=[False, True], kind="stable").head(k)
        r = rows.loc[q]
        opts = [(c, con[r.coding_system].preferred.get(c, "")) for c in g.code]
        out[q] = (list(g.code), adj.choose(r.coding_system, r.entity, r.chunk_text, opts))
        if i % 200 == 0:
            print("llm", i, flush=True)
    del adj
    return out


def apply_llm(X: pd.DataFrame, llm: dict, col: str, tau: float, alpha: float) -> dict:
    """Return {qid: code} for qids whose top prob < tau, re-ranked by log p + alpha * log p_llm."""
    out = {}
    for q, g in X.groupby("qid", sort=False):
        if q not in llm:
            continue
        codes, lp = llm[q]
        p = g.set_index("code")[col]
        if p.max() >= tau:
            continue
        s = [np.log(max(p.get(c, 1e-9), 1e-9)) + alpha * l for c, l in zip(codes, lp)]
        out[q] = codes[int(np.argmax(s))]
    return out


def run(use_llm: bool = C.USE_LLM) -> dict:
    C.seed_everything()
    con = concepts()
    train = load_split("train")
    fb = FeatureBuilder(con, train)
    Xd, Rd = assemble("dev", fb)
    Xt, Rt = assemble("test", fb)
    results = {"label_ceiling(train)": label_ceiling(train), "ablations_dev": {}, "ablations_test": {}}
    for R, X in ((Rd, Xd), (Rt, Xt)):
        R["in_cands"] = R.qid.map(X.groupby("qid").label.max()).fillna(0).astype(bool)

    # ---- ablations (each a top-1 rule over the same candidate set) ----------------------------
    def log_abl(name, Rd_, Rt_, col="pred"):
        results["ablations_dev"][name] = acc_min(Rd_, col)
        results["ablations_test"][name] = acc_min(Rt_, col)
        print(f"{name:40s} dev ICD/SN/min {results['ablations_dev'][name]}  test {results['ablations_test'][name]}")

    log_abl("candidate ceiling", Rd.assign(pred=np.where(Rd.in_cands, Rd.code, "")),
            Rt.assign(pred=np.where(Rt.in_cands, Rt.code, "")))
    log_abl("retrieval only (weighted RRF top-1)", predict_rows(Rd, Xd, "rrf"), predict_rows(Rt, Xt, "rrf"))
    log_abl("dense only top-1", predict_rows(Rd, Xd, "dense"), predict_rows(Rt, Xt, "dense"))
    log_abl("context cross-encoder top-1", predict_rows(Rd, Xd, "ce_ctx"), predict_rows(Rt, Xt, "ce_ctx"))
    log_abl("mention-only cross-encoder top-1", predict_rows(Rd, Xd, "ce_noctx"), predict_rows(Rt, Xt, "ce_noctx"))

    # comparator: R&D cascade = BM25 + SapBERT candidates, mention-only CE + local frequency
    def comp(X, lam):
        Y = X[(X.origin == 0) & ((X.bm25_rank < 999) | (X.dense_rank < 999))].copy()
        Y["cmp"] = Y.ce_noctx + lam * Y.prior_code
        return Y
    lam = max((0.0, 0.25, 0.5, 1.0, 2.0), key=lambda l: acc_min(predict_rows(Rd, comp(Xd, l), "cmp"))[2])
    results["comparator_lambda"] = lam
    Rd_c, Rt_c = predict_rows(Rd, comp(Xd, lam), "cmp"), predict_rows(Rt, comp(Xt, lam), "cmp")
    log_abl("COMPARATOR (R&D cascade re-impl.)", Rd_c, Rt_c)

    # ---- stacker -------------------------------------------------------------------------------
    feats_no_ce2 = [f for f in FEATURES if not f.startswith("ce2")]
    feats_no_ce = [f for f in FEATURES if not f.startswith("ce")]
    for name, fs in (("stacker w/o cross-encoders", feats_no_ce), ("stacker w/o mention-only CE", feats_no_ce2)):
        Xd["tmp"] = crossfit(Xd, fs)
        Xt["tmp"] = Stacker(fs).fit(Xd).score(Xt)
        log_abl(name, predict_rows(Rd, Xd, "tmp"), predict_rows(Rt, Xt, "tmp"))
    Xd["s"] = crossfit(Xd, FEATURES)
    stacker = Stacker(FEATURES).fit(Xd)
    Xt["s"] = stacker.score(Xt)
    stacker.save(C.MODELS / "stacker.pkl")
    imp = dict(zip(FEATURES, stacker.model.booster_.feature_importance("gain").round(1).tolist()))
    results["stacker_feature_gain"] = dict(sorted(imp.items(), key=lambda kv: -kv[1]))
    log_abl("STACKER (full features)", predict_rows(Rd, Xd, "s"), predict_rows(Rt, Xt, "s"))
    T = fit_temperature(Xd, "s")
    Xd["p"], Xt["p"] = mention_probs(Xd, "s", T), mention_probs(Xt, "s", T)

    # ---- joint consistency: three-arm test -------------------------------------------------
    bridge = Bridge(train, con)
    pd_, pt_ = pair_index(Rd), pair_index(Rt)
    grid = {}
    for w in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
        newp, _ = joint_marginals(Xd, pd_, bridge, w)
        apply_newp(Xd, newp)
        grid[w] = acc_min(predict_rows(Rd, Xd, "pj"))
    w_star = max(grid, key=lambda w: (grid[w][2], -w))
    joint = {"w_grid_dev": grid, "w_star": w_star}
    arms = {}
    for split, X, R, pairs in (("dev", Xd, Rd, pd_), ("test", Xt, Rt, pt_)):
        newp, agree = joint_marginals(X, pairs, bridge, w_star)
        apply_newp(X, newp)
        Ri = predict_rows(R, X, "p")
        Rj = predict_rows(R, X, "pj")
        Rj["agree"] = Rj.qid.map(agree)
        Rc = Ri.copy()  # crosswalk: ICD derived from the SNOMED prediction
        sn_pred = dict(zip(Ri.entity + "\x1f" + Ri.chunk_text + "\x1f" + Ri.coding_system, Ri.pred))
        bridge.fill([p for p in Ri[Ri.coding_system == C.SNOMED].pred if p])
        for i, r in Rc[Rc.coding_system == C.ICD].iterrows():
            s = sn_pred.get(r.entity + "\x1f" + r.chunk_text + "\x1f" + C.SNOMED)
            if s:
                Rc.at[i, "pred"] = bridge.crosswalk(s) or r.pred
        paired = set(q for pr in pairs for q in pr)
        sub = lambda Z: Z[Z.qid.isin(paired)]
        lt = lambda Z: summarize(sub(Z), train, con, conf_col=None)
        a = {"independent": acc_min(sub(Ri)), "joint(+agree)": acc_min(sub(Rj)), "crosswalk(ICD<-SNOMED)": acc_min(sub(Rc))}
        a["long_tail_independent"] = {s: lt(Ri)[s]["long_tail_acc(unseen+rare)"] for s in C.SYSTEMS}
        a["long_tail_joint"] = {s: lt(Rj)[s]["long_tail_acc(unseen+rare)"] for s in C.SYSTEMS}
        sj = sub(Rj).dropna(subset=["agree"])
        ok = sj.pred == sj.code
        a["acc_when_agree"] = float(ok[sj.agree >= 1].mean()) if (sj.agree >= 1).any() else None
        a["acc_when_disagree"] = float(ok[sj.agree < 1].mean()) if (sj.agree < 1).any() else None
        a["n_pairs"] = len(pairs)
        # signal ceiling: how often gold ICD is an 'agreeing' target of gold SNOMED
        gold = R.set_index("qid").code
        bridge.fill([gold[s] for _, s in pairs])
        a["gold_pair_agree_rate"] = float(np.mean([bridge.agree(gold[i], gold[s]) >= 1 for i, s in pairs])) if pairs else None
        arms[split] = a
    joint["arms"] = arms
    d_ind, d_j = arms["dev"]["independent"], arms["dev"]["joint(+agree)"]
    lt_ok = all(arms["dev"]["long_tail_joint"][s] >= arms["dev"]["long_tail_independent"][s] - 1e-9 for s in C.SYSTEMS)
    gap = (arms["dev"]["acc_when_agree"] or 0) - (arms["dev"]["acc_when_disagree"] or 0)
    joint["adopt"] = bool(d_j[2] - d_ind[2] >= 0.01 and lt_ok and gap >= 0.10)
    joint["adoption_rule"] = "min-acc gain >= 1pt on paired dev rows AND no long-tail regression AND agree-vs-disagree gap >= 10pt"
    joint["agree_gap_dev"] = gap
    results["joint"] = joint
    col = "pj" if joint["adopt"] else "p"
    if not joint["adopt"]:
        Xd["pj"], Xt["pj"] = Xd.p, Xt.p
    print("joint", json.dumps(joint, default=str)[:600])

    # ---- step 8: LLM adjudication on low-confidence rows ------------------------------------------
    llm_cfg = {"used": False}
    Rd_f, Rt_f = predict_rows(Rd, Xd, "pj"), predict_rows(Rt, Xt, "pj")
    if use_llm:
        import gc, torch
        gc.collect(); torch.cuda.empty_cache()
        Ld, Lt = llm_scores(Xd, Rd, "pj", con), llm_scores(Xt, Rt, "pj", con)
        base = acc_min(Rd_f)[2]
        best = (base, None, None)
        g = {}
        for tau in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.01):
            for alpha in (0.25, 0.5, 1.0, 2.0, 5.0):
                ch = apply_llm(Xd, Ld, "pj", tau, alpha)
                Rx = Rd_f.copy()
                Rx["pred"] = [ch.get(q, p) for q, p in zip(Rx.qid, Rx.pred)]
                a = acc_min(Rx)[2]
                g[f"tau={tau},alpha={alpha}"] = a
                if a > best[0] + 1e-9:
                    best = (a, tau, alpha)
        # LLM alone (argmax over the stacker's top-5) for reference
        Rx = Rd_f.copy(); Rx["pred"] = [Ld[q][0][int(np.argmax(Ld[q][1]))] if q in Ld else p for q, p in zip(Rx.qid, Rx.pred)]
        Ry = Rt_f.copy(); Ry["pred"] = [Lt[q][0][int(np.argmax(Lt[q][1]))] if q in Lt else p for q, p in zip(Ry.qid, Ry.pred)]
        log_abl("LLM alone over stacker top-5", Rx, Ry)
        llm_cfg = {"used": best[1] is not None, "tau": best[1], "alpha": best[2], "dev_min_before": base,
                   "dev_min_after": best[0], "grid": g}
        if best[1] is not None:
            for X, Rf, L in ((Xd, Rd_f, Ld), (Xt, Rt_f, Lt)):
                ch = apply_llm(X, L, "pj", best[1], best[2])
                Rf["pred"] = [ch.get(q, p) for q, p in zip(Rf.qid, Rf.pred)]
                Rf["llm_changed"] = Rf.qid.isin(ch.keys())
    results["llm"] = llm_cfg
    log_abl("FINAL (stacker + joint + LLM gate)", Rd_f, Rt_f)

    # ---- calibration (isotonic on dev out-of-fold top probability) --------------------------------
    pmax_d = Rd_f.qid.map(Xd.groupby("qid").pj.max()).fillna(0).to_numpy()
    pmax_t = Rt_f.qid.map(Xt.groupby("qid").pj.max()).fillna(0).to_numpy()
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(pmax_d, (Rd_f.pred == Rd_f.code).astype(float))
    Rd_f["conf"], Rt_f["conf"] = iso.predict(pmax_d), iso.predict(pmax_t)
    results["dev"] = summarize(Rd_f, train, con)
    results["test"] = summarize(Rt_f, train, con)
    results["comparator_test"] = summarize(Rt_c.assign(conf=Rt_c.pred_score), train, con)
    results["temperature"] = T

    # ---- error analysis on test --------------------------------------------------------------
    from .retrieval import load_frontend
    ab, _ = load_frontend()
    Rt_f["error_type"] = [error_category(r, con, set(ab.table)) for _, r in Rt_f.iterrows()]
    results["test_error_types"] = {s: g.error_type.value_counts().to_dict() for s, g in Rt_f.groupby("coding_system")}
    Rt_f["pred_desc"] = [con[s].preferred.get(p, "") for s, p in zip(Rt_f.coding_system, Rt_f.pred)]
    Rt_f["gold_desc"] = [con[s].preferred.get(c, "") for s, c in zip(Rt_f.coding_system, Rt_f.code)]
    cols = ["entity", "coding_system", "subconcept", "code", "gold_desc", "pred", "pred_desc", "conf", "in_cands", "error_type"]
    Rt_f[cols].to_csv(C.OUTPUTS / "test_predictions.csv", index=False)
    Rt_f[Rt_f.error_type != "correct"][cols + ["chunk_text"]].to_csv(C.OUTPUTS / "errors_test.csv", index=False)

    pickle.dump(dict(T=T, w=w_star if joint["adopt"] else 0.0, joint_adopted=joint["adopt"], llm=llm_cfg, iso=iso),
                SETTINGS.open("wb"))
    (C.OUTPUTS / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf8")
    return results


if __name__ == "__main__":
    run()

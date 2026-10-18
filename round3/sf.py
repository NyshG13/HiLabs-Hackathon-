"""Subconcept-free (SF) mode: at prediction nothing reads `subconcept`, not even the constant "DiagnosisCode".
Subconcept is still used to choose TRAINING rows (diagnosis train for kNN / priors / allowed tags; the stacker's
training rows) and to scope the evaluation to diagnosis gold rows.

  python -m round3.sf check            # text reconstruction + feature recompute reproduce the stored base tables
  python -m round3.sf tables none      # rescore every dev+test pool pair with X's context CE, no subconcept field
  python -m round3.sf run none         # tag options x SNOMED tag filter, vs xplus-v2c-base (non-inferiority)
  python -m round3.sf run diag         # same with the fixed task phrase "diagnosis" in the query template

Variants (CE template t):  t_const  = CE change alone, tag_compat still fed the constant (diagnostic only, not SF)
                           t_drop   = tag_compat dropped                    | t_drop_f   + SNOMED tag filter
                           t_dtag   = tag_compat -> cand_is_diag_tag        | t_dtag_f   + SNOMED tag filter
Gate: X's Qwen2.5 override on routed diagnosis SNOMED rows, from an EMPTY cache file per run (greedy next-token
log-probs, batch 1: bit-identical on GPU, round3.determinism). Writes artifacts/plus/sf/, results/ledger_sf.*.
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
from termmap.normalize import anchor
from termmap.reranker import SYS_NAME, cand_text
from termmap_plus import core as K
from termmap_plus import v2 as V
from termmap_plus import v2b as B

OUT = K.ART / "sf"
OUT.mkdir(parents=True, exist_ok=True)
LED_CSV, LED_MD = K.RES / "ledger_sf.csv", K.RES / "ledger_sf.md"
CE_CACHE = OUT / "ce_cache_sf.parquet"
PHRASE = {"none": None, "diag": "diagnosis"}  # fixed text written in code; no input column is read
SEEDS = V.SEEDS20
CE_FEATS = ["ce", "ce_margin_top", "ce_rank", "ce_prob", "ce_gap_next"]


def query_sf(system: str, mention: str, chunk: str, phrase: str | None) -> str:
    """X's context query with the subconcept field removed: "<system> : <mention> || <anchored window>"
    (X's own no-context query is "<system> : <mention>"). With a phrase: "<system> | diagnosis : ..."."""
    ctx, _ = anchor(mention, chunk, window=160)
    head = SYS_NAME[system] + (f" | {phrase}" if phrase else "")
    return f"{head} : {mention} || {ctx}"


# ---- tables ---------------------------------------------------------------------------------
def _texts(X: pd.DataFrame, qfn, with_sub: bool = False) -> tuple[list[str], list[str]]:
    """Query/candidate text per row. Only `check` (the base, subconcept query) reads sub_in."""
    fr, con = K.dev_frames(), K.concepts()
    qmap = {}
    for q in X.qid.unique():
        s, m = q.split(":")
        d = fr[s]
        e, ch = d.entity.iat[int(m)], d.chunk_text.iat[int(m)]
        qmap[q] = qfn(s, e, ch, d.sub_in.iat[int(m)]) if with_sub else qfn(s, e, ch)
    cmap = {}
    for s, code, a in X[["system", "code", "best_alias"]].drop_duplicates().itertuples(index=False):
        cmap[(s, code, a)] = cand_text(s, con[s].loc[code], a)
    return [qmap[q] for q in X.qid], [cmap[k] for k in zip(X.system, X.code, X.best_alias)]


def recompute_ce(X: pd.DataFrame) -> pd.DataFrame:
    """X's _softmax_rank on ce_ctx, per query (X builds it per system on `mention`; qid is the same grouping)."""
    X = X.copy()
    X["ce"] = X.ce_ctx
    g = X.groupby("qid", sort=False).ce
    mx = g.transform("max")
    X["ce_margin_top"] = X.ce - mx
    X["ce_rank"] = g.rank(ascending=False, method="first")
    e = np.exp(X.ce - mx)
    X["ce_prob"] = e / e.groupby(X.qid).transform("sum")
    second = g.transform(lambda s: s.nlargest(2).iloc[-1] if len(s) > 1 else s.iloc[0])
    X["ce_gap_next"] = np.where(X.ce >= mx, X.ce - second, X.ce - mx)
    return X


def check():
    """Base tables: rebuilding X's (subconcept) query text finds the stored ce_ctx in the X+ CE cache, and the
    recomputed ce_* features equal the stored ones -> the SF rescoring changes only the query text."""
    from termmap.reranker import query_text
    from termmap_plus.pools import CE_CACHE as BASE_CACHE
    XA, XI = _base()
    cache = pd.read_parquet(BASE_CACHE)
    cache = cache[cache.kind == "ctx"]
    lut = dict(zip(zip(cache.q, cache.c), cache.score))
    for name, X in (("XA", XA), ("XI", XI)):
        q, c = _texts(X, lambda s, e, ch, sub: query_text(s, e, ch, sub, True), with_sub=True)
        hit = np.array([(a, b) in lut for a, b in zip(q, c)])
        v = np.array([lut.get((a, b), np.nan) for a, b in zip(q, c)])
        same = np.isclose(v[hit], X.ce_ctx.to_numpy()[hit], atol=1e-4)
        R = recompute_ce(X)
        dif = max(float(np.nanmax(np.abs(R[f].to_numpy() - X[f].to_numpy()))) for f in CE_FEATS)
        print(f"{name}: {len(X)} rows; base query text found in CE cache {hit.mean():.4f}; ce_ctx equal {same.mean():.4f}; "
              f"recomputed ce_* max |diff| {dif:.2e}", flush=True)


def _base():
    from termmap_plus import v2c as VC
    B.LED_CSV, B.LED_MD = LED_CSV, LED_MD  # v2c's import points B at ledger_v2c; never write there
    return VC.XA(), VC.XI()


def rescore(tpl: str, tables: dict) -> dict:
    from termmap.reranker import CrossEncoder
    ph = PHRASE[tpl]
    cache = pd.read_parquet(CE_CACHE) if CE_CACHE.exists() else pd.DataFrame(columns=["tpl", "q", "c", "score"])
    lut = {(t, q, c): v for t, q, c, v in zip(cache.tpl, cache.q, cache.c, cache.score)}
    texts = {n: _texts(X, lambda s, e, ch: query_sf(s, e, ch, ph)) for n, X in tables.items()}
    todo = sorted({(tpl, q, c) for q_, c_ in texts.values() for q, c in zip(q_, c_)} - set(lut))
    if todo:
        t0 = time.perf_counter()
        ce = CrossEncoder(str(C.MODELS / "cross-ctx"), max_len=C.CROSS_MAX_LEN)
        sc = ce.score([k[1] for k in todo], [k[2] for k in todo])
        del ce
        import torch
        torch.cuda.empty_cache()
        print(f"  SF[{tpl}] scored {len(todo)} new context pairs on {C.device()} in {time.perf_counter() - t0:.0f}s", flush=True)
        new = pd.DataFrame({"tpl": tpl, "q": [k[1] for k in todo], "c": [k[2] for k in todo], "score": sc})
        pd.concat([cache, new], ignore_index=True).to_parquet(CE_CACHE)
        lut.update({k: float(v) for k, v in zip(todo, sc)})
    out = {}
    for n, X in tables.items():
        q, c = texts[n]
        X = X.copy()
        X["ce_ctx_base"] = X.ce_ctx
        X["ce_ctx"] = [lut[(tpl, a, b)] for a, b in zip(q, c)]
        out[n] = recompute_ce(X)
    return out


def tables(tpl: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    pa, pi = OUT / f"XA_{tpl}.parquet", OUT / f"XI_{tpl}.parquet"
    if pa.exists() and pi.exists():
        V.plug_pool()
        return pd.read_parquet(pa), pd.read_parquet(pi)
    XA, XI = _base()
    t = rescore(tpl, {"XA": XA, "XI": XI})
    t["XA"].to_parquet(pa)
    t["XI"].to_parquet(pi)
    for n, X in t.items():
        top = X.sort_values(["qid", "ce_ctx"], ascending=[True, False]).groupby("qid").head(1)
        topb = X.sort_values(["qid", "ce_ctx_base"], ascending=[True, False]).groupby("qid").head(1)
        same = (top.set_index("qid").code == topb.set_index("qid").code.reindex(top.qid)).mean()
        print(f"  {n}[{tpl}]: CE top-1 unchanged vs subconcept query on {same:.3f} of queries; "
              f"corr(ce, ce_base) {np.corrcoef(X.ce_ctx, X.ce_ctx_base)[0, 1]:.4f}", flush=True)
    return t["XA"], t["XI"]


# ---- tag handling -----------------------------------------------------------------------------
def allowed_tags() -> set:
    from termmap_plus.pools import allowed_tags as at
    return set(at(0.995))  # semantic tags covering >= 99.5% of diagnosis-TRAIN SNOMED gold


def with_tag(X: pd.DataFrame, opt: str, base_feats: list[str]) -> tuple[pd.DataFrame, list[str]]:
    feats = [f for f in base_feats if f != "tag_compat"]
    if opt == "const":
        return X, base_feats
    if opt == "drop":
        return X, feats
    tag = K.concepts()[C.SNOMED].tag
    X = X.copy()
    X["cand_is_diag_tag"] = ((X.system == C.SNOMED) & X.code.map(tag).isin(allowed_tags())).astype(int)
    i = base_feats.index("tag_compat")
    return X, feats[:i] + ["cand_is_diag_tag"] + feats[i:]


# ---- evaluation -------------------------------------------------------------------------------
class Gate:
    """X's Qwen2.5 override (tau 0.6, alpha 0.25), from an EMPTY cache file (one per run)."""

    def __init__(self, tag: str):
        from termmap import llm as L
        L.CACHE_FILE = OUT / f"llm_cache_sf_{tag}.jsonl"
        L.CACHE_FILE.unlink(missing_ok=True)
        self.adj = L.Adjudicator()
        assert not self.adj.cache
        self.calls, self.prompts = 0, set()

    def __call__(self, X: pd.DataFrame, tau=0.6, alpha=0.25) -> dict:
        con, fr, out = K.concepts(), K.dev_frames(), {}
        for q, g in X[X.is_diag & (X.system == C.SNOMED)].groupby("qid", sort=False):
            if g.p.max() >= tau:
                continue
            m = int(q.split(":")[1])
            g = g.sort_values(["p", "fused_rank"], ascending=[False, True], kind="stable").head(5)
            opts = [(c, con[C.SNOMED].preferred.get(c, "")) for c in g.code]
            lp = self.adj.choose(C.SNOMED, fr[C.SNOMED].entity.iat[m], fr[C.SNOMED].chunk_text.iat[m], opts)
            self.calls += 1
            self.prompts.add((q, tuple(g.code)))
            sc = np.log(np.clip(g.p.to_numpy(), 1e-9, 1)) + alpha * lp
            out[q] = g.code.iat[int(np.argmax(sc))]
        return out


def tag_filter(X: pd.DataFrame) -> pd.DataFrame:
    """Diagnosis SNOMED queries keep only candidates whose semantic tag is allowed (query left whole if none is)."""
    tag = K.concepts()[C.SNOMED].tag
    target = X.is_diag & (X.system == C.SNOMED)
    ok = X.code.map(tag).isin(allowed_tags())
    any_ok = ok.groupby(X.qid).transform("max")
    return X[~target | ok | ~any_ok].reset_index(drop=True)


def snomed_fit(XA: pd.DataFrame, feats) -> pd.DataFrame:
    X = XA.copy()
    X["s"] = V.oof(X, feats, SEEDS)  # shared stacker (both systems' rows), out-of-fold
    return X


def snomed_eval(X: pd.DataFrame, gate: Gate, filt: bool):
    Xs = tag_filter(X) if filt else X
    Xs = Xs.copy()
    Xs["p"] = K.x_probs(Xs)
    n0 = gate.calls
    P = K.predictions(Xs, override=gate(Xs))
    return P[P.system == C.SNOMED], gate.calls - n0


def icd_eval(XI: pd.DataFrame, feats, s_ref: pd.DataFrame, save: str | None = None):
    from termmap_plus import v2_qwen3 as Q
    Q.CACHE = OUT / "qwen3_cache_sf.parquet"  # private copy (keys row_id/system/code; prompt has no subconcept)
    if not Q.CACHE.exists():
        Q.CACHE.write_bytes((V.V2 / "qwen3_cache.parquet").read_bytes())
    X = XI.drop(columns=[c for c in ("s_ref",) if c in XI]).merge(s_ref, on=["qid", "code"], how="left")
    X = Q.qwen3_features(X, "s_ref").drop(columns=["s_ref"])
    if save:
        X.to_parquet(OUT / f"XIq_{save}.parquet")  # ICD stacker training table for the predict path
    X["s"] = B.oof_sys(X, {C.ICD: feats}, SEEDS)
    X["p"] = K.x_probs(X)
    P = K.predictions(X)
    return P[P.system == C.ICD]


def add_split(P):
    P = P.copy()
    P["split"] = [K.dev_frames()[s].split.iat[int(q.split(":")[1])] for s, q in zip(P.system, P.qid)]
    return P


def non_diag_rate(P: pd.DataFrame) -> float:
    tag = K.concepts()[C.SNOMED].tag
    s = P[P.system == C.SNOMED]
    return float((~s.pred.map(tag).isin(allowed_tags())).mean())


def compare(P: pd.DataFrame, base: pd.DataFrame) -> dict:
    """Non-inferiority: per system pooled acc >= base - 1.0 pt, pooled min >= base min - 1.0 pt,
    broken - fixed <= sqrt(fixed + broken) per system."""
    out, ok = {}, True
    mp, mb = K.metrics(P), K.metrics(base)
    for s, t in ((C.ICD, "ICD"), (C.SNOMED, "SNOMED")):
        j = P[P.system == s].merge(base[base.system == s], on="qid", suffixes=("", "_b"))
        fixed = int(((j.pred == j.gold) & (j.pred_b != j.gold)).sum())
        broken = int(((j.pred != j.gold) & (j.pred_b == j.gold)).sum())
        a = P[P.system == s]
        r = {"acc": mp[t], "d_acc": mp[t] - mb[t], "unseen": mp[f"unseen_{t}"], "d_unseen": mp[f"unseen_{t}"] - mb[f"unseen_{t}"],
             "ceiling": mp[f"ceiling_{t}"], "fixed": fixed, "broken": broken,
             "dev": float((a[a.split == "dev"].pred == a[a.split == "dev"].gold).mean()),
             "test": float((a[a.split == "test"].pred == a[a.split == "test"].gold).mean())}
        r["ok"] = bool(r["d_acc"] >= -0.010 - 1e-9 and broken - fixed <= math.sqrt(fixed + broken))
        ok &= r["ok"]
        out[t] = r
    out["min"], out["d_min"] = mp["min"], mp["min"] - mb["min"]
    out["accept"] = bool(ok and out["d_min"] >= -0.010 - 1e-9)
    return out


def ledger(code: str, name: str, cmp: dict, note: str):
    df = pd.read_csv(LED_CSV) if LED_CSV.exists() else pd.DataFrame()
    rows = []
    for t in ("ICD", "SNOMED"):
        r = cmp[t]
        rows.append({"step": code, "system": t, "name": name, "status": "ACCEPT" if cmp["accept"] else "REJECT",
                     "pool_acc": r["acc"], "d_acc": r["d_acc"], "unseen": r["unseen"], "d_unseen": r["d_unseen"],
                     "ceiling": r["ceiling"], "dev": r["dev"], "test": r["test"], "pool_min": cmp["min"], "d_min": cmp["d_min"],
                     "fixed": r["fixed"], "broken": r["broken"], "sys_ok": r["ok"], "note": note})
    df = pd.concat([df[df.step.astype(str) != code] if len(df) else df, pd.DataFrame(rows)], ignore_index=True)
    df.to_csv(LED_CSV, index=False)
    with LED_MD.open("w", encoding="utf8") as f:
        f.write("# Subconcept-free mode vs xplus-v2c-base — pooled dev+test, 20 seeds, fixed folds\n\n"
                "Non-inferiority: each system's pooled acc and the pooled min >= base - 1.0 pt, and broken - fixed <= "
                "sqrt(fixed + broken) per system. `*_const` rows still read the constant tag_compat (diagnostic only).\n\n"
                "| step | system | name | status | pool acc | Δacc | unseen | Δunseen | ceiling | dev | test | pool min | Δmin | fixed | broken | note |\n"
                "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in df.itertuples():
            f.write(f"| {r.step} | {r.system} | {r.name} | {r.status} | {r.pool_acc:.3f} | {r.d_acc:+.3f} | {r.unseen:.3f} | "
                    f"{r.d_unseen:+.3f} | {r.ceiling:.3f} | {r.dev:.3f} | {r.test:.3f} | {r.pool_min:.3f} | {r.d_min:+.3f} | "
                    f"{r.fixed} | {r.broken} | {'' if pd.isna(r.note) else r.note} |\n")


def run(tpl: str):
    base = pd.read_parquet(K.ART / "v2c" / "P_ref.parquet")
    XA, XI = tables(tpl)
    st = json.loads((K.ART / "v2c" / "state_v2c.json").read_text(encoding="utf8"))
    fA = list(FEATURES) + st["featsA"]
    fI = list(FEATURES) + V.V1_FEATS + st["featsI_extra"]
    gate = Gate(tpl)
    results = {"base": {"non_diag_tag_rate": non_diag_rate(base), **K.metrics(base)}}
    print(f"base xplus-v2c-base: ICD {K.metrics(base)['ICD']:.3f} SNOMED {K.metrics(base)['SNOMED']:.3f}; "
          f"top SNOMED pick non-diagnosis tag {results['base']['non_diag_tag_rate']:.3f}", flush=True)
    for opt in ("const", "drop", "dtag"):
        t0 = time.perf_counter()
        XAo, fAo = with_tag(XA, opt, fA)
        XIo, fIo = with_tag(XI, opt, fI)
        Xs = snomed_fit(XAo, fAo)
        s_ref = Xs[["qid", "code", "s"]].rename(columns={"s": "s_ref"})  # Qwen3 top-10 chooser (as v2.1 was for base)
        Pi = icd_eval(XIo, fIo, s_ref, save=f"{tpl}_{opt}")
        for filt in ((False, True) if opt != "const" else (False,)):
            Ps, calls = snomed_eval(Xs, gate, filt)
            code = f"{tpl}_{opt}" + ("_f" if filt else "")
            P = add_split(pd.concat([Pi, Ps], ignore_index=True))
            P.to_parquet(OUT / f"P_{code}.parquet")
            cmp = compare(P, base)
            cmp["non_diag_tag_rate"] = non_diag_rate(P)
            cmp["gate_calls"] = calls
            results[code] = cmp
            name = {"const": "CE without subconcept; tag_compat constant (CE change alone)",
                    "drop": "SF: CE without subconcept; tag_compat dropped",
                    "dtag": "SF: CE without subconcept; tag_compat -> cand_is_diag_tag"}[opt]
            name = (name.replace("CE without subconcept", 'CE with fixed phrase "diagnosis"') if tpl == "diag" else name) \
                + ("; SNOMED tag filter" if filt else "")
            note = (f"top SNOMED pick non-diag tag {cmp['non_diag_tag_rate']:.3f} (base {results['base']['non_diag_tag_rate']:.3f}); "
                    f"gate calls {calls} (empty cache, {len(gate.prompts)} distinct prompts so far)")
            ledger(code, name, cmp, note)
            print(f"{code}: ICD {cmp['ICD']['acc']:.3f} ({cmp['ICD']['d_acc']:+.3f}, {cmp['ICD']['fixed']}/{cmp['ICD']['broken']}) "
                  f"SNOMED {cmp['SNOMED']['acc']:.3f} ({cmp['SNOMED']['d_acc']:+.3f}, {cmp['SNOMED']['fixed']}/{cmp['SNOMED']['broken']}) "
                  f"min {cmp['min']:.3f} -> {'ACCEPT' if cmp['accept'] else 'REJECT'} | {note} | {time.perf_counter() - t0:.0f}s", flush=True)
    (OUT / f"results_{tpl}.json").write_text(json.dumps(results, indent=1), encoding="utf8")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "check":
        check()
    elif cmd == "tables":
        tables(sys.argv[2])
    elif cmd == "run":
        run(sys.argv[2])

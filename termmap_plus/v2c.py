"""X+ v2c: targeted fixes from the label audit.  python -m termmap_plus.v2c <ref|c1|det|c2|c3|c4|c5>

Hybrid state (C0): SNOMED = shared v2.1-style stacker (both systems' rows, v1 features, 20 seeds) + X's Qwen2.5
gate; ICD = v2b ICD-only stacker (v1 features + Qwen3-Reranker + convention kNN, kNN pool additions, gate off).
Pooled dev+test, fixed folds, per-system keep rule; one ledger row per system. Features use TRAIN rows only.
No new LLM calls except the explicit determinism check (existing caches)."""
from __future__ import annotations

import termmap_plus  # noqa: F401
import json
import math
import re
import sys

import numpy as np
import pandas as pd

from termmap import config as C
from termmap.normalize import clean, content_tokens
from termmap_plus import core as K
from termmap_plus import v2 as V
from termmap_plus import v2b as B

VC = K.ART / "v2c"
VC.mkdir(parents=True, exist_ok=True)
B.LED_CSV, B.LED_MD = K.RES / "ledger_v2c.csv", K.RES / "ledger_v2c.md"
STATE = VC / "state_v2c.json"


def state():
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf8"))
    sb = B.state()
    return {"featsA": list(V.V1_FEATS), "featsI_extra": [f for f in sb["feats"][C.ICD] if f not in _base()],
            "gate_mode": "override", "prev": None, "kept": {C.ICD: [], C.SNOMED: []}}


def _base():
    from termmap.features import FEATURES
    return list(FEATURES) + V.V1_FEATS


def save(st):
    STATE.write_text(json.dumps(st), encoding="utf8")


# ---- feature tables (cached) ---------------------------------------------------------------
def XA() -> pd.DataFrame:
    """Shared table (both systems, v1 pools/features) for the SNOMED shared stacker."""
    return V.base_X()


def XI() -> pd.DataFrame:
    """ICD state table: kNN-augmented ICD pools + Qwen3 + kNN features (ICD rows only)."""
    p = VC / "XI.parquet"
    if p.exists():
        V.plug_pool()
        return pd.read_parquet(p)
    from termmap_plus.v2b_steps import current_state_X
    X = current_state_X(B.state())
    X = X[X.system == C.ICD].reset_index(drop=True)
    X.to_parquet(p)
    return X


# ---- evaluation -----------------------------------------------------------------------------
def gate_info(X: pd.DataFrame, tau=0.6, alpha=0.25):
    """X's Qwen2.5 gate on diagnosis SNOMED rows, from the cache: {qid: (codes_top5, lp, choice)}."""
    from termmap import llm as L
    L.CACHE_FILE = K.ART / "llm_cache.jsonl"
    if "adj" not in K._cache:
        K._cache["adj"] = L.Adjudicator()
    adj, con, fr, out = K._cache["adj"], K.concepts(), K.dev_frames(), {}
    for q, g in X[X.is_diag & (X.system == C.SNOMED)].groupby("qid", sort=False):
        if g.p.max() >= tau:
            continue
        m = int(q.split(":")[1])
        g = g.sort_values(["p", "fused_rank"], ascending=[False, True], kind="stable").head(5)
        lp = adj.choose(C.SNOMED, fr[C.SNOMED].entity.iat[m], fr[C.SNOMED].chunk_text.iat[m],
                        [(c, con[C.SNOMED].preferred.get(c, "")) for c in g.code])
        sc = np.log(np.clip(g.p.to_numpy(), 1e-9, 1)) + alpha * lp
        out[q] = (list(g.code), lp, g.code.iat[int(np.argmax(sc))], g.p.to_numpy())
    return out


def snomed_eval(XA_, featsA, gate="override"):
    from termmap.features import FEATURES
    X = XA_.copy()
    X["s"] = V.oof(X, list(FEATURES) + featsA, V.SEEDS20)
    X["p"] = K.x_probs(X)
    gi = gate_info(X) if gate else {}
    choice = {q: v[2] for q, v in gi.items()} if gate == "override" else {}
    P = K.predictions(X, override=choice)
    return P[P.system == C.SNOMED], X, gi


def icd_eval(XI_, extra):
    from termmap.features import FEATURES
    X = XI_.copy()
    X["s"] = B.oof_sys(X, {C.ICD: list(FEATURES) + V.V1_FEATS + extra})
    X["p"] = K.x_probs(X)
    P = K.predictions(X)
    return P[P.system == C.ICD], X


def add_split(P):
    P = P.copy()
    P["split"] = [K.dev_frames()[s].split.iat[int(q.split(":")[1])] for s, q in zip(P.system, P.qid)]
    return P


def combine(Pi, Ps):
    return add_split(pd.concat([Pi, Ps], ignore_index=True))


def step_row(code, name, P, prev, targets, st, note="", systems=K.SYSTEMS):
    """Ledger rows for the systems this step touches; the untouched system is carried from prev."""
    P2 = pd.concat([P[P.system == s] if s in systems else prev[prev.system == s] for s in K.SYSTEMS], ignore_index=True)
    keeps = B.ledger(code, name, P2, prev, targets, note=note)
    return {s: (keeps[s] if s in systems else False) for s in K.SYSTEMS}, P2


# ---- steps ---------------------------------------------------------------------------------
def run_ref():
    st = state()
    Pi, _ = icd_eval(XI(), st["featsI_extra"])
    Ps, _, gi = snomed_eval(XA(), st["featsA"])
    P = combine(Pi, Ps)
    P.to_parquet(VC / "P_ref.parquet")
    B.ledger("C0", "hybrid base: ICD v2b stacker | SNOMED shared v2.1 stacker + Qwen2.5 gate", P, None, "v2c reference",
             forced="REFERENCE", note="SNOMED base = shared (0.677 >= separate 0.675)")
    st["prev"] = "P_ref.parquet"
    save(st)


def run_c1():
    st = state()
    prev = pd.read_parquet(VC / st["prev"])
    Xa = XA()
    Pg, X, gi = snomed_eval(Xa, st["featsA"], gate="override")  # reference gate behaviour
    P0 = K.predictions(X)  # stacker top-1 without gate
    P0 = P0[P0.system == C.SNOMED].set_index("qid")
    gold = P0.gold
    wrong = [q for q, v in gi.items() if P0.pred[q] == gold[q] and v[2] != gold[q]]
    fixes = [q for q, v in gi.items() if P0.pred[q] != gold[q] and v[2] == gold[q]]
    # (ii) gated override: tau / margin tuned inside folds
    f = dict(zip(X.qid, V.folds(X)))
    rows = P0.reset_index()
    rows["fold"] = rows.qid.map(f)
    def pick(df, tau, m):
        out = []
        for q, p in zip(df.qid, df.pred):
            v = gi.get(q)
            if v is None:
                out.append(p)
                continue
            p1, p2 = v[3][0], (v[3][1] if len(v[3]) > 1 else 0.0)
            out.append(v[2] if (p1 < tau or (p1 - p2) < m) else p)
        return out
    grid = [(t, m) for t in (0.2, 0.3, 0.4, 0.5, 0.6) for m in (0.0, 0.05, 0.1, 0.2, 0.3)]
    chosen = {}
    for k in range(5):
        tr = rows[rows.fold != k]
        chosen[k] = max(grid, key=lambda tm: (float((np.array(pick(tr, *tm)) == tr.gold.to_numpy()).mean()), -tm[0], -tm[1]))
    rows["pred"] = [pick(rows.iloc[[i]], *chosen[fk])[0] for i, fk in enumerate(rows.fold)]
    P_ii = rows.drop(columns=["fold"])
    # (i) gate as stacker features
    Xf = Xa.copy()
    Xf["p_ref"] = X.p.to_numpy()
    called = {q for q in gi}
    gpick = {(q, v[2]) for q, v in gi.items()}
    agree = {q for q, v in gi.items() if v[2] == v[0][0]}
    lpd = {(q, c): l for q, v in gi.items() for c, l in zip(v[0], v[1])}
    Xf["gate_called"] = Xf.qid.isin(called).astype(int)
    Xf["gate_pick"] = [int((q, c) in gpick) for q, c in zip(Xf.qid, Xf.code)]
    Xf["gate_agrees_top1"] = Xf.qid.isin(agree).astype(int)
    Xf["gate_lp"] = [lpd.get((q, c), -10.0) for q, c in zip(Xf.qid, Xf.code)]
    GF = ["gate_called", "gate_pick", "gate_agrees_top1", "gate_lp"]
    P_i, _, _ = snomed_eval(Xf, st["featsA"] + GF, gate=None)

    def tally(Pv):
        pv = Pv.set_index("qid").pred
        return sum(pv[q] == gold[q] for q in wrong), sum(pv[q] == gold[q] for q in fixes)
    res = {}
    for name, Pv in (("(i) gate as features", P_i), ("(ii) gated override", P_ii)):
        av, kept = tally(Pv)
        Pfull = pd.concat([prev[prev.system == C.ICD], Pv], ignore_index=True)
        res[name] = (float((Pv.pred == Pv.gold).mean()), Pfull, av, kept)
        print(f"{name}: SNOMED pooled {res[name][0]:.4f} | wrong overrides avoided {av}/{len(wrong)} | fixes kept {kept}/{len(fixes)}", flush=True)
    best = max(res, key=lambda k: res[k][0])
    acc, Pfull, av, kept = res[best]
    note = (f"variant {best}; wrong overrides avoided {av}/{len(wrong)}, gate fixes kept {kept}/{len(fixes)}; "
            f"(ii) tau/m per fold {sorted(set(chosen.values()))}")
    keeps, P2 = step_row("C1", f"smarter SNOMED gate — {best}", add_split(Pfull), prev, "gate overriding correct picks", st,
                         note=note, systems=[C.SNOMED])
    if keeps[C.SNOMED]:
        st["gate_mode"] = "features" if best.startswith("(i)") else "gated"
        st["kept"][C.SNOMED].append("C1" + best[:4])
        if best.startswith("(i)"):
            st["featsA"] = st["featsA"] + GF
        P2.to_parquet(VC / "P_C1.parquet")
        st["prev"] = "P_C1.parquet"
    save(st)


def run_det():
    """Determinism of X's Qwen2.5 gate WITHOUT its cache: 50 prompts twice on GPU, 8 twice on CPU."""
    import time
    import torch
    from termmap import llm as L
    Xa = XA()
    X = Xa.copy()
    from termmap.features import FEATURES
    X["s"] = V.oof(X, list(FEATURES) + state()["featsA"][:6], (13,))
    X["p"] = K.x_probs(X)
    con, fr = K.concepts(), K.dev_frames()
    qs = [q for q, g in X[X.is_diag & (X.system == C.SNOMED)].groupby("qid", sort=False) if g.p.max() < 0.6][:50]
    prompts = []
    for q in qs:
        g = X[X.qid == q].sort_values(["p", "fused_rank"], ascending=[False, True]).head(5)
        m = int(q.split(":")[1])
        prompts.append((fr[C.SNOMED].entity.iat[m], fr[C.SNOMED].chunk_text.iat[m], [(c, con[C.SNOMED].preferred.get(c, "")) for c in g.code]))
    out = {}
    for dev, n in (("cuda", 50), ("cpu", 8)):
        C.DEVICE = dev
        runs = []
        for r in range(2):
            L.CACHE_FILE = VC / f"det_tmp_{dev}_{r}.jsonl"
            if L.CACHE_FILE.exists():
                L.CACHE_FILE.unlink()
            adj = L.Adjudicator()
            adj.cache = {}
            t0 = time.perf_counter()
            runs.append([adj.choose(C.SNOMED, e, ch, o) for e, ch, o in prompts[:n]])
            el = (time.perf_counter() - t0) / n
            del adj
            torch.cuda.empty_cache()
        same_arg = sum(int(np.argmax(a) == np.argmax(b)) for a, b in zip(*runs))
        same_val = sum(int(np.allclose(a, b, atol=0)) for a, b in zip(*runs))
        out[dev] = (n, same_arg, same_val, el)
        print(f"Qwen2.5 gate {dev}: {same_arg}/{n} identical choices, {same_val}/{n} bit-identical log-probs, {el:.2f}s/prompt", flush=True)
    C.DEVICE = "auto"
    (VC / "determinism.json").write_text(json.dumps(out), encoding="utf8")
    K.log_decision(f"v2c gate determinism without cache: {out}")


# ---- C2 negation ---------------------------------------------------------------------------
_NEG = re.compile(r"\b(no|not|without|denies|denied|deny|negative for|absent|free of|no evidence of|non)\b")
_CAND_NEG = re.compile(r"^(no |absent|normal)|\b(absent|negative|not present|no evidence)\b")


def neg_features(X, frames):
    from termmap.context import locate
    X = X.copy()
    cache = {}
    sysm = X.qid.str.split(":", n=1, expand=True)
    neg = np.zeros(len(X), int)
    for i, (s, m) in enumerate(zip(sysm[0].to_numpy(), sysm[1].astype(int).to_numpy())):
        if (s, m) not in cache:
            d = frames[s]
            e, ch = d.entity.iat[m], d.chunk_text.iat[m]
            loc = locate(e, ch)
            before = clean(ch[:loc[0]]).split()[-5:] if loc else []
            cache[(s, m)] = int(bool(_NEG.search(clean(e))) or bool(_NEG.search(" ".join(before))))
        neg[i] = cache[(s, m)]
    con = K.concepts()
    cn = np.array([int(bool(_CAND_NEG.search(clean(con[s].preferred.get(c, ""))))) for s, c in zip(sysm[0], X.code)])
    X["neg_term"], X["cand_negative"] = neg, cn
    X["neg_match"] = (neg & cn).astype(int)
    X["neg_conflict"] = (neg ^ cn).astype(int)
    return X


NEG_FEATS = ["neg_term", "cand_negative", "neg_match", "neg_conflict"]


def run_c2():
    st = state()
    prev = pd.read_parquet(VC / st["prev"])
    fr = K.dev_frames()
    Pi, _ = icd_eval(neg_features(XI(), fr), st["featsI_extra"] + NEG_FEATS)
    Xa = with_current_A(neg_features(XA(), fr), st)
    Ps, _, _ = snomed_eval(Xa, st["featsA"] + NEG_FEATS, gate=gate_arg(st))
    keeps, _ = step_row("C2", "negation alignment (NegEx cue x candidate polarity)", combine(Pi, Ps), prev,
                        "negated mentions -> normal/absent concepts", st)
    newP = combine(Pi if keeps[C.ICD] else prev[prev.system == C.ICD].drop(columns=["split"]),
                   Ps if keeps[C.SNOMED] else prev[prev.system == C.SNOMED].drop(columns=["split"]))
    if keeps[C.ICD]:
        st["featsI_extra"] += NEG_FEATS
        st["kept"][C.ICD].append("C2")
    if keeps[C.SNOMED]:
        st["featsA"] += NEG_FEATS
        st["kept"][C.SNOMED].append("C2")
    newP.to_parquet(VC / "P_C2.parquet")
    st["prev"] = "P_C2.parquet"
    save(st)


def gate_arg(st):
    return "override" if st["gate_mode"] == "override" else None


def with_current_A(X, st):
    """Re-attach kept SNOMED-side extras (C1 gate features, C2 negation, C3 priors) to a shared table."""
    if "C2" in st["kept"][C.SNOMED] and "neg_term" not in X:
        X = neg_features(X, K.dev_frames())
    if any(k.startswith("C3") for k in st["kept"][C.SNOMED]) and "c3_tag_prior" not in X:
        X = c3_features(X)
    return X


# ---- C3 SNOMED convention priors -----------------------------------------------------------
def _head(e):
    from termmap_plus.features_plus import QUAL
    t = [w for w in content_tokens(e) if w not in QUAL]
    return t[-1] if t else ""


def c3_features(X):
    from termmap_plus.v2_features import tag_features
    con = K.concepts()[C.SNOMED]
    t = K.diag_train()
    t = t[t.coding_system == C.SNOMED]
    tags = t.code.map(con.tag).fillna("")
    heads = t.entity.map(_head)
    glob = tags.value_counts(normalize=True).to_dict()
    ct = pd.crosstab(heads, tags)
    a = 5.0
    acute_e = t.entity.map(lambda e: "acute" in clean(e))
    acute_g = t.code.map(lambda c: "acute" in clean(con.preferred.get(c, "")))
    base_ac = float(acute_g[~acute_e].mean())
    hp = pd.DataFrame({"h": heads[~acute_e], "g": acute_g[~acute_e]}).groupby("h").g.agg(["sum", "count"])
    fr = K.dev_frames()[C.SNOMED]
    X = X.copy()
    sysm = X.qid.str.split(":", n=1, expand=True)
    tp, ap, adds = np.zeros(len(X)), np.zeros(len(X)), np.zeros(len(X))
    for i in np.where(sysm[0].to_numpy() == C.SNOMED)[0]:
        e = fr.entity.iat[int(sysm[1].iat[i])]
        h, c = _head(e), X.code.iat[i]
        tg = con.tag.get(c, "")
        n = ct.loc[h].sum() if h in ct.index else 0
        k = ct.at[h, tg] if (h in ct.index and tg in ct.columns) else 0
        tp[i] = (k + a * glob.get(tg, 0.0)) / (n + a)
        if "acute" in clean(con.preferred.get(c, "")) and "acute" not in clean(e):
            s_, n_ = (hp.at[h, "sum"], hp.at[h, "count"]) if h in hp.index else (0, 0)
            ap[i] = (s_ + a * base_ac) / (n_ + a)
            adds[i] = 1
    X["c3_tag_prior"], X["c3_acute_prior"], X["c3_adds_acute"] = tp, ap, adds
    return tag_features(X, K.dev_frames())


C3_FEATS = ["c3_tag_prior", "c3_acute_prior", "c3_adds_acute", "tag_p"]


def run_c3():
    st = state()
    prev = pd.read_parquet(VC / st["prev"])
    Xa = c3_features(with_current_A(XA(), st))
    Ps, _, _ = snomed_eval(Xa, st["featsA"] + C3_FEATS, gate=gate_arg(st))
    Pfull = combine(prev[prev.system == C.ICD].drop(columns=["split"]), Ps)
    keeps, P2 = step_row("C3", "SNOMED convention priors (tag|head noun, acute|head, tag classifier)", Pfull, prev,
                         "coder conventions: semantic tag by head word, 'acute' by convention", st, systems=[C.SNOMED])
    if keeps[C.SNOMED]:
        st["featsA"] += C3_FEATS
        st["kept"][C.SNOMED].append("C3")
        P2.to_parquet(VC / "P_C3.parquet")
        st["prev"] = "P_C3.parquet"
    save(st)


# ---- C4 ICD release convention -------------------------------------------------------------
def c4_features(X):
    from termmap_plus.pools import build_icd_tree
    tree = build_icd_tree().set_index("code")
    con = K.concepts()[C.ICD]
    t = K.diag_train()
    gi = t[t.coding_system == C.ICD].code
    gold_cnt = gi.value_counts().to_dict()
    no7 = gi.groupby(gi.str[:3]).agg(lambda s: float((s.str.replace(".", "").str.len() < 7).mean())).to_dict()
    X = X.copy()
    par = X.code.map(tree.parent).fillna("")
    X["c4_exp_child"] = [int(bool(p) and p in tree.index and not tree.at[p, "is_leaf"]) for p in par]
    X["c4_parent_gold"] = [np.log1p(gold_cnt.get(p, 0)) for p in par]
    X["c4_has7"] = (X.code.str.replace(".", "").str.len() == 7).astype(int)
    X["c4_cat_no7"] = X.code.str[:3].map(no7).fillna(-1.0)
    X["c4_header"] = (~X.code.map(con.billable).fillna(True).astype(bool)).astype(int)
    return X


C4_FEATS = ["c4_exp_child", "c4_parent_gold", "c4_has7", "c4_cat_no7", "c4_header"]


def current_I(st):
    X = XI()
    if "C2" in st["kept"][C.ICD]:
        X = neg_features(X, K.dev_frames())
    return X


def run_c4():
    st = state()
    prev = pd.read_parquet(VC / st["prev"])
    Pi, _ = icd_eval(c4_features(current_I(st)), st["featsI_extra"] + C4_FEATS)
    Pfull = combine(Pi, prev[prev.system == C.SNOMED].drop(columns=["split"]))
    keeps, P2 = step_row("C4", "ICD release convention (parent/header/7th-character priors)", Pfull, prev,
                         "gold uses older/parent code or omits 7th character", st, systems=[C.ICD])
    if keeps[C.ICD]:
        st["featsI_extra"] += C4_FEATS
        st["kept"][C.ICD].append("C4")
        P2.to_parquet(VC / "P_C4.parquet")
        st["prev"] = "P_C4.parquet"
    save(st)


def run_c5():
    """CPU-only configuration: ICD without the Qwen3-Reranker feature (GPU-only); SNOMED unchanged."""
    from termmap_plus.v2_qwen3 import Q3_FEATS
    st = state()
    prev = pd.read_parquet(VC / st["prev"])
    X = current_I(st)
    if "C4" in st["kept"][C.ICD]:
        X = c4_features(X)
    Pi, _ = icd_eval(X, [f for f in st["featsI_extra"] if f not in Q3_FEATS])
    Pfull = combine(Pi, prev[prev.system == C.SNOMED].drop(columns=["split"]))
    m = K.metrics(Pfull)
    B.ledger("C5", "CPU-only configuration (no Qwen3-Reranker; gate as decided)", Pfull, prev, "README CPU-only numbers",
             forced="INFO", note=f"CPU-only pooled ICD {m['ICD']:.3f} SNOMED {m['SNOMED']:.3f} min {m['min']:.3f}")
    st["cpu_only"] = {"ICD": m["ICD"], "SNOMED": m["SNOMED"], "min": m["min"]}
    save(st)


if __name__ == "__main__":
    V.plug_pool()  # pooled dev+test frames before any step reads K.dev_frames()
    {"ref": run_ref, "c1": run_c1, "det": run_det, "c2": run_c2, "c3": run_c3, "c4": run_c4, "c5": run_c5}[sys.argv[1]]()

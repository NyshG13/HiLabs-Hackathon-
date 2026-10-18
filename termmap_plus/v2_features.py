"""X+ v2 features: picking-failure buckets (v2.2), ICD tree evidence (v2.3), specificity policy (v2.4),
SNOMED semantic-tag classifier (v2.5). Nothing here is fitted on dev/test rows (train only)."""
from __future__ import annotations

import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from termmap import config as C
from termmap.normalize import clean, content_tokens
from termmap_plus import core as K
from termmap_plus.features_plus import stem, stems, window

_LATW = re.compile(r"\b(left|right|bilateral)\b")


# ----------------------------------------------------------------------------------------------
# v2.2 buckets
# ----------------------------------------------------------------------------------------------
def bucket(system: str, entity: str, gold: str, pred: str) -> str:
    con = K.concepts()[system]
    gp, pp = clean(con.preferred.get(gold, "")), clean(con.preferred.get(pred, ""))
    te = clean(entity)
    if pp == te and gp != te:
        return "probable label noise (pred name == term)"
    hist = lambda s: bool(re.search(r"\b(history|status|family)\b", s))
    if hist(gp) != hist(pp):
        return "history/status"
    if system == C.ICD:
        if gold[:3] == pred[:3]:
            g7, p7 = gold.replace(".", ""), pred.replace(".", "")
            if len(g7) == len(p7) == 7 and g7[:-1] == p7[:-1]:
                return "7th character / laterality"
            gl, pl = _LATW.search(gp), _LATW.search(pp)
            if (gl and pl and gl.group(0) != pl.group(0)) or ((gl is None) != (pl is None) and len(g7) == len(p7)):
                return "7th character / laterality"
            if len(p7) > len(g7):
                return "ICD sibling, pred more specific"
            if len(p7) < len(g7):
                return "ICD sibling, pred less specific"
            return "ICD sibling, same depth"
        return "other wrong concept"
    gt, pt = set(content_tokens(gp)), set(content_tokens(pp))
    if gt and pt and (gt == pt or gt <= pt or pt <= gt) and con.tag.get(gold) != con.tag.get(pred):
        return "SNOMED same words, different tag"
    if gt and pt and (gt <= pt or pt <= gt):
        return "SNOMED nested (specificity)"
    return "SNOMED different concept"


# ----------------------------------------------------------------------------------------------
# v2.3 ICD tree evidence
# ----------------------------------------------------------------------------------------------
def _term_stems(frames, abbr):
    cache = {}

    def get(s, m):
        if (s, m) not in cache:
            d = frames[s]
            e, ch = d.entity.iat[m], d.chunk_text.iat[m]
            exp = []
            for t in clean(e).split():
                if len(t) <= 6:
                    exp += abbr.expansions(t, k=6)
            st = stems(e) | stems(window(e, ch))
            for x in exp:
                st |= stems(x)
            cache[(s, m)] = st
        return cache[(s, m)]
    return get


def tree_features(X: pd.DataFrame, frames, abbr) -> pd.DataFrame:
    from termmap_plus.pools import build_icd_tree
    tree = build_icd_tree().set_index("code")
    kids = tree.reset_index().groupby("parent").code.agg(list).to_dict()
    dw = {c: {stem(w) for w in str(v).split()} for c, v in tree.distinguishing_words.items()}
    X = X.copy()
    get = _term_stems(frames, abbr)
    sysm = X.qid.str.split(":", n=1, expand=True)
    is_icd = (sysm[0] == C.ICD).to_numpy()
    ment = sysm[1].astype(int).to_numpy()
    n = len(X)
    depth, unspec, evid, unev = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    for i in np.where(is_icd)[0]:
        c = X.code.iat[i]
        if c not in tree.index:
            continue
        st = get(C.ICD, ment[i])
        w = dw.get(c, set())
        hit = len(w & st)
        depth[i], unspec[i] = tree.at[c, "depth"], float(tree.at[c, "is_unspecified"])
        evid[i] = hit / len(w) if w else 1.0
        unev[i] = len(w) - hit
    X["t_depth"], X["t_unspec"], X["t_evidence"], X["t_unevidenced"] = depth, unspec, evid, unev
    # category vote: softmax of the cross-encoder within the mention, summed per 3-char category
    e = np.exp(X.ce_ctx - X.groupby("qid").ce_ctx.transform("max"))
    pce = e / e.groupby(X.qid).transform("sum")
    cat = X.code.str[:3]
    vote = pce.groupby([X.qid, cat]).transform("sum")
    X["t_cat_vote"] = np.where(is_icd, vote, 0.0)
    X["t_cat_rank"] = np.where(is_icd, X.assign(v=vote, cat=cat).groupby("qid").v.rank(ascending=False, method="dense"), 0)
    # descent pick
    desc = np.zeros(n)
    for q, g in X[is_icd].groupby("qid", sort=False):
        m = int(q.split(":")[1])
        st = get(C.ICD, m)
        top_cat = g.assign(v=X.loc[g.index, "t_cat_vote"]).sort_values("v", ascending=False).code.str[:3].iat[0]
        node = top_cat if top_cat in tree.index else None
        while node is not None and kids.get(node):
            ch = kids[node]
            sc = [(len(dw.get(x, set()) & st) / max(1, len(dw.get(x, set()))), -len(x), x) for x in ch]
            best = max(sc)
            if best[0] >= 0.5:
                node = best[2]
            else:  # detail not evidenced: take the unspecified child, else stop at this node
                un = [x for x in ch if tree.at[x, "is_unspecified"]]
                if not un:
                    break
                node = un[0]
        if node is not None:
            desc[g.index[g.code.to_numpy() == node]] = 1
    X["t_descent_pick"] = desc
    return X


TREE_FEATS = ["t_depth", "t_unspec", "t_evidence", "t_unevidenced", "t_cat_vote", "t_cat_rank", "t_descent_pick"]


# ----------------------------------------------------------------------------------------------
# v2.4 specificity policy + coding defaults (mined from diagnosis train)
# ----------------------------------------------------------------------------------------------
_FX = re.compile(r"\b(fracture|fx|fractured)\b")
_OPENCLOSED = re.compile(r"\b(open|closed|compound)\b")
_SUBSEQ = re.compile(r"\b(subsequent|sequela|sequelae|follow up|healing|late effect)\b")
_DM = re.compile(r"\b(diabetes|diabetic|dm|niddm|iddm|t2dm|t1dm)\b")
_DMTYPE = re.compile(r"\b(type 1|type i|type 2|type ii|t1dm|t2dm|dm1|dm2|iddm|niddm|gestational|juvenile)\b")


def mine_policy() -> dict:
    from termmap_plus.pools import build_icd_tree
    tree = build_icd_tree().set_index("code")
    t = K.diag_train()
    icd = t[t.coding_system == C.ICD]
    con = K.concepts()[C.ICD]
    cat = defaultdict(lambda: [0, 0])
    fx, inj, dm = [0, 0], [0, 0], [0, 0]
    for e, g in zip(icd.entity, icd.code):
        te = clean(e)
        cat[g[:3]][1] += 1
        cat[g[:3]][0] += int(bool(tree.at[g, "is_unspecified"])) if g in tree.index else 0
        g7 = g.replace(".", "")
        if _FX.search(te) and not _OPENCLOSED.search(te) and len(g7) == 7 and g[0] == "S":
            fx[1] += 1
            fx[0] += int(g7[-1] == "A")
        if g[0] in "ST" and len(g7) == 7 and not _SUBSEQ.search(te):
            inj[1] += 1
            inj[0] += int(g7[-1] == "A")
        if _DM.search(te) and not _DMTYPE.search(te) and g[:1] == "E" and g[:3] in ("E08", "E09", "E10", "E11", "E13"):
            dm[1] += 1
            dm[0] += int(g[:3] == "E11")
    pol = {"cat_unspec": {k: (v[0] / v[1], v[1]) for k, v in cat.items()},
           "fracture_closed_initial_A": (fx[0] / max(1, fx[1]), fx[1]),
           "injury_no_subsequent_A": (inj[0] / max(1, inj[1]), inj[1]),
           "diabetes_type_unstated_E11": (dm[0] / max(1, dm[1]), dm[1])}
    lines = ["# Specificity policy and coding defaults (mined from diagnosis train)", "",
             "| default | P(default holds) | training support |", "|---|---|---|",
             f"| fracture, open/closed not stated -> 7th character A (initial) | {pol['fracture_closed_initial_A'][0]:.3f} | {fx[1]} |",
             f"| injury (S/T, 7 chars) without subsequent/sequela -> A | {pol['injury_no_subsequent_A'][0]:.3f} | {inj[1]} |",
             f"| diabetes, type not stated -> E11 | {pol['diabetes_type_unstated_E11'][0]:.3f} | {dm[1]} |", "",
             "Per-category P(gold is an unspecified code), categories with support >= 5:", "",
             "| category | P(unspecified) | support |", "|---|---|---|"]
    for k, (p, n) in sorted(pol["cat_unspec"].items(), key=lambda kv: -kv[1][1]):
        if n >= 5:
            lines.append(f"| {k} | {p:.2f} | {n} |")
    (K.RES / "specificity_policy.md").write_text("\n".join(lines), encoding="utf8")
    return pol


def policy_features(X: pd.DataFrame, frames, pol: dict) -> pd.DataFrame:
    from termmap_plus.pools import build_icd_tree
    tree = build_icd_tree().set_index("code")
    con = K.concepts()[C.ICD]
    X = X.copy()
    sysm = X.qid.str.split(":", n=1, expand=True)
    is_icd = (sysm[0] == C.ICD).to_numpy()
    ment = sysm[1].astype(int).to_numpy()
    n = len(X)
    f = {k: np.zeros(n) for k in ("pol_cat_unspec", "pol_unspec_match", "pol_fx_A", "pol_inj_A", "pol_dm_E11")}
    for i in np.where(is_icd)[0]:
        c = X.code.iat[i]
        te = clean(frames[C.ICD].entity.iat[ment[i]])
        p, sup = pol["cat_unspec"].get(c[:3], (0.0, 0))
        unsp = bool(tree.at[c, "is_unspecified"]) if c in tree.index else False
        f["pol_cat_unspec"][i] = p if sup >= 3 else -1
        f["pol_unspec_match"][i] = (p if unsp else 1 - p) if sup >= 3 else 0.5
        c7 = c.replace(".", "")
        if _FX.search(te) and not _OPENCLOSED.search(te) and len(c7) == 7 and c[0] == "S":
            f["pol_fx_A"][i] = 1 if (c7[-1] == "A") else -1
        if c[0] in "ST" and len(c7) == 7 and not _SUBSEQ.search(te):
            f["pol_inj_A"][i] = 1 if c7[-1] == "A" else -1
        if _DM.search(te) and not _DMTYPE.search(te) and c[:3] in ("E08", "E09", "E10", "E11", "E13"):
            f["pol_dm_E11"][i] = 1 if c[:3] == "E11" else -1
    for k, v in f.items():
        X[k] = v
    return X


POLICY_FEATS = ["pol_cat_unspec", "pol_unspec_match", "pol_fx_A", "pol_inj_A", "pol_dm_E11"]


# ----------------------------------------------------------------------------------------------
# v2.5 SNOMED semantic-tag classifier (term + window -> P(tag)), trained on diagnosis train only
# ----------------------------------------------------------------------------------------------
def tag_features(X: pd.DataFrame, frames) -> pd.DataFrame:
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    con = K.concepts()[C.SNOMED]
    t = K.diag_train()
    t = t[t.coding_system == C.SNOMED]
    y = t.code.map(con.tag).fillna("")
    text = [clean(e) + " || " + window(e, c) for e, c in zip(t.entity, t.chunk_text)]
    vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)
    vw = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    A = hstack([vc.fit_transform(text), vw.fit_transform(text)]).tocsr()
    lr = LogisticRegression(max_iter=3000, C=2.0).fit(A, y)
    cls = {c: i for i, c in enumerate(lr.classes_)}
    X = X.copy()
    sysm = X.qid.str.split(":", n=1, expand=True)
    is_sn = (sysm[0] == C.SNOMED).to_numpy()
    ment = sysm[1].astype(int).to_numpy()
    d = frames[C.SNOMED]
    ms = sorted(set(ment[is_sn]))
    probs = {}
    if ms:
        tx = [clean(d.entity.iat[m]) + " || " + window(d.entity.iat[m], d.chunk_text.iat[m]) for m in ms]
        P = lr.predict_proba(hstack([vc.transform(tx), vw.transform(tx)]).tocsr())
        probs = dict(zip(ms, P))
    tagp = np.zeros(len(X))
    for i in np.where(is_sn)[0]:
        tg = con.tag.get(X.code.iat[i], "")
        tagp[i] = probs[ment[i]][cls[tg]] if tg in cls else 0.0
    X["tag_p"] = tagp
    return X


TAG_FEATS = ["tag_p"]

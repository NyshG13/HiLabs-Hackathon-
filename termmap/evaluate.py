"""Metrics: exact match per system (headline = min), hierarchical credit, long tail, abstention, errors."""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd

from . import config as C
from .normalize import clean, content_tokens


def long_tail_bucket(code: str, train_freq: Counter) -> str:
    n = train_freq.get(code, 0)
    return "unseen" if n == 0 else ("rare(1-2)" if n <= 2 else "frequent(3+)")


def snomed_related(a: str, b: str, pref: dict) -> bool:
    """Licence-free stand-in for IS-A distance <= 1: preferred-term token sets nested."""
    if a == b:
        return True
    ta, tb = set(content_tokens(pref.get(a, ""))), set(content_tokens(pref.get(b, "")))
    return bool(ta) and bool(tb) and (ta <= tb or tb <= ta)


def summarize(res: pd.DataFrame, train: pd.DataFrame, concepts: dict, pred_col: str = "pred",
              conf_col: str | None = "conf") -> dict:
    """res: rows with coding_system, code (gold), pred, optionally conf, in_cands."""
    out = {}
    freq = {s: Counter(train.loc[train.coding_system == s, "code"]) for s in C.SYSTEMS}
    sn_pref = concepts[C.SNOMED].preferred.to_dict()
    icd_blk = concepts[C.ICD].block_desc.to_dict()
    for s in C.SYSTEMS:
        r = res[res.coding_system == s]
        if r.empty:
            continue
        ok = (r[pred_col] == r.code).to_numpy()
        d = {"n": int(len(r)), "exact": float(ok.mean())}
        if s == C.ICD:
            d["category_match"] = float((r[pred_col].str[:3] == r.code.str[:3]).mean())
            d["block_match"] = float(np.mean([icd_blk.get(p, "?") == icd_blk.get(g, "!") for p, g in zip(r[pred_col], r.code)]))
        else:
            d["related_match"] = float(np.mean([snomed_related(p, g, sn_pref) for p, g in zip(r[pred_col], r.code)]))
        lt = r.code.map(lambda c: long_tail_bucket(c, freq[s]))
        d["long_tail"] = {b: dict(n=int((lt == b).sum()), acc=float(ok[(lt == b).to_numpy()].mean()) if (lt == b).any() else None)
                          for b in ("unseen", "rare(1-2)", "frequent(3+)")}
        d["long_tail_acc(unseen+rare)"] = float(ok[(lt != "frequent(3+)").to_numpy()].mean())
        if "in_cands" in r:
            d["candidate_ceiling"] = float(r.in_cands.mean())
        if "subconcept" in r:
            d["by_subconcept"] = {k: dict(n=int(len(g)), acc=float((g[pred_col] == g.code).mean()))
                                  for k, g in r.groupby("subconcept")}
        if conf_col and conf_col in r:
            conf = r[conf_col].to_numpy()
            o = np.argsort(-conf, kind="stable")
            cum = np.cumsum(ok[o]) / np.arange(1, len(o) + 1)
            d["risk_coverage"] = {f"{int(c * 100)}%": float(cum[max(0, int(round(c * len(o))) - 1)]) for c in (0.5, 0.8, 0.9, 0.95, 0.97, 1.0)}
            keep = conf >= C.ABSTAIN_THRESHOLD
            d["abstain"] = dict(threshold=C.ABSTAIN_THRESHOLD, abstain_rate=float(1 - keep.mean()),
                                acc_on_attempted=float(ok[keep].mean()) if keep.any() else None)
        out[s] = d
    out["headline_min_exact"] = min(out[s]["exact"] for s in C.SYSTEMS if s in out)
    return out


_LATW = re.compile(r"\b(left|right|bilateral|unspecified side|unspecified (ear|eye|knee|shoulder|hip|foot|hand|arm|leg|lower|upper))\b")


def error_category(row, concepts: dict, abbr_keys: set) -> str:
    s, g, p = row.coding_system, row.code, row.pred
    if p == g:
        return "correct"
    if not row.get("in_cands", True):
        return "retrieval_miss"
    con = concepts[s]
    gp, pp = clean(con.preferred.get(g, "")), clean(con.preferred.get(p, ""))
    toks = clean(row.entity).split()
    if any(t in abbr_keys and len(t) <= 5 for t in toks):
        return "abbreviation"
    if s == C.ICD and g[:3] == p[:3]:
        gl, pl = _LATW.search(gp), _LATW.search(pp)
        if (gl is None) != (pl is None) or (gl and pl and gl.group(0) != pl.group(0)):
            return "laterality/side"
        return "specificity(same category)"
    if s == C.SNOMED and (set(content_tokens(gp)) <= set(content_tokens(pp)) or set(content_tokens(pp)) <= set(content_tokens(gp))):
        return "specificity(nested concept)"
    if "unspecified" in gp or "other" in gp.split()[:1] or "other specified" in gp:
        return "convention(catch-all code)"
    return "wrong_concept"


def label_ceiling(train: pd.DataFrame) -> dict:
    """Consistency of gold within the labelled data: mentions whose bag of content tokens is identical
    but whose gold codes differ. An empirical lower bound on annotation noise."""
    out = {}
    for s in C.SYSTEMS:
        d = train[train.coding_system == s]
        key = d.entity.map(lambda e: " ".join(sorted(set(content_tokens(e)))))
        g = d.groupby(key).code.agg(list)
        g = g[g.map(len) > 1]
        rows = int(g.map(len).sum())
        agree = int(sum(Counter(v).most_common(1)[0][1] for v in g))
        out[s] = dict(duplicate_groups=int(len(g)), rows=rows,
                      majority_agreement=float(agree / rows) if rows else None)
    return out

"""Candidate features for the learned fusion stacker (plan section 3, step 6)."""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import pandas as pd

from . import config as C
from .normalize import clean, content_tokens

_LAT = [("bilateral", re.compile(r"\b(bilateral|bilat|both|bil)\b")),
        ("left", re.compile(r"\b(left|lt|l)\b")),
        ("right", re.compile(r"\b(right|rt|r)\b"))]
_ACU = [("acute", re.compile(r"\b(acute|acutely)\b")), ("chronic", re.compile(r"\b(chronic|chr|longstanding)\b"))]
_DMT = [("type1", re.compile(r"\b(type 1|type i|t1dm|dm1|iddm)\b")), ("type2", re.compile(r"\b(type 2|type ii|t2dm|dm2|niddm)\b"))]
_HIST = re.compile(r"\b(history of|hx of|h o|personal history|past history|status post|s p)\b")
_UNSPEC_SIDE = re.compile(r"unspecified (side|ear|eye|knee|shoulder|hip|foot|hand|arm|leg|lower|upper|limb|kidney|lung|breast|ankle|wrist|elbow)")

KIND = {"raw": 0, "clean": 1, "abbr": 2, "typo": 3, "none": 4, "span": 5}
SRC = {"dict": 0, "pref": 1, "cdc_index": 2, "expand": 3}

FEATURES = [
    "is_icd", "rrf", "fused_rank_log", "tfidf", "tfidf_rank_log", "bm25", "bm25_rank_log", "dense", "dense_rank_log",
    "origin", "best_kind", "best_src",
    "ce", "ce_rank", "ce_margin_top", "ce_gap_next", "ce_prob",
    "ce2", "ce2_rank", "ce2_prob",
    "mem_sim", "mem_exact", "mem_dsim", "prior_code", "prior_cat",
    "depth", "billable", "unspecified", "tag_compat", "tag_prior",
    "lat_match", "lat_conflict", "lat_unsupported", "unspec_side_ok",
    "acu_conflict", "acu_unsupported", "dmt_conflict", "hist_m", "hist_c",
    "jacc_pref", "cover_alias", "exact_alias", "exact_pref", "pref_len",
]
if not C.USE_SUBCONCEPT:  # tag_compat is the only feature that reads subconcept
    FEATURES.remove("tag_compat")


def _first(regs, s):
    for name, r in regs:
        if r.search(s):
            return name
    return None


def _softmax_rank(cand: pd.DataFrame, col: str, prefix: str) -> None:
    g = cand.groupby("mention")[col]
    mx = g.transform("max")
    cand[prefix + "_margin_top"] = cand[col] - mx
    cand[prefix + "_rank"] = g.rank(ascending=False, method="first")
    e = np.exp(cand[col] - mx)
    cand[prefix + "_prob"] = e / e.groupby(cand.mention).transform("sum")
    # gap to the best *other* candidate
    second = g.transform(lambda s: s.nlargest(2).iloc[-1] if len(s) > 1 else s.iloc[0])
    cand[prefix + "_gap_next"] = np.where(cand[col] >= mx, cand[col] - second, cand[col] - mx)


class FeatureBuilder:
    def __init__(self, concepts: dict, memory_df: pd.DataFrame):
        self.con = concepts
        m = memory_df
        self.prior = {s: Counter(m.loc[m.coding_system == s, "code"]) for s in C.SYSTEMS}
        self.cat_prior = Counter(c[:3] for c in m.loc[m.coding_system == C.ICD, "code"])
        sn = m[m.coding_system == C.SNOMED].copy()
        sn["tag"] = sn.code.map(concepts[C.SNOMED].tag)
        self.tag_sub = Counter(zip(sn.subconcept, sn.tag))
        self.sub_n = Counter(sn.subconcept)
        self.tag_n = Counter(sn.tag)
        self.n_tags = max(1, len(self.tag_n))
        self.n_sn = max(1, len(sn))

    def build(self, cand: pd.DataFrame, df: pd.DataFrame, system: str) -> pd.DataFrame:
        cand = cand.copy()
        con = self.con[system]
        info = con.loc[cand.code]
        cand["is_icd"] = int(system == C.ICD)
        for r in ("fused_rank", "tfidf_rank", "bm25_rank", "dense_rank"):
            cand[r + "_log"] = np.log1p(cand[r].astype(float))
        cand["best_kind"] = cand.best_kind.map(KIND).fillna(4)
        cand["best_src"] = cand.best_alias_src.map(SRC).fillna(0)
        if "ce_ctx" not in cand:
            cand["ce_ctx"] = 0.0
        if "ce_noctx" not in cand:
            cand["ce_noctx"] = 0.0
        cand["ce"] = cand.ce_ctx
        _softmax_rank(cand, "ce", "ce")
        cand["ce2"] = cand.ce_noctx
        _softmax_rank(cand, "ce2", "ce2")
        pr = self.prior[system]
        cand["prior_code"] = [math.log1p(pr.get(c, 0)) for c in cand.code]
        cand["prior_cat"] = [math.log1p(self.cat_prior.get(c[:3], 0)) if system == C.ICD else 0.0 for c in cand.code]
        if system == C.ICD:
            cand["depth"] = info.depth.to_numpy()
            cand["billable"] = info.billable.astype(int).to_numpy()
            cand["unspecified"] = info.unspecified.astype(int).to_numpy()
            cand["tag_compat"] = 0.0
            cand["tag_prior"] = 0.0
        else:
            cand["depth"] = 0
            cand["billable"] = 1
            cand["unspecified"] = 0
            subs = (df.subconcept.to_numpy()[cand.mention.to_numpy()] if C.USE_SUBCONCEPT and "subconcept" in df
                    else np.full(len(cand), "", dtype=object))
            tags = info.tag.to_numpy()
            cand["tag_compat"] = [math.log((self.tag_sub.get((s, t), 0) + 0.5) / (self.sub_n.get(s, 0) + 0.5 * self.n_tags))
                                  if s else 0.0 for s, t in zip(subs, tags)]
            cand["tag_prior"] = [math.log((self.tag_n.get(t, 0) + 0.5) / (self.n_sn + 0.5 * self.n_tags)) for t in tags]

        ments = [clean(x) for x in df.entity]
        ctxs = [clean(x)[:600] for x in df.chunk_text]
        m_lat = [_first(_LAT, m) for m in ments]
        m_acu = [_first(_ACU, m) for m in ments]
        m_dmt = [_first(_DMT, m + " " + c) for m, c in zip(ments, ctxs)]
        m_hist = [int(bool(_HIST.search(m))) for m in ments]
        m_tok = [set(content_tokens(m)) for m in ments]
        rows = {k: [] for k in ("lat_match", "lat_conflict", "lat_unsupported", "unspec_side_ok", "acu_conflict",
                                "acu_unsupported", "dmt_conflict", "hist_m", "hist_c", "jacc_pref", "cover_alias",
                                "exact_alias", "exact_pref", "pref_len")}
        prefs = info.preferred.to_numpy()
        for mi, pref, alias in zip(cand.mention.to_numpy(), prefs, cand.best_alias.to_numpy()):
            p = clean(pref)
            a = clean(alias)
            cl, ml = _first(_LAT, p), m_lat[mi]
            rows["lat_match"].append(int(cl is not None and cl == ml))
            rows["lat_conflict"].append(int(cl is not None and ml is not None and cl != ml))
            rows["lat_unsupported"].append(int(cl is not None and ml is None))
            rows["unspec_side_ok"].append(int(ml is None and bool(_UNSPEC_SIDE.search(p))))
            ca = _first(_ACU, p)
            rows["acu_conflict"].append(int(ca is not None and m_acu[mi] is not None and ca != m_acu[mi]))
            rows["acu_unsupported"].append(int(ca is not None and m_acu[mi] is None))
            cd = _first(_DMT, p)
            rows["dmt_conflict"].append(int(cd is not None and m_dmt[mi] is not None and cd != m_dmt[mi]))
            rows["hist_m"].append(m_hist[mi])
            rows["hist_c"].append(int("history" in p))
            pt, at, mt = set(content_tokens(p)), set(content_tokens(a)), m_tok[mi]
            rows["jacc_pref"].append(len(pt & mt) / max(1, len(pt | mt)))
            rows["cover_alias"].append(len(at & mt) / max(1, len(mt)))
            rows["exact_alias"].append(int(a == ments[mi]))
            rows["exact_pref"].append(int(p == ments[mi]))
            rows["pref_len"].append(len(pt))
        for k, v in rows.items():
            cand[k] = v
        return cand

"""Part 2 (light) features added on top of X's stacker features.

2.2 lexical traps : extra_words, missing_words, head-noun match, abbreviation-expansion match
2.3 checklist     : laterality, acuity, status (history/family/suspected), encounter (ICD 7th char),
                    diabetes type, stage/grade, with/without -> 0 none | 1 match | 2 term_only |
                    3 candidate_only | 4 contradiction; plus a veto mask for laterality/encounter/status.
Context window = +-12 words around the entity in chunk_text (rapidfuzz partial alignment; fallback first
40 words; empty without chunk_text).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from termmap import config as C
from termmap.normalize import STOP, clean, content_tokens

QUAL = {"left", "right", "bilateral", "acute", "chronic", "mild", "moderate", "severe", "unspecified", "other",
        "primary", "secondary", "history", "type", "stage", "grade", "with", "without", "initial", "subsequent"}


def stem(t: str) -> str:
    for suf in ("ies", "es", "s", "ing", "ed", "al", "ic"):
        if len(t) > 5 and t.endswith(suf):
            t = t[: -len(suf)]
            break
    return t[:7]


def stems(text: str) -> set:
    return {stem(t) for t in content_tokens(text)}


def window(entity: str, chunk: str, k: int = 12) -> str:
    if not isinstance(chunk, str) or not chunk.strip():
        return ""
    words = clean(chunk).split()
    low = " ".join(words)
    al = fuzz.partial_ratio_alignment(clean(entity), low, score_cutoff=60)
    if al is None:
        return " ".join(words[:40])
    a = len(low[: al.dest_start].split())
    b = len(low[: al.dest_end].split())
    return " ".join(words[max(0, a - k): b + k])


# ---- checklist slot extractors ------------------------------------------------------------
def _lat(s):
    s = " " + s + " "
    if re.search(r" (bilateral|bilat|both) ", s):
        return "bilateral"
    l, r = re.search(r" (left|lt) ", s), re.search(r" (right|rt) ", s)
    return "left" if l and not r else ("right" if r and not l else ("bilateral" if l and r else None))


def _acu(s):
    a, c = re.search(r"\bacute(ly)?\b", s), re.search(r"\b(chronic|chr|longstanding)\b", s)
    return "acute_on_chronic" if a and c else ("acute" if a else ("chronic" if c else None))


def _status(s):
    if re.search(r"\b(family history|fh|mother|father|sister|brother|maternal|paternal)\b", s):
        return "family"
    if re.search(r"\b(history of|hx|h o|personal history|past history|status post|s p|resolved)\b", s):
        return "history"
    if re.search(r"\b(suspected|possible|probable|rule out|r o|questionable|likely)\b", s):
        return "suspected"
    return None


def _enc_text(s):
    if re.search(r"\b(sequela|sequelae|late effect)\b", s):
        return "S"
    if re.search(r"\b(subsequent|follow up|healing|routine healing|f u)\b", s):
        return "D"
    if re.search(r"\b(initial|new|acute|today|ed visit|emergency)\b", s):
        return "A"
    return None


def _dm(s):
    if re.search(r"\b(type 1|type i|t1dm|dm1|iddm|juvenile)\b", s):
        return "1"
    if re.search(r"\b(type 2|type ii|t2dm|dm2|niddm)\b", s):
        return "2"
    return None


def _stage(s):
    m = re.search(r"\b(stage|grade|class|gold)\s*([0-9]+|i{1,3}v?|iv|v)\b", s)
    if not m:
        return None
    v = m.group(2)
    return {"i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5"}.get(v, v)


def _slot(t, c):
    if t is None and c is None:
        return 0
    if t is not None and c is not None:
        return 1 if t == c else 4
    return 2 if t is not None else 3


def add_features(X: pd.DataFrame, frames: dict, concepts: dict, abbr) -> pd.DataFrame:
    """X: X's feature table (qid = '<system>:<mention>'). frames: dev_frames()."""
    X = X.copy()
    sys_m = X.qid.str.split(":", n=1, expand=True)
    systems, mentions = sys_m[0].to_numpy(), sys_m[1].astype(int).to_numpy()
    cache = {}

    def term_info(s, m):
        k = (s, m)
        if k not in cache:
            d = frames[s]
            e, ch = d.entity.iat[m], d.chunk_text.iat[m]
            w = window(e, ch)
            te = clean(e)
            exp = []
            for t in te.split():
                exp += abbr.expansions(t, k=6) if len(t) <= 6 else []
            toks = [t for t in content_tokens(e) if t not in QUAL]
            cache[k] = dict(term=te, win=w, both=te + " " + w, st=stems(e), st_ctx=stems(e) | stems(w),
                            st_exp=set().union(*[stems(x) for x in exp]) if exp else set(),
                            head=stem(toks[-1]) if toks else "", has_abbr=bool(exp))
        return cache[k]

    feats = {k: [] for k in ("extra_words", "extra_frac", "missing_words", "missing_frac", "head_match", "abbr_match",
                            "ck_lat", "ck_acu", "ck_status", "ck_enc", "ck_dm", "ck_stage", "ck_without",
                            "veto")}
    for s, m, code, alias in zip(systems, mentions, X.code.to_numpy(), X.best_alias.to_numpy()):
        ti = term_info(s, m)
        pref = clean(concepts[s].preferred.get(code, ""))
        cst = stems(pref) | stems(alias)
        pst = {x for x in stems(pref) if x not in {stem(q) for q in QUAL}}
        support = ti["st_ctx"] | ti["st_exp"]
        extra = [x for x in pst if x not in support]
        term_need = (ti["st"] | ti["st_exp"]) if ti["has_abbr"] else ti["st"]
        miss = [x for x in ti["st"] if x not in cst and x not in {stem(q) for q in QUAL}]
        feats["extra_words"].append(len(extra))
        feats["extra_frac"].append(len(extra) / max(1, len(pst)))
        feats["missing_words"].append(len(miss))
        feats["missing_frac"].append(len(miss) / max(1, len(ti["st"])))
        feats["head_match"].append(int(bool(ti["head"]) and ti["head"] in cst))
        feats["abbr_match"].append(int(ti["has_abbr"] and bool(ti["st_exp"]) and len(ti["st_exp"] & cst) >= max(1, len(ti["st_exp"]) // 2)))
        lat = _slot(_lat(ti["term"]), _lat(pref))
        acu = _slot(_acu(ti["term"]), _acu(pref))
        stt = _slot(_status(ti["both"]), _status(pref))
        c7 = code.replace(".", "")
        enc = _slot(_enc_text(ti["both"]), c7[-1] if (s == C.ICD and len(c7) == 7 and c7[-1] in "ADS") else None)
        dm = _slot(_dm(ti["both"]), _dm(pref))
        stg = _slot(_stage(ti["term"]), _stage(pref))
        wo = 0
        mw = re.search(r"\bwithout ([a-z ]+)", pref)
        if mw and stems(mw.group(1)) & ti["st"]:
            wo = 4
        elif re.search(r"\bwith\b", pref) and not re.search(r"\bwith\b", ti["term"]):
            wo = 3
        for k, v in (("ck_lat", lat), ("ck_acu", acu), ("ck_status", stt), ("ck_enc", enc), ("ck_dm", dm),
                     ("ck_stage", stg), ("ck_without", wo)):
            feats[k].append(v)
        feats["veto"].append(int(lat == 4 or enc == 4 or stt == 4))
    for k, v in feats.items():
        X[k] = v
    return X


TRAP_FEATURES = ["extra_words", "extra_frac", "missing_words", "missing_frac", "head_match", "abbr_match"]
CHECK_FEATURES = ["ck_lat", "ck_acu", "ck_status", "ck_enc", "ck_dm", "ck_stage", "ck_without"]

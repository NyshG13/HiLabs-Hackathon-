"""Chunk-context helpers (context_v2): section heading, clause and co-entities. Pure functions.

Mention location = first case-insensitive match of the entity in the raw chunk. If the entity is not
found: section = "none", clause = the entity itself.
"""
from __future__ import annotations

import re

SECTIONS = ["history", "family", "social", "current_hpi", "ros", "assessment_plan", "meds_allergies",
            "labs_vitals_exam", "none"]

_HEADERS = [
    ("family", r"family history|family hx|fhx|fh"),
    ("social", r"social history|social hx|shx|sh"),
    ("history", r"past medical history|past surgical history|medical history|pmhx|pmh|pshx|psh"),
    ("current_hpi", r"history of present illness|hpi"),
    ("ros", r"review of systems|ros"),
    ("assessment_plan", r"assessment and plan|assessment & plan|assessment/plan|assessment|impression|a/p|plan|"
                        r"diagnosis|diagnoses|problem list|active problems"),
    ("meds_allergies", r"medications|medication|meds|allergies|allergy"),
    ("labs_vitals_exam", r"physical exam|physical examination|labs|lab results|vitals|vital signs|pe"),
]
# a header is a known heading followed by ':' / '-' (optionally after a few spaces), not inside a word
_HEADER_RE = re.compile(r"(?<![a-z0-9])(" + "|".join(f"(?P<{n}>{p})" for n, p in _HEADERS) + r")\s*(?::|-(?!\d))",
                        re.IGNORECASE)

# clause delimiters: the spec's , ; \n • | and numbered-list markers, plus this corpus's own list/table
# separators: '�' (mis-encoded bullet), '<br>', '::', ' - ' list dashes and '. ' sentence ends (the corpus
# has no newlines, so sentence ends play that role)
_DELIM_RE = re.compile(r"<br\s*/?>|::|[,;\n|•�]|(?:(?<=\s)|^)\d{1,2}[.)](?=\s)|\s-+\s|\.\s",
                       re.IGNORECASE)
_TABLE_LABEL = re.compile(r"header_\d+:\s*", re.IGNORECASE)
MAX_CLAUSE_WORDS = 25


def locate(entity: str, chunk: str) -> tuple[int, int] | None:
    if not entity or not chunk:
        return None
    i = chunk.lower().find(entity.strip().lower())
    if i < 0:
        return None
    return i, i + len(entity.strip())


def find_section(chunk: str, mention_start: int | None) -> str:
    """Nearest preceding section header -> coarse section label."""
    if mention_start is None or not chunk:
        return "none"
    best = "none"
    for m in _HEADER_RE.finditer(chunk, 0, mention_start):
        best = next(n for n, _ in _HEADERS if m.group(n))
    return best


def _pieces(chunk: str) -> list[tuple[int, int]]:
    spans, last = [], 0
    for m in _DELIM_RE.finditer(chunk):
        spans.append((last, m.start()))
        last = m.end()
    spans.append((last, len(chunk)))
    return spans


def get_clause(chunk: str, start: int | None, end: int | None, entity: str = "") -> str:
    """The delimited piece containing the mention; never cuts the mention. Under 3 words: extend by one
    neighbouring piece on each side."""
    if start is None or end is None or not chunk:
        return entity.strip()
    pieces = _pieces(chunk)
    idx = [k for k, (a, b) in enumerate(pieces) if a < end and b > start]
    if not idx:  # mention sits exactly on delimiters
        return entity.strip()
    lo, hi = idx[0], idx[-1]

    def span(lo, hi):
        a, b = min(pieces[lo][0], start), max(pieces[hi][1], end)
        heads = list(_HEADER_RE.finditer(chunk, a, start))  # drop a section header inside the clause
        if heads:
            a = heads[-1].end()
        return a, b

    a, b = span(lo, hi)
    text = chunk[a:b]
    if len(re.sub(r"<br\s*/?>|::", " ", text).split()) < 3:
        nonempty = lambda k: bool(_TABLE_LABEL.sub("", chunk[pieces[k][0]:pieces[k][1]]).strip())
        l2 = next((k for k in range(lo - 1, -1, -1) if nonempty(k)), lo)
        h2 = next((k for k in range(hi + 1, len(pieces)) if nonempty(k)), hi)
        a, b = span(l2, h2)
        text = chunk[a:b]
    if len(text.split()) > MAX_CLAUSE_WORDS:  # trim to +-10 words around the mention, never cutting it
        left, mid, right = chunk[a:start].split(), chunk[start:end], chunk[end:b].split()
        text = " ".join(left[-10:] + [mid] + right[:10])
    text = _TABLE_LABEL.sub(" ", re.sub(r"<br\s*/?>|::|[•�|]", " ", text, flags=re.IGNORECASE))
    return re.sub(r"\s+", " ", text).strip(" -:.\t")


def co_entities(entity: str, chunk_entities: list[str], limit: int | None = None) -> list[str]:
    """Other extracted entities of the same chunk (deduplicated, case-insensitive), excluding this one."""
    me = entity.strip().lower()
    out, seen = [], {me}
    for e in chunk_entities:
        k = e.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(e.strip())
    return out[:limit] if limit else out


def annotate(df) -> "pd.DataFrame":
    """Add m_start, m_end, section, clause, coents (list) to a frame with entity, chunk_text, group."""
    import pandas as pd
    df = df.copy()
    loc = [locate(e, c) for e, c in zip(df.entity, df.chunk_text)]
    df["m_start"] = [l[0] if l else -1 for l in loc]
    df["m_end"] = [l[1] if l else -1 for l in loc]
    df["section"] = [find_section(c, l[0]) if l else "none" for c, l in zip(df.chunk_text, loc)]
    df["clause"] = [get_clause(c, l[0], l[1], e) if l else e.strip() for e, c, l in zip(df.entity, df.chunk_text, loc)]
    ents = df.groupby("group", sort=False).entity.agg(list).to_dict()
    df["coents"] = [co_entities(e, ents.get(g, [])) for e, g in zip(df.entity, df.group)]
    return df

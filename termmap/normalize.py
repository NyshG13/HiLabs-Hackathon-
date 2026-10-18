"""Front end: cleaning, abbreviation expansion, high-precision typo gate and mention anchoring.

Variants are *added* as extra queries; the raw mention is always kept (plan section 3, step 2).
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict

from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein

_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"[a-z0-9]+(?:[+'][a-z0-9]*)?")
_HTML = re.compile(r"<br\s*/?>|<[^>]{1,20}>", re.I)
_HEADER = re.compile(r"header_\d+:\s*", re.I)

STOP = {"of", "the", "and", "or", "with", "without", "in", "on", "to", "for", "a", "an", "by", "at", "due",
        "as", "from", "is", "nos", "not", "other", "unspecified", "specified"}


def clean(s: str) -> str:
    """Lower-case, strip accents/markup/punctuation, collapse whitespace. Digits and '+' survive."""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = _HTML.sub(" ", s.lower())
    s = s.replace("&", " and ").replace("w/o", " without ").replace("w/", " with ").replace("h/o", " history of ")
    s = s.replace("s/p", " status post ").replace("c/o", " complains of ")
    s = re.sub(r"[^a-z0-9+'/ ]+", " ", s)
    s = s.replace("/", " ")
    return _WS.sub(" ", s).strip()


def tokens(s: str) -> list[str]:
    return _TOKEN.findall(clean(s))


def content_tokens(s: str) -> list[str]:
    return [t for t in tokens(s) if t not in STOP]


def clean_context(chunk: str) -> str:
    if not isinstance(chunk, str):
        return ""
    s = _HEADER.sub(" ", chunk)
    s = _HTML.sub(" ", s).replace("::", " | ")
    return _WS.sub(" ", s).strip()


def anchor(mention: str, chunk: str, window: int = 220) -> tuple[str, bool]:
    """Return the chunk window with the mention wrapped in [ms] ... [me] markers.

    Exact (case-insensitive) match first, fuzzy alignment second; if neither works the
    mention is prepended so the reranker always sees an anchored mention.
    """
    ctx = clean_context(chunk)
    m = mention.strip()
    if not ctx:
        return f"[ms] {m} [me]", False
    low = ctx.lower()
    i = low.find(m.lower())
    found = i >= 0
    if found:
        j = i + len(m)
    else:
        al = fuzz.partial_ratio_alignment(m.lower(), low, score_cutoff=80)
        if al is not None:
            i, j, found = al.dest_start, al.dest_end, True
    if not found:
        return f"[ms] {m} [me] | " + ctx[: 2 * window], False
    a, b = max(0, i - window), min(len(ctx), j + window)
    return (("... " if a > 0 else "") + ctx[a:i] + " [ms] " + ctx[i:j] + " [me] " + ctx[j:b]
            + (" ..." if b < len(ctx) else "")), True


# ----------------------------------------------------------------------------------------------
# Abbreviations
# ----------------------------------------------------------------------------------------------
_ABBR_DESC = re.compile(r"^([A-Za-z][A-Za-z0-9+]{1,7})\s+-\s+(.+)$")
_ABBR_PAREN = re.compile(r"^(.+?)\s+\(([A-Z][A-Z0-9+]{1,7})\)$")


def _initials(words: list[str]) -> str:
    return "".join(w[0] for w in words if w)


class AbbreviationTable:
    """abbr -> Counter(expansion). Mined from dictionary descriptions ("AR - Aortic regurgitation",
    "Chronic kidney disease (CKD)") and from training gold (entity token whose letters are the initials
    of consecutive words of the gold display)."""

    def __init__(self, table: dict[str, Counter] | None = None):
        self.table: dict[str, Counter] = table or defaultdict(Counter)

    def add(self, abbr: str, expansion: str, w: int = 1) -> None:
        a, e = clean(abbr), clean(expansion)
        if not a or not e or a == e or len(a) > 8 or len(e) <= len(a):
            return
        self.table.setdefault(a, Counter())[e] += w

    def mine_descriptions(self, descriptions) -> None:
        for d in descriptions:
            if not isinstance(d, str):
                continue
            m = _ABBR_DESC.match(d)
            if m and sum(c.isupper() for c in m.group(1)) >= 2:
                self.add(m.group(1), m.group(2))
                continue
            m = _ABBR_PAREN.match(d)
            if m:
                self.add(m.group(2), m.group(1))

    def mine_gold(self, entities, displays) -> None:
        for ent, disp in zip(entities, displays):
            et, dw = tokens(ent), [w for w in tokens(disp) if w not in {"of", "the", "and", "with", "in"}]
            if not dw:
                continue
            for t in et:
                if not (2 <= len(t) <= 6) or not t.isalpha():
                    continue
                for i in range(len(dw)):
                    for n in range(2, min(6, len(dw) - i) + 1):
                        if _initials(dw[i:i + n]) == t:
                            self.add(t, " ".join(dw[i:i + n]), 2)

    def expansions(self, abbr: str, k: int = 3) -> list[str]:
        c = self.table.get(abbr)
        return [e for e, _ in c.most_common(k)] if c else []

    def to_json(self) -> dict:
        return {a: dict(c.most_common(8)) for a, c in self.table.items()}

    @classmethod
    def from_json(cls, d: dict) -> "AbbreviationTable":
        return cls({a: Counter(c) for a, c in d.items()})


# ----------------------------------------------------------------------------------------------
# Typo gate
# ----------------------------------------------------------------------------------------------
class TypoGate:
    """High-precision spelling repair against the dictionary vocabulary.

    A token is repaired only when it is out of vocabulary, long enough, and has a *unique* best
    in-vocabulary neighbour at edit distance 1 (or 2 for tokens of 8+ characters). Run-together words
    ("chronickidney") are split when both halves are frequent vocabulary words.
    """

    def __init__(self, vocab: Counter, min_freq: int = 3):
        self.vocab = vocab
        self.common = {w: c for w, c in vocab.items() if c >= min_freq and len(w) >= 3}
        self.by_first: dict[str, list[str]] = defaultdict(list)
        for w in self.common:
            self.by_first[w[0]].append(w)

    def fix_token(self, t: str) -> str | None:
        if len(t) < 4 or not t.isalpha() or t in self.vocab:
            return None
        maxd = 2 if len(t) >= 8 else 1
        pool = [w for w in self.by_first.get(t[0], []) if abs(len(w) - len(t)) <= maxd]
        hits = process.extract(t, pool, scorer=Levenshtein.distance, score_cutoff=maxd, limit=3)
        if hits:
            best = hits[0][1]
            ties = [h for h in hits if h[1] == best]
            if len(ties) == 1 or self.common[ties[0][0]] >= 5 * self.common[ties[1][0]]:
                return ties[0][0]
        for i in range(3, len(t) - 2):  # run-together words
            a, b = t[:i], t[i:]
            if self.common.get(a, 0) >= 20 and self.common.get(b, 0) >= 20:
                return f"{a} {b}"
        return None

    def fix(self, text: str) -> str | None:
        toks, changed = tokens(text), False
        out = []
        for t in toks:
            f = self.fix_token(t)
            out.append(f or t)
            changed |= f is not None
        return " ".join(out) if changed else None


def variants(mention: str, abbr: AbbreviationTable, typo: TypoGate | None, context: str = "") -> list[tuple[str, str]]:
    """Return [(kind, query)], raw first. Abbreviation expansions are ranked by overlap with context."""
    raw = mention.strip()
    out = [("raw", raw)]
    c = clean(raw)
    if c and c != raw.lower():
        out.append(("clean", c))
    toks = tokens(raw)
    ctx = set(content_tokens(context))
    seen = {q for _, q in out}
    for i, t in enumerate(toks):
        if t in STOP or t.isdigit() or len(t) > 6:
            continue
        exps = abbr.expansions(t, k=6)
        if not exps or (t in (typo.vocab if typo else {}) and len(t) > 4):
            continue
        exps.sort(key=lambda e: -len(set(e.split()) & ctx))  # context-conditioned order (stable)
        for e in exps[:3]:
            q = " ".join(toks[:i] + [e] + toks[i + 1:])
            if q not in seen:
                out.append(("abbr", q))
                seen.add(q)
    if typo is not None:
        f = typo.fix(raw)
        if f and f not in seen:
            out.append(("typo", f))
    return out

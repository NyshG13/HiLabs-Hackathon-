"""Synthetic noisy mentions for every dictionary concept (SynCABEL-style, plan step 5).

Meaning-preserving corruptions only: OCR/typo edits, run-together words, truncation, stopword drop,
abbreviation contraction (expansion -> short form, from the mined table) and word re-ordering.
"""
from __future__ import annotations

import random

from .normalize import AbbreviationTable, STOP, clean

_KEYB = "qwertyuiopasdfghjklzxcvbnm"


def _typo(w: str, rng: random.Random) -> str:
    if len(w) < 4:
        return w
    i = rng.randrange(1, len(w) - 1)
    op = rng.randrange(4)
    if op == 0:
        return w[:i] + w[i + 1:]
    if op == 1:
        return w[:i] + w[i + 1] + w[i] + w[i + 2:]
    if op == 2:
        return w[:i] + rng.choice(_KEYB) + w[i:]
    return w[:i] + rng.choice(_KEYB) + w[i + 1:]


def noisy_variant(text: str, rng: random.Random, abbr: AbbreviationTable | None = None,
                  reverse: dict[str, str] | None = None) -> str:
    words = clean(text).split()
    if not words:
        return text
    ops = rng.sample(range(7), k=rng.choice([1, 1, 2]))
    for op in ops:
        if op == 0:
            j = rng.randrange(len(words))
            words[j] = _typo(words[j], rng)
        elif op == 1 and len(words) >= 2:
            j = rng.randrange(len(words) - 1)
            words[j:j + 2] = [words[j] + words[j + 1]]
        elif op == 2 and len(words[-1]) >= 6:
            words[-1] = words[-1][: rng.randrange(4, len(words[-1]))]
        elif op == 3:
            kept = [w for w in words if w not in STOP]
            words = kept or words
        elif op == 4 and reverse:
            s = " ".join(words)
            for exp, ab in reverse.items():
                if exp in s:
                    s = s.replace(exp, ab)
                    break
            words = s.split()
        elif op == 5 and len(words) >= 2:
            j = rng.randrange(len(words) - 1)
            words[j], words[j + 1] = words[j + 1], words[j]
        elif op == 6:
            words = [w.upper() if len(w) <= 4 and rng.random() < 0.3 else w for w in words]
    return " ".join(words)


def reverse_abbreviations(abbr: AbbreviationTable, min_count: int = 2) -> dict[str, str]:
    """expansion -> abbreviation, for the most frequent unambiguous pairs (multi-word expansions)."""
    rev = {}
    for a, c in abbr.table.items():
        e, n = c.most_common(1)[0]
        if n >= min_count and " " in e and len(a) >= 2:
            rev.setdefault(e, a)
    return dict(sorted(rev.items(), key=lambda kv: -len(kv[0])))

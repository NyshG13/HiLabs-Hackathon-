"""Shared candidate generation for training and inference."""
from __future__ import annotations

import random

import pandas as pd

from . import config as C
from .retrieval import GoldMemory, Mention, SystemRetriever
from .synth import noisy_variant, reverse_abbreviations


def mentions_for(df: pd.DataFrame) -> list[Mention]:
    return [Mention(e, c, g) for e, c, g in zip(df.entity, df.chunk_text, df.group)]


def generate(df: pd.DataFrame, system: str, R: SystemRetriever, enc, mem: GoldMemory | None, ab, tg,
             leave_group_out: bool = False) -> pd.DataFrame:
    """df rows are mentions (entity, chunk_text, group). Returns candidates with `mention` = df position."""
    cand = R.retrieve(mentions_for(df), ab, tg, encoder=enc, memory=mem, use_memory_groups=leave_group_out)
    cand["system"] = system
    return cand


def synthetic_queries(n: int, train: pd.DataFrame, ab, seed: int = C.SEED) -> pd.DataFrame:
    """Noisy variants of dictionary aliases, sampled per system in proportion to training rows and,
    for SNOMED, in proportion to the semantic-tag mix of the training gold."""
    from .resources import load_aliases, load_concepts
    rng = random.Random(seed)
    rev = reverse_abbreviations(ab)
    rows = []
    share = train.coding_system.value_counts(normalize=True)
    for system in C.SYSTEMS:
        k = int(round(n * share.get(system, 0)))
        con = load_concepts(system)
        al = load_aliases(system)
        by_code = al.groupby("code").alias.apply(list)
        t = train[train.coding_system == system]
        if system == C.ICD:
            pool = con[con.billable].code.tolist()
            sub_of = lambda c: "DiagnosisCode"
        else:
            tag = con.set_index("code").tag
            gold_tags = t.code.map(tag).value_counts(normalize=True)
            sub_mode = t.assign(tag=t.code.map(tag)).groupby("tag").subconcept.agg(lambda s: s.mode().iat[0])
            by_tag = con.groupby("tag").code.apply(list)
            pool = []
            for tg_, p in gold_tags.items():
                if tg_ in by_tag.index:
                    pool += rng.sample(by_tag[tg_], min(len(by_tag[tg_]), max(1, int(p * k * 3))))
            sub_of = lambda c: sub_mode.get(tag.get(c, ""), "")
        for c in rng.sample(pool, min(k, len(pool))):
            alias = rng.choice(by_code[c])
            rows.append(dict(entity=noisy_variant(alias, rng, ab, rev), chunk_text="", group=f"synth-{len(rows)}",
                             coding_system=system, subconcept=sub_of(c), code=c, synthetic=1))
    return pd.DataFrame(rows)

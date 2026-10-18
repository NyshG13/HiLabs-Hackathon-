"""Training stages. Each stage caches its output so stages can be re-run independently.

  cands       candidates for train (+synthetic) rows and for dev/test rows
  cross       context-aware cross-encoder        -> models/cross-ctx
  comparator  mention-only cross-encoder          -> models/cross-noctx  (R&D cascade re-implementation)
  score       cross-encoder scores for dev/test candidates
  stacker     cross-fitted LightGBM stacker, joint head, LLM adjudicator gate (see stack.py)
"""
from __future__ import annotations

import gc
import random

import pandas as pd

from . import config as C
from .data import load_split

CAND = C.CACHE / "candidates"
CAND.mkdir(parents=True, exist_ok=True)


def _free():
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass


def stage_cands(splits=("train", "dev", "test")) -> None:
    from .candidates import generate, synthetic_queries
    from .dense import Encoder
    from .retrieval import GoldMemory, SystemRetriever, load_frontend
    C.seed_everything()
    ab, tg = load_frontend()
    train = load_split("train")
    enc = Encoder()
    mem = GoldMemory(train)
    mem.attach_dense(enc)
    synth = synthetic_queries(int(len(train) * C.SYNTH_RATIO), train, ab)
    synth.to_parquet(CAND / "synthetic_rows.parquet")
    for system in C.SYSTEMS:
        R = SystemRetriever(system)
        for name in splits:
            df = load_split(name)
            df = df[df.coding_system == system].reset_index(drop=True)
            cand = generate(df, system, R, enc, mem, ab, tg, leave_group_out=(name == "train"))
            cand.to_parquet(CAND / f"{name}_{system}.parquet")
            print(name, system, len(df), "rows", len(cand), "cands")
        s = synth[synth.coding_system == system].reset_index(drop=True)
        generate(s, system, R, enc, None, ab, tg).to_parquet(CAND / f"synth_{system}.parquet")
        del R
        _free()


def _concepts():
    from .resources import load_concepts
    return {s: load_concepts(s).set_index("code") for s in C.SYSTEMS}


def stage_cross(context: bool = True) -> None:
    from .reranker import build_groups, train_cross
    rng = random.Random(C.SEED)
    con = _concepts()
    train = load_split("train")
    synth = pd.read_parquet(CAND / "synthetic_rows.parquet")
    groups = []
    for system in C.SYSTEMS:
        parts = [(train[train.coding_system == system].reset_index(drop=True), f"train_{system}")]
        if context:  # synthetic variants only for the full system (the comparator has no such step)
            parts.append((synth[synth.coding_system == system].reset_index(drop=True), f"synth_{system}"))
        for df, name in parts:
            cand = pd.read_parquet(CAND / f"{name}.parquet")
            q = [dict(system=system, entity=e, chunk=c, subconcept=s, code=k)
                 for e, c, s, k in zip(df.entity, df.chunk_text, df.subconcept, df.code)]
            groups += build_groups(q, cand, con, rng, context=context)
    print(f"cross-encoder groups: {len(groups)} (context={context})")
    out = C.MODELS / ("cross-ctx" if context else "cross-noctx")
    train_cross(groups, out, max_len=C.CROSS_MAX_LEN if context else C.COMPARATOR_MAX_LEN)


def stage_score(splits=("dev", "test")) -> None:
    """Score every candidate of dev/test with both cross-encoders."""
    from .reranker import CrossEncoder, cand_text, query_text
    con = _concepts()
    for model_name, ctx in (("cross-ctx", True), ("cross-noctx", False)):
        path = C.MODELS / model_name
        if not (path / "config.json").exists():
            continue
        ce = CrossEncoder(str(path), max_len=C.CROSS_MAX_LEN if ctx else C.COMPARATOR_MAX_LEN)
        for name in splits:
            df = load_split(name)
            for system in C.SYSTEMS:
                d = df[df.coding_system == system].reset_index(drop=True)
                p = CAND / f"{name}_{system}.parquet"
                cand = pd.read_parquet(p)
                qa = [query_text(system, d.entity.iat[m], d.chunk_text.iat[m], d.subconcept.iat[m], ctx)
                      for m in cand.mention]
                cb = [cand_text(system, con[system].loc[c], a) for c, a in zip(cand.code, cand.best_alias)]
                cand["ce_ctx" if ctx else "ce_noctx"] = ce.score(qa, cb)
                cand.to_parquet(p)
                print("scored", model_name, name, system)
        del ce
        _free()

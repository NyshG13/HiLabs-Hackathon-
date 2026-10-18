"""Context-aware cross-encoder reranker (plan section 3, step 5).

Input A: system, CodEx subconcept, the mention, and the chunk window with the mention anchored by
[ms] ... [me]. Input B: candidate preferred term, the alias that matched, and its parent path
(ICD: category and block; SNOMED: semantic tag). Trained listwise (1 positive vs 7 hard negatives:
top retrieved, hierarchy siblings, specificity-only differences) on gold rows mixed with synthetic noisy
variants. `context=False` gives the comparator's mention-only cross-encoder.
"""
from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from . import config as C
from .normalize import anchor, clean

SYS_NAME = {C.ICD: "icd-10-cm", C.SNOMED: "snomed ct"}


def query_text(system: str, mention: str, chunk: str, subconcept: str, context: bool = True) -> str:
    sub = subconcept if (subconcept and C.USE_SUBCONCEPT) else "unknown"
    if not context:
        return f"{SYS_NAME[system]} : {mention}"
    ctx, _ = anchor(mention, chunk, window=160)
    return f"{SYS_NAME[system]} | {sub} : {mention} || {ctx}"


V2_TAG = {C.ICD: "[ICD]", C.SNOMED: "[SNOMED]"}
V2_TOKENS = ["[M]", "[/M]", "[ICD]", "[SNOMED]"]


def _mark(text: str, start: int, end: int) -> str:
    return text[:start] + "[M] " + text[start:end] + " [/M]" + text[end:]


def query_text_v2(system: str, mention: str, chunk: str, section: str, clause: str, coents: list[str],
                  tok=None, budget: int | None = None, ctx_chars: int = 200) -> str:
    """context_v2 query side:
    "[ICD]/[SNOMED] | section: S | clause: ..[M] mention [/M].. | context: ~200 chars marked | also in note: e1; ..e5"
    With a tokenizer and a token budget, only the context part is shortened to fit; clause and mention never."""
    from .context import locate
    from .normalize import clean_context
    i = clause.lower().find(mention.strip().lower())
    cl = _mark(clause, i, i + len(mention.strip())) if i >= 0 else f"[M] {mention.strip()} [/M] {clause}"
    raw = clean_context(chunk)
    loc = locate(mention, raw)
    also = "; ".join(e[:40] for e in coents[:5])

    def build(nchars: int) -> str:
        ctx = ""
        if loc and nchars > 0:
            a, b = loc
            half = max(0, (nchars - (b - a)) // 2)
            lo, hi = max(0, a - half), min(len(raw), b + half)
            ctx = _mark(raw[lo:hi], a - lo, b - lo)
        parts = [V2_TAG[system], f"section: {section}", f"clause: {cl}"]
        if ctx:
            parts.append(f"context: {ctx}")
        if also:
            parts.append(f"also in note: {also}")
        return " | ".join(parts)

    q = build(ctx_chars)
    if tok is not None and budget is not None:
        n = ctx_chars
        while n > 0 and len(tok(q, add_special_tokens=False).input_ids) > budget:
            n = n // 2 if n > 40 else 0
            q = build(n)
    return q


def cand_text(system: str, concept: pd.Series | dict, alias: str) -> str:
    pref = concept["preferred"]
    parts = [pref]
    if alias and clean(alias) != clean(pref):
        parts.append(alias)
    if system == C.ICD:
        parts.append(f"{concept['category_desc']} ; {concept['block_desc']}")
        if not concept["billable"]:
            parts.append("category header")
    else:
        parts.append(f"({concept['tag']})")
    return " | ".join(parts)


def encode_pairs(tok, a, b, max_len):
    """Truncate the query side first; fall back to truncating both when a candidate alone is too long."""
    try:
        return tok(a, b, padding=True, truncation="only_first", max_length=max_len, return_tensors="pt")
    except Exception:
        return tok(a, b, padding=True, truncation="longest_first", max_length=max_len, return_tensors="pt")


class CrossEncoder:
    def __init__(self, path: str, max_len: int = C.CROSS_MAX_LEN):
        self.dev = C.device()
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=1).to(self.dev).eval()
        if self.dev.type == "cuda":
            self.model.half()
        self.max_len = max_len

    @torch.inference_mode()
    def score(self, a: list[str], b: list[str], batch: int = 64) -> np.ndarray:
        out = np.zeros(len(a), np.float32)
        order = np.argsort([len(x) + len(y) for x, y in zip(a, b)], kind="stable")
        for i in range(0, len(a), batch):
            idx = order[i:i + batch]
            enc = encode_pairs(self.tok, [a[j] for j in idx], [b[j] for j in idx], self.max_len).to(self.dev)
            out[idx] = self.model(**enc).logits[:, 0].float().cpu().numpy()
        return out


# ----------------------------------------------------------------------------------------------
def build_groups(queries: list[dict], cands: pd.DataFrame, concepts: dict, rng: random.Random,
                 n_neg: int = C.CROSS_NEG_PER_QUERY, context: bool = True,
                 query_fn=None) -> list[tuple[str, list[str]]]:
    """queries[i]: dict(system, entity, chunk, subconcept, code). cands: retrieval output for them."""
    groups = []
    by_m = {m: g for m, g in cands.groupby("mention", sort=True)}
    for mi, q in enumerate(queries):
        system, gold = q["system"], q["code"]
        con = concepts[system]
        if gold not in con.index:
            continue
        g = by_m.get(mi)
        neg_pool = [] if g is None else [r for r in g.itertuples() if r.code != gold]
        gold_pref = clean(con.at[gold, "preferred"])
        neg_pool = [r for r in neg_pool if clean(con.at[r.code, "preferred"]) != gold_pref]  # false negatives
        hard = [r for r in neg_pool if r.fused_rank < 10]
        sib = [r for r in neg_pool if r.origin == 1]
        rest = [r for r in neg_pool if r.fused_rank >= 10 and r.origin != 1]
        chosen = hard[:3] + rng.sample(sib, min(2, len(sib)))
        chosen = list({r.code: r for r in chosen}.values())
        pool = [r for r in hard[3:] + rest + sib if r.code not in {c.code for c in chosen}]
        chosen += rng.sample(pool, min(n_neg - len(chosen), len(pool)))
        if not chosen:
            continue
        sub = q["subconcept"] if rng.random() > 0.3 else ""  # robust to a missing subconcept column
        qa = query_fn(q, sub) if query_fn is not None else query_text(system, q["entity"], q["chunk"], sub, context)
        gold_alias = ""
        if g is not None and (g.code == gold).any():
            gold_alias = g[g.code == gold].best_alias.iat[0]
        texts = [cand_text(system, con.loc[gold], gold_alias)]
        texts += [cand_text(system, con.loc[r.code], r.best_alias) for r in chosen[:n_neg]]
        groups.append((qa, texts))
    return groups


def train_cross(groups: list[tuple[str, list[str]]], out_dir, max_len: int = C.CROSS_MAX_LEN,
                epochs: int = C.CROSS_EPOCHS, special_tokens=("[ms]", "[me]")) -> None:
    C.seed_everything()
    dev = C.device()
    tok = AutoTokenizer.from_pretrained(C.CROSS_BASE)
    tok.add_special_tokens({"additional_special_tokens": list(special_tokens)})
    model = AutoModelForSequenceClassification.from_pretrained(C.CROSS_BASE, num_labels=1)
    model.resize_token_embeddings(len(tok))
    model.gradient_checkpointing_enable()
    model.to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=C.CROSS_LR, weight_decay=0.01)
    bq = C.CROSS_BATCH_QUERIES
    steps = (len(groups) // bq) * epochs // C.CROSS_ACCUM
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.06 * steps)) * max(0.0, 1 - s / steps))
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    rng = random.Random(C.SEED)
    it = 0
    for ep in range(epochs):
        order = list(range(len(groups)))
        rng.shuffle(order)
        for i in range(0, len(order) - bq + 1, bq):
            a, b, sizes = [], [], []
            for gi in order[i:i + bq]:
                qa, texts = groups[gi]
                a += [qa] * len(texts)
                b += texts
                sizes.append(len(texts))
            enc = encode_pairs(tok, a, b, max_len).to(dev)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                logits = model(**enc).logits[:, 0].float()
            loss, o = 0.0, 0
            for s in sizes:  # listwise softmax, positive is index 0 of each group
                loss = loss + torch.nn.functional.cross_entropy(logits[o:o + s][None], torch.zeros(1, dtype=torch.long, device=dev))
                o += s
            loss = loss / len(sizes) / C.CROSS_ACCUM
            scaler.scale(loss).backward()
            it += 1
            if it % C.CROSS_ACCUM == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            if it % 500 == 0:
                print(f"cross ep {ep} it {it}/{len(order) // bq * epochs} loss {loss.item() * C.CROSS_ACCUM:.3f}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

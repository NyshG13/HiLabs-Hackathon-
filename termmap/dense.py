"""Dense retrieval with SapBERT: encoder, alias index, contrastive fine-tuning (plan step 4)."""
from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from . import config as C
from .normalize import clean
from .synth import noisy_variant

DENSE_DIR = C.MODELS / "sapbert-ft"


class Encoder:
    def __init__(self, path: str | None = None):
        path = path or (str(DENSE_DIR) if (DENSE_DIR / "config.json").exists() else C.SAPBERT)
        self.dev = C.device()
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path).to(self.dev).eval()
        if self.dev.type == "cuda":
            self.model.half()
        self.path = path

    @torch.inference_mode()
    def encode(self, texts: list[str], batch: int = 256, max_len: int = C.DENSE_MAX_LEN) -> np.ndarray:
        out = []
        order = np.argsort([len(t) for t in texts], kind="stable")  # length bucketing; result re-ordered
        for i in range(0, len(texts), batch):
            idx = order[i:i + batch]
            enc = self.tok([texts[j] for j in idx], padding=True, truncation=True, max_length=max_len,
                           return_tensors="pt").to(self.dev)
            v = self.model(**enc).last_hidden_state[:, 0]
            v = torch.nn.functional.normalize(v.float(), dim=-1)
            out.append(v.cpu().numpy().astype(np.float16))
        emb = np.concatenate(out) if out else np.zeros((0, 768), np.float16)
        res = np.empty_like(emb)
        res[order] = emb
        return res


class DenseIndex:
    """Alias embeddings kept as float16; search is a blocked matrix product on GPU (or CPU)."""

    def __init__(self, emb: np.ndarray):
        self.dev = C.device()
        self.emb = torch.from_numpy(emb).to(self.dev)
        if self.dev.type != "cuda":
            self.emb = self.emb.float()

    @torch.inference_mode()
    def search(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        qs = torch.from_numpy(q).to(self.dev).to(self.emb.dtype)
        S, I = [], []
        for i in range(0, len(qs), 64):
            s = qs[i:i + 64] @ self.emb.T
            v, ix = torch.topk(s.float(), k, dim=1)
            S.append(v.cpu().numpy())
            I.append(ix.cpu().numpy())
        return np.concatenate(S), np.concatenate(I)


def build_dense_index(system: str, aliases: pd.DataFrame, enc: Encoder, out_path=None) -> np.ndarray:
    p = out_path or (C.TRAINDEP / f"dense_{system}.npy")
    emb = enc.encode(aliases.alias_clean.tolist())
    np.save(p, emb)
    return emb


def load_dense_index(system: str) -> np.ndarray:
    return np.load(C.TRAINDEP / f"dense_{system}.npy")


# ----------------------------------------------------------------------------------------------
def fine_tune(train: pd.DataFrame, n_synth: int = 30000, n_syn_pairs: int = 30000, out_dir=None,
              abbr_path=None) -> None:
    """In-batch contrastive (InfoNCE) fine-tuning of SapBERT on:
    gold (mention -> gold description), dictionary synonym pairs, and synthetic noisy variants."""
    from .resources import load_aliases, load_concepts
    from .normalize import AbbreviationTable
    import json

    C.seed_everything()
    rng = random.Random(C.SEED)
    ab = AbbreviationTable.from_json(json.loads((abbr_path or C.TRAINDEP / "abbreviations.json").read_text(encoding="utf8")))
    pairs: list[tuple[str, str, str]] = []
    for system in C.SYSTEMS:
        al = load_aliases(system)
        by_code = al.groupby("code").alias_clean.apply(list).to_dict()
        con = load_concepts(system).set_index("code").preferred
        t = train[train.coding_system == system]
        for ent, code in zip(t.entity, t.code):
            if code not in by_code:
                continue
            pairs.append((clean(ent), clean(con.get(code, by_code[code][0])), code))
            pairs.append((clean(ent), rng.choice(by_code[code]), code))
        codes = [c for c, v in by_code.items() if len(v) > 1]
        gold = list(set(t.code) & set(by_code))
        sample = gold + rng.sample(codes, min(len(codes), n_syn_pairs // 2))
        for c in sample:
            v = by_code[c]
            if len(v) > 1:
                a, b = rng.sample(v, 2)
                pairs.append((a, b, c))
        for c in rng.sample(list(by_code), min(len(by_code), n_synth // 2)):
            a = rng.choice(by_code[c])
            pairs.append((noisy_variant(a, rng, ab), clean(con.get(c, a)), c))
    rng.shuffle(pairs)
    print(f"dense fine-tune pairs: {len(pairs)}")

    dev = C.device()
    tok = AutoTokenizer.from_pretrained(C.SAPBERT)
    model = AutoModel.from_pretrained(C.SAPBERT).to(dev).train()
    # avoid two pairs of the same concept in one batch (they would be false in-batch negatives)
    batches, cur, seen = [], [], set()
    for a, b, c in pairs:
        if c in seen:
            continue
        cur.append((a, b))
        seen.add(c)
        if len(cur) == C.DENSE_BATCH:
            batches.append(cur)
            cur, seen = [], set()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total = len(batches) * C.DENSE_EPOCHS
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100) * max(0.0, 1 - s / total))
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")

    def emb(texts):
        enc = tok(texts, padding=True, truncation=True, max_length=C.DENSE_MAX_LEN, return_tensors="pt").to(dev)
        return torch.nn.functional.normalize(model(**enc).last_hidden_state[:, 0].float(), dim=-1)

    step = 0
    for ep in range(C.DENSE_EPOCHS):
        for bt in batches:
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                qa, qb = emb([x[0] for x in bt]), emb([x[1] for x in bt])
                logits = qa @ qb.T * 20.0
                lab = torch.arange(len(bt), device=dev)
                loss = (torch.nn.functional.cross_entropy(logits, lab) + torch.nn.functional.cross_entropy(logits.T, lab)) / 2
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 200 == 0:
                print(f"dense step {step}/{total} loss {loss.item():.3f}", flush=True)
    out_dir = out_dir or DENSE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

"""Step 8: selective LLM adjudication with a local open-weight model (Qwen2.5-1.5B-Instruct, Apache-2.0).

The LLM never writes codes. It sees the anchored chunk, the term and up to five dictionary-valid
candidates in a fixed order (stacker rank) and we read the next-token distribution over the option
letters: one forward pass, no sampling, batch size 1, so the output is deterministic for a fixed input.
Decisions are cached on disk keyed by the prompt hash. No data leaves the machine.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import torch

from . import config as C
from .normalize import anchor

LETTERS = "ABCDE"
CACHE_FILE = C.CACHE / "llm_cache.jsonl"
SYS = {C.ICD: "ICD-10-CM", C.SNOMED: "SNOMED CT"}


def build_prompt(system: str, mention: str, chunk: str, options: list[tuple[str, str]]) -> list[dict]:
    ctx, _ = anchor(mention, chunk, window=250)
    ctx = ctx.replace("[ms]", "<<").replace("[me]", ">>")
    lines = "\n".join(f"{LETTERS[i]}. {code} - {desc}" for i, (code, desc) in enumerate(options))
    user = (f"Chart excerpt (the term is marked << >>):\n{ctx}\n\nTerm: {mention}\nTarget terminology: {SYS[system]}\n\n"
            f"Which option is the correct {SYS[system]} code for this term as used in the excerpt? "
            "Expand abbreviations using the excerpt. Choose the most specific option the text supports; if a "
            "qualifier (side, acuity, type, complication) is not stated, prefer the unspecified option.\n\n"
            f"{lines}\n\nAnswer with the letter only.")
    return [{"role": "system", "content": "You are an experienced certified medical coder."},
            {"role": "user", "content": user}]


class Adjudicator:
    def __init__(self, model: str = C.LLM):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.dev = C.device()
        self.tok = AutoTokenizer.from_pretrained(model)
        dtype = torch.float16 if self.dev.type == "cuda" else torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=dtype).to(self.dev).eval()
        self.letter_ids = [self.tok.encode(x, add_special_tokens=False)[0] for x in LETTERS]
        self.cache = {}
        if CACHE_FILE.exists():
            for line in CACHE_FILE.read_text(encoding="utf8").splitlines():
                d = json.loads(line)
                self.cache[d["k"]] = d["p"]

    @torch.inference_mode()
    def choose(self, system: str, mention: str, chunk: str, options: list[tuple[str, str]]) -> np.ndarray:
        msgs = build_prompt(system, mention, chunk, options)
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        key = hashlib.sha1((C.LLM + text).encode("utf8")).hexdigest()
        if key in self.cache:
            return np.asarray(self.cache[key])
        ids = self.tok(text, return_tensors="pt").to(self.dev)
        logits = self.model(**ids).logits[0, -1].float()
        lp = torch.log_softmax(logits[self.letter_ids[:len(options)]], dim=-1).cpu().numpy()
        self.cache[key] = lp.round(5).tolist()
        with CACHE_FILE.open("a", encoding="utf8") as f:
            f.write(json.dumps({"k": key, "p": self.cache[key]}) + "\n")
        return np.asarray(self.cache[key])

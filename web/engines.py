"""Engine adapter: TermMap-X+ (github.com/NyshG13/HiLabs-Hackathon-).

The console talks to this one interface:  map_rows(rows) -> (results, timing)
    rows     [{chunk_id, clinical_entity, chunk_text}, ...]
    results  one dict per row, same order:
             {chunk_id, clinical_entity, icd: {code, name, confidence, top3}, snomed: {...}}

X+ is called through its own subconcept-free entry point, predict_frame(), exactly as `python predict.py` does, so the
console and the CLI give the same codes. X+ returns one calibrated probability per system (no top-3 list), so `top3`
is empty and the console simply shows no alternatives.
"""
import os
import sys
from pathlib import Path


class XPlusEngine:
    name = "TermMap-X+"

    def __init__(self, root, use_llm=True):
        root = Path(root).expanduser().resolve()
        if not (root / "termmap_plus" / "predict_sf.py").exists():
            raise FileNotFoundError(f"no TermMap-X+ checkout at {root} (set --engine-dir or TERMMAP_XPLUS_DIR)")
        # the weights are Git LFS objects: a clone made without LFS holds ~130-byte pointer files instead
        probe = root / "models" / "cross-ctx" / "model.safetensors"
        if probe.exists() and probe.stat().st_size < 1_000_000:
            raise RuntimeError("model weights are Git LFS pointers; run `git lfs install && git lfs pull` in " + str(root))
        sys.path.insert(0, str(root))
        import pandas as pd
        from termmap import config as C
        from termmap_plus import core as K
        from termmap_plus import predict_sf as P
        self.pd, self.C, self.K, self.P = pd, C, K, P
        self.use_llm = bool(use_llm)
        self.root = root
        self.con = K.concepts()                       # dictionaries: the legal answer space, and the code names

    def info(self):
        return {"name": self.name, "llm": self.use_llm, "device": str(self.C.device())}

    def _name(self, system, code):
        try:
            return str(self.con[system].at[code, "preferred"])
        except Exception:
            return ""

    def map_rows(self, rows):
        df = self.pd.DataFrame({"entity": [r["clinical_entity"] for r in rows],
                                "chunk_text": [r["chunk_text"] for r in rows],
                                "chunk_id": [r["chunk_id"] for r in rows]})
        final, timing = self.P.predict_frame(df, use_llm=self.use_llm)
        if len(final) != len(rows):
            raise RuntimeError(f"engine returned {len(final)} rows for {len(rows)} inputs")
        out = []
        for r, (_, f) in zip(rows, final.iterrows()):
            def side(system, code_col, conf_col):
                code = str(f[code_col])
                return {"code": code, "name": self._name(system, code), "confidence": float(f[conf_col]), "top3": []}
            out.append({"chunk_id": r["chunk_id"], "clinical_entity": r["clinical_entity"],
                        "icd": side(self.C.ICD, "icd10cm_code", "confidence_icd"),
                        "snomed": side(self.C.SNOMED, "snomed_code", "confidence_snomed")})
        return out, {k: timing.get(k) for k in ("terms", "seconds", "ms_per_term", "device", "llm", "routed_to_gate")}

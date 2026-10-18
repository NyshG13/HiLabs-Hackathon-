"""Determinism of the subconcept-free predict path: 50 diagnosis rows whose gold is unseen in train (entity +
chunk_text only), predicted twice, each run with EVERY cache empty (CE, Qwen3, gate). Writes
submission/determinism.json with byte-equality, latency and peak GPU memory.
  python -m round3.sf_determinism"""
from __future__ import annotations

import termmap_plus  # noqa: F401
import filecmp
import json
import shutil

import pandas as pd

from termmap.data import load_split
from termmap_plus import core as K

OUT = K.ROOT / "submission"
TMP = K.ART / "sf" / "det"


def rows(n=50) -> pd.DataFrame:
    seen = K.seen_codes()
    t = load_split("test")
    t = t[t.subconcept == K.CONST_SUB]  # selection of evaluation rows only; the input file has no subconcept
    t = t[[c not in seen[s] for s, c in zip(t.coding_system, t.code)]]
    t = t.drop_duplicates(["entity", "chunk_text"]).sample(n=n, random_state=13)
    return t[["entity", "chunk_text"]].reset_index(drop=True)


def main():
    from termmap_plus import predict_sf as PS
    OUT.mkdir(exist_ok=True)
    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True)
    inp = TMP / "unseen50.csv"
    rows().to_csv(inp, index=False)
    t = []
    for r in (1, 2):
        fresh = TMP / f"cache_run{r}"
        t.append(PS.main(["--input", str(inp), "--output", str(TMP / f"pred_run{r}.csv"), "--fresh-cache", str(fresh)]))
    same = filecmp.cmp(TMP / "pred_run1.csv", TMP / "pred_run2.csv", shallow=False)
    res = {"rows": 50, "byte_identical": same, "runs": t}
    (OUT / "determinism.json").write_text(json.dumps(res, indent=1), encoding="utf8")
    print(f"DETERMINISM: byte-identical {same}; " + "; ".join(
        f"run{i + 1} {x['seconds']} s ({x['ms_per_term']} ms/term), retrieval p50/p95 {x['retrieval_p50_p95_ms']} ms, "
        f"peak GPU {x['peak_gpu_mb']} MB, gate {x['routed_to_gate']}" for i, x in enumerate(t)), flush=True)


if __name__ == "__main__":
    main()

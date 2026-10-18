"""Score a predictions CSV against the internal test split (diagnosis rows), the way the README reports results.

  python score.py --pred submission/evaluation_predictions.csv

Exact match per system, ICD 3-character category match, accuracy on unseen codes (gold code never a gold code of a
diagnosis row in data/splits/train.csv), each with a 95% Wilson interval. Rows are matched on (entity, chunk_text),
case- and whitespace-insensitive.
"""
import argparse
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent


def norm(s: str) -> str:
    return " ".join(str(s).lower().split())


def wilson(k: int, n: int, z: float = 1.96) -> str:
    if n == 0:
        return "n/a"
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"{100 * p:.1f}% [{100 * (c - h):.1f}, {100 * (c + h):.1f}] (n={n})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gold", default=str(ROOT / "data" / "splits" / "test.csv"))
    a = ap.parse_args()
    gold = pd.read_csv(a.gold, dtype=str, keep_default_na=False)
    gold = gold[gold.subconcept == "DiagnosisCode"].copy()  # scoring scope only; the engine never reads this column
    train = pd.read_csv(ROOT / "data" / "splits" / "train.csv", dtype=str, keep_default_na=False)
    train = train[train.subconcept == "DiagnosisCode"]
    pred = pd.read_csv(a.pred, dtype=str, keep_default_na=False)
    ent = "clinical_entity" if "clinical_entity" in pred else "entity"
    pred["k"] = [norm(e) + "||" + norm(c) for e, c in zip(pred[ent], pred["chunk_text"])]
    pred = pred.drop_duplicates("k").set_index("k")
    gold["k"] = [norm(e) + "||" + norm(c) for e, c in zip(gold.entity, gold.chunk_text)]
    for system, col in (("ICD10CM", "icd10cm_code"), ("SNOMEDCT_US", "snomed_code")):
        g = gold[gold.coding_system == system]
        p = g.k.map(pred[col]).fillna("")
        code = g.code.str.strip()
        ok = p == code
        seen = set(train.code[train.coding_system == system].str.strip())
        unseen = ~code.isin(seen)
        print(f"{system:12s} exact      {wilson(int(ok.sum()), len(g))}   missing predictions: {int((p == '').sum())}")
        print(f"{'':12s} unseen     {wilson(int(ok[unseen].sum()), int(unseen.sum()))}")
        if system == "ICD10CM":
            cat = p.str[:3] == code.str[:3]
            print(f"{'':12s} category   {wilson(int(cat.sum()), len(g))}   (first 3 characters)")


if __name__ == "__main__":
    main()

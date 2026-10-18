"""Input loading, schema normalisation and the grouped 80/10/10 split."""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


def download_raw(force: bool = False) -> None:
    C.RAW.mkdir(parents=True, exist_ok=True)
    for name, url in C.DATA_URLS.items():
        p = C.RAW / name
        if p.exists() and not force:
            continue
        print(f"downloading {name}")
        urllib.request.urlretrieve(url, p)


def load_input(path: str | Path) -> pd.DataFrame:
    """Read an input file and map it to the canonical schema.

    Accepted columns: entity | clinical_entity, chunk_text (optional), chunk_id (optional),
    coding_system (optional; ICD10CM / SNOMEDCT_US), subconcept (optional). Gold columns
    code / icd10cm_code / snomed_code are kept when present (used only for evaluation).
    """
    path = Path(path)
    df = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path, dtype=str, keep_default_na=False)
    df = df.rename(columns={"clinical_entity": "entity"})
    if "entity" not in df.columns:
        raise ValueError(f"{path} needs an 'entity' or 'clinical_entity' column; got {list(df.columns)}")
    for col in ("chunk_text", "chunk_id", "coding_system", "subconcept"):
        if col not in df.columns:
            df[col] = ""
    df = df.fillna("")
    for col in ("entity", "chunk_text", "chunk_id", "coding_system", "subconcept"):
        df[col] = df[col].astype(str)
    if "code" in df.columns:
        df["code"] = df["code"].astype(str).str.strip()
    df["group"] = np.where(df.chunk_id.str.len() > 0, df.chunk_id, df.chunk_text)
    df["row_id"] = np.arange(len(df))
    return df


def mention_key(entity: str, chunk_text: str) -> str:
    return hashlib.md5((entity.strip() + "\x1f" + chunk_text).encode("utf8")).hexdigest()


def make_split(seed: int = C.SEED) -> dict[str, pd.DataFrame]:
    """Grouped by chunk: 80% train / 10% dev / 10% untouched internal test.

    All rows (both coding systems, every entity) of a chunk land in the same part, which stops
    near-copies from leaking across parts.
    """
    df = load_input(C.TRAIN_CSV)
    groups = df.group.drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups))
    n = len(groups)
    a, b = int(C.SPLIT_FRACTIONS[0] * n), int((C.SPLIT_FRACTIONS[0] + C.SPLIT_FRACTIONS[1]) * n)
    part = {}
    for i, gi in enumerate(order):
        part[groups[gi]] = "train" if i < a else ("dev" if i < b else "test")
    df["split"] = df.group.map(part)
    if C.VARIANT == "diag":  # same chunk-grouped partition as the full run, restricted to diagnoses
        df = df[df.subconcept == "DiagnosisCode"].reset_index(drop=True)
    out = {}
    for s in ("train", "dev", "test"):
        d = df[df.split == s].drop(columns=["split"]).reset_index(drop=True)
        d.to_csv(C.SPLITS / f"{s}.csv", index=False)
        out[s] = d
    # evaluation-style input (no gold) for the internal test, to exercise the predict command
    out["test"][["entity", "chunk_text", "coding_system", "subconcept"]].to_csv(C.SPLITS / "test_input.csv", index=False)
    return out


def load_split(name: str) -> pd.DataFrame:
    p = C.SPLITS / f"{name}.csv"
    if not p.exists():
        make_split()
    return load_input(p)

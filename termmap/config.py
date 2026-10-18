"""Central configuration: paths, seeds and hyper-parameters.

Everything that influences predictions lives here so a run is reproducible from one place.
Values can be overridden with environment variables (see .env.example).
"""
from __future__ import annotations

import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

DATA = ROOT / "data"
RAW = DATA / "raw"
EXTERNAL = DATA / "external"
PROCESSED = DATA / "processed"
SPLITS = DATA / "splits"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
CACHE = ROOT / "cache"

os.environ.setdefault("HF_HOME", str(MODELS / "hf_cache"))

# Experiment variant: TERMMAP_VARIANT=diag keeps only subconcept == DiagnosisCode rows and redirects every
# train-dependent artefact (splits, abbreviations, vocab, SapBERT, dense indexes, candidates, cross-encoders,
# stacker, outputs) to artifacts/<variant>/ and outputs/<variant>/. Dictionary tables stay shared.
VARIANT = os.environ.get("TERMMAP_VARIANT", "")
TRAINDEP = PROCESSED  # abbreviations.json, vocab.json, dense_<system>.npy
if VARIANT:
    _VROOT = ROOT / "artifacts" / VARIANT
    SPLITS, MODELS, CACHE, TRAINDEP = _VROOT / "splits", _VROOT / "models", _VROOT / "cache", _VROOT / "processed"
    OUTPUTS = ROOT / "outputs" / VARIANT

TRAIN_CSV = RAW / "hackathon_train.csv"
ICD_XLSX = RAW / "icd10cm_codes.xlsx"
SNOMED_XLSX = RAW / "SNOMEDCT_US_codes_and_descriptions.xlsx"
DATA_URLS = {
    "hackathon_train.csv": "https://hiringdrive2026.blob.core.windows.net/hackathon/hackathon_train.csv",
    "icd10cm_codes.xlsx": "https://hiringdrive2026.blob.core.windows.net/hackathon/icd10cm_codes.xlsx",
    "SNOMEDCT_US_codes_and_descriptions.xlsx":
        "https://hiringdrive2026.blob.core.windows.net/hackathon/SNOMEDCT_US_codes_and_descriptions.xlsx",
}
CDC_YEAR = os.environ.get("TERMMAP_CDC_YEAR", "2026")
CDC_BASE = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM"
CDC_DIR = EXTERNAL / "cdc" / f"fy{CDC_YEAR}"

ICD = "ICD10CM"
SNOMED = "SNOMEDCT_US"
SYSTEMS = (ICD, SNOMED)

SEED = int(os.environ.get("TERMMAP_SEED", "13"))
SPLIT_FRACTIONS = (0.8, 0.1, 0.1)  # train / dev / internal test, grouped by chunk

# models (all open weight, downloaded from the Hugging Face hub)
SAPBERT = os.environ.get("TERMMAP_SAPBERT", "cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
CROSS_BASE = os.environ.get("TERMMAP_CROSS_BASE", "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
LLM = os.environ.get("TERMMAP_LLM", "Qwen/Qwen2.5-1.5B-Instruct")
USE_LLM = os.environ.get("TERMMAP_USE_LLM", "1") == "1"
DEVICE = os.environ.get("TERMMAP_DEVICE", "auto")

# SNOMED semantic tags that never carry a chart "Problem/Procedure/Lab/Device/Encounter" target.
SNOMED_EXCLUDED_TAGS = {
    "organism", "medicinal product", "medicinal product form", "clinical drug", "cell", "cell structure",
    "geographic location", "namespace concept", "foundation metadata concept", "ISBT symbol",
    "religion/philosophy", "dose form", "product", "AJCC", "linkage concept", "number",
    "OWL metadata concept", "core metadata concept", "inactive concept", "racial group", "ethnic group",
    "unit of presentation", "basic dose form", "state of matter", "transformation", "role",
    "supplier", "administration method", "release characteristic", "intended site",
}
# but keep any concept that is a gold code in training regardless of tag
# retrieval
TOPK_PER_RETRIEVER = 100
TOPK_FUSED = 50
RRF_K = 60
RRF_WEIGHTS = {"dense": 1.0, "tfidf": 0.5, "bm25": 0.25}
MAX_CANDIDATES = 64
RERANK_TOPN = 32
# reranker
CROSS_MAX_LEN = 192
COMPARATOR_MAX_LEN = 96
CROSS_NEG_PER_QUERY = 7
CROSS_EPOCHS = int(os.environ.get("TERMMAP_CROSS_EPOCHS", "2"))
CROSS_LR = 2e-5
CROSS_BATCH_QUERIES = 2
CROSS_ACCUM = 4
SYNTH_RATIO = 0.5  # synthetic noisy variants mixed with gold rows
# dense fine-tuning
DENSE_EPOCHS = 1
DENSE_BATCH = 64
DENSE_MAX_LEN = 32
# The evaluation file has no `subconcept` column, so by default nothing at inference reads it: the cross-encoder
# query gets "unknown", the stacker drops tag_compat, the pipeline neither predicts nor uses it. (Training data
# may still carry it.) TERMMAP_USE_SUBCONCEPT=1 restores the original vanilla behaviour.
USE_SUBCONCEPT = os.environ.get("TERMMAP_USE_SUBCONCEPT", "0") == "1"
# context_v2 flags (all OFF by default; the vanilla pipeline is unchanged when they are off)
SPAN_EXTENSION = os.environ.get("TERMMAP_SPAN_EXTENSION", "0") == "1"    # clause as an extra retrieval query
STACKER_CONTEXT = os.environ.get("TERMMAP_STACKER_CONTEXT", "0") == "1"  # section/clause/co-entity stacker features
CE_CONTEXT_V2 = os.environ.get("TERMMAP_CE_CONTEXT_V2", "0") == "1"      # enriched cross-encoder query side
ARTIFACTS_V2 = (ROOT / "artifacts" / VARIANT / "context_v2") if VARIANT else (ROOT / "artifacts" / "context_v2")
OUTPUTS_V2 = OUTPUTS / "context_v2"
# abstention: below this calibrated confidence a prediction is flagged
ABSTAIN_THRESHOLD = float(os.environ.get("TERMMAP_ABSTAIN", "0.5"))


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    except ImportError:
        pass


def device():
    import torch
    if DEVICE != "auto":
        return torch.device(DEVICE)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


for _p in (PROCESSED, SPLITS, MODELS, OUTPUTS, CACHE, TRAINDEP):
    _p.mkdir(parents=True, exist_ok=True)

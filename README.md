# TermMap-X+: clinical mention → ICD-10-CM + SNOMED CT

Give it a CSV of clinical mentions, each with the chart chunk it appears in. It returns a CSV with one ICD-10-CM code and one SNOMED CT concept per row. Everything runs locally: there are no API calls and no data leaves the machine.

## Abstract

We map a clinical mention, read in its chunk of the chart, to one ICD-10-CM code and one SNOMED CT concept, using only the entity and the chunk text.

The system extends Approach X, a retrieve–score–stack cascade.

- **Candidate pool.** One shared pool comes from BM25, character TF-IDF and a fine-tuned SapBERT, fused by reciprocal rank. It adds a memory of training terms and hierarchy expansion, and for ICD also tree siblings and convention-kNN codes. The gold code is in the pool for 94.3% of ICD rows and 89.6% of SNOMED rows.
- **Scoring.** Cross-encoders and lexical features score every candidate.
- **Decision.** A separate LightGBM stacker per system, averaged over 20 seeds, picks the answer. A Qwen2.5 gate re-checks only the low-confidence SNOMED picks.

No single model exceeds 58% top-1. The stacker lifts this to 74.9% ICD and 67.5% SNOMED out-of-fold, on 470 ICD and 541 SNOMED diagnosis rows.¹

On a blind test set (grouped by chunk, scored once), top-1 is 72.4% ICD [66.5, 77.7] and 70.6% SNOMED [64.9, 75.6]. Changes made after the blind test were kept only if they passed a pre-declared keep rule.

¹ This repository ships the `xplus-sf` build. The 68.4% SNOMED figure in our architecture diagram includes an extra CE-2 A reranker that is **not** part of this build.

## Quick start: run the engine end to end

**Requirements:**
- Python 3.11
- [Git LFS](https://git-lfs.com). The trained weights are stored in this repository with it.
- About 8 GB of free disk space.
- An NVIDIA GPU is optional but much faster; the code was tested on a 4 GB RTX 2050.

```bash
git lfs install                                             # once per machine, BEFORE cloning
git clone <this repo> termmap-xplus && cd termmap-xplus     # also downloads the 2.9 GB of trained weights (LFS)
python -m venv .venv && source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt                             # torch 2.14.1 + CUDA 12.6 wheel (see requirements.txt)
python setup_model.py                                       # checks the weights, downloads the 2 public Qwen models
```

**One command reproduces the predictions from the evaluation input file:**

```bash
python predict.py --input <evaluation_file.csv> --output predictions.csv
```

`make predict INPUT=<evaluation_file.csv> OUTPUT=predictions.csv` does the same.

It runs without manual intervention and is deterministic: the same input always gives the same output.

**Input:** a CSV (or XLSX), one row per term.

| column | required | meaning |
|---|---|---|
| `clinical_entity` (or `entity`) | yes | the extracted term, verbatim |
| `chunk_text` | strongly recommended | the chart text the term was extracted from; the engine uses it to disambiguate (e.g. *MS*, *RA*) |
| `chunk_id` | optional | echoed to the output |

Any other column (e.g. `display`, `subconcept`, `coding_system`, gold codes) is dropped unread before anything runs. The output is byte-identical with or without such columns.

```csv
clinical_entity,chunk_text
htn,"pmh: htn, hld, dm2. bp 150/90 today, continue lisinopril."
MS,"echo: rheumatic mitral valve with moderate ms, mean gradient 8 mmhg."
```

**Output:** one row per input row.

| column | meaning |
|---|---|
| `chunk_id` and/or `chunk_text` | echoed from the input (whichever the input had) |
| `clinical_entity` | echoed from the input |
| `icd10cm_code` | predicted ICD-10-CM code: one code, always a valid code from the supplied ICD-10-CM table |
| `snomed_code` | predicted SNOMED CT concept id: one concept, always a valid concept from the supplied SNOMED CT table |
| `confidence_icd`, `confidence_snomed` | the stacker's calibrated probability for each pick |
| `confidence` | the lower of the two. Low values flag likely errors and can be used for abstention or review. |

**Check your install:** `python predict.py --input data/splits/test_input.csv --output out.csv` should reproduce `submission/reference_predictions.csv` (our internal test set) exactly on a CUDA GPU.

## Environment variables

None are required, and no API keys are used anywhere: every model runs locally, so no data leaves the machine. Optional overrides go in a `.env` file. Copy [`.env.example`](.env.example); `.env` is git-ignored.

| variable | default | effect |
|---|---|---|
| `TERMMAP_DEVICE` | `auto` | `auto` (GPU if available) / `cuda` / `cpu` |
| `TERMMAP_USE_LLM` | `1` | `0` skips the Qwen2.5 SNOMED gate (same as `--no-llm`; faster, about 2 pt lower SNOMED) |
| `TERMMAP_SEED` | `13` | the fixed seed of every sampled step |
| `HF_HOME` | `./models/hf_cache` | where the Qwen2.5 cache lives |

## Model weights

| what | where |
|---|---|
| **Our trained weights:** fine-tuned SapBERT (`models/sapbert-ft/`), X's context and mention-only cross-encoders (`models/cross-ctx/`, `models/cross-noctx/`), dense + lexical alias indexes (`data/processed/dense_*.npy`, `lexical_*.pkl`). 2.9 GB, 16 files | **in this repository, via Git LFS.** Every file's sha256 is in [`artifacts/WEIGHTS_MANIFEST.json`](artifacts/WEIGHTS_MANIFEST.json) and is checked by `python setup_model.py`. |
| 20-seed LightGBM stackers + recipe | in this repository (`artifacts/plus/sf/`) |
| Qwen2.5-1.5B-Instruct: SNOMED gate, used as is (not fine-tuned) | downloaded by `setup_model.py` from https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct @ `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` (Apache-2.0; its 3.1 GB weight file exceeds GitHub's LFS file limit) |
| Qwen3-Reranker-0.6B: ICD zero-shot scorer, used as is | downloaded by `setup_model.py` from https://huggingface.co/Qwen/Qwen3-Reranker-0.6B @ `e61197ed45024b0ed8a2d74b80b4d909f1255473` (Apache-2.0) |
| Bases of our fine-tuned models | `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` (Apache-2.0), `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` (MIT) |

If `setup_model.py` reports "Git LFS pointers", the clone was made without Git LFS. Run `git lfs install && git lfs pull`.

**Windows:** clone into a short folder (e.g. `C:\work	ermmap-xplus`). Some model-cache paths are about 120 characters deep inside the repo, and Windows refuses paths over 260 characters unless long paths are enabled (`LongPathsEnabled` registry setting).

## Demo video

Link to be added (2–3 min walkthrough: architecture, live hard cases, confidence output).

## Architecture

```
INPUT                1 · RETRIEVE (shared)            2 · SCORE (shared)                   3 · DECIDE (per system)           OUTPUT
entity          ┌──────────────────────────┐   ┌───────────────────────────────┐   ┌──────────────────────────────┐
+ chunk_text ──▶│ BM25 · char TF-IDF ·      │──▶│ X's cross-encoder with and     │──▶│ ICD stacker: LightGBM, own    │──▶ ICD-10-CM code
(no other       │ fine-tuned SapBERT, RRF   │   │ without chart context          │   │ model, 20 seeds; LLM gate off │    74.9% OOF · blind 72.4%
 column read)   │ + memory of training terms│   │ retrieval scores/ranks, memory │   ├──────────────────────────────┤
                │ + hierarchy expansion     │   │ hits, kNN votes                │   │ SNOMED stacker: LightGBM,     │──▶ SNOMED CT concept
                │ ICD: tree siblings (unspec│   │ lexical-trap features,         │   │ shared, 20 seeds; then the    │    67.5% OOF · blind 70.6%
                │ laterality, 7th char)     │   │ diagnosis-type tag flag        │   │ Qwen2.5 gate re-checks only   │
                │ + convention-kNN codes    │   │ ICD: Qwen3-Reranker-0.6B       │   │ low-confidence picks (p<0.6)  │
                └──────────────────────────┘   └───────────────────────────────┘   └──────────────────────────────┘
                gold in pool: ICD 94.3 · SNOMED 89.6   best single scorer: ICD 52.3 · SNOMED 55.8
                fused top-1:  ICD 41.5 · SNOMED 48.2   (X's cross-encoder, subconcept removed from the query)
```

| stage | component | code |
|---|---|---|
| retrieve | char TF-IDF + BM25 + fine-tuned SapBERT, reciprocal-rank fusion, ≤ 64 per system; training-term memory; hierarchy expansion | `termmap/retrieval.py`, `termmap/dense.py` |
| retrieve (ICD) | tree siblings (unspecified / laterality / 7th character); top-3 convention-kNN codes | `termmap_plus/pools.py`, `termmap_plus/v2b_knn.py` |
| score | X's BiomedBERT cross-encoders (context query `system : mention ‖ window`, no subconcept; mention-only) | `termmap/reranker.py`, `round3/sf.py` |
| score | X's 43 features (`tag_compat` → `cand_is_diag_tag`); lexical traps; kNN votes | `termmap/features.py`, `termmap_plus/features_plus.py` |
| score (ICD) | Qwen3-Reranker-0.6B zero-shot P(yes) on the top-10 | `termmap_plus/v2_qwen3.py` |
| decide | LightGBM lambdarank, 20 seeds averaged: ICD-only stacker; shared stacker for SNOMED | `termmap/stack.py`, `artifacts/plus/sf/stackers_none_dtag.pkl` |
| decide (SNOMED) | Qwen2.5-1.5B gate on rows with p < 0.6: top-5, log p + 0.25 · LLM letter log-prob | `termmap/llm.py`, `termmap_plus/predict_sf.py` |
| entry point | input guard → pipeline → CSV | `predict.py` → `termmap_plus/predict_sf.py` |

The frozen recipe is [`artifacts/plus/sf/sf_recipe.json`](artifacts/plus/sf/sf_recipe.json). It holds the features, the 20 seeds, the diagnosis-type tag list and the memory splits.

**No subconcept at prediction.** Nothing reads `subconcept`, not even a constant:
- the input guard keeps only `entity`, `chunk_text` and `chunk_id`;
- the cross-encoder query has no subconcept field;
- the one feature derived from subconcept (`tag_compat`) is replaced by a flag computed from the candidate alone.

`make test` checks that a file with extra `subconcept` / `coding_system` columns gives byte-identical output.

## Results

| | ICD-10-CM | SNOMED CT |
|---|---|---|
| gold code in the candidate pool | 94.3 | 89.6 |
| top-1 of the fused retrieval ranking | 41.5 | 48.2 |
| best single scorer (X's cross-encoder) | 52.3 | 55.8 |
| **this build, out-of-fold** (pooled dev+test, 20 seeds) | **74.9** | **67.5** |
| blind internal test, scored once (v1, 243 / 275 rows) | 72.4 [66.5, 77.7] | 70.6 [64.9, 75.6] |

The out-of-fold numbers were also used to select components. A held-out replay of that selection gives 68.7% ICD and 68.9% SNOMED: components were chosen on 60% of the chunks and scored on the other 40%. That is our conservative estimate for unseen data.

## Repository layout

| path | contents |
|---|---|
| `predict.py`, `setup_model.py`, `Makefile` | entry points |
| `termmap/` | Approach X engine (retrieval, cross-encoders, features, stacker, LLM gate) |
| `termmap_plus/`, `round3/` | X+ additions and the subconcept-free predict path (`termmap_plus/predict_sf.py`) |
| `artifacts/plus/sf/` | frozen recipe + 20-seed stackers (`stackers_none_dtag.pkl`) |
| `artifacts/WEIGHTS_MANIFEST.json` | sha256 of every trained weight file |
| `models/sapbert-ft/`, `models/cross-ctx/`, `models/cross-noctx/` | trained models (Git LFS) |
| `models/stack_settings.pkl` | stacker temperature |
| `data/raw/` | hackathon training CSV, ICD-10-CM and SNOMED CT code tables |
| `data/splits/` | train / dev / test split (seed 13, grouped by chunk); training memory |
| `data/processed/` | concept / alias tables, abbreviations, typo vocabulary, dense + lexical alias indexes (Git LFS) |
| `data/external/cdc/` | CDC FY2026 ICD-10-CM order file + Alphabetic Index (public domain) |
| `shared/` | ICD tree, scope |
| `tests/test_sf_guard.py` | input-guard test |
| `submission/reference_predictions.csv` | expected output for `data/splits/test_input.csv` |

## Security

- **No API keys or tokens are in this repository**, and none are needed to run it.
- The two Qwen models are public; downloading them needs no account or token.
- `.env` is git-ignored; `.env.example` lists the optional settings and contains no secrets.

## Licences and data

- **Models:**
  - SapBERT: Apache-2.0
  - BiomedBERT: MIT
  - Qwen2.5 and Qwen3-Reranker: Apache-2.0
- **CDC ICD-10-CM files:** public domain.
- **Hackathon data:** `data/raw/` and `data/splits/` are the hackathon's data and are included because this repository is private. Do not make the repository public with them inside.
- **SNOMED CT:** use of the SNOMED CT table is subject to your SNOMED licence.

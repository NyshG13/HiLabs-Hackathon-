# TermMap-X+ (cut-down plan): summary

Everything here is built on top of vanilla TermMap-X, which is untouched (`python vanilla_snapshot/verify.py` passes; 49 files). New code is in `termmap_plus/`, flags in `configs/plus.yaml`, artefacts in `artifacts/plus/` and `shared/`, decisions in `DECISIONS.md`, and the step-by-step ledger in `results/ledger.md`.

Evaluation format matches the judges' file: diagnosis rows only (subconcept = DiagnosisCode) and input = entity + chunk_text only. The leak fix: X read `subconcept` in its cross-encoder query and in the stacker feature `tag_compat`. Every diagnosis row has the same single value, so X+ feeds the constant `DiagnosisCode`. `coding_system` is never read, and both systems are predicted for every row.

## Dev ledger (decisions made here; 227 ICD / 266 SNOMED gold rows)

| step | change | status | ICD | SNOMED | min | Δmin |
|---|---|---|---|---|---|---|
| 0 | X baseline (single-seed stacker, LLM gate both) | baseline | 0.670 | 0.632 | 0.632 | |
| 1.1 | X's LLM gate per system (ICD off, SNOMED on) | KEPT (per system) | 0.696 | 0.632 | 0.632 | 0 (ICD +2.6) |
| 1.R | reference: rebuilt pools, X stacker averaged over 5 seeds | reference | 0.696 | 0.602 | 0.602 | |
| 1.3 | SNOMED semantic-tag scope (99.5% coverage) | DROPPED | 0.700 | 0.605 | 0.605 | +0.4 |
| 1.9 | ICD tree + sibling expansion | KEPT | 0.709 | 0.617 | 0.617 | +1.5 |
| 2.2 | lexical-trap features | KEPT | 0.709 | 0.632 | 0.632 | +1.5 |
| 2.3 | clinical checklist + veto | DROPPED | 0.696 | 0.620 | 0.620 | −1.1 |

Why 1.R exists: refitting X's single-seed LightGBM stacker on near-identical features flipped about 9% of predictions. Every step is therefore compared against a 5-seed reference with the same retrieval path.

## Internal test, scored once after the dev decisions (243 ICD / 275 SNOMED gold rows)

| run | ICD | SNOMED | min | unseen | ICD ceiling |
|---|---|---|---|---|---|
| X as-is (saved vanilla stacker, LLM gate both) | 0.683 | 0.695 | 0.683 | 0.628 | 0.934 |
| X reference (5-seed stacker, gate ICD off / SNOMED on) | 0.716 | 0.709 | **0.709** | 0.628 | 0.934 |
| **X+ final** (+ ICD tree/siblings + lexical traps) | **0.724** | 0.706 | 0.706 | 0.628 | **0.955** |

Against X as-is, the final model gains +2.3 points of min, with 19 rows fixed and 6 broken. Most of that gain comes from the cheap changes: the per-system LLM gate and seed averaging. The ICD tree raises the ICD ceiling by 2.1 points and ICD accuracy by 0.8 over the reference. On test, the lexical-trap gain seen on dev (SNOMED +1.5) does not repeat (−0.4). With 243–275 rows per system, differences under about 3 points are within noise.

## Not done (cut for time)
1.2, 1.4–1.8, 1.10, 2.0–2.1, 2.4–2.9, the leave-one-out ablation, and the train+dev+test refit of non-neural parts. The predict command offers `--memory train,dev,test` for the gold memory.

## One command
`python -m termmap_plus predict --input eval.csv --output preds.csv` (input: entity/clinical_entity + optional chunk_text / chunk_id; every other column is dropped unread).

## Predict command checks (artifacts/plus/checks/)
- 10-row file with only entity,chunk_text: 10 rows out, every code valid in both dictionaries.
- Same rows with wrong subconcept (ProcedureCode) and coding_system (ICD10CM) columns: byte-identical output.
- 50 test rows predicted twice: byte-identical (deterministic).
- Speed with models already loaded: about 2 s/term for small files, dominated by model loading. The first run also fits the X+ stackers (one-time, about 4 min).
- Hard cases: MS -> multiple sclerosis in the neurology note and rheumatic mitral stenosis in the echo note; 'chronickidney disease stag 3' -> N18.30; 'diabetes' -> E11.9; '1-2+ ai' -> I35.1. Known miss: 'fx of right distal radius' -> S52.91XB (open fracture, whole forearm) instead of the closed distal-radius code.

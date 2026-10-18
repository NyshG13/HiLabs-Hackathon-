# TermMap-X+ web console

Apple-style multi-page front end for the engine in this repository: Home (pipeline walkthrough), Lab (single term and
CSV upload) and Results. Plain HTML/CSS/JS, no build step.

```bash
# from the repository root, with the engine's environment active (weights via Git LFS, see the root README)
python web/server.py [--no-llm] [--port 8787]        # http://127.0.0.1:8787
```

- **Lab → Upload CSV:** `chunk_id`, `clinical_entity` (or `entity`), optional `chunk_text`. Rows are mapped 20 per request
  with progress; rows under 0.5 confidence are flagged; download `predictions.csv`
  (`chunk_id, clinical_entity, icd10cm_code, snomed_code`) or a detailed CSV with names and confidences.
- **Engine:** `web/engines.py` calls `termmap_plus.predict_sf.predict_frame()`, the same path as `python predict.py`.
- `--engine-dir` / `TERMMAP_XPLUS_DIR` override the engine location (default: this repository).
- Without the engine the pages still open and map only built-in hard cases with hand-written illustrative samples,
  badged as such. Everything runs on `127.0.0.1`; no data leaves the machine.

| endpoint | purpose |
|---|---|
| `GET /api/status` | engine state, name, LLM gate on/off, device |
| `POST /api/map` | one term |
| `POST /api/batch` | up to 100 rows, results in input order |

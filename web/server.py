"""TermMap-X console - local server.

    python web/server.py --engine-dir /path/to/HiLabs-Hackathon- [--port 8787] [--no-llm] [--no-engine]

Serves the static console in web/ and, when the TermMap-X+ engine is available, the live endpoints:

    GET  /api/status                      ->  {"engine": "ready" | "loading" | "off", "name", "llm", "device", "error"}
    POST /api/map    {term, context}      ->  one term, both systems (shape used by the single-term lab)
    POST /api/batch  {rows: [{chunk_id, clinical_entity, chunk_text}, ...]}   ->  one result per row, in order

The engine is NyshG13/HiLabs-Hackathon- ("TermMap-X+"). Run this server with that repo's Python environment
(torch, lightgbm, transformers; weights from Git LFS) and point --engine-dir (or TERMMAP_XPLUS_DIR) at the clone.
Everything runs on this machine (no hosted API). Without the engine the console still opens and falls back to its
built-in illustrative samples; nothing is fabricated as engine output.
"""
import argparse
import json
import os
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WEB = Path(__file__).resolve().parent
sys.path.insert(0, str(WEB))

MAX_BODY = 2_000_000
MAX_ROWS = 100            # per request; the console sends 20 at a time so it can show progress

STATE = {"engine": "off", "error": None, "eng": None}
LOCK = threading.Lock()   # the engine is not thread-safe: one request at a time


def load_engine(engine_dir, use_llm):
    STATE["engine"] = "loading"
    try:
        from engines import XPlusEngine
        STATE["eng"] = XPlusEngine(engine_dir, use_llm)
        STATE["engine"] = "ready"
    except BaseException as e:   # missing weights / env is the normal case for a fresh clone; SystemExit included
        STATE["engine"], STATE["error"] = "off", f"{type(e).__name__}: {e}"


def clean_rows(rows):
    out = []
    for r in rows[:MAX_ROWS]:
        term = str(r.get("clinical_entity", r.get("term", ""))).strip()[:300]
        out.append({"chunk_id": str(r.get("chunk_id", ""))[:200], "clinical_entity": term,
                    "chunk_text": str(r.get("chunk_text", r.get("context", "")))[:4000]})
    return out


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(WEB), **kw)

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/status":
            eng = STATE["eng"]
            info = eng.info() if eng else {}
            return self._json(200, {"engine": STATE["engine"], "error": STATE["error"], **info})
        return super().do_GET()

    def do_POST(self):
        if self.path not in ("/api/map", "/api/batch"):
            return self._json(404, {"error": "not found"})
        if STATE["engine"] != "ready":
            return self._json(503, {"error": "engine " + STATE["engine"], "detail": STATE["error"]})
        try:
            n = int(self.headers.get("Content-Length", "0"))
            if n > MAX_BODY:
                return self._json(413, {"error": "request too large"})
            req = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/api/map":
                rows = clean_rows([{"clinical_entity": req.get("term", ""), "chunk_text": req.get("context", "")}])
                if not rows[0]["clinical_entity"]:
                    return self._json(400, {"error": "term is required"})
                with LOCK:
                    res, timing = STATE["eng"].map_rows(rows)
                r = res[0]
                return self._json(200, {"term": rows[0]["clinical_entity"], "timing": timing, "systems": {
                    "ICD10CM": r["icd"], "SNOMEDCT_US": r["snomed"]}})
            rows = clean_rows(req.get("rows", []))
            if not rows or any(not r["clinical_entity"] for r in rows):
                return self._json(400, {"error": "every row needs a clinical_entity"})
            with LOCK:
                res, timing = STATE["eng"].map_rows(rows)
            return self._json(200, {"results": res, "timing": timing})
        except Exception as e:
            return self._json(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt, *args):
        sys.stderr.write("[web] " + fmt % args + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8787)
    # inside the X+ repo (web/ next to termmap_plus/) the engine is the parent folder; otherwise a sibling clone
    here = WEB.parent if (WEB.parent / "termmap_plus").exists() else WEB.parent / "HiLabs-Hackathon-"
    ap.add_argument("--engine-dir", default=os.environ.get("TERMMAP_XPLUS_DIR", str(here)),
                    help="clone of NyshG13/HiLabs-Hackathon- (TermMap-X+), weights pulled with Git LFS")
    ap.add_argument("--no-llm", action="store_true", help="skip the Qwen2.5 SNOMED gate (faster, ~2 pt lower SNOMED)")
    ap.add_argument("--no-engine", action="store_true", help="serve the console only (illustrative samples)")
    a = ap.parse_args()
    if not a.no_engine:
        use_llm = not a.no_llm and os.environ.get("TERMMAP_USE_LLM", "1") != "0"
        threading.Thread(target=load_engine, args=(a.engine_dir, use_llm), daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"TermMap-X console -> http://127.0.0.1:{a.port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

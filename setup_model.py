"""One-time setup after cloning: check the trained weights that came with the repo, and download the two public
base LLMs the pipeline uses.

  python setup_model.py            # verify trained weights (sha256) + download Qwen models (pinned revisions)
  python setup_model.py --verify   # only re-check the trained weights

The trained weights (fine-tuned SapBERT, X's two cross-encoders, dense + lexical alias indexes) are stored in this
repository with Git LFS; each file is checked against artifacts/WEIGHTS_MANIFEST.json. The Qwen models were not
trained by us; they are public and are downloaded from their official Hugging Face pages at fixed revisions into
the caches the code reads (no account or token needed).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "artifacts" / "WEIGHTS_MANIFEST.json"
QWEN = {  # repo id: (revision, cache dir the code loads from)
    "Qwen/Qwen2.5-1.5B-Instruct": ("989aa7980e4cf806f80c7fef2b1adb7bc71aa306", ROOT / "models" / "hf_cache" / "hub"),
    "Qwen/Qwen3-Reranker-0.6B": ("e61197ed45024b0ed8a2d74b80b4d909f1255473", ROOT / "artifacts" / "plus" / "hf_cache"),
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def verify() -> bool:
    m = json.loads(MANIFEST.read_text(encoding="utf8"))
    bad, stubs = [], []
    for rel, info in m["files"].items():
        p = ROOT / rel
        if p.exists() and p.stat().st_size < 1024 and p.read_bytes().startswith(b"version https://git-lfs"):
            stubs.append(rel)
        elif not p.exists() or p.stat().st_size != info["bytes"] or sha256(p) != info["sha256"]:
            bad.append(rel)
    print(f"trained weights: {len(m['files']) - len(bad) - len(stubs)}/{len(m['files'])} files verified (sha256)")
    if stubs:
        print(f"  {len(stubs)} file(s) are Git LFS pointers, not the real weights. Run:  git lfs install && git lfs pull")
    if bad:
        print(f"  missing or modified: {bad}")
    return not bad and not stubs


def qwen() -> None:
    from huggingface_hub import snapshot_download
    for repo, (rev, cache) in QWEN.items():
        cache.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=repo, revision=rev, cache_dir=str(cache),
                          ignore_patterns=["*.msgpack", "*.h5", "onnx/*", "*.onnx"])
        ref = cache / f"models--{repo.replace('/', '--')}" / "refs" / "main"
        ref.parent.mkdir(parents=True, exist_ok=True)
        ref.write_text(rev)  # the code loads by repo id -> "main" points at the pinned revision
        print(f"{repo} @ {rev[:10]} ready in {cache.relative_to(ROOT)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="only re-check the trained weights")
    a = ap.parse_args()
    ok = verify()
    if not ok:
        sys.exit(1)
    if not a.verify:
        qwen()
    print("setup complete" if not a.verify else "weights OK")


if __name__ == "__main__":
    main()

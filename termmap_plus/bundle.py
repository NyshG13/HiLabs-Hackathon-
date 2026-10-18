"""Artifact bundle for the subconcept-free submission (xplus-sf).

  python -m termmap_plus.bundle manifest            # artifacts/MANIFEST.json (sha256 + bytes of every bundled file)
  python -m termmap_plus.bundle pack [out.zip]      # stored zip of the bundled files + the manifest
  python -m termmap_plus.bundle fetch --url URL     # download the zip, extract into the repo, verify every sha256
  python -m termmap_plus.bundle verify              # re-check local files against artifacts/MANIFEST.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts" / "MANIFEST.json"
BUNDLE = [
    "models/cross-ctx", "models/cross-noctx", "models/sapbert-ft", "models/stack_settings.pkl",
    "models/hf_cache/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots",
    "models/hf_cache/hub/models--Qwen--Qwen2.5-1.5B-Instruct/refs",
    "artifacts/plus/hf_cache/models--Qwen--Qwen3-Reranker-0.6B/snapshots",
    "artifacts/plus/hf_cache/models--Qwen--Qwen3-Reranker-0.6B/refs",
    "artifacts/plus/sf/sf_recipe.json", "artifacts/plus/sf/stackers_none_dtag.pkl",
    "artifacts/plus/state.json", "artifacts/plus/v2c/state_v2c.json",
    "shared/icd_tree.parquet", "shared/scope.json", "configs/plus.yaml",
    "data/raw", "data/processed", "data/splits", "data/external",
]


def files() -> list[Path]:
    out = []
    for b in BUNDLE:
        p = ROOT / b
        if not p.exists():
            raise SystemExit(f"bundle entry missing: {b}")
        out += sorted(x for x in p.rglob("*") if x.is_file()) if p.is_dir() else [p]
    return [f for f in out if not f.name.endswith(".incomplete")]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def manifest() -> dict:
    fs = files()
    m = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "tag": "xplus-sf", "n_files": len(fs),
         "bytes": sum(f.stat().st_size for f in fs),
         "files": {f.relative_to(ROOT).as_posix(): {"sha256": sha256(f), "bytes": f.stat().st_size} for f in fs}}
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m, indent=1), encoding="utf8")
    print(f"wrote {MANIFEST}: {m['n_files']} files, {m['bytes'] / 2**30:.2f} GiB")
    return m


def pack(out: Path) -> None:
    m = json.loads(MANIFEST.read_text(encoding="utf8"))
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        z.write(MANIFEST, "artifacts/MANIFEST.json")
        for rel in m["files"]:
            z.write(ROOT / rel, rel)
    print(f"wrote {out} ({out.stat().st_size / 2**30:.2f} GiB); sha256 {sha256(out)}")


def verify() -> bool:
    m = json.loads(MANIFEST.read_text(encoding="utf8"))
    bad = [rel for rel, v in m["files"].items()
           if not (ROOT / rel).exists() or (ROOT / rel).stat().st_size != v["bytes"] or sha256(ROOT / rel) != v["sha256"]]
    print(f"verify: {len(m['files']) - len(bad)}/{len(m['files'])} files OK" + (f"; BAD: {bad[:10]}" if bad else ""))
    return not bad


def fetch(url: str) -> None:
    dst = ROOT / "artifacts" / "bundle.zip"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        print(f"downloading {url} -> {dst}")
        with urllib.request.urlopen(url) as r, dst.open("wb") as f:
            shutil.copyfileobj(r, f, 1 << 22)
    with zipfile.ZipFile(dst) as z:
        for n in z.namelist():  # refuse paths that escape the repo
            if n.startswith(("/", "\\")) or ".." in Path(n).parts:
                raise SystemExit(f"unsafe path in bundle: {n}")
        z.extractall(ROOT)
    if not verify():
        raise SystemExit("bundle verification failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["manifest", "pack", "fetch", "verify"])
    ap.add_argument("out", nargs="?", default=str(ROOT / "artifacts" / "termmap_xplus_sf_bundle.zip"))
    ap.add_argument("--url")
    a = ap.parse_args()
    if a.cmd == "manifest":
        manifest()
    elif a.cmd == "pack":
        pack(Path(a.out))
    elif a.cmd == "fetch":
        if not a.url:
            sys.exit("--url is required (make setup BUNDLE_URL=...)")
        fetch(a.url)
    else:
        sys.exit(0 if verify() else 1)

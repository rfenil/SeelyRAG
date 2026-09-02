#!/usr/bin/env python
"""Run the API on Render, downloading the demo RAG data when needed."""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
REQUIRED_PATHS = (
    DATA_ROOT / "02_processed" / "chunks.jsonl",
    DATA_ROOT / "02_processed" / "codes.jsonl",
    DATA_ROOT / "03_index" / "chunks.lance",
)


def data_ready() -> bool:
    """Return whether the files required by Ask/Search are already present."""
    return all(path.exists() for path in REQUIRED_PATHS)


def ensure_data() -> None:
    """Download and extract the hosted data bundle if the data tree is missing."""
    if data_ready():
        return

    url = os.environ.get("SEELEY_DATA_BUNDLE_URL")
    if not url:
        missing = ", ".join(str(path.relative_to(REPO_ROOT)) for path in REQUIRED_PATHS)
        raise RuntimeError(
            "RAG data is missing and SEELEY_DATA_BUNDLE_URL is not set. "
            f"Expected: {missing}"
        )

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "seeley-rag-data.zip"
        print(f"Downloading RAG data bundle: {url}", flush=True)
        urllib.request.urlretrieve(url, archive)

        print("Extracting RAG data bundle...", flush=True)
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(REPO_ROOT)

    if not data_ready():
        raise RuntimeError("RAG data bundle extracted, but required files are still missing.")


def main() -> int:
    """Prepare data and then run the normal server entry point."""
    ensure_data()

    serve_path = REPO_ROOT / "scripts" / "08_serve.py"
    spec = importlib.util.spec_from_file_location("seeley_rag_serve", serve_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load server script: {serve_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return int(module.main())


if __name__ == "__main__":
    sys.exit(main())

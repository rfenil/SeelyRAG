#!/usr/bin/env python
"""Run the API on Render, downloading the demo RAG data when needed."""

from __future__ import annotations

import importlib.util
import os
import shutil
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


def find_data_root(root: Path) -> Path | None:
    """Find a directory containing the required demo data layout."""
    candidates = [root / "data", *root.rglob("data")]
    for candidate in candidates:
        if (candidate / "02_processed" / "chunks.jsonl").exists() and (
            candidate / "03_index" / "chunks.lance"
        ).exists():
            return candidate
    return None


def install_extracted_data(extracted_root: Path) -> None:
    """Move extracted demo data into the repository's ``data`` directory."""
    source = find_data_root(extracted_root)
    if source is None:
        sample = sorted(
            str(path.relative_to(extracted_root))
            for path in extracted_root.rglob("*")
            if path.is_file()
        )[:20]
        raise RuntimeError(
            "RAG data bundle extracted, but no data/02_processed + data/03_index layout "
            f"was found. First files seen: {sample}"
        )

    if source.resolve() == DATA_ROOT.resolve():
        return

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    for dirname in ("02_processed", "03_index"):
        destination = DATA_ROOT / dirname
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source / dirname, destination)


def extract_bundle(archive: Path, destination: Path) -> None:
    """Extract a zip bundle, normalising Windows member paths on Linux."""
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            name = member.filename.replace("\\", "/").lstrip("/")
            if not name or name.endswith("/"):
                continue
            target = (destination / name).resolve()
            if destination not in target.parents:
                raise RuntimeError(f"Refusing to extract unsafe bundle member: {member.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, target.open("wb") as handle:
                shutil.copyfileobj(source, handle)


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
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 SeeleyRAGDemo/0.1",
                "Accept": "application/zip,application/octet-stream,*/*",
            },
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            with archive.open("wb") as handle:
                shutil.copyfileobj(response, handle)

        print("Extracting RAG data bundle...", flush=True)
        extract_root = Path(tmp) / "extracted"
        extract_root.mkdir()
        extract_bundle(archive, extract_root)
        install_extracted_data(extract_root)

    if not data_ready():
        missing = [str(path.relative_to(REPO_ROOT)) for path in REQUIRED_PATHS if not path.exists()]
        raise RuntimeError(
            "RAG data bundle extracted, but required files are still missing: "
            + ", ".join(missing)
        )


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

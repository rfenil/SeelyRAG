"""Shared test fixtures.

**No test may make a real network request.** Every HTTP interaction is served by
``pytest_httpx``'s mock transport. The ``no_network`` autouse fixture is a
belt-and-braces tripwire: it blocks outbound connections, so a test that slips
past the mock fails loudly instead of quietly hitting Seeley's server.

⚠ The tripwire guards ``connect``, and deliberately **allows loopback**. An
earlier version replaced ``socket.socket`` itself, which broke two things the
moment Stage 4 arrived: Python's ``ssl`` module does ``class SSLSocket(socket)``
and cannot subclass a function, and Windows' ``ProactorEventLoop`` builds its
self-pipe from a loopback socketpair and then calls
``isinstance(conn, socket.socket)``. With the constructor replaced, that
``isinstance`` raises inside the event loop and the test hangs forever rather
than failing. LanceDB is async-backed, so every store test hit it. Guarding the
connection instead keeps the type intact, and loopback is not the network we are
protecting.

Fixtures in ``tests/fixtures/`` are trimmed samples of real portal pages. They
are the contract with the live site: if the portal's markup changes, these are
what you re-capture and diff.
"""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any, Iterator

import pytest

from seeley_rag.settings import Settings

FIXTURE_DIR = Path(__file__).parent / "fixtures"

BASE_URL = "https://seeleyinternationalhelp.freshdesk.com"


def read_fixture(name: str) -> str:
    """Read a fixture file.

    Args:
        name: Filename inside ``tests/fixtures/``.

    Returns:
        The file's contents.
    """
    return (FIXTURE_DIR / name).read_text(encoding="utf-8")


#: Hosts a test may still reach. Loopback only, and only because embedded
#: libraries build their internal plumbing on it -- asyncio's event-loop
#: self-pipe on Windows, for one. Nothing outside the machine is reachable.
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", "0.0.0.0", "::"})


def _is_loopback(address: Any) -> bool:
    """Whether a socket address is local to this machine.

    Args:
        address: The address passed to ``connect``.

    Returns:
        True for loopback addresses and for non-IP address families such as
        Unix sockets, which cannot reach another host.
    """
    if isinstance(address, (str, bytes)):
        return True
    if isinstance(address, tuple) and address:
        return str(address[0]) in LOOPBACK_HOSTS
    return False


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any test that opens a connection off this machine.

    ``pytest_httpx`` already intercepts httpx, so this only fires if a test
    reaches the network some other way. That is exactly the case worth catching:
    the crawl targets someone else's production server, and Stage 4 spends real
    money per request.

    Guards ``connect`` rather than replacing ``socket.socket`` -- see the module
    docstring for why the latter hangs the suite instead of failing it.
    """
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def guard(self: Any, address: Any, *args: Any, **kwargs: Any) -> Any:
        if _is_loopback(address):
            return real_connect(self, address, *args, **kwargs)
        raise RuntimeError(
            f"A test attempted a real network connection to {address!r}. Tests must mock "
            "HTTP via pytest_httpx; never hit the live Seeley portal or a paid API "
            "from the suite."
        )

    def guard_ex(self: Any, address: Any, *args: Any, **kwargs: Any) -> Any:
        if _is_loopback(address):
            return real_connect_ex(self, address, *args, **kwargs)
        raise RuntimeError(
            f"A test attempted a real network connection to {address!r}. Tests must mock "
            "HTTP via pytest_httpx; never hit the live Seeley portal or a paid API "
            "from the suite."
        )

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket.socket, "connect_ex", guard_ex)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Return settings pointed at a temporary data root.

    Tests that write to ``data/`` must never touch the real tree, which is
    write-once and holds a real crawl.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        A settings object rooted at ``tmp_path``.
    """
    resolved = Settings.from_yaml()
    resolved.data_root = tmp_path / "data"
    return resolved


@pytest.fixture
def temp_data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Redirect every module-level path constant at a temporary tree.

    ``paths.py`` resolves its constants at import time, so overriding settings
    alone is not enough -- the constants must be patched directly.

    Args:
        tmp_path: pytest's per-test temporary directory.
        monkeypatch: pytest's patcher.

    Yields:
        The temporary data root.
    """
    from seeley_rag import paths

    root = tmp_path / "data"
    raw = root / "00_raw"
    replacements = {
        "DATA_ROOT": root,
        "RAW_DIR": raw,
        "RAW_HTML_DIR": raw / "html",
        "RAW_PDF_DIR": raw / "pdf",
        "MANIFEST_PATH": raw / "manifest.jsonl",
        "INTERIM_DIR": root / "01_interim",
        "PAGES_PATH": root / "01_interim" / "pages.jsonl",
        "PAGE_IMAGES_DIR": root / "01_interim" / "page_images",
        "PROCESSED_DIR": root / "02_processed",
        "CHUNKS_PATH": root / "02_processed" / "chunks.jsonl",
        "CODES_PATH": root / "02_processed" / "codes.jsonl",
        "INDEX_DIR": root / "03_index",
        "CACHE_DIR": root / "cache",
        "LLM_CACHE_DIR": root / "cache" / "llm",
        "EMBEDDING_CACHE_DIR": root / "cache" / "embeddings",
        "REPORTS_DIR": root / "reports",
        "QUERY_LOG_PATH": root / "reports" / "queries.jsonl",
        "FEEDBACK_LOG_PATH": root / "reports" / "feedback.jsonl",
    }
    for name, value in replacements.items():
        monkeypatch.setattr(paths, name, value)
    monkeypatch.setattr(
        paths, "ALL_DIRS", tuple(v for k, v in replacements.items() if not k.endswith("_PATH"))
    )
    # DERIVED_DIRS must be redirected too. Left unpatched, clean_derived() would
    # delete the real repository's derived stages during a test run.
    monkeypatch.setattr(
        paths,
        "DERIVED_DIRS",
        (
            replacements["INTERIM_DIR"],
            replacements["PROCESSED_DIR"],
            replacements["INDEX_DIR"],
            replacements["CACHE_DIR"],
        ),
    )

    # Modules that imported the constants by value need patching too.
    from seeley_rag.acquire import attachments as attachments_module
    from seeley_rag.acquire import manifest as manifest_module
    from seeley_rag.acquire import portal as portal_module
    from seeley_rag.parse import triage as triage_module

    monkeypatch.setattr(portal_module, "RAW_HTML_DIR", replacements["RAW_HTML_DIR"])
    monkeypatch.setattr(attachments_module, "RAW_PDF_DIR", replacements["RAW_PDF_DIR"])
    monkeypatch.setattr(manifest_module, "MANIFEST_PATH", replacements["MANIFEST_PATH"])
    monkeypatch.setattr(manifest_module, "RAW_DIR", replacements["RAW_DIR"])
    monkeypatch.setattr(triage_module, "REPORTS_DIR", replacements["REPORTS_DIR"])

    paths.ensure_dirs()
    yield root


@pytest.fixture
def folder_page_html() -> str:
    """A trimmed real folder listing page, with pagination markup."""
    return read_fixture("folder_page.html")


@pytest.fixture
def article_stub_html() -> str:
    """A trimmed real stub article: boilerplate, "Pdf attached", one PDF."""
    return read_fixture("article_stub.html")


@pytest.fixture
def article_content_html() -> str:
    """A trimmed real diagnostic article: substantial body, no attachments."""
    return read_fixture("article_content.html")

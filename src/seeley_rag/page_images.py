"""URL helpers for rendered manual page images.

Local development serves images through the FastAPI ``/pages`` endpoint. A
hosted demo can point citation links directly at object storage by setting
``PAGE_IMAGE_BASE_URL`` to the public prefix that contains the document
directories.
"""

from __future__ import annotations

from urllib.parse import quote

from seeley_rag.settings import get_settings


def page_image_directory(doc_id: str) -> str:
    """Return the storage directory name for a document id."""
    return doc_id.split(":", 1)[1] if doc_id.startswith("article:") else doc_id


def page_image_url(doc_id: str, page_index: int) -> str | None:
    """Return the browser URL for a rendered page image.

    In local mode this returns the API route, which accepts an unpadded page
    index and resolves it to ``0000.png`` on disk. In hosted mode this returns
    the actual object key, because object stores serve static files directly.
    """
    if not doc_id or page_index < 0:
        return None

    base_url = get_settings().page_image_base_url
    if base_url:
        directory = quote(page_image_directory(doc_id), safe="")
        return f"{base_url.rstrip('/')}/{directory}/{page_index:04d}.png"

    return f"/pages/{quote(doc_id, safe=':')}/{page_index}.png"

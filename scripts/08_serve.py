#!/usr/bin/env python
"""Stage 7 -- run the API.

build-plan.md section 9.

    python scripts/08_serve.py            # http://127.0.0.1:8000
    python scripts/08_serve.py --reload   # development

Interactive docs are at ``/docs``; the corpus inventory is at
``/docs-inventory``, because FastAPI owns the former and an integration seam
should not remove its own documentation.
"""

from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Serve the Seeley installer assistant API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address.")
    parser.add_argument("--port", type=int, default=8000, help="Bind port.")
    parser.add_argument("--reload", action="store_true", help="Reload on source changes.")
    return parser.parse_args()


def main() -> int:
    """Run the ASGI server.

    Returns:
        Process exit code.
    """
    args = parse_args()
    try:
        import uvicorn
    except ImportError:
        print('uvicorn is not installed. Run: pip install -e ".[downstream]"')
        return 1

    print(f"Docs:      http://{args.host}:{args.port}/docs")
    print(f"Health:    http://{args.host}:{args.port}/health")
    print(f"Inventory: http://{args.host}:{args.port}/docs-inventory\n")

    uvicorn.run(
        "seeley_rag.api.main:get_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

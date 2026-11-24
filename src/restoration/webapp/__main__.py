"""``python -m restoration.webapp [--host 0.0.0.0] [--port 5000]``."""

from __future__ import annotations

import argparse

from restoration.config import logs_dir
from restoration.utils.logging import setup_logging
from restoration.webapp import create_app


def main() -> None:
    """Start the development server (single-threaded GPU work is serialised by a lock)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    setup_logging(logs_dir(), "webapp")
    create_app().run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()

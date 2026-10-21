"""Thin CLI wrapper, e.g. 02_discover.py --config config/default.toml discover ..."""

from insider_research.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

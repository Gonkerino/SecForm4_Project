"""Command-line interface for the first implementation milestone."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from . import __version__
from .catalog import Catalog
from .config import load_config
from .discovery import discover
from .locking import publisher_lock
from .sources.indexes import ACCESSION, plan_sources, source_for
from .validation import validate


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="insider", description="SEC ownership filing catalogue (Forms 3/4/5 and amendments)")
    root.add_argument("--version", action="version", version=__version__)
    root.add_argument("--config", type=Path, help="Optional TOML settings file")
    root.add_argument("--data-root", type=Path, help="Dedicated data directory (overrides config/environment)")
    root.add_argument("--offline", action="store_true", help="Guarantee no network calls; use only validated cached indexes")
    sub = root.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="Initialize a fresh data root and SQLite catalogue")
    sub.add_parser("status", help="Show committed catalogue counts")
    sub.add_parser("coverage", help="Show source-period validation and latest failures")
    sub.add_parser("validate", help="Verify committed index hashes and catalogue relationships")
    sub.add_parser("recover", help="Mark abandoned runs interrupted after acquiring the publisher lock")
    listings = sub.add_parser("filings", help="List unique filings from current source revisions")
    listings.add_argument("--start", type=date.fromisoformat)
    listings.add_argument("--end", type=date.fromisoformat)
    listings.add_argument("--form", choices=("3", "3/A", "4", "4/A", "5", "5/A"))
    listings.add_argument("--cik", help="Indexed entity CIK; not necessarily the issuer")
    listings.add_argument("--limit", type=int, default=100)
    inspect = sub.add_parser("inspect", help="Show all source associations for an accession")
    inspect.add_argument("accession")
    for name in ("discover", "plan"):
        command = sub.add_parser(name, help="Fetch source indexes" if name == "discover" else "Preview discovery without writes/network")
        command.add_argument("--start", required=True, type=date.fromisoformat)
        command.add_argument("--end", required=True, type=date.fromisoformat)
        command.add_argument("--source", choices=("quarterly", "daily"), default="quarterly")
        command.add_argument("--accept-removals", action="store_true", help="Activate revisions with reviewed accession removals")
    ingest = sub.add_parser("ingest-index", help="Import an original master.idx offline")
    ingest.add_argument("--file", required=True, type=Path)
    ingest.add_argument("--source", choices=("quarterly", "daily"), required=True)
    ingest.add_argument("--period", required=True, help="2024Q1 for quarterly; YYYY-MM-DD for daily")
    ingest.add_argument("--accept-removals", action="store_true")
    return root


def main(argv: list[str] | None = None) -> int:
    args_parser = parser()
    try:
        args = args_parser.parse_args(argv)
    except SystemExit as exc:
        return 64 if exc.code == 2 else exc.code
    try:
        cfg = load_config(args.config, args.data_root)
        if args.command in {"discover", "plan"}:
            sources = plan_sources(args.start, args.end, args.source,
                                   today=datetime.now(ZoneInfo("America/New_York")).date())
            if not sources:
                raise ValueError("Selection contains no planned source periods (daily discovery skips weekends)")
            if args.command == "plan":
                result = {"settings": cfg.public_settings(), "source_count": len(sources),
                          "sources": [s.as_dict() for s in sources],
                          "note": "Quarterly discovery imports whole source quarters, including dates outside a partial-quarter selection. No submissions will be downloaded."}
                print(json.dumps(result, indent=2))
                return 0
        if args.command == "init":
            Catalog.initialize(cfg)
            result = {"initialized": True, "database": str(cfg.database), "settings": cfg.public_settings()}
        elif args.command == "validate":
            result = validate(cfg)
            print(json.dumps(result, indent=2))
            return 0 if result["status"] == "valid" else 2
        elif args.command == "recover":
            with publisher_lock(cfg.data_root), Catalog(cfg.database) as catalog:
                result = catalog.recover()
        elif args.command in {"discover", "ingest-index"}:
            if args.command == "ingest-index":
                sources = [source_for(args.source, args.period)]
            result = discover(cfg, sources, local_file=args.file if args.command == "ingest-index" else None,
                              offline=args.offline, accept_removals=args.accept_removals,
                              progress=lambda message: print(message, file=sys.stderr, flush=True))
            print(json.dumps(result, indent=2))
            return 0 if result["status"] == "succeeded" else 2
        else:
            with Catalog(cfg.database) as catalog:
                if args.command == "status":
                    result = catalog.status()
                elif args.command == "coverage":
                    result = {"sources": catalog.coverage(),
                              "note": "Validated means source index parsed; it does not mean submissions were downloaded or ownership records verified. Only requested periods are listed."}
                elif args.command == "filings":
                    if args.start and args.end and args.start > args.end:
                        raise ValueError("Start date must be on or before end date")
                    if args.cik is not None and (len(args.cik) > 10 or not args.cik.isascii() or not args.cik.isdigit() or not 0 < int(args.cik) < 10**10):
                        raise ValueError("CIK must contain 1–10 digits and be nonzero")
                    result = {"filings": catalog.list_filings(start=args.start.isoformat() if args.start else None,
                                                              end=args.end.isoformat() if args.end else None,
                                                              form=args.form, cik=args.cik.lstrip('0') if args.cik else None,
                                                              limit=args.limit),
                              "note": "Unique accessions matching current index associations; results are limited. Inspect an accession for full provenance."}
                else:
                    if not ACCESSION.fullmatch(args.accession):
                        raise ValueError("Accession must look like 0001234567-24-000001")
                    result = catalog.inspect(args.accession)
        print(json.dumps(result, indent=2))
        return 0
    except KeyboardInterrupt:
        print("Interrupted; already committed catalogue revisions are retained.", file=sys.stderr)
        return 130
    except (ValueError, OSError, sqlite3.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

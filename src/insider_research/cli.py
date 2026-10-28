"""Small public CLI with explicit stages and backwards-compatible aliases."""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures.process import BrokenProcessPool
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import psycopg

from . import __version__, health, legacy_cli
from .legacy_cli import json_value
from .catalog import Catalog
from .config import load_config
from .discovery import discover
from .ownership_store import (plan_parse, parse_ownership, parsing_status,
                              inspect_ownership, validate_ownership)
from .progress import Progress
from .sources.indexes import ACCESSION, plan_sources
from .submissions import download_submissions

PUBLIC = {
    'discover': 'Catalogue filings from SEC indexes',
    'download': 'Download pending full submissions',
    'parse': 'Extract downloaded Forms 3/4/5 and amendments',
    'status': 'See acquisition, parsing, verification and outstanding work',
    'check': 'Verify data and record version-specific findings',
    'issues': 'List integrity failures, failed attempts and source review cases',
    'repair': 'Recheck and repair recorded integrity failures',
    'show': 'Inspect one filing across its full lifecycle',
    'dashboard': 'Open the coverage dashboard',
    'admin': 'Setup, local imports, listings and interrupted-run recovery',
}
OWNERSHIP_ALIASES = {'plan-parse', 'parse-ownership', 'parsing-status', 'inspect-ownership', 'validate-ownership'}
ADMIN = {'init': 'init', 'import-index': 'ingest-index', 'recover': 'recover',
         'filings': 'filings', 'coverage': 'coverage'}


def _filters(command):
    command.add_argument('--start', type=date.fromisoformat, help='First indexed filing date (inclusive; default all)')
    command.add_argument('--end', type=date.fromisoformat, help='Last indexed filing date (inclusive)')
    command.add_argument('--form', choices=('3', '3/A', '4', '4/A', '5', '5/A'))
    command.add_argument('--accession')


def parser():
    root = argparse.ArgumentParser(prog='insider', description='SEC ownership filings: discover -> download -> parse',
        epilog='Global configuration options go before the command. Existing command names remain supported as compatibility aliases.')
    root.add_argument('--version', action='version', version=__version__)
    root.add_argument('--config', type=Path)
    root.add_argument('--data-root', type=Path)
    root.add_argument('--offline', action='store_true', help='No SEC requests; PostgreSQL is still required')
    sub = root.add_subparsers(dest='command', required=True, metavar='COMMAND')
    for name, help_text in PUBLIC.items():
        cmd = sub.add_parser(name, help=help_text)
        if name not in ('admin', 'dashboard'):
            cmd.add_argument('--json', action='store_true', help='Machine-readable JSON output')
        if name == 'discover':
            cmd.add_argument('--start', required=True, type=date.fromisoformat)
            cmd.add_argument('--end', required=True, type=date.fromisoformat)
            cmd.add_argument('--source', choices=('quarterly', 'daily'), default='quarterly')
            cmd.add_argument('--accept-removals', action='store_true')
            cmd.add_argument('--dry-run', action='store_true')
        elif name in ('download', 'parse', 'check'):
            _filters(cmd)
            if name != 'check':
                cmd.add_argument('--limit', type=int)
                cmd.add_argument('--retry-failed', action='store_true')
                cmd.add_argument('--dry-run', action='store_true')
            if name == 'download':
                cmd.add_argument('--workers', type=int, default=10)
            if name == 'parse':
                cmd.add_argument('--workers', type=int, default=4,
                    help='Concurrent local preparation workers, 1-20 (default: 4)')
            if name == 'check':
                cmd.add_argument('--stage', choices=health.STAGES)
                cmd.add_argument('--workers', type=int, default=8,
                    help='Concurrent archive checks, 1-20 (default: 8); other stages run sequentially')
                cmd.add_argument('--executor', choices=('auto', 'thread', 'process'), default='auto',
                    help='Archive workers (default: auto uses processes for 4096+ files, threads for smaller checks)')
        elif name in ('issues', 'repair'):
            cmd.add_argument('--stage', choices=health.STAGES)
            cmd.add_argument('--accession')
            cmd.add_argument('--limit', type=int, default=100)
            if name == 'repair':
                cmd.add_argument('--dry-run', action='store_true')
            else:
                cmd.add_argument('--severity', choices=('error', 'review', 'warning', 'info'))
                cmd.add_argument('--include-covered', action='store_true', help='Also show unresolved daily requests covered by quarterly indexes')
        elif name == 'show':
            cmd.add_argument('accession')
            cmd.add_argument('--result-id', type=int)
            cmd.add_argument('--details', action='store_true', help='Include every extracted field and preserved evidence row')
        elif name == 'dashboard':
            cmd.add_argument('--port', type=int, default=8765)
            cmd.add_argument('--no-browser', action='store_true')
        elif name == 'admin':
            admin = cmd.add_subparsers(dest='admin_command', required=True)
            old_sub = next(a for a in legacy_cli.parser()._actions if isinstance(a, argparse._SubParsersAction))
            for action, old in ADMIN.items():
                admin.add_parser(action, parents=[old_sub.choices[old]], add_help=False)
    old_sub = next(a for a in legacy_cli.parser()._actions if isinstance(a, argparse._SubParsersAction))
    for name, old in old_sub.choices.items():
        if name not in PUBLIC:
            sub.add_parser(name, parents=[old], add_help=False)
    for name in sorted(OWNERSHIP_ALIASES):
        cmd = sub.add_parser(name)
        if name in ('parse-ownership', 'plan-parse'):
            _filters(cmd)
            cmd.add_argument('--retry-failed', action='store_true')
            if name == 'parse-ownership':
                cmd.add_argument('--limit', type=int)
                cmd.add_argument('--workers', type=int, default=4)
        if name == 'inspect-ownership':
            cmd.add_argument('accession')
            cmd.add_argument('--result-id', type=int)
    return root


def render(result, *, as_json=False):
    if as_json:
        print(json.dumps(result, indent=2, default=json_value))
        return

    def emit(value, indent=0):
        prefix = ' ' * indent
        if isinstance(value, dict):
            for key, item in value.items():
                label = key.replace('_', ' ').capitalize()
                if isinstance(item, (dict, list)):
                    print(f'{prefix}{label}:' if item else f'{prefix}{label}: none')
                    if item:
                        emit(item, indent + 2)
                else:
                    display = f'{item:,}' if isinstance(item, int) and not isinstance(item, bool) else item
                    print(f"{prefix}{label}: {display if display is not None else '-'}")
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    print(prefix + '- ' + ' | '.join(f"{k.replace('_', ' ')}: {json.dumps(v, default=json_value) if isinstance(v, (dict,list)) else v}" for k, v in item.items()))
                else:
                    print(prefix + '- ' + str(item))
    emit(result)


def _command_position(argv):
    i = 0
    while i < len(argv):
        if argv[i] in ('--config', '--data-root'):
            i += 2
        elif argv[i].startswith('-'):
            i += 1
        else:
            return i
    return None


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    pos = _command_position(argv)
    if pos is not None:
        command = argv[pos]
        if command not in PUBLIC and command not in OWNERSHIP_ALIASES:
            return legacy_cli.main(argv)
        if command == 'admin' and len(argv) > pos + 1 and argv[pos + 1] in ADMIN:
            return legacy_cli.main(argv[:pos] + [ADMIN[argv[pos+1]]] + argv[pos+2:])
    try:
        args = parser().parse_args(argv)
    except SystemExit as exc:
        return 64 if exc.code == 2 else exc.code
    try:
        cfg = load_config(args.config, args.data_root)
        accession = getattr(args, 'accession', None)
        if accession is not None and not ACCESSION.fullmatch(accession):
            raise ValueError('Accession must look like 0001234567-24-000001')
        start, end = getattr(args, 'start', None), getattr(args, 'end', None)
        if start and end and start > end:
            raise ValueError('Start date must be on or before end date')
        limit = getattr(args, 'limit', None)
        if limit is not None and not 1 <= limit <= 10000:
            raise ValueError('Limit must be between 1 and 10000')
        filters = dict(start=start, end=end, form=getattr(args, 'form', None), accession=accession)
        if args.command == 'status':
            result = health.status(cfg)
        elif args.command == 'show':
            result = health.show(cfg, accession, result_id=args.result_id)
            if not args.json and not args.details:
                extracted = result['extraction']
                result = dict(accession=accession, download=result['download'],
                    extraction=extracted.get('result'), report=extracted.get('report', []),
                    owners=extracted.get('owners', []), transactions=extracted.get('transactions', []),
                    holdings=extracted.get('holdings', []), issues=extracted.get('issues', []),
                    source_associations=len(result['associations']),
                    download_attempts=len(result['download_attempts']), parse_attempts=len(extracted['attempts']),
                    verification_observations=len(result['verification']), repair_attempts=len(result['repairs']),
                    next='Use show --details or --json for all fields, footnotes, signatures and history')
        elif args.command == 'issues':
            result = health.issues(cfg, stage=args.stage, accession=accession, limit=limit, severity=args.severity, include_covered=args.include_covered)
        elif args.command == 'repair':
            with Progress('Repairing integrity', 'object') as progress:
                result = health.repair(cfg, stage=args.stage, accession=accession, limit=limit,
                                      dry_run=args.dry_run, offline=args.offline, progress=progress)
        elif args.command == 'check':
            with Progress('Checking integrity', 'object') as progress:
                result = health.check(cfg, stage=args.stage, workers=args.workers,
                                      executor=args.executor, progress=progress, **filters)
        elif args.command in ('parse', 'parse-ownership', 'plan-parse'):
            if not 1 <= getattr(args, 'workers', 4) <= 20:
                raise ValueError('Workers must be between 1 and 20')
            if args.command == 'plan-parse' or getattr(args, 'dry_run', False):
                result = plan_parse(cfg, retry_failed=args.retry_failed, **filters)
                result.update(dry_run=True, queued=min(result['eligible'], limit) if limit else result['eligible'])
            else:
                with Progress('Parsing ownership', 'filing') as progress:
                    result = parse_ownership(cfg, retry_failed=args.retry_failed, limit=limit,
                                             workers=args.workers, progress=progress, **filters)
        elif args.command == 'parsing-status':
            result = parsing_status(cfg)
        elif args.command == 'inspect-ownership':
            result = inspect_ownership(cfg, accession, result_id=args.result_id)
        elif args.command == 'validate-ownership':
            with Progress('Validating ownership', 'result') as progress:
                result = validate_ownership(cfg, progress=progress)
        elif args.command == 'download':
            if not 1 <= args.workers <= 20:
                raise ValueError('Workers must be between 1 and 20')
            if args.dry_run:
                with Catalog(cfg) as cat:
                    counts = cat.submission_counts(**filters)
                    result = dict(counts, dry_run=True, eligible=cat.submission_candidate_count(retry_failed=args.retry_failed, **filters))
                    result['queued'] = min(result['eligible'], limit) if limit else result['eligible']
            else:
                if args.offline:
                    raise ValueError('download requires SEC access; --dry-run is available offline')
                with Progress('Downloading submissions', 'filing') as progress:
                    result = download_submissions(cfg, limit=limit, workers=args.workers, retry_failed=args.retry_failed, progress=progress, **filters)
        elif args.command == 'discover':
            sources = plan_sources(start, end, args.source, today=datetime.now(ZoneInfo('America/New_York')).date())
            if not sources:
                raise ValueError('Selection contains no source periods; daily discovery skips weekends')
            if args.dry_run:
                result = dict(dry_run=True, source_count=len(sources), sources=[s.as_dict() for s in sources],
                              note='Quarterly discovery imports whole quarters. No submissions are downloaded.')
            else:
                with Progress('Discovering indexes', 'source', len(sources)) as progress:
                    result = discover(cfg, sources, offline=args.offline, accept_removals=args.accept_removals, progress=progress)
        elif args.command == 'dashboard':
            from .dashboard import serve
            serve(cfg, port=args.port, open_browser=not args.no_browser, offline=args.offline)
            return 0
        render(result, as_json=getattr(args, 'json', False) or args.command in OWNERSHIP_ALIASES)
        return 2 if result.get('status') in ('incomplete', 'invalid') else 0
    except KeyboardInterrupt:
        print('Interrupted; committed work is retained. Use insider admin recover if needed.', file=sys.stderr)
        return 130
    except psycopg.Error:
        print('Error: PostgreSQL operation failed. Check permissions, schema and server availability.', file=sys.stderr)
        return 1
    except BrokenProcessPool:
        print('Error: archive verification worker exited unexpectedly. Retry check with --executor thread.', file=sys.stderr)
        return 1
    except (ValueError, OSError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

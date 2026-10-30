"""Read-only monthly acquisition and extraction coverage served on loopback."""

from __future__ import annotations

import json
import webbrowser
import time
from collections import OrderedDict
from threading import Lock
from datetime import date, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from urllib.parse import parse_qs, urlsplit
from zoneinfo import ZoneInfo

import psycopg

from .catalog import Catalog
from .config import RESEARCH_START
from .ownership import PARSER_VERSION, current_result_sql

HISTORY_START = RESEARCH_START
CACHE_SECONDS = 30


def next_month(month):
    return date(month.year + (month.month == 12), month.month % 12 + 1, 1)


def monthly_coverage(rows, sources, *, today, start=HISTORY_START):
    """Never infer a complete month from an absent or partial index."""
    counts = {row["month"]: row for row in rows}
    quarter_dates = {}
    daily_months = set()
    for source in sources:
        if source["kind"] == "quarterly":
            observed = source["last_observed_at"].astimezone(ZoneInfo("America/New_York")).date()
            quarter_dates[source["period"]] = observed
        elif source["kind"] == "daily":
            daily_months.add(source["period"][:7])
    result = []
    month = start
    while month <= today:
        key = month.strftime("%Y-%m")
        row = counts.get(key, {})
        indexed = row.get("indexed", 0)
        downloaded = row.get("downloaded", 0)
        observed = quarter_dates.get(f"{month.year}Q{(month.month - 1) // 3 + 1}")
        complete_index = observed is not None and observed >= next_month(month)
        expected = indexed if complete_index else None
        has_index = indexed > 0 or key in daily_months or (observed is not None and observed >= month)
        # A zero-filing index is not evidence that any files were downloaded.
        if not has_index:
            state = "unindexed"
        elif downloaded == 0:
            state = "empty"
        elif complete_index and downloaded == indexed:
            state = "complete"
        else:
            state = "partial"
        result.append({"month": key, "indexed": indexed, "downloaded": downloaded,
                       "expected": expected, "state": state,
                       "open_month": next_month(month) > today,
                       "parsing": parsing_coverage(row, downloaded)})
        month = next_month(month)
    return result


def parsing_coverage(row, downloaded):
    """Selected current extractions take priority over older failed attempts."""
    parsed = row.get('parsed', 0)
    review = row.get('needs_review', 0)
    failed = row.get('failed', 0)
    unsupported = row.get('unsupported', 0)
    running = row.get('running', 0)
    remaining = downloaded - parsed
    state = ('unavailable' if not downloaded else
             'review' if parsed == downloaded and review else
             'complete' if parsed == downloaded else
             'attention' if failed or unsupported else
             'partial' if parsed or running else 'empty')
    return dict(parsed=parsed, completed=parsed-review,
                with_warnings=row.get('with_warnings', 0), needs_review=review,
                failed=failed, unsupported=unsupported, running=running,
                pending=remaining-failed-unsupported-running, remaining=remaining, state=state)


class Dashboard:
    def __init__(self, cfg):
        self.cfg = cfg
        self._cache = OrderedDict()
        self._cache_lock = Lock()

    def coverage(self, *, start=None):
        # A second browser tab must not multiply scans of a million-filing
        # catalogue. No background jobs or writes are introduced.
        with self._cache_lock:
            today = datetime.now(ZoneInfo('America/New_York')).date()
            start = start or HISTORY_START
            if start.day != 1 or not HISTORY_START <= start <= today.replace(day=1):
                raise ValueError('Start must be the first day of a month between July 2003 and the current month')
            interval = 60 if start == HISTORY_START else CACHE_SECONDS
            cached = self._cache.get(start)
            if cached is not None and time.monotonic() - cached[1] < interval:
                self._cache.move_to_end(start)
                return cached[0]
            result = self._read_coverage(start)
            self._cache[start] = (result, time.monotonic())
            self._cache.move_to_end(start)
            while len(self._cache) > 8:
                self._cache.popitem(last=False)
            return result

    def _read_coverage(self, start):
        today = datetime.now(ZoneInfo("America/New_York")).date()
        with Catalog(self.cfg) as catalog, catalog.connection.transaction():
            catalog.connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            # Bound memory to this read-only transaction; avoid the many small
            # hash/sort spills caused by the server's general-purpose default.
            catalog.connection.execute("SET LOCAL work_mem = '96MB'")
            catalog.connection.execute("SET LOCAL statement_timeout = '60s'")
            as_of = datetime.now(ZoneInfo('UTC')).isoformat()
            rows = list(catalog.connection.execute(f"""
                WITH monthly_filings AS (
                    SELECT DISTINCT date_trunc('month', filed_date)::date AS month, accession
                    FROM current_associations WHERE filed_date>=%s
                ), current_extractions AS MATERIALIZED (
                    SELECT r.accession,r.status,r.parsed_at
                    FROM ownership_current_results c JOIN parse_results r ON r.id=c.result_id
                    JOIN submissions current_source ON current_source.accession=r.accession
                        AND current_source.sha256=r.source_sha256
                    WHERE {current_result_sql(PARSER_VERSION)}
                ), outstanding_attempts AS (
                    SELECT p.accession,p.status FROM parse_attempts p
                    JOIN submissions s ON s.accession=p.accession AND s.sha256=p.source_sha256
                    WHERE p.parser_version=%s AND p.status IN ('running','failed','unsupported')
                      AND NOT EXISTS(SELECT 1 FROM parse_attempts newer
                        WHERE newer.accession=p.accession AND newer.source_sha256=p.source_sha256
                          AND newer.parser_version=p.parser_version AND newer.id>p.id)
                )
                SELECT to_char(m.month, 'YYYY-MM') AS month, count(*) AS indexed,
                       count(s.accession) AS downloaded, count(r.accession) AS parsed,
                       count(*) FILTER(WHERE r.status='complete_with_warnings') AS with_warnings,
                       count(*) FILTER(WHERE r.status='needs_review') AS needs_review,
                       count(*) FILTER(WHERE r.accession IS NULL AND p.status='failed') AS failed,
                       count(*) FILTER(WHERE r.accession IS NULL AND p.status='unsupported') AS unsupported,
                       count(*) FILTER(WHERE r.accession IS NULL AND p.status='running') AS running,
                       max(r.parsed_at) AS last_parsed_at
                FROM monthly_filings m LEFT JOIN submissions s USING(accession)
                LEFT JOIN current_extractions r ON r.accession=s.accession
                LEFT JOIN outstanding_attempts p ON p.accession=s.accession
                GROUP BY m.month ORDER BY m.month
            """, (start, PARSER_VERSION, PARSER_VERSION)))
            sources = list(catalog.connection.execute("""
                SELECT s.kind, s.period,
                       COALESCE((SELECT max(t.finished_at) FROM attempts t
                                 WHERE t.artifact_id=a.id AND t.status='validated'
                                   AND t.response_json->>'offline' IS DISTINCT FROM 'true'),
                                a.first_observed_at) AS last_observed_at
                FROM sources s JOIN artifacts a ON a.id=s.current_artifact_id
            """))
            latest = catalog.connection.execute("""
                SELECT s.kind, s.period, t.finished_at,
                       COALESCE(t.response_json->>'offline' = 'true', false) AS offline
                FROM attempts t JOIN sources s ON s.id=t.source_id
                WHERE t.status='validated' AND t.finished_at IS NOT NULL
                ORDER BY t.finished_at DESC, t.id DESC LIMIT 1
            """).fetchone()
        last_indexed = ({"kind": latest["kind"], "period": latest["period"],
                         "indexed_at": latest["finished_at"].isoformat(), "offline": latest["offline"]}
                        if latest else None)
        latest_parse = max((r['last_parsed_at'] for r in rows if r['last_parsed_at']), default=None)
        return {"months": monthly_coverage(rows, sources, today=today, start=start), "last_indexed": last_indexed,
                "last_parsed": latest_parse.isoformat() if latest_parse else None,
                "as_of": as_of}


def make_server(cfg, *, port=8765, offline=False):
    app = Dashboard(cfg)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, status, body, content_type="application/json; charset=utf-8"):
            if not isinstance(body, bytes):
                body = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.headers.get("Host") != f"127.0.0.1:{self.server.server_port}":
                return self.send(403, {"error": "Use the dashboard's 127.0.0.1 address"})
            parsed_url = urlsplit(self.path)
            path = parsed_url.path
            try:
                if path == "/":
                    page = files("insider_research").joinpath("web/dashboard.html").read_bytes()
                    return self.send(200, page, "text/html; charset=utf-8")
                if path == "/api/coverage":
                    params = parse_qs(parsed_url.query, keep_blank_values=True)
                    if set(params) - {'start'} or ('start' in params and len(params['start']) != 1):
                        raise ValueError('Use one optional start date')
                    start = date.fromisoformat(params['start'][0]) if 'start' in params else None
                    return self.send(200, app.coverage(start=start))
                return self.send(404, {"error": "Not found"})
            except psycopg.errors.QueryCanceled:
                self.send(503, {"error": "Coverage query timed out. Choose a shorter range or try again later."})
            except psycopg.Error:
                self.send(503, {"error": "Cannot read the catalogue. Check PostgreSQL is running."})
            except (ValueError, OSError) as exc:
                self.send(400, {"error": str(exc)})

        def do_POST(self):
            self.send(405, {"error": "This dashboard is read-only"})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.app = app
    return server


def serve(cfg, *, port=8765, open_browser=True, offline=False):
    if not 1 <= port <= 65535:
        raise ValueError("Port must be between 1 and 65535")
    server = make_server(cfg, port=port, offline=offline)
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"Insider Research dashboard: {url}\nPress Ctrl+C to stop.", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

"""Read-only monthly submission coverage served on loopback."""

from __future__ import annotations

import json
import webbrowser
from datetime import date, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import psycopg

from .catalog import Catalog
from .config import RESEARCH_START

HISTORY_START = RESEARCH_START


def next_month(month):
    return date(month.year + (month.month == 12), month.month % 12 + 1, 1)


def monthly_coverage(rows, sources, *, today):
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
    month = HISTORY_START
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
                       "open_month": next_month(month) > today})
        month = next_month(month)
    return result


class Dashboard:
    def __init__(self, cfg):
        self.cfg = cfg

    def coverage(self):
        today = datetime.now(ZoneInfo("America/New_York")).date()
        with Catalog(self.cfg) as catalog, catalog.connection.transaction():
            catalog.connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            rows = list(catalog.connection.execute("""
                WITH monthly_filings AS (
                    SELECT DISTINCT date_trunc('month', filed_date)::date AS month, accession
                    FROM current_associations
                )
                SELECT to_char(m.month, 'YYYY-MM') AS month, count(*) AS indexed,
                       count(s.accession) AS downloaded
                FROM monthly_filings m LEFT JOIN submissions s USING(accession)
                GROUP BY m.month ORDER BY m.month
            """))
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
        return {"months": monthly_coverage(rows, sources, today=today), "last_indexed": last_indexed}


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
            path = urlsplit(self.path).path
            try:
                if path == "/":
                    page = files("insider_research").joinpath("web/dashboard.html").read_bytes()
                    return self.send(200, page, "text/html; charset=utf-8")
                if path == "/api/coverage":
                    return self.send(200, app.coverage())
                return self.send(404, {"error": "Not found"})
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

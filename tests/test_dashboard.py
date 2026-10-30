import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

from insider_research.catalog import Catalog
from insider_research.config import Config
from insider_research.dashboard import Dashboard, HISTORY_START, make_server, monthly_coverage
from insider_research.discovery import discover
from insider_research.http import Response
from insider_research.sources.indexes import source_for
from insider_research.submissions import _publish_zip
from insider_research.ownership_store import parse_ownership
from insider_research.ownership import PARSER_VERSION
from tests.test_ownership import FORM4, source as filing_source
from tests.helpers import HEADER, postgres_config, record


def source(period="2025Q1", observed="2025-04-02", kind="quarterly"):
    return {"period": period, "kind": kind,
            "last_observed_at": datetime.fromisoformat(observed).replace(tzinfo=timezone.utc)}


def january(rows, sources, today=date(2025, 4, 2)):
    return next(month for month in monthly_coverage(rows, sources, today=today) if month["month"] == "2025-01")


class DashboardTests(unittest.TestCase):
    def test_every_month_from_mandatory_electronic_research_start(self):
        months = monthly_coverage([], [], today=date(2026, 10, 21))
        self.assertEqual(months[0]["month"], HISTORY_START.strftime("%Y-%m"))
        self.assertEqual(months[-1]["month"], "2026-10")
        self.assertEqual(months[0]["month"], "2003-07")
        self.assertEqual(len(months), (2026 - 2003) * 12 + 10 - 7 + 1)
        self.assertEqual(len({month["month"] for month in months}), len(months))
        self.assertTrue(all(month["state"] == "unindexed" and month["expected"] is None for month in months))

    def test_partial_and_complete_downloads_have_real_denominators(self):
        rows = [{"month": "2025-01", "indexed": 10, "downloaded": 3}]
        month = january(rows, [source()])
        self.assertEqual((month["expected"], month["state"]), (10, "partial"))
        rows[0]["downloaded"] = 10
        self.assertEqual(january(rows, [source()])["state"], "complete")
        rows[0]["downloaded"] = 0
        self.assertEqual(january(rows, [source()])["state"], "empty")

    def test_absent_daily_only_stale_and_open_month_indexes_cannot_be_green(self):
        rows = [{"month": "2025-01", "indexed": 10, "downloaded": 10}]
        for sources in ([], [source(kind="daily")], [source(observed="2025-01-20")]):
            with self.subTest(sources=sources):
                month = january(rows, sources)
                self.assertIsNone(month["expected"])
                self.assertEqual(month["state"], "partial")
        month = january(rows, [source(observed="2025-01-20")], today=date(2025, 1, 20))
        self.assertTrue(month["open_month"])
        self.assertIsNone(month["expected"])

    def test_known_zero_is_distinguished_from_unknown(self):
        month = january([], [source()])
        self.assertEqual(month["expected"], 0)
        self.assertEqual(month["state"], "empty")
        self.assertIsNone(january([], [])["expected"])
        self.assertEqual(january([], [])["state"], "unindexed")
        self.assertEqual(january([], [source(period="2025-01-02", kind="daily")])["state"], "empty")
        self.assertEqual(january([], [source(observed="2024-12-31")])["state"], "unindexed")

    def test_parsing_denominator_and_disjoint_states(self):
        row = dict(month='2025-01', indexed=12, downloaded=10, parsed=5,
                   with_warnings=1, needs_review=2, failed=1, unsupported=1, running=1)
        month = january([row], [])
        p = month['parsing']
        self.assertIsNone(month['expected'])
        self.assertEqual((p['parsed'], p['completed'], p['remaining'], p['pending']), (5, 3, 5, 2))
        self.assertEqual(p['state'], 'attention')
        self.assertEqual(p['completed']+p['needs_review']+p['failed']+p['unsupported']+p['running']+p['pending'], 10)
        row.update(downloaded=5, failed=0, unsupported=0, running=0)
        self.assertEqual(january([row], [])['parsing']['state'], 'review')
        row['needs_review'] = 0
        self.assertEqual(january([row], [])['parsing']['state'], 'complete')
        self.assertEqual(january([], [])['parsing']['state'], 'unavailable')

    def test_cache_shares_reads_and_does_not_cache_errors(self):
        app = Dashboard(Config(data_root=Path('dashboard-test')))
        start = date(datetime.now(ZoneInfo('America/New_York')).year, 1, 1)
        with patch.object(app, '_read_coverage', side_effect=[{'months': []}, {'months': [1]}, ValueError('unavailable'), {'months': [2]}]) as read:
            with patch('insider_research.dashboard.time.monotonic', return_value=100):
                self.assertEqual(app.coverage(start=start), app.coverage(start=start))
                self.assertEqual(read.call_count, 1)
            with patch('insider_research.dashboard.time.monotonic', return_value=131):
                self.assertEqual(app.coverage(start=start), {'months': [1]})
            with patch('insider_research.dashboard.time.monotonic', return_value=162):
                with self.assertRaises(ValueError):
                    app.coverage(start=start)
                self.assertEqual(app.coverage(start=start), {'months': [2]})

    def test_month_range_and_validation(self):
        months = monthly_coverage([], [], today=date(2026,10,28), start=date(2026,1,1))
        self.assertEqual((len(months), months[0]['month'], months[-1]['month']), (10, '2026-01', '2026-10'))
        app = Dashboard(Config(data_root=Path('dashboard-test')))
        for start in (date(2000,1,1), date(2026,1,2), date(9999,1,1)):
            with self.subTest(start=start), self.assertRaises(ValueError):
                app.coverage(start=start)

    def test_database_parsing_counts_current_sources_versions_and_latest_attempts(self):
        with tempfile.TemporaryDirectory() as temp, postgres_config(Path(temp)/'data') as cfg:
            Catalog.initialize(cfg)
            accessions = [f'0000001750-26-{800000+i:06d}' for i in range(9)]
            body = HEADER + b''.join(record(accession=a, filed='2026-01-06') for a in accessions)
            body += record(accession=accessions[0], cik='456', filed='2026-01-06')
            class Client:
                def get(self, _):
                    return Response(body, {'status': 200})
            discover(cfg, [source_for('quarterly', '2026Q1'), source_for('daily', '2026-01-06')], client=Client())
            for i, accession in enumerate(accessions[:8]):
                text = filing_source().replace(FORM4.encode(), accession.encode())
                # A real source discrepancy, independent of the retired
                # checkbox/keyword heuristic, still counts as a warning.
                text = text.replace(b'<periodOfReport>2026-01-02', b'<periodOfReport>2026-01-01')
                if i == 2:
                    text = text.replace(b'</ownershipDocument>', b'<newField>review</newField></ownershipDocument>')
                elif i == 3:
                    text = text.replace(b'</issuer>', b'</invalid>')
                elif i == 4:
                    text = text.replace(b'ownershipDocument', b'unsupportedDocument')
                digest, relative, size, source_size = _publish_zip(cfg, accession, text)
                with Catalog(cfg) as cat:
                    run = cat.start_run('fixture', cfg)
                    attempt = cat.begin_submission_attempt(run, accession)
                    cat.register_submission(accession, relative, digest, size, source_size, 'https://www.sec.gov/fixture', attempt)
                    cat.finish_run(run, 'succeeded', {})
                if i < 5:
                    parse_ownership(cfg, accession=accession, workers=1)
            with Catalog(cfg) as cat:
                run = cat.start_run('fixture', cfg)
                for i, state in ((6, 'running'), (7, 'interrupted')):
                    cat.connection.execute('''INSERT INTO parse_attempts
                        (run_id,accession,source_sha256,parser_version,started_at,status)
                        SELECT %s,accession,sha256,%s,now(),%s FROM submissions WHERE accession=%s''',
                        (run, PARSER_VERSION, state, accessions[i]))
                # An unsuccessful later attempt must not hide a selected current extraction.
                cat.connection.execute('''INSERT INTO parse_attempts
                    (run_id,accession,source_sha256,parser_version,started_at,status)
                    SELECT %s,accession,sha256,%s,now(),'failed' FROM submissions WHERE accession=%s''',
                    (run, PARSER_VERSION, accessions[0]))
            coverage = Dashboard(cfg).coverage()
            month = next(m for m in coverage['months'] if m['month'] == '2026-01')
            self.assertEqual((month['indexed'], month['downloaded']), (9, 8))
            p = month['parsing']
            self.assertEqual((p['parsed'], p['with_warnings'], p['needs_review']), (3, 2, 1))
            self.assertEqual((p['failed'], p['unsupported'], p['running'], p['pending'], p['remaining']), (1, 1, 1, 2, 5))
            self.assertIsNotNone(coverage['last_parsed'])
            with patch('insider_research.dashboard.PARSER_VERSION', 'future-parser'):
                stale = next(m for m in Dashboard(cfg).coverage()['months'] if m['month'] == '2026-01')['parsing']
            self.assertEqual((stale['parsed'], stale['failed'], stale['running'], stale['pending']), (0, 0, 0, 8))
            with Catalog(cfg) as cat:
                cat.connection.execute('UPDATE submissions SET sha256=%s WHERE accession=%s', ('b'*64, accessions[1]))
            changed = next(m for m in Dashboard(cfg).coverage()['months'] if m['month'] == '2026-01')['parsing']
            self.assertEqual((changed['parsed'], changed['with_warnings'], changed['pending']), (2, 1, 3))

    def test_http_is_read_only_and_rejects_wrong_hosts(self):
        server = make_server(Config(data_root=Path("dashboard-test")), port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with urllib.request.urlopen(base) as response:
                page = response.read().decode()
                self.assertIn("Monthly coverage", page)
                self.assertNotIn("Download selected filings", page)
            with patch.object(server.app, "coverage", return_value={"months": []}):
                with urllib.request.urlopen(base + "/api/coverage") as response:
                    self.assertEqual(json.load(response), {"months": []})
                with urllib.request.urlopen(base + '/api/coverage?start=2026-01-01') as response:
                    self.assertEqual(json.load(response), {'months': []})
            for request, status in (
                (urllib.request.Request(base, headers={"Host": "example.com"}), 403),
                (urllib.request.Request(base + "/api/jobs", data=b"{}"), 405),
                (urllib.request.Request(base + "/api/filings"), 404),
                (urllib.request.Request(base + '/api/coverage?start=invalid'), 400),
                (urllib.request.Request(base + '/api/coverage?start=2026-01-02'), 400),
                (urllib.request.Request(base + '/api/coverage?start=2026-01-01&start=2026-02-01'), 400),
            ):
                with self.assertRaises(urllib.error.HTTPError) as raised:
                    urllib.request.urlopen(request)
                self.assertEqual(raised.exception.code, status)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_database_counts_unique_accessions_across_entities_and_source_overlap(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = self.enterContext(postgres_config(Path(temp) / "data"))
            Catalog.initialize(cfg)
            class Client:
                def get(self, _):
                    return Response(HEADER + record() + record(cik="456"), {"status": 200})
            discover(cfg, [source_for("quarterly", "2024Q1"), source_for("daily", "2024-01-02")], client=Client())
            with Catalog(cfg) as catalog:
                run = catalog.start_run("download-submissions", cfg)
                accession = "0000000123-24-000001"
                attempt = catalog.begin_submission_attempt(run, accession)
                catalog.register_submission(accession, "raw/test.zip", "a" * 64, 100, 200,
                                            "https://www.sec.gov/test.txt", attempt)
                catalog.finish_run(run, "succeeded", {})
            coverage = Dashboard(cfg).coverage()
            self.assertEqual(coverage["last_indexed"]["period"], "2024-01-02")
            self.assertFalse(coverage["last_indexed"]["offline"])
            months = coverage["months"]
            month = next(month for month in months if month["month"] == "2024-01")
            self.assertEqual((month["indexed"], month["downloaded"], month["expected"]), (1, 1, 1))
            self.assertEqual(month["state"], "complete")
            # Replaying a cached mid-month index later must not make it complete.
            with Catalog(cfg) as catalog:
                catalog.connection.execute("UPDATE artifacts SET first_observed_at='2024-01-15', last_observed_at='2024-01-15'")
                catalog.connection.execute("UPDATE attempts SET finished_at='2024-01-15'")
            discover(cfg, [source_for("quarterly", "2024Q1")], offline=True)
            coverage = Dashboard(cfg).coverage()
            self.assertEqual(coverage["last_indexed"]["period"], "2024Q1")
            self.assertTrue(coverage["last_indexed"]["offline"])
            month = next(month for month in coverage["months"] if month["month"] == "2024-01")
            self.assertIsNone(month["expected"])
            self.assertEqual(month["state"], "partial")


if __name__ == "__main__":
    unittest.main()

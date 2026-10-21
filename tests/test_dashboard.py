import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import patch

from insider_research.catalog import Catalog
from insider_research.config import Config
from insider_research.dashboard import Dashboard, HISTORY_START, make_server, monthly_coverage
from insider_research.discovery import discover
from insider_research.http import Response
from insider_research.sources.indexes import source_for
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
            for request, status in (
                (urllib.request.Request(base, headers={"Host": "example.com"}), 403),
                (urllib.request.Request(base + "/api/jobs", data=b"{}"), 405),
                (urllib.request.Request(base + "/api/filings"), 404),
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

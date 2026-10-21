import gzip
import sqlite3
import tempfile
import unittest
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from insider_research.catalog import Catalog
from insider_research.discovery import discover
from insider_research.http import FetchError, Response
from insider_research.locking import publisher_lock
from insider_research.sources.indexes import source_for, parse_master_index
from insider_research.validation import validate
from tests.helpers import HEADER, config, record


class Client:
    def __init__(self, content):
        self.content = content

    def get(self, _):
        if isinstance(self.content, BaseException):
            raise self.content
        return Response(self.content, {"status": 200})


class CatalogueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cfg = config(Path(self.temp.name))
        self.source = source_for("quarterly", "2024Q1")
        Catalog.initialize(self.cfg)

    def run_content(self, content, **kwargs):
        return discover(self.cfg, [self.source], client=Client(content), **kwargs)

    def test_initialize_idempotent_and_keeps_rows(self):
        self.run_content(HEADER + record())
        Catalog.initialize(self.cfg)
        with Catalog(self.cfg.database) as cat:
            self.assertEqual(cat.status()["unique_filings_ever_observed"], 1)

    def test_nonempty_unrelated_root_refused(self):
        other = Path(self.temp.name) / "other"
        other.mkdir()
        (other / "important.txt").write_text("preserve")
        with self.assertRaises(ValueError):
            Catalog.initialize(type(self.cfg)(data_root=other))

    def test_unique_accession_multiple_associations_and_rerun(self):
        body = HEADER + record() + record(cik="456")
        self.run_content(body)
        self.run_content(body)
        with Catalog(self.cfg.database) as cat:
            status = cat.status()
            self.assertEqual(status["unique_filings_ever_observed"], 1)
            self.assertEqual(status["current_index_associations"], 2)
            self.assertEqual(status["source_revisions"], 1)
            self.assertEqual(len(cat.inspect("0000000123-24-000001")["associations"]), 2)

    def test_preserved_bytes_roundtrip(self):
        body = HEADER + record()
        self.run_content(body)
        paths = list((self.cfg.data_root / "raw/indexes").rglob("*.gz"))
        self.assertEqual(len(paths), 1)
        self.assertEqual(gzip.decompress(paths[0].read_bytes()), body)

    def test_revision_preserves_old_observations(self):
        self.run_content(HEADER + record())
        self.run_content(HEADER + record(company="Renamed Corporation"))
        with Catalog(self.cfg.database) as cat:
            rows = cat.inspect("0000000123-24-000001")["associations"]
            self.assertEqual(len(rows), 2)
            self.assertEqual(sum(r["in_current_source"] for r in rows), 1)

    def test_revision_removals_need_explicit_activation(self):
        self.run_content(HEADER + record())
        changed = HEADER + record(form="10-K", accession="0000000123-24-000002")
        result = self.run_content(changed)
        self.assertEqual(result["status"], "incomplete")
        with Catalog(self.cfg.database) as cat:
            self.assertEqual(cat.status()["unique_filings_in_current_sources"], 1)
        result = self.run_content(changed, accept_removals=True)
        self.assertEqual(result["removed_accessions"], 1)
        with Catalog(self.cfg.database) as cat:
            self.assertEqual(cat.status()["unique_filings_in_current_sources"], 0)
            self.assertEqual(cat.status()["unique_filings_ever_observed"], 1)

    def test_malformed_revision_never_replaces_good_index(self):
        self.run_content(HEADER + record())
        self.run_content(b"<html>Error</html>")
        with Catalog(self.cfg.database) as cat:
            self.assertEqual(cat.status()["unique_filings_in_current_sources"], 1)
            self.assertEqual(cat.coverage()[0]["latest_attempt_status"], "failed")

    def test_http_failure_is_not_zero_filings(self):
        result = self.run_content(FetchError("Blocked", 403))
        self.assertEqual(result["status"], "incomplete")
        with Catalog(self.cfg.database) as cat:
            self.assertEqual(cat.coverage()[0]["http_status"], 403)
            self.assertIsNone(cat.coverage()[0]["ownership_rows"])

    def test_404_remains_unresolved(self):
        self.run_content(FetchError("Not found", 404))
        with Catalog(self.cfg.database) as cat:
            self.assertEqual(cat.coverage()[0]["latest_attempt_status"], "unavailable")

    def test_offline_cache_no_http(self):
        self.run_content(HEADER + record())
        with patch("insider_research.discovery.HttpClient", side_effect=AssertionError("network")):
            result = discover(self.cfg, [self.source], offline=True)
        self.assertEqual(result["status"], "succeeded")

    def test_missing_offline_cache_is_incomplete(self):
        result = discover(self.cfg, [self.source], offline=True)
        self.assertEqual(result["status"], "incomplete")

    def test_corrupt_cache_detected(self):
        self.run_content(HEADER + record())
        path = next((self.cfg.data_root / "raw/indexes").rglob("*.gz"))
        path.write_bytes(gzip.compress(HEADER + record(form="5")))
        result = discover(self.cfg, [self.source], offline=True)
        self.assertEqual(result["status"], "incomplete")

    def test_transaction_rollback_does_not_activate_partial_rows(self):
        with patch.object(Catalog, "commit_index", side_effect=sqlite3.IntegrityError("simulated failure")):
            with self.assertRaises(sqlite3.IntegrityError):
                self.run_content(HEADER + record())
        with Catalog(self.cfg.database) as cat:
            self.assertEqual(cat.status()["unique_filings_ever_observed"], 0)
            self.assertEqual(cat.coverage()[0]["latest_attempt_status"], "failed")

    def test_constraint_failure_rolls_back_inserted_filings(self):
        parsed = parse_master_index(HEADER + record() + record(accession="0000000123-24-000002"))
        parsed.rows[1] = replace(parsed.rows[1], form_type="INVALID")
        with patch("insider_research.discovery.parse_master_index", return_value=parsed):
            with self.assertRaises(sqlite3.IntegrityError):
                self.run_content(HEADER + record())
        with Catalog(self.cfg.database) as cat:
            self.assertEqual(cat.status()["unique_filings_ever_observed"], 0)
            self.assertEqual(cat.status()["source_revisions"], 0)

    def test_blocked_access_preserves_unattempted_periods(self):
        sources = [self.source, source_for("quarterly", "2024Q2")]
        result = discover(self.cfg, sources, client=Client(FetchError("Blocked", 403)))
        self.assertEqual(result["sources_not_attempted"], 1)
        with Catalog(self.cfg.database) as cat:
            coverage = cat.coverage()
            self.assertEqual(len(coverage), 2)
            self.assertIsNone(coverage[1]["latest_attempt_status"])

    def test_validate_artifacts_and_unresolved_sources(self):
        self.run_content(HEADER + record())
        self.assertEqual(validate(self.cfg)["status"], "valid")
        next((self.cfg.data_root / "raw/indexes").rglob("*.gz")).unlink()
        result = validate(self.cfg)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(len(result["problems"]), 1)

    def test_recover_abandoned_writer_records(self):
        with Catalog(self.cfg.database) as cat:
            run_id = cat.start_run("simulated-kill", self.cfg)
            cat.begin_attempt(run_id, self.source.kind, self.source.period, self.source.url)
            with publisher_lock(self.cfg.data_root):
                result = cat.recover()
            self.assertEqual(result["attempts_reconciled"], 1)
            self.assertEqual(result["runs_reconciled"], 1)
            self.assertEqual(cat.status()["running_runs"], 0)

    def test_list_filings_filters_without_owner_duplication(self):
        self.run_content(HEADER + record() + record(cik="456") + record(form="5", accession="0000000123-24-000002"))
        with Catalog(self.cfg.database) as cat:
            self.assertEqual(len(cat.list_filings()), 2)
            self.assertEqual(len(cat.list_filings(form="4")), 1)
            self.assertEqual(cat.list_filings(form="4")[0]["indexed_entity_count"], 2)
            self.assertEqual(cat.list_filings(start="2025-01-01"), [])

    def test_interrupt_marks_run_and_attempt(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_content(KeyboardInterrupt())
        with Catalog(self.cfg.database) as cat:
            self.assertEqual(cat.status()["running_runs"], 0)
            self.assertEqual(cat.coverage()[0]["latest_attempt_status"], "interrupted")

    def test_single_publisher_lock(self):
        with publisher_lock(self.cfg.data_root):
            with self.assertRaises(ValueError):
                with publisher_lock(self.cfg.data_root):
                    pass

    def test_newer_schema_refused(self):
        with closing(sqlite3.connect(self.cfg.database)) as conn:
            conn.execute("PRAGMA user_version=999")
        with self.assertRaises(ValueError):
            Catalog(self.cfg.database)

    def test_metadata_conflicts_are_reported(self):
        self.run_content(HEADER + record())
        source = source_for("daily", "2024-01-02")
        discover(self.cfg, [source], client=Client(HEADER + record(form="5")))
        with Catalog(self.cfg.database) as cat:
            self.assertTrue(cat.inspect("0000000123-24-000001")["metadata_conflict"])

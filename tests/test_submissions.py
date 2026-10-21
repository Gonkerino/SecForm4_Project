import tempfile
import uuid
import unittest
import zipfile
import hashlib
import os
from dataclasses import replace
from pathlib import Path
from datetime import datetime, timezone
from importlib.resources import files

import psycopg
from psycopg import sql

from insider_research.catalog import Catalog
from insider_research.discovery import discover
from insider_research.http import FetchError, Response
from insider_research.locking import publisher_lock
from insider_research.sources.indexes import source_for
from insider_research.submissions import download_submissions, verify_submissions, _verify_zip
from tests.helpers import HEADER, postgres_config, record


DOCUMENT = b"<SEC-DOCUMENT>0000000123-24-000001.txt\n<SEC-HEADER>header</SEC-HEADER>\n<TEXT>body</TEXT>\n"


class FakeClient:
    def __init__(self, body):
        self.body = body
        self.urls = []

    def get(self, url, *, max_bytes=None):
        self.urls.append((url, max_bytes))
        if isinstance(self.body, BaseException):
            raise self.body
        body = self.body(url) if callable(self.body) else self.body
        return Response(body, {"status": 200})


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cfg = self.enterContext(postgres_config(Path(self.temp.name) / "data"))
        Catalog.initialize(self.cfg)
        source = source_for("quarterly", "2024Q1")
        discover(self.cfg, [source], client=FakeClient(HEADER + record()))

    def test_existing_catalogue_schema_migrates_additively(self):
        legacy = replace(self.cfg, data_root=Path(self.temp.name) / "legacy-data",
                         database_schema="test_migration_" + uuid.uuid4().hex)
        dsn = self.cfg.database_url
        with psycopg.connect(dsn, autocommit=True) as connection, connection.transaction():
            connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(legacy.database_schema)))
            connection.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(legacy.database_schema)))
            migration = files("insider_research").joinpath("migrations/001_catalog.sql").read_text(encoding="utf-8")
            connection.execute(migration)
            connection.execute("INSERT INTO metadata(key,value) VALUES('data_root',%s)", (str(legacy.data_root),))
            connection.execute("INSERT INTO filings VALUES(%s,%s,%s)",
                ("0000000123-24-000001", datetime.now(timezone.utc), datetime.now(timezone.utc)))
        try:
            Catalog.initialize(legacy)
            with Catalog(legacy) as catalog:
                self.assertEqual(catalog.status()["schema_version"], 2)
                self.assertEqual(catalog.status()["unique_filings_ever_observed"], 1)
                self.assertEqual(catalog.status()["submissions_downloaded"], 0)
        finally:
            with psycopg.connect(dsn, autocommit=True) as connection:
                connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(legacy.database_schema)))

    def test_download_zip_registered_and_repeat_skips_existing(self):
        client = FakeClient(DOCUMENT)
        result = download_submissions(self.cfg, client=client)
        self.assertEqual(result["submissions_downloaded"], 1)
        self.assertEqual(len(client.urls), 1)
        self.assertTrue(client.urls[0][0].endswith("edgar/data/123/0000000123-24-000001.txt"))
        with Catalog(self.cfg) as cat:
            row = cat.connection.execute("SELECT * FROM submissions").fetchone()
            self.assertEqual(row["source_bytes"], len(DOCUMENT))
            self.assertEqual(cat.status()["submissions_downloaded"], 1)
        archive_path = self.cfg.data_root / row["archive_path"]
        with zipfile.ZipFile(archive_path) as archive:
            self.assertEqual(archive.namelist(), ["0000000123-24-000001.txt"])
            self.assertEqual(archive.read(archive.namelist()[0]), DOCUMENT)
        self.assertEqual(verify_submissions(self.cfg)["status"], "valid")
        second = FakeClient(DOCUMENT)
        self.assertEqual(download_submissions(self.cfg, client=second)["submissions_requested"], 0)
        self.assertEqual(second.urls, [])

    def test_default_download_processes_all_matching_filings_and_reports_progress(self):
        source = source_for("quarterly", "2024Q1")
        accessions = ["0000000123-24-000002", "0000000123-24-000003"]
        discover(self.cfg, [source], client=FakeClient(HEADER + record() +
            record(accession=accessions[0], filed="2024-02-01") +
            record(accession=accessions[1], filed="2024-03-01")))
        events = []

        def document_for_url(url):
            accession = url.rsplit("/", 1)[1][:-4]
            return (f"<SEC-DOCUMENT>{accession}.txt\n<SEC-HEADER>header</SEC-HEADER>\n"
                    "<TEXT>body</TEXT>\n").encode()

        result = download_submissions(self.cfg, client=FakeClient(document_for_url), progress=events.append)
        self.assertEqual(result["total_indexed"], 3)
        self.assertEqual(result["pending_before_run"], 3)
        self.assertEqual(result["submissions_requested"], 3)
        self.assertEqual(result["submissions_downloaded"], 3)
        self.assertTrue(result["coverage_complete"])
        self.assertEqual(events[0]["event"], "start")
        self.assertEqual(events[0]["total"], 3)
        self.assertEqual(sum(event["event"] == "progress" for event in events), 3)

    def test_bad_response_never_registers_and_explicit_retry_can_recover(self):
        first = download_submissions(self.cfg, client=FakeClient(b"<html>blocked</html>"))
        self.assertEqual(first["status"], "incomplete")
        with Catalog(self.cfg) as cat:
            self.assertEqual(cat.connection.execute("SELECT count(*) AS n FROM submissions").fetchone()["n"], 0)
        skipped = download_submissions(self.cfg, client=FakeClient(DOCUMENT))
        self.assertEqual(skipped["submissions_requested"], 0)
        retried = download_submissions(self.cfg, client=FakeClient(DOCUMENT), retry_failed=True)
        self.assertEqual(retried["submissions_downloaded"], 1)
        self.assertEqual(verify_submissions(self.cfg)["status"], "valid")

    def test_failed_latest_attempt_is_not_retried_unless_requested(self):
        download_submissions(self.cfg, client=FakeClient(FetchError("temporary", 503)))
        with Catalog(self.cfg) as cat:
            self.assertEqual(cat.submission_candidates(), [])
            self.assertEqual(len(cat.submission_candidates(retry_failed=True)), 1)

    def test_recovery_marks_abandoned_filing_attempt(self):
        client = FakeClient(KeyboardInterrupt())
        with self.assertRaises(KeyboardInterrupt):
            download_submissions(self.cfg, client=client)
        with Catalog(self.cfg) as cat, publisher_lock(self.cfg.data_root, cat):
            recovered = cat.recover()
            self.assertEqual(recovered["submission_attempts_reconciled"], 1)
            attempt = cat.connection.execute("SELECT status FROM submission_attempts").fetchone()
            self.assertEqual(attempt["status"], "interrupted")
        resumed = download_submissions(self.cfg, client=FakeClient(DOCUMENT))
        self.assertEqual(resumed["submissions_downloaded"], 1)

    def test_corrupt_archive_is_reported(self):
        download_submissions(self.cfg, client=FakeClient(DOCUMENT))
        with Catalog(self.cfg) as cat:
            relative = cat.connection.execute("SELECT archive_path FROM submissions").fetchone()["archive_path"]
        (self.cfg.data_root / relative).write_bytes(b"not a zip")
        events = []
        self.assertEqual(verify_submissions(self.cfg, progress=events.append)["status"], "incomplete")
        self.assertEqual([event["event"] for event in events], ["loading", "start", "progress"])
        self.assertEqual(events[-1]["current"], 1)
        self.assertEqual(events[-1]["total"], 1)
        self.assertEqual(events[-1]["problems"], 1)

    def test_parallel_validation_matches_serial_for_missing_and_mismatched_archives(self):
        source = source_for("quarterly", "2024Q1")
        discover(self.cfg, [source], client=FakeClient(HEADER + record() +
            record(accession="0000000123-24-000002") + record(accession="0000000123-24-000003")))
        def document(url):
            return DOCUMENT.replace(b"0000000123-24-000001", url.rsplit("/", 1)[-1][:-4].encode())
        download_submissions(self.cfg, client=FakeClient(document))
        with Catalog(self.cfg) as cat:
            rows = list(cat.connection.execute("SELECT * FROM submissions ORDER BY accession"))
            cat.connection.execute("UPDATE submissions SET source_bytes=source_bytes+1 WHERE accession=%s",
                                   (rows[1]["accession"],))
        (self.cfg.data_root / rows[0]["archive_path"]).unlink()
        serial = verify_submissions(self.cfg, workers=1)
        events = []
        parallel = verify_submissions(self.cfg, workers=4, progress=events.append)
        self.assertEqual(serial, parallel)
        self.assertEqual(parallel["submissions_checked"], 3)
        self.assertEqual(len(parallel["problems"]), 2)
        self.assertEqual(events[-1]["current"], 3)


class ZipVerificationTests(unittest.TestCase):
    def test_small_and_large_archives_retain_all_checks(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "test.zip"
            for body in (b"small", os.urandom(9 * 1024 * 1024)):
                with self.subTest(size=len(body)):
                    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
                        archive.writestr("accession.txt", body)
                    expected = hashlib.sha256(body).hexdigest()
                    result = _verify_zip(path, "accession", expected)
                    self.assertEqual(result, (hashlib.sha256(path.read_bytes()).hexdigest(),
                                              path.stat().st_size, len(body)))
                    self.assertEqual(_verify_zip(path, "accession"), result)
                    with self.assertRaises(ValueError):
                        _verify_zip(path, "accession", "0" * 64)
                    with self.assertRaises(ValueError):
                        _verify_zip(path, "wrong-name")
                    with path.open("r+b") as stream:
                        stream.seek(len("accession.txt") + 30)
                        stream.write(bytes([body[0] ^ 255]))
                    with self.assertRaises(zipfile.BadZipFile):
                        _verify_zip(path, "accession")


if __name__ == "__main__":
    unittest.main()

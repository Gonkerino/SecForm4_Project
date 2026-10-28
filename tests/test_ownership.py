"""Reviewed public filing fixtures plus adversarial and persistence checks."""
from decimal import Decimal
from importlib.resources import files
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from types import SimpleNamespace
from unittest.mock import patch

from insider_research.catalog import Catalog, connect
from insider_research.config import Config
from insider_research.cli import main, parser
from insider_research.ownership import parse_submission, UnsupportedSubmission
from insider_research.ownership_store import (parse_ownership, plan_parse, inspect_ownership,
    parsing_status, validate_ownership, read_archive, TABLES)
from insider_research.submissions import _publish_zip
from tests.helpers import postgres_config

FIXTURES = Path(__file__).parent / "fixtures" / "ownership"
FORM3 = "0000002178-20-000004"
FORM4 = "0000001750-26-000002"
FORM5 = "0000002230-20-000001"


def source(accession=FORM4):
    return (FIXTURES / (accession + ".txt")).read_bytes()


class OwnershipParserTests(unittest.TestCase):
    def test_reviewed_form3_derivative_holdings_and_field_notes(self):
        x = parse_submission(source(FORM3), FORM3)
        self.assertEqual(x.status, "complete")
        self.assertEqual(x.report["issuer_cik"], "0000002178")
        self.assertEqual(x.report["form_type"], "3")
        self.assertEqual(x.owners[0]["officer_title"], "President, Service Transport")
        self.assertFalse(x.owners[0]["is_ten_percent_owner"])
        self.assertEqual(len(x.transactions), 0)
        self.assertEqual([h["underlying_security_shares"] for h in x.holdings], [Decimal("263.0"), Decimal("262.0")])
        self.assertTrue(all(h["is_derivative"] for h in x.holdings))
        self.assertIsNone(x.holdings[0]["conversion_or_exercise_price"])
        price = next(f for f in x.fields if f["source_path"].endswith("derivativeHolding[1]/conversionOrExercisePrice[1]"))
        self.assertEqual(price["value_state"], "footnote_only")
        self.assertEqual(len(x.footnotes), 3)
        self.assertEqual(len(x.field_notes), 6)
        self.assertEqual(x.signatures[0]["signature_name"], "/s/ Wade M. Harrison")

    def test_reviewed_form4_price_precision_signature_and_exhibit(self):
        x = parse_submission(source(), FORM4)
        self.assertEqual(x.status, "complete_with_warnings")
        self.assertEqual([i["code"] for i in x.issues], ["plan_evidence_conflict"])
        self.assertEqual(len(x.documents), 2)
        self.assertEqual(x.documents[1]["document_type"], "EX-24")
        sic = next(f for f in x.header_fields if f["field_name"] == "STANDARD INDUSTRIAL CLASSIFICATION")
        self.assertEqual(sic["value"], "AIRCRAFT & PARTS [3720]")
        self.assertEqual(sic["section_path"], "/ISSUER[1]/COMPANY DATA[1]")
        self.assertEqual(x.header["acceptance_at"].isoformat(), "2026-01-06T22:45:53+00:00")
        row = x.transactions[0]
        self.assertEqual(row["transaction_shares"], Decimal("23077"))
        self.assertEqual(row["transaction_price_per_share"], Decimal("84.1229"))
        self.assertEqual(row["shares_owned_following"], Decimal("237064"))
        self.assertEqual(row["transaction_code"], "S")
        self.assertIsNone(x.owners[0]["is_ten_percent_owner"])
        self.assertEqual(len(x.holdings), 0)
        for doc in x.documents:
            data = source()[doc["byte_offset"]:doc["byte_offset"] + doc["byte_length"]]
            self.assertEqual(hashlib.sha256(data).hexdigest(), doc["sha256"])

    def test_reviewed_form5_fractional_quantities_and_explicit_zero(self):
        x = parse_submission(source(FORM5), FORM5)
        self.assertEqual(x.status, "complete")
        self.assertEqual(x.report["form_type"], "5")
        self.assertFalse(x.report["form4_transactions_reported"])
        self.assertEqual(x.transactions[0]["transaction_shares"], Decimal("4101.766"))
        self.assertEqual(x.transactions[0]["transaction_price_per_share"], Decimal(0))
        self.assertEqual(x.transactions[0]["shares_owned_following"], Decimal("71264.469"))
        self.assertEqual(x.transactions[0]["transaction_form_type"], "5")

    def test_all_amendment_forms_preserved_independently(self):
        for accession, form in ((FORM3, b"3"), (FORM4, b"4"), (FORM5, b"5")):
            body = source(accession).replace(b"<documentType>" + form + b"</documentType>", b"<documentType>" + form + b"/A</documentType>")
            body = body.replace(b"<TYPE>" + form + b"\r\n", b"<TYPE>" + form + b"/A\r\n")
            # Fixtures retain LF or CRLF as originally downloaded.
            body = body.replace(b"<TYPE>" + form + b"\n", b"<TYPE>" + form + b"/A\n")
            x = parse_submission(body, accession)
            self.assertEqual(x.report["form_type"], form.decode() + "/A")

    def test_namespace_and_unknown_field_preservation(self):
        body = source().replace(b"<ownershipDocument>", b'<ownershipDocument xmlns="urn:ownership">')
        body = body.replace(b"</ownershipDocument>", b'<newField extra="x">future value</newField></ownershipDocument>')
        x = parse_submission(body, FORM4)
        self.assertEqual(x.status, "needs_review")
        self.assertEqual(x.transactions[0]["transaction_price_per_share"], Decimal("84.1229"))
        f = next(f for f in x.fields if f["tag"] == "{urn:ownership}newField")
        self.assertEqual(f["text"], "future value")
        self.assertEqual(f["attributes"], {"extra": "x"})
        self.assertFalse(f["mapped"])

    def test_missing_empty_zero_invalid_and_footnote_only_distinct(self):
        for replacement, expected, issue in ((b"<value>0</value>", Decimal(0), False),
                (b"<value></value>", None, False), (b"<footnoteId id=\"F2\"/>", None, False),
                (b"<value>NaN</value>", None, True), (b"<value>1,000</value>", None, True)):
            body = source().replace(b"<value>84.1229</value>", replacement)
            x = parse_submission(body, FORM4)
            self.assertEqual(x.transactions[0]["transaction_price_per_share"], expected)
            self.assertEqual(any(i["code"] == "invalid_scalar" for i in x.issues), issue)

    def test_owners_do_not_multiply_transactions(self):
        import re
        body = source()
        owner = re.search(rb"<reportingOwner>.*?</reportingOwner>", body, re.S).group()
        body = body.replace(owner, owner + owner.replace(b"0001554880", b"0001554881"))
        x = parse_submission(body, FORM4)
        self.assertEqual(len(x.owners), 2)
        self.assertEqual(len(x.transactions), 1)

    def test_duplicate_scalar_and_dangling_footnote_need_review(self):
        body = source().replace(b"<value>84.1229</value>", b"<value>84.1229</value><value>1</value>")
        body = body.replace(b'id="F2"/>', b'id="missing"/>')
        x = parse_submission(body, FORM4)
        self.assertEqual(x.status, "needs_review")
        self.assertTrue({"repeated_value", "unresolved_footnote"}.issubset({i["code"] for i in x.issues}))

    def test_dtd_and_entities_rejected(self):
        body = source().replace(b"<ownershipDocument>", b'<!DOCTYPE ownershipDocument [<!ENTITY x "bad">]><ownershipDocument>')
        with self.assertRaisesRegex(ValueError, "DTD"):
            parse_submission(body, FORM4)

    def test_malformed_and_ambiguous_submissions(self):
        with self.assertRaises(ValueError):
            parse_submission(source().replace(b"</issuer>", b"</bad>"), FORM4)
        with self.assertRaisesRegex(ValueError, "accession"):
            parse_submission(source(), FORM3)
        body = source().replace(b"<TYPE>EX-24", b"<TYPE>4")
        with self.assertRaises(UnsupportedSubmission):
            parse_submission(body, FORM4)

    def test_truncation_depth_and_foreign_namespaces(self):
        with self.assertRaisesRegex(ValueError, "final SEC-DOCUMENT"):
            parse_submission(source().replace(b"</SEC-DOCUMENT>", b""), FORM4)
        body = source().replace(b"</ownershipDocument>", b"<nested>" * 70 + b"</nested>" * 70 + b"</ownershipDocument>")
        with self.assertRaisesRegex(ValueError, "depth/node"):
            parse_submission(body, FORM4)
        body = source().replace(b"<value>84.1229</value>", b'<value xmlns="urn:foreign">84.1229</value>')
        x = parse_submission(body, FORM4)
        self.assertIn("foreign_namespace", {i["code"] for i in x.issues})
        self.assertEqual(x.status, "needs_review")

    def test_cli_contract_all_local_commands_and_date_defaults(self):
        for command in ("plan-parse", "parse-ownership", "parsing-status", "validate-ownership"):
            args = parser().parse_args(["--offline", command])
            self.assertTrue(args.offline)
        args = parser().parse_args(["parse-ownership"])
        self.assertIsNone(args.start)
        self.assertIsNone(args.limit)
        self.assertEqual(parser().parse_args(['parse']).workers, 4)
        self.assertEqual(parser().parse_args(['parse', '--workers', '8']).workers, 8)
        self.assertEqual(parser().parse_args(['parse-ownership', '--workers', '1']).workers, 1)

    def test_invalid_workers_fail_before_database_access(self):
        for workers in (0, 21):
            with self.subTest(workers=workers), self.assertRaisesRegex(ValueError, 'Workers'):
                parse_ownership(None, workers=workers)


class ArchiveReaderTests(unittest.TestCase):
    def test_sized_read_retains_checksum_length_and_snapshot_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = Config(data_root=Path(directory))
            path = cfg.data_root / 'test.zip'
            body = source()
            with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr(FORM4 + '.txt', body)
            archive_bytes = path.read_bytes()
            row = dict(accession=FORM4, archive_path=path.name, zip_bytes=len(archive_bytes),
                source_bytes=len(body), sha256=hashlib.sha256(archive_bytes).hexdigest())
            sizes = []
            original_open = Path.open

            class Reader:
                def __init__(self, stream):
                    self.stream = stream
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    self.stream.close()
                def fileno(self):
                    return self.stream.fileno()
                def read(self, size):
                    sizes.append(size)
                    return self.stream.read(size)

            with patch.object(Path, 'open', lambda p, *a, **k: Reader(original_open(p, *a, **k))):
                self.assertEqual(read_archive(cfg, row), (body, archive_bytes))
            self.assertEqual(sizes, [len(archive_bytes) + 1])
            snapshot = dict(row, source_snapshot_path=path.name, source_sha256=row['sha256'])
            self.assertEqual(read_archive(cfg, snapshot, snapshot=True), (body, archive_bytes))
            for key, value in (('sha256', '0' * 64), ('zip_bytes', len(archive_bytes) + 1),
                               ('source_bytes', len(body) + 1)):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    read_archive(cfg, dict(row, **{key: value}))
            with patch('insider_research.ownership_store.os.fstat',
                       return_value=SimpleNamespace(st_size=len(archive_bytes) - 1)):
                with self.assertRaisesRegex(ValueError, 'changed size'):
                    read_archive(cfg, row)
            with patch('insider_research.ownership_store.os.fstat',
                       return_value=SimpleNamespace(st_size=cfg.max_submission_bytes * 2 + 1024 * 1024 + 1)):
                with self.assertRaisesRegex(ValueError, 'bounded archive'):
                    read_archive(cfg, row)


class OwnershipDatabaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cfg = self.enterContext(postgres_config(Path(self.temp.name) / "data"))
        Catalog.initialize(self.cfg)

    def seed(self, accession=FORM4, body=None):
        body = source(accession) if body is None else body
        digest_path = _publish_zip(self.cfg, accession, body)
        digest, relative, zip_size, source_size = digest_path
        with Catalog(self.cfg) as catalog:
            catalog.connection.execute("INSERT INTO filings VALUES(%s,now(),now()) ON CONFLICT DO NOTHING", (accession,))
            run = catalog.start_run("fixture", self.cfg)
            attempt = catalog.begin_submission_attempt(run, accession)
            catalog.register_submission(accession, relative, digest, zip_size, source_size, "https://www.sec.gov/fixture", attempt)
            catalog.finish_run(run, "succeeded", {})

    def test_real_fixtures_roundtrip_idempotency_and_full_validation(self):
        for accession in (FORM3, FORM4, FORM5):
            self.seed(accession)
        self.assertEqual(plan_parse(self.cfg)["eligible"], 3)
        result = parse_ownership(self.cfg)
        self.assertEqual(result["processed"], 3)
        self.assertEqual(result["complete"], 2)
        self.assertEqual(result["complete_with_warnings"], 1)
        self.assertEqual(parse_ownership(self.cfg)["processed"], 0)
        self.assertEqual(parsing_status(self.cfg)["current_source_and_parser"], 3)
        x = inspect_ownership(self.cfg, FORM5)
        self.assertEqual(x["transactions"][0]["transaction_shares"], Decimal("4101.766"))
        self.assertEqual(validate_ownership(self.cfg)["status"], "valid")
        self.assertEqual(plan_parse(self.cfg, accession=FORM4)["already_extracted"], 1)

    def test_parallel_batches_match_serial_in_every_typed_and_evidence_table(self):
        accessions = (FORM3, FORM4, FORM5)
        for accession in accessions:
            self.seed(accession)
        with patch('insider_research.ownership_store.PARSE_BATCH_SIZE', 2):
            self.assertEqual(parse_ownership(self.cfg, workers=1)['processed'], 3)
            serial = {a: inspect_ownership(self.cfg, a) for a in accessions}
            with Catalog(self.cfg) as cat:
                cat.connection.execute('DELETE FROM ownership_current_results')
            self.assertEqual(parse_ownership(self.cfg, workers=4)['processed'], 3)
        for accession in accessions:
            parallel = inspect_ownership(self.cfg, accession)
            for table in ('header', 'report', *TABLES):
                normalize = lambda rows: [{k: v for k, v in row.items() if k != 'result_id'} for row in rows]
                self.assertEqual(normalize(serial[accession][table]), normalize(parallel[table]), (accession, table))
        self.assertEqual(validate_ownership(self.cfg)['results_checked'], 6)
        self.assertEqual(validate_ownership(self.cfg)['status'], 'valid')
        self.assertEqual(parse_ownership(self.cfg)['processed'], 0)

    def test_parallel_failure_and_unsupported_cases_do_not_discard_valid_results(self):
        self.seed(FORM3, source(FORM3).replace(b'</issuer>', b'</bad>'))
        self.seed(FORM4)
        self.seed(FORM5, source(FORM5).replace(b'ownershipDocument', b'unsupportedDocument'))
        result = parse_ownership(self.cfg, workers=4)
        self.assertEqual((result['processed'], result['failed'], result['unsupported']), (3, 1, 1))
        self.assertIsNotNone(inspect_ownership(self.cfg, FORM4)['result'])
        self.assertEqual(plan_parse(self.cfg)['previous_failures_excluded'], 2)
        with Catalog(self.cfg) as cat:
            failed = list(cat.connection.execute("SELECT * FROM parse_attempts WHERE status IN ('failed','unsupported')"))
        self.assertTrue(all(r['source_snapshot_path'] and r['source_text_sha256'] for r in failed))

    def test_pipeline_interruption_preserves_committed_group_and_resumes_prefetch(self):
        for accession in (FORM3, FORM4, FORM5):
            self.seed(accession)
        def progress(event):
            if event['event'] == 'progress':
                raise KeyboardInterrupt()
        with patch('insider_research.ownership_store.PARSE_BATCH_SIZE', 2), self.assertRaises(KeyboardInterrupt):
            parse_ownership(self.cfg, workers=4, progress=progress)
        with Catalog(self.cfg) as cat:
            run = cat.connection.execute("SELECT * FROM runs WHERE command='parse-ownership'").fetchone()
            self.assertEqual((run['status'], run['summary_json']['processed']), ('interrupted', 2))
            self.assertEqual(cat.connection.execute('SELECT count(*) AS n FROM parse_results').fetchone()['n'], 2)
            self.assertEqual(cat.connection.execute("SELECT count(*) AS n FROM parse_attempts WHERE status='interrupted'").fetchone()['n'], 1)
        self.assertEqual(parse_ownership(self.cfg, workers=4)['processed'], 1)
        self.assertEqual(validate_ownership(self.cfg)['status'], 'valid')

    def test_byte_budget_and_limit_keep_prefetch_bounded(self):
        for accession in (FORM3, FORM4, FORM5):
            self.seed(accession)
        events = []
        with patch('insider_research.ownership_store.PARSE_BATCH_BYTES', 1):
            result = parse_ownership(self.cfg, workers=4, limit=2, progress=events.append)
        self.assertEqual((result['processed'], result['unattempted_eligible']), (2, 1))
        self.assertEqual([e['current'] for e in events if e['event'] == 'progress'], [1, 2])
        with Catalog(self.cfg) as cat:
            self.assertEqual(cat.connection.execute('SELECT count(*) AS n FROM parse_attempts').fetchone()['n'], 2)
        self.assertEqual(parse_ownership(self.cfg)['processed'], 1)

    def test_copy_interruption_rolls_back_entire_group(self):
        from insider_research.ownership_store import _insert
        for accession in (FORM3, FORM4, FORM5):
            self.seed(accession)
        def interrupted(connection, table, rows):
            if table == 'ownership_fields':
                raise KeyboardInterrupt()
            return _insert(connection, table, rows)
        with patch('insider_research.ownership_store._insert', side_effect=interrupted), self.assertRaises(KeyboardInterrupt):
            parse_ownership(self.cfg, workers=4)
        with Catalog(self.cfg) as cat:
            for table in ('parse_results', 'ownership_current_results', 'ownership_reports', *TABLES.values()):
                self.assertEqual(cat.connection.execute(f'SELECT count(*) AS n FROM {table}').fetchone()['n'], 0)
            self.assertEqual(cat.connection.execute("SELECT count(*) AS n FROM parse_attempts WHERE status='interrupted'").fetchone()['n'], 3)
        self.assertEqual(parse_ownership(self.cfg)['processed'], 3)
        self.assertEqual(validate_ownership(self.cfg)['status'], 'valid')

    def test_failures_retry_and_successful_results_survive(self):
        self.seed(FORM4)
        self.seed(FORM5, source(FORM5).replace(b"</issuer>", b"</bad>"))
        result = parse_ownership(self.cfg)
        self.assertEqual(result["failed"], 1)
        self.assertEqual(result["complete_with_warnings"], 1)
        self.assertEqual(plan_parse(self.cfg)["previous_failures_excluded"], 1)
        self.assertEqual(plan_parse(self.cfg, retry_failed=True)["eligible"], 1)
        self.seed(FORM5)  # A changed checksum is eligible without retry-failed.
        self.assertEqual(parse_ownership(self.cfg)["complete"], 1)

    def test_cli_json_roundtrip_and_local_offline_operation(self):
        self.seed(FORM5)
        for args in (["plan-parse"], ["--offline", "parse-ownership"],
                     ["inspect-ownership", FORM5], ["parsing-status"], ["validate-ownership"]):
            output, errors = io.StringIO(), io.StringIO()
            with patch("insider_research.cli.load_config", return_value=self.cfg), patch("sys.stdout", output), patch("sys.stderr", errors):
                self.assertEqual(main(args), 0)
            result = json.loads(output.getvalue())
            if args[0] == "inspect-ownership":
                self.assertEqual(result["transactions"][0]["transaction_shares"], "4101.766")

    def test_failed_new_source_does_not_replace_previous_success(self):
        self.seed()
        parse_ownership(self.cfg)
        first = inspect_ownership(self.cfg, FORM4)["result"]["id"]
        self.seed(body=source().replace(b"</issuer>", b"</bad>"))
        self.assertEqual(parse_ownership(self.cfg)["failed"], 1)
        self.assertEqual(inspect_ownership(self.cfg, FORM4)["result"]["id"], first)
        self.assertEqual(parsing_status(self.cfg)["current_source_and_parser"], 0)
        self.assertEqual(validate_ownership(self.cfg)["status"], "valid")

    def test_missing_invalid_and_empty_filter_bounds(self):
        from datetime import date
        self.seed()
        with self.assertRaises(ValueError):
            plan_parse(self.cfg, start=date(2026, 2, 1), end=date(2026, 1, 1))
        with self.assertRaises(ValueError):
            parse_ownership(self.cfg, limit=0)
        self.assertEqual(plan_parse(self.cfg, form="3")["downloaded_selected"], 0)
        # No index associations is still eligible by default: registered downloads
        # survive removal from active indexes and require no date cutoff.
        self.assertEqual(plan_parse(self.cfg)["downloaded_selected"], 1)

    def test_versions_preserved_and_source_snapshot_independent(self):
        self.seed()
        parse_ownership(self.cfg)
        with patch("insider_research.ownership_store.PARSER_VERSION", "ownership-test-upgrade"):
            self.assertEqual(parse_ownership(self.cfg)["processed"], 1)
        with Catalog(self.cfg) as cat:
            self.assertEqual(cat.connection.execute("SELECT count(*) AS n FROM parse_results").fetchone()["n"], 2)
        old = inspect_ownership(self.cfg, FORM4, result_id=1)
        self.assertIsNotNone(old["result"])
        snapshot = self.cfg.data_root / old["result"]["source_snapshot_path"]
        self.assertTrue(snapshot.is_file())
        with Catalog(self.cfg) as cat:
            archive = cat.connection.execute("SELECT archive_path FROM submissions").fetchone()["archive_path"]
        (self.cfg.data_root / archive).unlink()
        body, _ = read_archive(self.cfg, old["result"], snapshot=True)
        self.assertEqual(body, source())

    def test_atomic_publication_failure_and_interruption_recovery(self):
        self.seed()
        with patch("insider_research.ownership_store._insert", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                parse_ownership(self.cfg)
        with Catalog(self.cfg) as cat:
            self.assertEqual(cat.connection.execute("SELECT count(*) AS n FROM parse_results").fetchone()["n"], 0)
            self.assertEqual(cat.connection.execute("SELECT status FROM parse_attempts").fetchone()["status"], "interrupted")
        self.assertEqual(parse_ownership(self.cfg)["processed"], 1)

    def test_validation_detects_typed_and_field_tampering(self):
        self.seed()
        parse_ownership(self.cfg)
        with Catalog(self.cfg) as cat:
            cat.connection.execute("UPDATE ownership_transactions SET transaction_shares=1")
            cat.connection.execute("UPDATE ownership_fields SET text='changed' WHERE source_path='/ownershipDocument[1]/schemaVersion[1]'")
        result = validate_ownership(self.cfg)
        self.assertEqual(result["status"], "invalid")
        self.assertEqual(result["problem_count"], 1)
        self.assertEqual(len(result["problems"][0]["messages"]), 2)

    def test_path_escape_is_a_filing_failure(self):
        self.seed()
        with Catalog(self.cfg) as cat:
            cat.connection.execute("UPDATE submissions SET archive_path='../escape.zip'")
        self.assertEqual(parse_ownership(self.cfg)["failed"], 1)

    def test_upgrade_from_v1_v2_v3_preserves_catalogue(self):
        # Separate temporary schemas; rebuild only this test-owned schema.
        from psycopg import sql
        for version in (1, 2, 3):
            with connect(self.cfg) as connection:
                connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(self.cfg.database_schema)))
                connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(self.cfg.database_schema)))
                connection.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(self.cfg.database_schema)))
                connection.execute(files("insider_research").joinpath("migrations/001_catalog.sql").read_text())
                connection.execute("INSERT INTO metadata VALUES('data_root',%s)", (str(self.cfg.data_root),))
                connection.execute("INSERT INTO filings VALUES(%s,now(),now())", (FORM4,))
                if version >= 2:
                    connection.execute(files("insider_research").joinpath("migrations/002_submissions.sql").read_text())
                if version >= 3:
                    connection.execute(files("insider_research").joinpath("migrations/003_ownership.sql").read_text())
            Catalog.initialize(self.cfg)
            Catalog.initialize(self.cfg)
            with Catalog(self.cfg) as cat:
                self.assertEqual(cat.status()["schema_version"], 4)
                self.assertEqual(cat.status()["unique_filings_ever_observed"], 1)


if __name__ == "__main__":
    unittest.main()

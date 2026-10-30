"""Reviewed public filing fixtures plus adversarial and persistence checks."""
from decimal import Decimal
from datetime import date
from functools import partial
import re
import threading
import subprocess
import sys
from importlib.resources import files
import hashlib
import copy
import io
import json
from pathlib import Path, PureWindowsPath
import tempfile
import unittest
import zipfile
import psycopg
from types import SimpleNamespace
from unittest.mock import patch
from concurrent.futures.process import BrokenProcessPool

from insider_research.catalog import Catalog, connect
from insider_research.config import Config
from insider_research.cli import main, parser
from insider_research.ownership import parse_submission, UnsupportedSubmission, PARSER_VERSION, LEGACY_PARSER_VERSION
from insider_research.ownership_store import (parse_ownership, plan_parse, inspect_ownership,
    parsing_status, validate_ownership, read_archive, TABLES)
from insider_research.ownership_store import _comparison_path, _safe_path, _preparation_mode, parsing_execution
from insider_research.submissions import _publish_zip
from tests.helpers import postgres_config

FIXTURES = Path(__file__).parent / "fixtures" / "ownership"
FORM3 = "0000002178-20-000004"
FORM4 = "0000001750-26-000002"
FORM5 = "0000002230-20-000001"


def source(accession=FORM4):
    return (FIXTURES / (accession + ".txt")).read_bytes()


class OwnershipParserTests(unittest.TestCase):
    def test_omitted_optional_coding_is_a_warning_without_inferred_values(self):
        for accession in (FORM4, FORM5):
            for amended in (False, True):
                with self.subTest(accession=accession, amended=amended):
                    body = re.sub(rb'<transactionCoding>.*?</transactionCoding>', b'', source(accession), count=1, flags=re.S)
                    if amended:
                        form = b'4' if accession == FORM4 else b'5'
                        body = body.replace(b'<documentType>'+form+b'</documentType>', b'<documentType>'+form+b'/A</documentType>')
                        body = re.sub(rb'(<TYPE>)'+form+rb'(?=\r?\n)', lambda m:m[1]+form+b'/A', body)
                        body = re.sub(rb'(CONFORMED SUBMISSION TYPE:[\t ]*)'+form+rb'(?=\r?\n)', lambda m:m[1]+form+b'/A', body)
                    x = parse_submission(body, accession)
                    self.assertEqual(x.status, 'complete_with_warnings')
                    self.assertIsNone(x.transactions[0]['transaction_code'])
                    self.assertIsNone(x.transactions[0]['transaction_form_type'])
                    self.assertIsNone(x.transactions[0]['equity_swap_involved'])
                    self.assertEqual([(i['code'],i['severity']) for i in x.issues], [('unreported_transaction_code','warning')])
                    for version in (LEGACY_PARSER_VERSION, 'ownership-2'):
                        prior = parse_submission(body, accession, parser_version=version)
                        self.assertEqual(prior.status, 'needs_review')
                        self.assertEqual(prior.transactions, x.transactions)
                        self.assertEqual(prior.fields, x.fields)

    def test_present_coding_block_without_required_code_still_needs_review(self):
        body = re.sub(rb'<transactionCode>[^<]*</transactionCode>', b'', source(), count=1)
        x = parse_submission(body, FORM4)
        self.assertEqual(x.status, 'needs_review')
        self.assertIn('missing_transaction_field', {i['code'] for i in x.issues})
        self.assertNotIn('unreported_transaction_code', {i['code'] for i in x.issues})

    def test_xml_dates_with_timezones_preserve_calendar_dates_and_raw_evidence(self):
        for suffix in ('Z', '-05:00', '+05:30', '+14:00', '-14:00', '+00:00'):
            with self.subTest(suffix=suffix):
                body = re.sub(rb'(2026-01-0[26])(?=<)', lambda m: m[0]+suffix.encode(), source())
                x = parse_submission(body, FORM4)
                self.assertEqual(x.status, 'complete')
                self.assertEqual(x.report['period_of_report'], date(2026, 1, 2))
                self.assertEqual(x.transactions[0]['transaction_date'], date(2026, 1, 2))
                self.assertEqual(x.signatures[0]['signature_date'], date(2026, 1, 6))
                raw = next(f for f in x.fields if f['source_path'].endswith('/periodOfReport[1]'))
                self.assertEqual(raw['text'], '2026-01-02'+suffix)

    def test_invalid_calendar_dates_and_timezone_offsets_still_need_review(self):
        for value in ('2026-02-29', '2026-01-02+14:01', '2026-01-02-15:00',
                      '2026-01-02+05:60', '2026-01-02z', '2026-01-02T00:00:00Z'):
            with self.subTest(value=value):
                body = source().replace(b'<periodOfReport>2026-01-02', b'<periodOfReport>'+value.encode())
                x = parse_submission(body, FORM4)
                self.assertIsNone(x.report['period_of_report'])
                self.assertEqual(x.status, 'needs_review')
                self.assertIn('invalid_scalar', {i['code'] for i in x.issues})

    def test_plan_mentions_do_not_infer_checkbox_conflicts(self):
        for text in ('Transactions were not effected pursuant to a Rule 10b5-1 plan.',
                     'This amendment corrects the incorrectly checked 10b5-1 box.',
                     'Mandatory sell-to-cover transactions under Rule 10b5-1.',
                     'Transactions pursuant to a Rule 10b5-1 plan adopted in 2025.'):
            with self.subTest(text=text):
                body = re.sub(rb'(<footnote id="F1">).*?(</footnote>)',
                              lambda m: m[1]+text.encode()+m[2], source(), flags=re.S)
                x = parse_submission(body, FORM4)
                self.assertFalse(x.report['aff10b5_one'])
                self.assertEqual(x.footnotes[0]['text'], text)
                self.assertEqual(x.status, 'complete')
                legacy = parse_submission(body, FORM4, parser_version=LEGACY_PARSER_VERSION)
                self.assertEqual([i['code'] for i in legacy.issues], ['plan_evidence_conflict'])

    def test_legacy_replay_retains_date_and_warning_behaviour(self):
        body = source().replace(b'<periodOfReport>2026-01-02', b'<periodOfReport>2026-01-02-05:00')
        x = parse_submission(body, FORM4, parser_version=LEGACY_PARSER_VERSION)
        self.assertIsNone(x.report['period_of_report'])
        self.assertEqual(x.status, 'needs_review')
        self.assertIn('plan_evidence_conflict', {i['code'] for i in x.issues})
        with self.assertRaisesRegex(ValueError, 'Unsupported parser version'):
            parse_submission(body, FORM4, parser_version='future-parser')

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
        self.assertEqual(x.status, "complete")
        self.assertEqual(x.issues, [])
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
        self.assertIsNone(parser().parse_args(['parse']).write_workers)
        self.assertEqual(parser().parse_args(['parse', '--write-workers', '1']).write_workers, 1)
        self.assertEqual(parser().parse_args(['parse-ownership', '--write-workers', '2']).write_workers, 2)
        self.assertEqual(parser().parse_args(['parse']).executor, 'auto')
        self.assertEqual(parser().parse_args(['parse', '--executor', 'process']).executor, 'process')
        self.assertEqual(parser().parse_args(['parse', '--workers', '8']).workers, 8)
        self.assertEqual(parser().parse_args(['parse-ownership', '--workers', '1']).workers, 1)

    def test_invalid_workers_fail_before_database_access(self):
        for workers in (0, 21):
            with self.subTest(workers=workers), self.assertRaisesRegex(ValueError, 'Workers'):
                parse_ownership(None, workers=workers)
        for workers in (0, 5):
            with self.subTest(write_workers=workers), self.assertRaisesRegex(ValueError, 'Write workers'):
                parse_ownership(None, write_workers=workers)
        with self.assertRaisesRegex(ValueError, 'Executor'):
            parse_ownership(None, executor='invalid')

    def test_parse_preparation_mode_keeps_small_runs_lightweight(self):
        self.assertEqual(_preparation_mode('auto', 4, 4095), 'thread')
        self.assertEqual(_preparation_mode('auto', 4, 4096), 'process')
        self.assertEqual(_preparation_mode('thread', 4, 1000000), 'thread')
        self.assertEqual(_preparation_mode('process', 4, 3), 'process')
        self.assertEqual(_preparation_mode('process', 1, 1000000), 'serial')

    def test_large_run_defaults_and_explicit_overrides(self):
        self.assertEqual(parsing_execution(total=4096),dict(executor='process',workers=4,write_workers=2))
        self.assertEqual(parsing_execution(total=4095),dict(executor='thread',workers=4,write_workers=1))
        self.assertEqual(parsing_execution(total=1000000,workers=1),dict(executor='serial',workers=1,write_workers=1))
        self.assertEqual(parsing_execution(total=1000000,executor='thread',write_workers=1),
                         dict(executor='thread',workers=4,write_workers=1))
        self.assertEqual(parsing_execution(total=10,executor='process',write_workers=2),
                         dict(executor='process',workers=4,write_workers=2))

    def test_dry_run_reports_the_same_large_and_limited_defaults(self):
        from contextlib import redirect_stdout
        for extra,expected in (([],('process',2)),(['--limit','1000'],('thread',1)),
                               (['--executor','thread','--write-workers','1'],('thread',1))):
            with self.subTest(options=extra):
                output=io.StringIO()
                with patch('insider_research.cli.load_config'), \
                     patch('insider_research.cli.plan_parse',return_value=dict(eligible=100000,parser_version=PARSER_VERSION)), \
                     redirect_stdout(output):
                    self.assertEqual(main(['parse','--start','2020-01-01','--dry-run','--json',*extra]),0)
                result=json.loads(output.getvalue())
                self.assertEqual((result['executor'],result['write_workers']),expected)
                self.assertEqual(result['workers'],4)


class ArchiveReaderTests(unittest.TestCase):
    def test_windows_extended_path_containment_preserves_escape_detection(self):
        for root, child in ((r'D:\data', r'\\?\D:\data\raw\filing.zip'),
                            (r'\\?\D:\data', r'D:\data\raw\filing.zip'),
                            (r'\\server\share\data', r'\\?\UNC\server\share\data\raw\filing.zip')):
            with self.subTest(root=root):
                self.assertTrue(_comparison_path(PureWindowsPath(child)).is_relative_to(_comparison_path(PureWindowsPath(root))))
                with patch.object(Path, 'resolve', side_effect=[PureWindowsPath(child), PureWindowsPath(root)]):
                    self.assertEqual(_safe_path(Path('data'), 'raw/filing.zip'), PureWindowsPath(child))
        root = _comparison_path(PureWindowsPath(r'D:\data'))
        for child in (r'\\?\D:\database\filing.zip', r'\\?\D:\outside\filing.zip', r'\\?\C:\data\filing.zip'):
            self.assertFalse(_comparison_path(PureWindowsPath(child)).is_relative_to(root))
            with patch.object(Path, 'resolve', side_effect=[PureWindowsPath(child), root]):
                with self.assertRaisesRegex(ValueError, 'escapes data root'):
                    _safe_path(Path('data'), 'raw/filing.zip')
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'escapes data root'):
                _safe_path(Path(directory), '../outside.zip')

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

    def test_parse_queue_excludes_downloads_committed_after_start(self):
        from insider_research.locking import publisher_lock
        self.seed(FORM4)
        def progress(event):
            if event['event'] == 'start':
                with Catalog(self.cfg) as downloader, publisher_lock(self.cfg.data_root, downloader, lane="download"):
                    self.seed(FORM3)
                with Catalog(self.cfg) as maintenance, self.assertRaises(ValueError):
                    with publisher_lock(self.cfg.data_root, maintenance):
                        maintenance.recover()
        result = parse_ownership(self.cfg, progress=progress, workers=1)
        self.assertEqual(result['processed'], 1)
        self.assertEqual(result['queued'], 1)
        self.assertEqual(parse_ownership(self.cfg, workers=1)['processed'], 1)

    def test_frozen_queue_pages_resume_after_interrupt_with_new_download(self):
        from insider_research.locking import publisher_lock
        self.seed(FORM3)
        self.seed(FORM4)
        def progress(event):
            if event['event'] == 'progress':
                with Catalog(self.cfg) as downloader, publisher_lock(self.cfg.data_root, downloader, lane="download"):
                    self.seed(FORM5)
                raise KeyboardInterrupt()
        with patch('insider_research.ownership_store.PARSE_BATCH_SIZE', 1), self.assertRaises(KeyboardInterrupt):
            parse_ownership(self.cfg, workers=1, progress=progress)
        with Catalog(self.cfg) as cat:
            self.assertEqual(cat.connection.execute("SELECT count(*) AS n FROM ownership_current_results").fetchone()['n'], 1)
            self.assertEqual(cat.connection.execute("SELECT count(*) AS n FROM parse_attempts WHERE status='running'").fetchone()['n'], 0)
            with publisher_lock(self.cfg.data_root, cat):
                cat.recover()
        with patch('insider_research.ownership_store.PARSE_BATCH_SIZE', 1):
            self.assertEqual(parse_ownership(self.cfg, workers=1)['processed'], 2)
        self.assertEqual(parse_ownership(self.cfg, workers=1)['processed'], 0)

    def test_real_fixtures_roundtrip_idempotency_and_full_validation(self):
        for accession in (FORM3, FORM4, FORM5):
            self.seed(accession)
        self.assertEqual(plan_parse(self.cfg)["eligible"], 3)
        result = parse_ownership(self.cfg)
        self.assertEqual(result["processed"], 3)
        self.assertEqual(result["complete"], 3)
        self.assertEqual(result["complete_with_warnings"], 0)
        self.assertEqual(parse_ownership(self.cfg)["processed"], 0)
        self.assertEqual(parsing_status(self.cfg)["current_source_and_parser"], 3)
        x = inspect_ownership(self.cfg, FORM5)
        self.assertEqual(x["transactions"][0]["transaction_shares"], Decimal("4101.766"))
        self.assertEqual(validate_ownership(self.cfg)["status"], "valid")
        self.assertEqual(plan_parse(self.cfg, accession=FORM4)["already_extracted"], 1)

    def test_targeted_upgrade_preserves_compatible_results_and_historical_replay(self):
        self.seed(FORM3)
        self.seed(FORM4)
        # Use a known date from the fixture rather than assume its value.
        body = source(FORM5)
        body = re.sub(rb'(<periodOfReport>)([^<]+)', lambda m: m[1]+m[2]+b'-05:00', body)
        self.seed(FORM5, body)
        with patch('insider_research.ownership_store.PARSER_VERSION', LEGACY_PARSER_VERSION), \
             patch('insider_research.ownership_store.parse_submission', partial(parse_submission, parser_version=LEGACY_PARSER_VERSION)):
            self.assertEqual(parse_ownership(self.cfg)['processed'], 3)
        before = {a: inspect_ownership(self.cfg, a) for a in (FORM3, FORM4, FORM5)}
        self.assertEqual(before[FORM4]['result']['status'], 'complete_with_warnings')
        self.assertEqual(before[FORM5]['result']['status'], 'needs_review')
        self.assertEqual(parsing_status(self.cfg)['current_source_and_parser'], 1)
        self.assertEqual(plan_parse(self.cfg)['eligible'], 2)
        self.assertEqual(plan_parse(self.cfg, upgrade_only=True)['eligible'], 2)
        # An unparsed download must not be swept into the upgrade.
        unparsed = '0000002178-20-999999'
        self.seed(unparsed, source(FORM3).replace(FORM3.encode(), unparsed.encode()))
        self.assertEqual(parse_ownership(self.cfg, upgrade_only=True, workers=1)['processed'], 2)
        self.assertEqual(parse_ownership(self.cfg, upgrade_only=True)['processed'], 0)
        self.assertIsNone(inspect_ownership(self.cfg, unparsed)['result'])
        after = {a: inspect_ownership(self.cfg, a) for a in (FORM3, FORM4, FORM5)}
        self.assertEqual(after[FORM3]['result']['id'], before[FORM3]['result']['id'])
        for a in (FORM4, FORM5):
            self.assertEqual(after[a]['result']['parser_version'], PARSER_VERSION)
            self.assertEqual(after[a]['result']['supersedes_result_id'], before[a]['result']['id'])
            self.assertEqual(after[a]['result']['status'], 'complete')
            self.assertEqual(after[a]['fields'], [dict(r, result_id=after[a]['result']['id']) for r in before[a]['fields']])
            self.assertEqual(inspect_ownership(self.cfg, a, result_id=before[a]['result']['id'])['result'], before[a]['result'])
        self.assertEqual(parsing_status(self.cfg)['current_source_and_parser'], 3)
        self.assertEqual(validate_ownership(self.cfg)['status'], 'valid')

    def test_optional_coding_upgrade_from_v2_changes_only_issue_classification(self):
        body = re.sub(rb'<transactionCoding>.*?</transactionCoding>', b'', source(), count=1, flags=re.S)
        self.seed(FORM4, body)
        self.seed(FORM3)
        with patch('insider_research.ownership_store.PARSER_VERSION', 'ownership-2'), \
             patch('insider_research.ownership_store.parse_submission', partial(parse_submission, parser_version='ownership-2')):
            self.assertEqual(parse_ownership(self.cfg)['processed'], 2)
        before = inspect_ownership(self.cfg, FORM4)
        compatible = inspect_ownership(self.cfg, FORM3)['result']['id']
        self.assertEqual(before['result']['status'], 'needs_review')
        self.assertEqual(plan_parse(self.cfg, upgrade_only=True)['eligible'], 1)
        result = parse_ownership(self.cfg, upgrade_only=True)
        self.assertEqual((result['processed'],result['complete_with_warnings'],result['needs_review']), (1,1,0))
        after = inspect_ownership(self.cfg, FORM4)
        self.assertEqual(after['result']['supersedes_result_id'], before['result']['id'])
        for name in ('header','report',*TABLES):
            if name == 'issues':continue
            self.assertEqual([{k:v for k,v in r.items() if k!='result_id'} for r in before[name]],
                             [{k:v for k,v in r.items() if k!='result_id'} for r in after[name]])
        self.assertEqual(after['issues'][0]['code'], 'unreported_transaction_code')
        self.assertEqual(inspect_ownership(self.cfg, FORM3)['result']['id'], compatible)
        self.assertEqual(parsing_status(self.cfg)['current_source_and_parser'], 2)
        self.assertEqual(plan_parse(self.cfg, upgrade_only=True)['eligible'], 0)
        self.assertEqual(validate_ownership(self.cfg)['status'], 'valid')

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

    def test_spawned_chunk_processes_match_serial_typed_values_and_evidence(self):
        accessions = (FORM3, FORM4, FORM5)
        for a in accessions:self.seed(a)
        parse_ownership(self.cfg, workers=1)
        before = {a:inspect_ownership(self.cfg,a) for a in accessions}
        with Catalog(self.cfg) as cat:cat.connection.execute('DELETE FROM ownership_current_results')
        with patch('insider_research.ownership_store.PARSE_PROCESS_CHUNK',1), \
             patch('insider_research.ownership_store.PARSE_PROCESS_THRESHOLD',1), \
             patch('insider_research.ownership_store.PARSE_BATCH_SIZE',2):
            result = parse_ownership(self.cfg,executor='auto',workers=2)
        self.assertEqual((result['processed'],result['executor'],result['write_workers']), (3,'process',2))
        for a in accessions:
            after=inspect_ownership(self.cfg,a)
            for name in ('header','report',*TABLES):
                normalize=lambda rows:[{k:v for k,v in r.items() if k!='result_id'} for r in rows]
                self.assertEqual(normalize(before[a][name]),normalize(after[name]),(a,name))
        self.assertEqual(validate_ownership(self.cfg)['status'],'valid')

    def test_process_content_failures_remain_distinct_and_valid_files_survive(self):
        self.seed(FORM3,source(FORM3).replace(b'</issuer>',b'</invalid>'))
        self.seed(FORM4)
        self.seed(FORM5,source(FORM5).replace(b'ownershipDocument',b'unsupportedDocument'))
        result=parse_ownership(self.cfg,executor='process',workers=2)
        self.assertEqual((result['processed'],result['failed'],result['unsupported']),(3,1,1))
        self.assertIsNotNone(inspect_ownership(self.cfg,FORM4)['result'])
        self.assertEqual(validate_ownership(self.cfg)['status'],'valid')

    def test_process_interruption_preserves_committed_group_and_resumes(self):
        for a in (FORM3,FORM4,FORM5):self.seed(a)
        def interrupt(event):
            if event['event']=='progress':raise KeyboardInterrupt()
        with patch('insider_research.ownership_store.PARSE_BATCH_SIZE',2),self.assertRaises(KeyboardInterrupt):
            parse_ownership(self.cfg,executor='process',workers=2,progress=interrupt)
        with Catalog(self.cfg) as cat:
            run=cat.connection.execute("SELECT * FROM runs WHERE command='parse-ownership'").fetchone()
            self.assertEqual((run['status'],run['summary_json']['processed']),('interrupted',2))
            self.assertEqual(cat.connection.execute("SELECT count(*) AS n FROM parse_attempts WHERE status='running'").fetchone()['n'],0)
        self.assertEqual(parse_ownership(self.cfg,executor='process',workers=2)['processed'],1)
        self.assertEqual(validate_ownership(self.cfg)['status'],'valid')

    def test_broken_process_pool_is_resumable_with_thread_fallback(self):
        self.seed(FORM4)
        with patch('insider_research.ownership_store.ProcessPoolExecutor') as pool:
            pool.return_value.submit.side_effect=BrokenProcessPool('Simulated child failure')
            with self.assertRaises(BrokenProcessPool):parse_ownership(self.cfg,executor='process',workers=2)
            pool.return_value.shutdown.assert_called_once_with(wait=True,cancel_futures=True)
        with Catalog(self.cfg) as cat:
            run=cat.connection.execute("SELECT * FROM runs WHERE command='parse-ownership'").fetchone()
            self.assertEqual((run['status'],run['summary_json']['processed']),('failed',0))
            self.assertEqual(cat.connection.execute('SELECT status FROM parse_attempts').fetchone()['status'],'interrupted')
        self.assertEqual(parse_ownership(self.cfg,executor='thread')['processed'],1)
        self.assertEqual(validate_ownership(self.cfg)['status'],'valid')

    def test_command_line_spawn_entrypoint_and_parallel_publication(self):
        for a in (FORM3,FORM4,FORM5):self.seed(a)
        result=subprocess.run([sys.executable,'-m','insider_research','--data-root',str(self.cfg.data_root),
            'parse','--executor','process','--workers','2','--write-workers','2','--json'],
            capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
        summary=json.loads(result.stdout)
        self.assertEqual((summary['processed'],summary['executor'],summary['write_workers']),(3,'process',2))
        self.assertEqual(validate_ownership(self.cfg)['status'],'valid')

    def test_bulk_publication_preserves_bigint_precision_unicode_and_evidence(self):
        expected = Decimal('12345678901234567890.12345678901234567890')
        template = source(FORM4).replace(b'84.1229', str(expected).encode())
        template = template.replace(b'Transaction in accordance with',
            'Δ\tline\nliteral \\N &amp; Transaction in accordance with'.encode())
        accessions = [f'0000001750-26-{900000+i:06d}' for i in range(10)]
        for accession in accessions:
            self.seed(accession, template.replace(FORM4.encode(), accession.encode()))
        with Catalog(self.cfg) as cat:
            cat.connection.execute("SELECT setval(pg_get_serial_sequence('parse_results','id'),%s,false)", (2**40,))
        self.assertEqual(parse_ownership(self.cfg, workers=4)['processed'], 10)
        for accession in accessions:
            result = inspect_ownership(self.cfg, accession)
            self.assertGreaterEqual(result['result']['id'], 2**40)
            actual = result['transactions'][0]['transaction_price_per_share']
            self.assertEqual(actual.as_tuple(), expected.as_tuple())
            self.assertTrue(result['footnotes'][0]['text'].startswith('Δ\tline\nliteral \\N & '))
        self.assertEqual(validate_ownership(self.cfg)['status'], 'valid')

    def test_bulk_statements_respect_connection_types_and_constraints_stay_atomic(self):
        from insider_research.ownership_store import _insert
        with Catalog(self.cfg) as first, Catalog(self.cfg) as second:
            first.connection.execute('CREATE TEMP TABLE copy_type_probe(value bigint PRIMARY KEY CHECK(value>0))')
            second.connection.execute('CREATE TEMP TABLE copy_type_probe(value text NOT NULL)')
            _insert(first.connection, 'copy_type_probe', [dict(value=2**40+i) for i in range(8)])
            _insert(second.connection, 'copy_type_probe', [dict(value=f'Δ\\N\t{i}') for i in range(8)])
            self.assertEqual(first.connection.execute('SELECT min(value) AS value FROM copy_type_probe').fetchone()['value'], 2**40)
            self.assertEqual(second.connection.execute('SELECT value FROM copy_type_probe ORDER BY value LIMIT 1').fetchone()['value'], 'Δ\\N\t0')
            with self.assertRaises(psycopg.errors.CheckViolation):
                with first.connection.transaction():
                    _insert(first.connection, 'copy_type_probe', [dict(value=100+i) for i in range(8)])
                    _insert(first.connection, 'copy_type_probe', [dict(value=-1-i) for i in range(8)])
            self.assertEqual(first.connection.execute('SELECT count(*) AS n FROM copy_type_probe').fetchone()['n'], 8)

    def test_grouped_writes_keep_source_rows_unchanged_and_column_values_aligned(self):
        from insider_research.ownership_store import _insert, _ResultGroup
        rows = [dict(value='a\\N\tline\n', attributes={}),
                dict(attributes={'key': 'Δ\n\\N'}, value='b'),
                dict(value='c', attributes=[]), dict(attributes=None, value='d')]
        original = copy.deepcopy(rows)
        with Catalog(self.cfg) as cat:
            cat.connection.execute('CREATE TEMP TABLE grouped_write_probe(result_id bigint,value text,attributes jsonb NOT NULL)')
            _insert(cat.connection, 'grouped_write_probe', [_ResultGroup(2**40, rows), _ResultGroup(2**40+1, rows)])
            actual = list(cat.connection.execute('SELECT * FROM grouped_write_probe ORDER BY result_id,value'))
        expected = [dict(result_id=result_id, **row) for result_id in (2**40, 2**40+1) for row in original]
        self.assertEqual(actual, expected)
        self.assertEqual(rows, original)

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
            parse_ownership(self.cfg, workers=4, write_workers=1, progress=progress)
        with Catalog(self.cfg) as cat:
            run = cat.connection.execute("SELECT * FROM runs WHERE command='parse-ownership'").fetchone()
            self.assertEqual((run['status'], run['summary_json']['processed']), ('interrupted', 2))
            self.assertEqual(cat.connection.execute('SELECT count(*) AS n FROM parse_results').fetchone()['n'], 2)
            self.assertEqual(cat.connection.execute("SELECT count(*) AS n FROM parse_attempts WHERE status='interrupted'").fetchone()['n'], 1)
        self.assertEqual(parse_ownership(self.cfg, workers=4)['processed'], 1)
        self.assertEqual(validate_ownership(self.cfg)['status'], 'valid')

    def test_two_publishers_use_independent_connections_and_identical_values(self):
        from insider_research.ownership_store import _publish_batch
        accessions = [f'0000001750-26-{910000+i:06d}' for i in range(6)]
        for a in accessions:
            self.seed(a, source().replace(FORM4.encode(), a.encode()))
        gate = threading.Barrier(2)
        connections, active, peak = set(), 0, 0
        lock = threading.Lock()
        def publish(cat, group):
            nonlocal active, peak
            with lock:
                connections.add(cat.connection.info.backend_pid)
                active += 1; peak = max(peak, active)
            gate.wait(timeout=10)
            try:
                return _publish_batch(cat, group)
            finally:
                with lock:active -= 1
        with patch('insider_research.ownership_store.PARSE_BATCH_SIZE', 1), \
             patch('insider_research.ownership_store._publish_batch', side_effect=publish):
            result = parse_ownership(self.cfg, write_workers=2)
        self.assertEqual((result['processed'],peak,len(connections)), (6,2,2))
        self.assertEqual(validate_ownership(self.cfg)['status'], 'valid')
        self.assertEqual(parse_ownership(self.cfg, write_workers=2)['processed'], 0)

    def test_parallel_interruption_drains_commits_before_reconciling(self):
        from insider_research.ownership_store import _publish_batch
        accessions = [f'0000001750-26-{920000+i:06d}' for i in range(3)]
        for a in accessions:self.seed(a, source().replace(FORM4.encode(), a.encode()))
        second_started, finish_second = threading.Event(), threading.Event()
        def publish(cat, group):
            accession = group[0]['row']['accession']
            if accession == accessions[1]:
                second_started.set()
                self.assertTrue(finish_second.wait(10))
            _publish_batch(cat, group)
            if accession == accessions[0]:self.assertTrue(second_started.wait(10))
        def progress(event):
            if event['event'] == 'progress':
                finish_second.set()
                raise KeyboardInterrupt()
        with patch('insider_research.ownership_store.PARSE_BATCH_SIZE', 1), \
             patch('insider_research.ownership_store._publish_batch', side_effect=publish), \
             self.assertRaises(KeyboardInterrupt):
            parse_ownership(self.cfg, write_workers=2, progress=progress)
        with Catalog(self.cfg) as cat:
            run = cat.connection.execute("SELECT * FROM runs WHERE command='parse-ownership'").fetchone()
            self.assertEqual((run['status'],run['summary_json']['processed']), ('interrupted',2))
            self.assertEqual(cat.connection.execute('SELECT count(*) AS n FROM parse_results').fetchone()['n'], 2)
            self.assertEqual(cat.connection.execute("SELECT count(*) AS n FROM parse_attempts WHERE status='running'").fetchone()['n'], 0)
            self.assertEqual(cat.connection.execute("SELECT count(*) AS n FROM parse_attempts WHERE status='interrupted'").fetchone()['n'], 1)
        self.assertEqual(parse_ownership(self.cfg, write_workers=2)['processed'], 1)
        self.assertEqual(validate_ownership(self.cfg)['status'], 'valid')

    def test_parallel_write_failure_retains_and_accounts_for_sibling_commit(self):
        from insider_research.ownership_store import _publish_batch
        accessions = [f'0000001750-26-{930000+i:06d}' for i in range(3)]
        for a in accessions:self.seed(a, source().replace(FORM4.encode(), a.encode()))
        sibling_committed = threading.Event()
        def publish(cat, group):
            if group[0]['row']['accession'] == accessions[0]:
                with cat.connection.transaction():
                    _publish_batch(cat, group)
                    self.assertTrue(sibling_committed.wait(10))
                    raise ValueError('Simulated transaction failure')
            _publish_batch(cat, group)
            sibling_committed.set()
        with patch('insider_research.ownership_store.PARSE_BATCH_SIZE', 1), \
             patch('insider_research.ownership_store._publish_batch', side_effect=publish), \
             self.assertRaisesRegex(ValueError, 'Simulated transaction failure'):
            parse_ownership(self.cfg, write_workers=2)
        with Catalog(self.cfg) as cat:
            run = cat.connection.execute("SELECT * FROM runs WHERE command='parse-ownership'").fetchone()
            self.assertEqual((run['status'],run['summary_json']['processed']), ('failed',1))
            self.assertEqual(cat.connection.execute('SELECT accession FROM parse_results').fetchone()['accession'], accessions[1])
            self.assertEqual(cat.connection.execute("SELECT count(*) AS n FROM parse_attempts WHERE status='running'").fetchone()['n'], 0)
        self.assertEqual(parse_ownership(self.cfg, write_workers=2)['processed'], 2)
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
        self.assertEqual(result["complete"], 1)
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

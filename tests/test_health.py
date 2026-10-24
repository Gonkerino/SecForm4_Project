"""Integrity lifecycle tests against isolated PostgreSQL schemas and real filings."""
import contextlib
import io
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from insider_research import health
from insider_research.catalog import Catalog
from insider_research.coverage import coverage_counts
from insider_research.cli import main, parser
from insider_research.discovery import discover
from insider_research.http import Response, FetchError
from insider_research.locking import publisher_lock
from insider_research.ownership_store import parse_ownership, inspect_ownership
from insider_research.sources.indexes import source_for
from insider_research.submissions import _publish_zip, download_submissions
from tests.helpers import postgres_config, HEADER, record
from tests.test_ownership import FORM4, FORM5, source


class InterfaceTests(unittest.TestCase):
    def test_help_is_small_and_aliases_still_parse(self):
        help_text = parser().format_help()
        self.assertIn('repair', help_text)
        self.assertNotIn('parse-ownership', help_text)
        self.assertEqual(parser().parse_args(['parse-ownership', '--limit', '1']).limit, 1)
        self.assertEqual(parser().parse_args(['download', '--dry-run']).start, None)

    def test_discovery_preview_needs_no_database(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(['--offline', 'discover', '--start', '2024-01-01', '--end', '2024-01-02', '--dry-run', '--json']), 0)
        self.assertTrue(json.loads(output.getvalue())['dry_run'])


class HealthTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cfg = self.enterContext(postgres_config(Path(self.temp.name) / 'data'))
        Catalog.initialize(self.cfg)

    def seed(self, accession=FORM5, *, parse=True):
        digest, relative, size, source_size = _publish_zip(self.cfg, accession, source(accession))
        with Catalog(self.cfg) as cat:
            cat.connection.execute('INSERT INTO filings VALUES(%s,now(),now()) ON CONFLICT DO NOTHING', (accession,))
            run = cat.start_run('fixture', self.cfg)
            attempt = cat.begin_submission_attempt(run, accession)
            cat.register_submission(accession, relative, digest, size, source_size,
                f'https://www.sec.gov/Archives/edgar/data/2230/{accession}.txt', attempt)
            cat.finish_run(run, 'succeeded', {})
        if parse:
            parse_ownership(self.cfg, accession=accession)
        return self.cfg.data_root / relative

    def index(self):
        body = HEADER + record(accession=FORM5, form='5', filed='2020-01-02')
        client = unittest.mock.Mock()
        client.get.return_value = Response(body, {'status': 200})
        discover(self.cfg, [source_for('quarterly', '2020Q1')], client=client)
        return body, client

    def test_check_and_status_do_not_confuse_download_parse_verify(self):
        self.seed(parse=False)
        before = health.status(self.cfg)
        self.assertEqual(before['downloads']['registered'], 1)
        self.assertEqual(before['parsing']['selected_extractions'], 0)
        self.assertEqual(before['verification']['receipts'], [])
        checked = health.check(self.cfg, stage='download')
        self.assertEqual((checked['status'], checked['checked']), ('valid', 1))
        self.assertEqual(health.status(self.cfg)['verification']['receipts'][0]['verified_objects'], 1)

    def test_quarterly_coverage_separates_failed_requests_from_gaps(self):
        self.index()
        with Catalog(self.cfg) as cat:
            cat.register_sources([source_for('daily', day) for day in ('2020-01-01','2020-01-02','2020-01-03')])
            cat.connection.execute("UPDATE attempts SET finished_at='2020-01-03 12:00:00+00'")
            counts = coverage_counts(cat)
        self.assertEqual(counts['unresolved_sources'], 1)
        self.assertEqual(counts['covered_daily_sources'], 2)
        self.assertEqual(counts['unresolved_source_requests'], 3)
        result = health.issues(self.cfg, stage='discover')
        self.assertEqual(result['total'], 1)
        self.assertEqual(result['issues'][0]['target'], 'daily:2020-01-03')
        included = health.issues(self.cfg, stage='discover', include_covered=True)
        self.assertEqual(included['total'], 3)
        covered = [r for r in included['issues'] if r['category']=='covered_by_quarter']
        self.assertEqual(len(covered), 2)
        self.assertTrue(all(r['severity']=='info' and r['action']=='none' for r in covered))
        self.assertEqual(health.status(self.cfg)['discovery']['unresolved_sources'], 1)
        with Catalog(self.cfg) as cat:
            self.assertEqual(sum(r['latest_attempt_status'] is None for r in cat.coverage()), 3)

    def test_offline_replay_cannot_extend_quarterly_coverage(self):
        self.index()
        with Catalog(self.cfg) as cat:
            cat.register_sources([source_for('daily','2020-01-03')])
            cat.connection.execute("UPDATE attempts SET finished_at='2020-01-03 12:00:00+00'")
        discover(self.cfg, [source_for('quarterly','2020Q1')], offline=True)
        with Catalog(self.cfg) as cat:
            self.assertEqual(coverage_counts(cat)['unresolved_sources'], 1)
            cat.connection.execute("UPDATE attempts SET finished_at='2020-04-01 12:00:00+00' WHERE response_json->>'offline' IS DISTINCT FROM 'true'")
            self.assertEqual(coverage_counts(cat)['covered_daily_sources'], 1)

    def test_failed_daily_history_and_new_york_day_boundary(self):
        self.index()
        with Catalog(self.cfg) as cat:
            run = cat.start_run('fixture', self.cfg)
            day = source_for('daily','2020-01-02')
            _, attempt = cat.begin_attempt(run,day.kind,day.period,day.url)
            cat.fail_attempt(attempt,'SEC returned HTTP 403',http_status=403)
            cat.finish_run(run,'incomplete',{})
            cat.connection.execute("UPDATE attempts SET finished_at='2020-01-03 02:00:00+00' WHERE status='validated'")
            self.assertEqual(coverage_counts(cat)['unresolved_sources'], 1)
            cat.connection.execute("UPDATE attempts SET finished_at='2020-01-03 06:00:00+00' WHERE status='validated'")
            self.assertEqual(coverage_counts(cat)['covered_daily_sources'], 1)
            self.assertEqual(cat.connection.execute('SELECT status FROM attempts WHERE id=%s',(attempt,)).fetchone()['status'],'failed')
        issue = health.issues(self.cfg,stage='discover',include_covered=True)['issues'][0]
        self.assertIn('HTTP 403',issue['message'])

    def test_damaged_or_unselected_quarter_does_not_hide_daily_gap(self):
        self.index()
        with Catalog(self.cfg) as cat:
            cat.register_sources([source_for('daily','2020-01-01')])
            path = self.cfg.data_root / cat.connection.execute('SELECT relative_path FROM artifacts').fetchone()['relative_path']
        path.write_bytes(b'corrupt')
        health.check(self.cfg, stage='discover')
        result = health.issues(self.cfg, stage='discover')
        self.assertEqual({r['category'] for r in result['issues']}, {'coverage','integrity'})
        self.assertEqual(result['covered_daily_sources'], 0)
        with Catalog(self.cfg) as cat:
            cat.connection.execute('UPDATE sources SET current_artifact_id=NULL')
            self.assertEqual(coverage_counts(cat)['covered_daily_sources'], 0)

    def test_download_repair_restores_snapshot_and_preserves_failed_observation(self):
        path = self.seed()
        original = path.read_bytes()
        path.write_bytes(b'corrupt')
        self.assertEqual(health.check(self.cfg)['problems'], 1)
        dry = health.repair(self.cfg, dry_run=True, offline=True)
        self.assertEqual(dry['selected'], 1)
        self.assertEqual(path.read_bytes(), b'corrupt')
        repaired = health.repair(self.cfg, offline=True)
        self.assertEqual(repaired['outcomes'][0]['status'], 'repaired')
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(health.issues(self.cfg)['total'], 0)
        with Catalog(self.cfg) as cat:
            observations = list(cat.connection.execute("SELECT valid FROM health_observations WHERE stage='download' ORDER BY id"))
        self.assertEqual([r['valid'] for r in observations], [False, True])
        self.assertEqual(health.repair(self.cfg, offline=True)['selected'], 0)

    def test_extraction_repair_retains_old_result_and_selects_verified_revision(self):
        self.seed()
        before = inspect_ownership(self.cfg, FORM5)
        with Catalog(self.cfg) as cat:
            cat.connection.execute('UPDATE ownership_transactions SET transaction_shares=999')
        self.assertEqual(health.check(self.cfg, stage='parse')['problems'], 1)
        result = health.repair(self.cfg, stage='parse', offline=True)
        self.assertEqual(result['status'], 'succeeded')
        after = inspect_ownership(self.cfg, FORM5)
        self.assertNotEqual(after['result']['id'], before['result']['id'])
        self.assertEqual(after['result']['supersedes_result_id'], before['result']['id'])
        self.assertEqual(after['transactions'][0]['transaction_shares'], before['transactions'][0]['transaction_shares'])
        old = inspect_ownership(self.cfg, FORM5, result_id=before['result']['id'])
        self.assertEqual(old['transactions'][0]['transaction_shares'], 999)
        self.assertEqual(health.check(self.cfg, stage='parse')['status'], 'valid')
        self.assertEqual(parse_ownership(self.cfg)['processed'], 0)

    def test_failed_replacement_validation_rolls_back_selection(self):
        self.seed()
        original = inspect_ownership(self.cfg, FORM5)['result']['id']
        with Catalog(self.cfg) as cat:
            cat.connection.execute('UPDATE ownership_transactions SET transaction_shares=999')
        health.check(self.cfg, stage='parse')
        with patch('insider_research.health.verify', return_value=['forced validation failure']):
            result = health.repair(self.cfg, stage='parse', offline=True)
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(inspect_ownership(self.cfg, FORM5)['result']['id'], original)
        with Catalog(self.cfg) as cat:
            self.assertEqual(cat.connection.execute('SELECT count(*) AS n FROM parse_results').fetchone()['n'], 1)

    def test_repair_rechecks_and_does_not_touch_already_fixed_file(self):
        path = self.seed()
        original = path.read_bytes()
        path.unlink()
        health.check(self.cfg, stage='download')
        path.write_bytes(original)
        self.assertEqual(health.repair(self.cfg, offline=True)['outcomes'][0]['status'], 'already_valid')

    def test_offline_failure_leaves_finding_open(self):
        path = self.seed(parse=False)
        path.unlink()
        health.check(self.cfg, stage='download')
        result = health.repair(self.cfg, offline=True)
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(health.issues(self.cfg)['total'], 1)

    def test_network_repair_exact_bytes_and_refuses_changed_source(self):
        path = self.seed(parse=False)
        original = path.read_bytes()
        path.write_bytes(b'corrupt')
        health.check(self.cfg, stage='download')
        client = unittest.mock.Mock()
        client.get.return_value = Response(source(FORM5).replace(b'4101.766', b'4101.767'), {})
        self.assertEqual(health.repair(self.cfg, client=client)['status'], 'incomplete')
        self.assertEqual(path.read_bytes(), b'corrupt')
        client.get.return_value = Response(source(FORM5), {})
        self.assertEqual(health.repair(self.cfg, client=client)['status'], 'succeeded')
        self.assertEqual(path.read_bytes(), original)

    def test_filters_and_cli_json_progress_and_show(self):
        self.index()
        self.seed()
        self.seed(FORM4)
        self.assertEqual(health.check(self.cfg, stage='download', accession=FORM5)['checked'], 1)
        self.assertEqual(health.check(self.cfg, stage='parse', start=date(2020,1,1), end=date(2020,12,31))['checked'], 1)
        self.assertEqual(health.check(self.cfg, stage='discover', accession=FORM5)['status'], 'valid')
        for command in (['status'], ['issues'], ['show', FORM5], ['check', '--stage', 'download', '--accession', FORM5],
                        ['download', '--dry-run'], ['parse', '--dry-run'], ['repair', '--dry-run']):
            output = io.StringIO()
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(['--data-root', str(self.cfg.data_root), '--offline', *command, '--json']), 0)
            self.assertIsInstance(json.loads(output.getvalue()), dict)

    def test_index_source_repair_and_relationship_mismatch(self):
        body, client = self.index()
        self.assertEqual(health.check(self.cfg, stage='discover')['status'], 'valid')
        with Catalog(self.cfg) as cat:
            row = cat.connection.execute('SELECT * FROM artifacts').fetchone()
        path = self.cfg.data_root / row['relative_path']
        path.write_bytes(b'bad gzip')
        self.assertEqual(health.check(self.cfg, stage='discover')['problems'], 1)
        self.assertEqual(health.repair(self.cfg, stage='discover', client=client)['status'], 'succeeded')
        with Catalog(self.cfg) as cat:
            cat.connection.execute("UPDATE associations SET company_name='tampered'")
        self.assertEqual(health.check(self.cfg, stage='discover')['problems'], 1)
        self.assertEqual(health.repair(self.cfg, stage='discover', offline=True)['status'], 'succeeded')
        self.assertEqual(health.check(self.cfg, stage='discover')['status'], 'valid')
        self.assertTrue(list((self.cfg.data_root / 'reports' / 'repairs').glob('*-index-before.json.gz')))

    def test_snapshot_repair_uses_registered_copy(self):
        self.seed()
        extraction = inspect_ownership(self.cfg, FORM5)
        snapshot = self.cfg.data_root / extraction['result']['source_snapshot_path']
        original = snapshot.read_bytes()
        snapshot.write_bytes(b'bad snapshot')
        self.assertEqual(health.check(self.cfg, stage='parse')['problems'], 1)
        self.assertEqual(health.repair(self.cfg, offline=True)['status'], 'succeeded')
        self.assertEqual(snapshot.read_bytes(), original)
        self.assertEqual(health.check(self.cfg)['status'], 'valid')
        self.assertTrue(list((self.cfg.data_root / 'raw' / 'repair_quarantine').glob('*.zip')))

    def test_recovery_reconciles_abandoned_health_and_acquisition(self):
        with Catalog(self.cfg) as cat:
            cat.connection.execute("INSERT INTO health_checks(scope,status) VALUES('{}','running')")
            cat.connection.execute("INSERT INTO repair_attempts(stage,target,version,status) VALUES('parse','1','test','running')")
            cat.start_run('test abandoned run', self.cfg)
            with publisher_lock(self.cfg.data_root, cat):
                recovered = cat.recover()
        self.assertEqual(recovered['checks_reconciled'], 1)
        self.assertEqual(recovered['repairs_reconciled'], 1)
        self.assertEqual(recovered['runs_reconciled'], 1)

    def test_missing_selected_result_is_eligible_to_parse_again(self):
        self.seed()
        with Catalog(self.cfg) as cat:
            cat.connection.execute('DELETE FROM ownership_current_results')
        self.assertEqual(parse_ownership(self.cfg)['processed'], 1)
        self.assertEqual(health.status(self.cfg)['parsing']['selected_extractions'], 1)

    def test_writer_lock_blocks_check_repair_and_recovery(self):
        with Catalog(self.cfg) as cat, publisher_lock(self.cfg.data_root, cat):
            for action in (lambda: health.check(self.cfg), lambda: health.repair(self.cfg, dry_run=True)):
                with self.assertRaisesRegex(ValueError, 'writer'):
                    action()

    def test_interrupted_check_retains_partial_receipts(self):
        self.seed()
        def progress(event):
            if event['event'] == 'progress':
                raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            health.check(self.cfg, stage='download', progress=progress)
        with Catalog(self.cfg) as cat:
            self.assertEqual(cat.connection.execute('SELECT status FROM health_checks').fetchone()['status'], 'interrupted')
            self.assertEqual(cat.connection.execute('SELECT count(*) AS n FROM health_receipts').fetchone()['n'], 1)

    def test_blocked_download_stops_undispatched_queue(self):
        body = HEADER + record(accession=FORM4) + record(accession=FORM5)
        client = unittest.mock.Mock()
        client.get.return_value = Response(body, {})
        discover(self.cfg, [source_for('quarterly', '2024Q1')], client=client)
        client.get.side_effect = FetchError('blocked', 403)
        result = download_submissions(self.cfg, workers=1, client=client)
        self.assertEqual(result['submissions_failed'], 1)
        self.assertEqual(result['submissions_not_attempted'], 1)

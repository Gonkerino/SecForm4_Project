"""Research storage retains typed facts and can be recovered from its backup."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from psycopg import sql

from insider_research.catalog import Catalog, connect
from insider_research.cli import main, parser
from insider_research import health, storage
from insider_research.database_control import control, disable_autostart
from insider_research.locking import publisher_lock
from insider_research.ownership_store import parse_ownership, inspect_ownership, validate_ownership
from insider_research.submissions import _publish_zip
from tests.helpers import postgres_config
from tests.test_ownership import FORM3, FORM4, FORM5, source


class LifecycleTests(unittest.TestCase):
    def test_dispatch_does_not_connect_to_database_to_activate(self):
        with patch('insider_research.database_control.control', return_value={'status':'running'}) as start, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['activate']), 0)
            start.assert_called_once()
        self.assertTrue(parser().parse_args(['deactivate', '--force']).force)
        self.assertFalse(parser().parse_args(['admin', 'compact']).apply)

    def test_start_is_idempotent(self):
        with patch('insider_research.database_control.installation', return_value=dict(program_root='C:/postgres',data_directory='D:/data/postgres')), patch('insider_research.database_control.subprocess.run') as run:
            run.return_value.returncode = 0
            self.assertFalse(control(None, 'start')['changed'])
            self.assertEqual(run.call_count, 1)

    def test_shutdown_refuses_clients(self):
        with patch('insider_research.database_control.installation', return_value=dict(program_root='C:/postgres',data_directory='D:/data/postgres')), patch('insider_research.database_control.subprocess.run') as run, patch('insider_research.database_control.psycopg.connect') as conn:
            run.return_value.returncode = 0
            conn.return_value.__enter__.return_value.execute.return_value.fetchone.return_value = (2,)
            from types import SimpleNamespace
            with self.assertRaisesRegex(ValueError, '2 database client'):
                control(SimpleNamespace(database_url='unused'), 'stop')
            self.assertEqual(run.call_count, 1)

    def test_disable_startup_moves_only_exact_shortcut_and_is_repeatable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            startup = root / 'Microsoft/Windows/Start Menu/Programs/Startup'
            startup.mkdir(parents=True)
            shortcut = startup / 'Insider Research PostgreSQL.lnk'
            shortcut.write_bytes(b'test shortcut')
            other = startup / 'Other.lnk'
            other.write_bytes(b'other')
            from types import SimpleNamespace
            with patch('insider_research.database_control.installation'), patch.dict(os.environ, APPDATA=temp):
                cfg = SimpleNamespace(data_root=root / 'data')
                result = disable_autostart(cfg)
                self.assertFalse(shortcut.exists())
                self.assertEqual(Path(result['backup']).read_bytes(), b'test shortcut')
                self.assertTrue(other.exists())
                self.assertFalse(disable_autostart(cfg)['changed'])


class CompactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cfg = self.enterContext(postgres_config(Path(self.temp.name) / 'data'))
        Catalog.initialize(self.cfg)

    def seed(self, accession):
        digest, relative, size, source_size = _publish_zip(self.cfg, accession, source(accession))
        with Catalog(self.cfg) as cat:
            cat.connection.execute('INSERT INTO filings VALUES(%s,now(),now())', (accession,))
            run = cat.start_run('fixture', self.cfg)
            attempt = cat.begin_submission_attempt(run, accession)
            cat.register_submission(accession, relative, digest, size, source_size, 'https://www.sec.gov/fixture', attempt)
            cat.finish_run(run, 'succeeded', {})

    def migrate(self):
        def backup(cfg, directory, constraints):
            directory.mkdir(parents=True)
            return {'test':'backup'}
        with patch('insider_research.storage._backup', side_effect=backup):
            return storage.compact(self.cfg, apply=True)

    def test_existing_and_new_results_validate_and_keep_footnote_links(self):
        self.seed(FORM3)
        parse_ownership(self.cfg, workers=1)
        before = inspect_ownership(self.cfg, FORM3)
        plan = storage.compact(self.cfg)
        self.assertFalse(plan['applied'])
        self.assertGreater(plan['reclaimable_bytes'], 0)
        result = self.migrate()
        self.assertTrue(result['applied'])
        after = inspect_ownership(self.cfg, FORM3)
        for name in ('header','report','owners','transactions','holdings','footnotes','field_notes','signatures','documents','issues','result'):
            self.assertEqual(before[name], after[name], name)
        self.assertEqual(after['storage_profile'], 'research')
        self.assertNotIn('fields', after)
        self.assertFalse(storage.compact(self.cfg, apply=True)['applied'])
        self.seed(FORM4)
        self.seed(FORM5)
        self.assertEqual(parse_ownership(self.cfg, workers=2, write_workers=2)['processed'], 2)
        for accession in (FORM4, FORM5):
            current = inspect_ownership(self.cfg, accession)
            self.assertEqual(current['result']['counts_json']['fields'], 0)
        self.assertEqual(validate_ownership(self.cfg)['status'], 'valid')
        self.assertEqual(health.check(self.cfg, stage='parse')['status'], 'valid')
        # Retained fact corruption must still fail replay.
        with Catalog(self.cfg) as cat:
            cat.connection.execute("UPDATE ownership_transactions SET transaction_code='Z'")
        self.assertEqual(validate_ownership(self.cfg)['status'], 'invalid')

    def test_migration_failure_rolls_back_data_and_foreign_keys(self):
        self.seed(FORM3)
        parse_ownership(self.cfg, workers=1)
        original = storage._validate
        def reject_research(cfg, cat, rows):
            if cat.storage_profile == 'research':
                raise ValueError('simulated validation failure')
            original(cfg, cat, rows)
        with patch('insider_research.storage._validate', side_effect=reject_research), self.assertRaisesRegex(ValueError, 'simulated'):
            self.migrate()
        with Catalog(self.cfg) as cat:
            self.assertEqual(cat.storage_profile, 'full')
            self.assertGreater(cat.connection.execute('SELECT count(*) AS n FROM ownership_fields').fetchone()['n'], 0)
            self.assertEqual(cat.connection.execute("SELECT count(*) AS n FROM pg_constraint WHERE confrelid='ownership_fields'::regclass").fetchone()['n'], 2)

    def test_active_publisher_blocks_migration(self):
        with Catalog(self.cfg) as cat, publisher_lock(self.cfg.data_root, cat, lane='parse'), self.assertRaisesRegex(ValueError, 'writer'):
            self.migrate()

    def test_compact_backup_can_restore_research_database(self):
        if os.name != 'nt':
            self.skipTest('Local PostgreSQL backup binaries are Windows-specific')
        binaries = Path(os.environ['LOCALAPPDATA']) / 'Programs/PostgreSQL/18/bin'
        self.seed(FORM3)
        parse_ownership(self.cfg, workers=1)
        backup_dir = Path(self.temp.name) / 'backup'
        with patch('insider_research.storage.installation', return_value={'program_root':str(binaries.parent)}):
            result = storage.compact(self.cfg, apply=True, backup_dir=backup_dir)
        self.assertTrue(result['applied'])
        with connect(self.cfg) as conn:
            conn.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(self.cfg.database_schema)))
        restored = subprocess.run([str(binaries/'pg_restore.exe'), '--no-password', '--exit-on-error', '--dbname', self.cfg.database_url, '--use-list', str(backup_dir/'restore.list'), str(backup_dir/'research-core.dump')], capture_output=True)
        self.assertEqual(restored.returncode, 0, restored.stderr.decode(errors='replace'))
        with connect(self.cfg) as conn:
            conn.execute((backup_dir/'restore-profile.sql').read_text())
        self.assertEqual(validate_ownership(self.cfg)['status'], 'valid')
        self.assertGreater(len(inspect_ownership(self.cfg, FORM3)['field_notes']), 0)
        manifest = json.loads((backup_dir/'manifest.json').read_text())
        self.assertEqual(manifest['retained_counts'], manifest['post_counts'])

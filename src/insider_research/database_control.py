"""Explicit lifecycle for the per-user Windows PostgreSQL installation."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import psycopg
from psycopg.conninfo import conninfo_to_dict


def installation(cfg):
    if os.name != 'nt':
        raise ValueError('Local database lifecycle commands require Windows')
    manifest = Path(os.environ['LOCALAPPDATA']) / 'Programs/PostgreSQL/18/insider-research-install.json'
    settings = json.loads(manifest.read_text(encoding='utf-8'))
    root = Path(settings['data_directory']).resolve()
    if root != (cfg.data_root / 'postgres').resolve():
        raise ValueError('Installation manifest does not match the configured data root')
    dsn = conninfo_to_dict(cfg.database_url)
    if dsn.get('host') not in ('127.0.0.1', 'localhost') or int(dsn.get('port', 5432)) != int(settings['port']):
        raise ValueError('Lifecycle commands only control the configured local installation')
    return settings


def control(cfg, action, *, force=False):
    settings = installation(cfg)
    command = [str(Path(settings['program_root']) / 'bin/pg_ctl.exe'), '-D', settings['data_directory']]
    hidden = dict(creationflags=subprocess.CREATE_NO_WINDOW)
    status = subprocess.run([*command, 'status'], capture_output=True, **hidden)
    if action == 'status':
        return dict(status='running' if status.returncode == 0 else 'stopped')
    if action == 'start' and status.returncode == 0:
        return dict(status='running', changed=False)
    if action == 'stop' and status.returncode == 3:
        return dict(status='stopped', changed=False)
    if status.returncode not in (0, 3):
        raise ValueError('Cannot read local PostgreSQL status')
    if action == 'stop' and not force:
        with psycopg.connect(cfg.database_url, connect_timeout=5) as connection:
            count = connection.execute("SELECT count(*) FROM pg_stat_activity WHERE backend_type='client backend' AND pid<>pg_backend_pid()").fetchone()[0]
        if count:
            raise ValueError(f'{count} database client(s) connected; close them or use deactivate --force')
    if action == 'start':
        command += ['-l', settings['server_log'], '-w', '-t', '30', 'start']
    else:
        # Smart shutdown will not disconnect a client that connects after the check.
        command += ['-m', 'fast' if force else 'smart', '-w', '-t', '30', 'stop']
    with tempfile.TemporaryFile() as output:
        result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=output,
                                stderr=subprocess.STDOUT, **hidden)
        if result.returncode:
            raise ValueError('PostgreSQL lifecycle operation failed; inspect the local server log')
    return dict(status='running' if action == 'start' else 'stopped', changed=True)


def disable_autostart(cfg):
    installation(cfg)
    shortcut = Path(os.environ['APPDATA']) / 'Microsoft/Windows/Start Menu/Programs/Startup/Insider Research PostgreSQL.lnk'
    backup = cfg.data_root / 'reports' / 'startup-backup' / shortcut.name
    if not shortcut.exists():
        return dict(status='disabled', changed=False, backup=str(backup))
    backup.parent.mkdir(parents=True, exist_ok=True)
    if backup.exists():
        raise ValueError('Startup backup already exists; inspect it before replacing')
    shutil.move(str(shortcut), str(backup))
    return dict(status='disabled', changed=True, backup=str(backup))

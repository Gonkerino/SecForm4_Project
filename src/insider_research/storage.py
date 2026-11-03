"""Explicit, recoverable removal of reproducible XML/header projections."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

from psycopg import sql

from .catalog import Catalog, connect
from .database_control import installation
from .locking import publisher_lock
from .ownership_store import validate_result

OMITTED = ('ownership_fields', 'submission_header_fields')


def _sizes(connection, schema):
    return list(connection.execute('''SELECT c.relname AS table_name,
        pg_total_relation_size(c.oid) AS total_bytes, c.reltuples::bigint AS estimated_rows
        FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
        WHERE n.nspname=%s AND c.relkind='r' ORDER BY total_bytes DESC''', (schema,)))


def _counts(connection, tables):
    return {name: connection.execute(sql.SQL('SELECT count(*) AS n FROM {}').format(
        sql.Identifier(name))).fetchone()['n'] for name in tables}


def _sample(connection):
    # Cover both ends of ingestion plus every recorded parser and result status.
    rows = list(connection.execute('SELECT * FROM parse_results ORDER BY id LIMIT 3'))
    rows += list(connection.execute('SELECT * FROM parse_results ORDER BY id DESC LIMIT 3'))
    for column in ('parser_version', 'status'):
        for value in connection.execute(sql.SQL('SELECT DISTINCT {} AS value FROM parse_results').format(sql.Identifier(column))):
            rows += list(connection.execute(sql.SQL('SELECT * FROM parse_results WHERE {}=%s ORDER BY id LIMIT 1').format(sql.Identifier(column)), (value['value'],)))
    return list({r['id']: r for r in rows}.values())


def _validate(cfg, catalog, rows):
    for row in rows:
        messages = validate_result(cfg, catalog, row)
        if messages:
            raise ValueError(f"Source replay failed for {row['accession']}: {messages}")


def _backup(cfg, directory, removed_constraints):
    settings = installation(cfg)
    binaries = Path(settings['program_root']) / 'bin'
    directory.mkdir(parents=True, exist_ok=False)
    archive = directory / 'research-core.dump'
    command = [str(binaries / 'pg_dump.exe'), '--dbname', cfg.database_url,
               '--no-password', '--format=custom', '--compress=6', '--no-owner',
               '--no-privileges', '--schema', cfg.database_schema, '--file', str(archive)]
    for table in OMITTED:
        command += ['--exclude-table-data', f'{cfg.database_schema}.{table}']
    print(f'Backing up retained data to {archive}', file=sys.stderr, flush=True)
    # Do not expose the DSN or command in subprocess errors.
    result = subprocess.run(command, capture_output=True)
    if result.returncode:
        raise ValueError(f'Compact backup failed; evidence was not removed. Inspect {directory}')
    result = subprocess.run([str(binaries / 'pg_restore.exe'), '--list', str(archive)], capture_output=True)
    if result.returncode or not archive.stat().st_size:
        raise ValueError('Compact backup archive verification failed')
    toc = result.stdout.decode('utf-8')
    lines = []
    removed = set()
    for line in toc.splitlines():
        matching = [name for name in removed_constraints if ' FK CONSTRAINT ' in line and f' {name} ' in line]
        if matching:
            removed.update(matching)
            lines.append('; omitted XML-node dependency: ' + line)
        else:
            lines.append(line)
    if removed != set(removed_constraints):
        raise ValueError('Could not identify every omitted XML foreign key in backup TOC')
    (directory / 'restore.list').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    # The compact backup still has pre-migration metadata; restore this contract
    # after loading it using restore.list. Counts for omitted tables are historical.
    profile_sql = sql.SQL("INSERT INTO {}.metadata(key,value) VALUES ('storage_profile','research') ON CONFLICT(key) DO UPDATE SET value=excluded.value;\n").format(sql.Identifier(cfg.database_schema)).as_string()
    (directory / 'restore-profile.sql').write_text(profile_sql, encoding='utf-8')
    with archive.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(archive=str(archive), bytes=archive.stat().st_size, sha256=digest,
                restore_list=str(directory / 'restore.list'), restore_profile=str(directory / 'restore-profile.sql'))


def compact(cfg, *, apply=False, backup_dir=None):
    with connect(cfg) as connection:
        metadata = {r['key']: r['value'] for r in connection.execute('SELECT * FROM metadata')}
        if metadata.get('data_root') != str(cfg.data_root) or metadata.get('schema_version') not in ('4','5'):
            raise ValueError('Compact migration requires this initialized schema 4 or 5 data root')
        sizes = _sizes(connection, cfg.database_schema)
        plan = dict(storage_profile=metadata.get('storage_profile', 'full'),
                    omitted_tables=list(OMITTED),
                    reclaimable_bytes=sum(r['total_bytes'] for r in sizes if r['table_name'] in OMITTED),
                    tables=sizes, applied=False)
    if not apply:
        return plan
    if plan['storage_profile'] == 'research':
        return dict(plan, note='Already using research storage; nothing removed')
    # Initialization changes only the storage contract version, never data.
    Catalog.initialize(cfg)
    with Catalog(cfg) as catalog, publisher_lock(cfg.data_root, catalog):
        connection = catalog.connection
        connection.execute("SET statement_timeout='120s'")
        connection.execute("SET lock_timeout='5s'")
        constraints = list(connection.execute('''SELECT conname,conrelid::regclass::text AS child
            FROM pg_constraint WHERE contype='f' AND confrelid='ownership_fields'::regclass'''))
        for constraint in constraints:
            if constraint['child'].split('.')[-1] != 'ownership_field_notes':
                raise ValueError('Unexpected XML field dependency; migration requires review')
        tables = sorted(r['table_name'] for r in sizes if r['table_name'] not in OMITTED)
        print('Checking retained row counts and source replay samples', file=sys.stderr, flush=True)
        before = _counts(connection, tables)
        samples = _sample(connection)
        _validate(cfg, catalog, samples)
        directory = Path(backup_dir) if backup_dir else cfg.data_root / 'reports' / ('compact-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
        backup_parent = directory.parent
        while not backup_parent.exists():
            backup_parent = backup_parent.parent
        required_free = max(64 * 1024**2, min(5 * 1024**3, sum(r['total_bytes'] for r in sizes if r['table_name'] not in OMITTED)))
        if shutil.disk_usage(backup_parent).free < required_free:
            raise ValueError('Insufficient free space on backup volume; choose --backup-dir on a larger drive')
        backup = _backup(cfg, directory, [r['conname'] for r in constraints])
        manifest = dict(plan=plan, backup=backup, retained_counts=before,
                        replay_sample_ids=[r['id'] for r in samples], applied=False)
        manifest_path = directory / 'manifest.json'
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print('Backup complete; applying compact storage transaction', file=sys.stderr, flush=True)
        old_profile = catalog.storage_profile
        try:
            with connection.transaction():
                for constraint in constraints:
                    connection.execute(sql.SQL('ALTER TABLE ownership_field_notes DROP CONSTRAINT {}').format(sql.Identifier(constraint['conname'])))
                connection.execute('TRUNCATE ownership_fields, submission_header_fields')
                connection.execute("UPDATE metadata SET value='research' WHERE key='storage_profile'")
                catalog.storage_profile = 'research'
                _validate(cfg, catalog, samples)
                after = _counts(connection, tables)
                if before != after:
                    raise ValueError('Retained row counts changed; compact migration rolled back')
        except BaseException:
            catalog.storage_profile = old_profile
            raise
        manifest.update(applied=True, post_counts=after, post_tables=_sizes(connection, cfg.database_schema))
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        return dict(storage_profile='research', applied=True, backup=backup,
                    retained_counts=after, sample_results_checked=len(samples),
                    database_bytes=connection.execute('SELECT pg_database_size(current_database()) AS n').fetchone()['n'],
                    manifest=str(manifest_path))

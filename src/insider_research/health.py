"""Version-specific verification, bounded issue browsing and audited repairs.

Checks and repairs hold the publisher lock: receipts never certify an artifact
that an application writer replaced while it was being inspected.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import multiprocessing
import re
import signal
import shutil
import tempfile
import uuid
import zipfile
import zlib
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from dataclasses import replace
from pathlib import Path

from psycopg.types.json import Jsonb

from .catalog import Catalog, utc_now
from .coverage import SOURCE_COVERAGE_SQL, coverage_counts
from .discovery import atomic_write
from .http import FetchError, HttpClient
from .locking import publisher_lock
from .ownership import PARSER_VERSION, parse_submission
from .ownership_store import (_safe_path, _selection, publish, read_archive,
                              validate_result, parsing_status, inspect_ownership)
from .sources.indexes import parse_master_index, source_for
from .submissions import _publish_zip, _verify_registered_submission

STAGES = ('discover', 'download', 'parse')
CONTENT_ERRORS = (ValueError, OSError, EOFError, zipfile.BadZipFile, zlib.error, RuntimeError)
CHECK_BATCH_SIZE = 256
CHECK_PAGE_SIZE = 1024
CHECK_THREAD_CHUNK = 16
CHECK_PROCESS_CHUNK = 64
CHECK_GROUP_BYTES = 8 * 1024 * 1024
CHECK_PROCESS_THRESHOLD = 4096
DOWNLOAD_COLUMNS = 's.accession,s.archive_path,s.sha256,s.zip_bytes,s.source_bytes'


def targets(catalog, stage, *, count_only=False, **filters):
    """Keyset pages keep million-filing checks bounded in memory."""
    where, params = _selection(**filters)
    if stage == 'download':
        query = f'SELECT {DOWNLOAD_COLUMNS} FROM submissions s WHERE {where}'
        key, after = 'accession', ''
    elif stage == 'parse':
        where = where.replace('s.accession', 'r.accession')
        query = f'''SELECT r.* FROM parse_results r JOIN ownership_current_results c
                    ON c.result_id=r.id WHERE {where}'''
        key, after = 'id', 0
    else:
        # Discovery filters select source artifacts containing matching observations.
        where = where.replace('s.accession', 'f.accession')
        query = 'SELECT r.* FROM artifacts r WHERE TRUE'
        if any(value is not None for value in filters.values()):
            query += f''' AND EXISTS(SELECT 1 FROM associations f WHERE f.artifact_id=r.id AND {where})'''
        key, after = 'id', 0
    alias = 's' if stage == 'download' else 'r'
    if count_only:
        return catalog.connection.execute('SELECT count(*) AS n FROM (' + query + ') t', params).fetchone()['n']

    def pages():
        nonlocal after
        while True:
            if stage == 'download':
                page_where, page_params = _selection(**filters, accession_after=after)
                rows = list(catalog.connection.execute(
                    f'SELECT {DOWNLOAD_COLUMNS} FROM submissions s WHERE {page_where} ORDER BY s.accession LIMIT %s',
                    [*page_params, CHECK_PAGE_SIZE]))
            else:
                rows = list(catalog.connection.execute(
                    query + f' AND {alias}.{key}>%s ORDER BY {alias}.{key} LIMIT %s', [*params, after, CHECK_PAGE_SIZE]))
            if not rows:
                return
            for row in rows:
                after = row[key]
                yield row
    return pages()


def identity(stage, row):
    return (row['accession'], row['sha256']) if stage == 'download' else (
        str(row['id']), row['source_sha256'] if stage == 'parse' else row['sha256'])


def verify(cfg, catalog, stage, row):
    try:
        if stage == 'download':
            # Verification needs full decompression/CRC, but not a retained
            # in-memory copy of the entire source as extraction does.
            problem = _verify_registered_submission(cfg.data_root, row,
                max_source_bytes=min(cfg.max_submission_bytes, 128 * 1024 * 1024))
            return [problem['message']] if problem else []
        elif stage == 'parse':
            return validate_result(cfg, catalog, row)
        else:
            path = _safe_path(cfg.data_root, row['relative_path'])
            with gzip.open(path, 'rb') as stream:
                body = stream.read(cfg.max_index_bytes + 1)
            if len(body) != row['bytes'] or hashlib.sha256(body).hexdigest() != row['sha256']:
                raise ValueError('Index checksum or size mismatch')
            parsed = parse_master_index(body)
            expected = [(r.row_number, r.accession, r.cik, r.company_name, r.form_type,
                         r.filed_date, r.archive_path) for r in parsed.rows]
            actual = [tuple(r.values()) for r in catalog.connection.execute('''SELECT row_number,
                accession,indexed_cik,company_name,form_type,filed_date,archive_path
                FROM associations WHERE artifact_id=%s ORDER BY row_number''', (row['id'],))]
            # Parser dates are ISO strings; database dates are date objects.
            normalize = lambda rows: [tuple(str(v) for v in r) for r in rows]
            if normalize(expected) != normalize(actual) or len(expected) != row['ownership_rows'] or parsed.total_rows != row['total_rows']:
                raise ValueError('Catalogue index associations differ from preserved source')
    except CONTENT_ERRORS as exc:
        return [str(exc)]
    return []


def record(catalog, check_id, stage, row, messages):
    target, version = identity(stage, row)
    values = (stage, target, version, row.get('accession'), check_id, not messages, Jsonb(messages))
    with catalog.connection.transaction():
        catalog.connection.execute('''INSERT INTO health_observations
            (stage,target,version,accession,check_id,valid,messages) VALUES(%s,%s,%s,%s,%s,%s,%s)''', values)
        catalog.connection.execute('''INSERT INTO health_receipts
            (stage,target,version,accession,check_id,valid,messages) VALUES(%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT(stage,target,version) DO UPDATE SET check_id=excluded.check_id,
            checked_at=now(),valid=excluded.valid,messages=excluded.messages''', values)


def record_batch(catalog, check_id, results):
    """Publish observations and latest receipts atomically in one round trip."""
    values = []
    for stage, row, messages in results:
        target, version = identity(stage, row)
        values.append(dict(stage=stage, target=target, version=version,
                           accession=row.get('accession'), valid=not messages, messages=messages))
    catalog.connection.execute('''WITH observed AS (
        INSERT INTO health_observations(stage,target,version,accession,check_id,valid,messages)
        SELECT stage,target,version,accession,%s,valid,messages
        FROM jsonb_to_recordset(%s) AS t(stage text,target text,version text,
            accession text,valid boolean,messages jsonb)
        RETURNING stage,target,version,accession,check_id,checked_at,valid,messages
    ), receipted AS (
      INSERT INTO health_receipts(stage,target,version,accession,check_id,checked_at,valid,messages)
      SELECT stage,target,version,accession,check_id,checked_at,valid,messages FROM observed
      ON CONFLICT(stage,target,version) DO UPDATE SET check_id=excluded.check_id,
        checked_at=excluded.checked_at,valid=excluded.valid,messages=excluded.messages
      RETURNING valid
    ) UPDATE health_checks SET checked=checked+(SELECT count(*) FROM receipted),
      problems=problems+(SELECT count(*) FROM receipted WHERE NOT valid) WHERE id=%s''',
        (check_id, Jsonb(values), check_id))


def _check_worker_init():
    # The coordinator handles Ctrl+C, records interruption and cancels queued
    # groups. Child processes finish only their bounded in-flight work.
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _verify_download_group(cfg, rows):
    return [verify(cfg, None, 'download', row) for row in rows]


def _download_checks(cfg, rows, workers, executor='thread'):
    """Bound queued file work; only the caller accesses PostgreSQL/progress."""
    if workers == 1:
        for row in rows:
            yield row, verify(cfg, None, 'download', row)
        return
    pool = (ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                initializer=_check_worker_init) if executor == 'process' else
            ThreadPoolExecutor(max_workers=workers, thread_name_prefix='integrity'))
    chunk_size = CHECK_PROCESS_CHUNK if executor == 'process' else CHECK_THREAD_CHUNK
    with pool:
        pending = {}
        rows = iter(rows)
        exhausted = False
        carry = None
        try:
            while pending or not exhausted:
                while not exhausted and len(pending) < workers * 2:
                    group, source_bytes = [], 0
                    while len(group) < chunk_size:
                        row = carry if carry is not None else next(rows, None)
                        carry = None
                        if row is None:
                            exhausted = True
                            break
                        if group and source_bytes + row['source_bytes'] > CHECK_GROUP_BYTES:
                            carry = row
                            break
                        group.append(row)
                        source_bytes += row['source_bytes']
                    if group:
                        pending[pool.submit(_verify_download_group, cfg, group)] = group
                if not pending:
                    break
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    group = pending.pop(future)
                    yield from zip(group, future.result(), strict=True)
        finally:
            for future in pending:
                future.cancel()


def check(cfg, *, stage=None, progress=None, workers=8, executor='thread', **filters):
    if not 1 <= workers <= 20:
        raise ValueError('Workers must be between 1 and 20')
    if executor not in ('auto', 'thread', 'process'):
        raise ValueError('Executor must be auto, thread or process')
    scope = dict(stage=stage or 'all', **{k: str(v) for k, v in filters.items() if v is not None})
    counts = dict(checked=0, problems=0)
    with Catalog(cfg) as cat, publisher_lock(cfg.data_root, cat):
        check_id = cat.connection.execute('''INSERT INTO health_checks(scope,status)
            VALUES(%s,'running') RETURNING id''', (Jsonb(scope),)).fetchone()['id']
        try:
            selected_stages = (stage,) if stage else STAGES
            totals = {}
            if progress or executor == 'auto':
                totals = {selected: targets(cat, selected, count_only=True, **filters) for selected in selected_stages}
            if progress:
                progress({'event': 'start', 'total': sum(totals.values())})
            download_executor = ('process' if totals.get('download', 0) >= CHECK_PROCESS_THRESHOLD else 'thread') if executor == 'auto' else executor
            if workers == 1:
                download_executor = 'thread'
            for selected in selected_stages:
                rows = targets(cat, selected, **filters)
                results = (_download_checks(cfg, rows, workers, executor=download_executor) if selected == 'download' else
                           ((row, verify(cfg, cat, selected, row)) for row in rows))
                batch = []

                def flush():
                    if not batch:
                        return
                    record_batch(cat, check_id, batch)
                    counts['checked'] += len(batch)
                    counts['problems'] += sum(bool(messages) for _, _, messages in batch)
                    batch.clear()
                    if progress:
                        progress(dict(event='progress', current=counts['checked'], problems=counts['problems']))

                try:
                    for row, messages in results:
                        batch.append((selected, row, messages))
                        if len(batch) >= (CHECK_BATCH_SIZE if selected == 'download' else 1):
                            flush()
                    flush()
                finally:
                    # Close/cancel a partially consumed worker generator before
                    # releasing the publisher lock on an error or interruption.
                    results.close()
            status = 'invalid' if counts['problems'] else 'valid'
        except BaseException:
            # Durable counters were committed with the receipts. Do not replace
            # them with possibly stale Python counters if interruption occurred
            # just after PostgreSQL committed a batch.
            cat.connection.execute("UPDATE health_checks SET status='interrupted',finished_at=now() WHERE id=%s",
                                   (check_id,))
            raise
        cat.connection.execute('UPDATE health_checks SET status=%s,finished_at=now(),checked=%s,problems=%s WHERE id=%s',
                               (status, counts['checked'], counts['problems'], check_id))
    result = dict(status=status, check_id=check_id, scope=scope, **counts,
                next='insider issues' if counts['problems'] else 'insider status',
                note='Checks registered indexes, registered downloads and selected extractions. Missing acquisition work is reported by status/issues. No records selected means nothing was verified.')
    if 'download' in selected_stages:
        result.update(download_executor=download_executor, workers=workers)
    return result


# Only findings that still describe a current object are actionable. Retained
# extraction versions remain inspectable without repeatedly repairing history.
ACTIVE = '''((h.stage='download' AND EXISTS(SELECT 1 FROM submissions s
             WHERE s.accession=h.target AND s.sha256=h.version))
          OR (h.stage='parse' AND EXISTS(SELECT 1 FROM ownership_current_results c
             WHERE c.result_id=CASE WHEN h.stage='parse' THEN h.target::bigint END))
          OR (h.stage='discover' AND EXISTS(SELECT 1 FROM artifacts a
             WHERE a.id=CASE WHEN h.stage='discover' THEN h.target::bigint END AND a.sha256=h.version)))'''


def issues(cfg, *, stage=None, accession=None, limit=100, severity=None, include_covered=False):
    if not 1 <= limit <= 10000:
        raise ValueError('Limit must be between 1 and 10000')
    # UNION is evaluated on the server; counts remain complete even when output is limited.
    query = f'''WITH source_coverage AS ({SOURCE_COVERAGE_SQL}), items AS (
      SELECT h.stage,h.accession,h.target,'integrity' AS category,'error' AS severity,
             'repairable' AS action,h.messages::text AS message FROM health_receipts h
      WHERE NOT h.valid AND {ACTIVE}
      UNION ALL SELECT 'parse',r.accession,r.id::text,'source_review',
        CASE WHEN i.severity='review' THEN 'review' ELSE 'warning' END,
        'needs_review',i.code || ': ' || i.message FROM parse_issues i
        JOIN ownership_current_results c ON c.result_id=i.result_id JOIN parse_results r ON r.id=c.result_id
      UNION ALL SELECT 'download',p.accession,p.id::text,'attempt','error','retryable',p.error
        FROM submission_attempts p WHERE p.status IN ('failed','interrupted')
        AND p.id=(SELECT max(id) FROM submission_attempts q WHERE q.accession=p.accession)
        AND (SELECT s.accession FROM submissions s WHERE s.accession=p.accession LIMIT 1) IS NULL
      UNION ALL SELECT 'parse',p.accession,p.id::text,'attempt','error',
        CASE WHEN p.status='unsupported' THEN 'needs_review' ELSE 'retryable' END,p.error
        FROM parse_attempts p JOIN submissions s ON s.accession=p.accession AND s.sha256=p.source_sha256
        WHERE p.parser_version=%s AND p.status IN ('failed','unsupported','interrupted')
        AND p.id=(SELECT max(id) FROM parse_attempts q WHERE q.accession=p.accession
          AND q.source_sha256=p.source_sha256 AND q.parser_version=p.parser_version)
        AND NOT EXISTS(SELECT 1 FROM parse_results r WHERE r.accession=p.accession
          AND r.source_sha256=p.source_sha256 AND r.parser_version=p.parser_version)
      UNION ALL SELECT 'discover',NULL,kind || ':' || period,
        CASE WHEN coverage_status='covered_by_quarter' THEN 'covered_by_quarter' ELSE 'coverage' END,
        CASE WHEN coverage_status='covered_by_quarter' THEN 'info' ELSE 'error' END,
        CASE WHEN coverage_status='covered_by_quarter' THEN 'none' ELSE 'retryable' END,
        CASE WHEN coverage_status='covered_by_quarter' THEN
          'Covered by quarterly index ' || covering_quarter || ' (artifact ' || covering_artifact_id ||
          '); daily request remains ' || coalesce(latest_attempt_status,'not attempted') ||
          '. Original daily-attempt error: ' || coalesce(error,'none (not attempted)')
        ELSE coalesce(error,'Registered source has no completed discovery attempt') END
        FROM source_coverage WHERE coverage_status!='validated'
    ) SELECT * FROM items WHERE (%s::text IS NULL OR stage=%s)
        AND (%s::text IS NULL OR accession=%s) AND (%s::text IS NULL OR severity=%s)
        AND (%s OR category!='covered_by_quarter')'''
    params = [PARSER_VERSION, stage, stage, accession, accession, severity, severity, include_covered]
    with Catalog(cfg) as cat, cat.connection.transaction():
        cat.connection.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        covered = coverage_counts(cat)['covered_daily_sources'] if stage in (None, 'discover') and accession is None else 0
        total = cat.connection.execute('SELECT count(*) AS n FROM (' + query + ') q', params).fetchone()['n']
        rows = list(cat.connection.execute(query + ' ORDER BY stage,accession NULLS FIRST,target,category,message LIMIT %s', [*params, limit]))
    return dict(total=total, shown=len(rows), issues=rows, covered_daily_sources=covered,
                covered_sources_included=include_covered,
                next='Use repair for integrity findings; download/parse --retry-failed for failed attempts; discover for coverage gaps. Source review requires investigation.')


def status(cfg):
    extraction = parsing_status(cfg)
    with Catalog(cfg) as cat:
        acquisition = cat.submission_counts()
        latest = list(cat.connection.execute('SELECT * FROM health_checks ORDER BY id DESC LIMIT 5'))
        verified = list(cat.connection.execute(f'''SELECT h.stage,count(*) AS verified_objects,
            max(h.checked_at) AS last_verified_at FROM health_receipts h
            WHERE h.valid AND {ACTIVE} GROUP BY h.stage ORDER BY h.stage'''))
        running = list(cat.connection.execute("SELECT id,command,started_at FROM runs WHERE status='running' ORDER BY started_at"))
        coverage = coverage_counts(cat)
    pending_download = acquisition['total_indexed'] - acquisition['already_downloaded']
    pending_parse = extraction['downloaded'] - extraction['current_source_and_parser']
    return dict(discovery=dict(indexed=acquisition['total_indexed'], **coverage),
                downloads=dict(registered=extraction['downloaded'], pending=pending_download,
                               failed_latest=acquisition['latest_failed']), parsing=extraction,
                verification=dict(receipts=verified, recent_checks=latest,
                                  note='Verification is as of the recorded time, not a live disk scan.'),
                issues=issues(cfg, limit=1)['total'], running_runs=running,
                next='insider download' if pending_download else 'insider parse' if pending_parse else 'insider check')


def show(cfg, accession, *, result_id=None):
    with Catalog(cfg) as cat:
        result = cat.inspect(accession)
        result['download'] = cat.connection.execute('SELECT * FROM submissions WHERE accession=%s', (accession,)).fetchone()
        result['download_attempts'] = list(cat.connection.execute('SELECT * FROM submission_attempts WHERE accession=%s ORDER BY id', (accession,)))
        result['verification'] = list(cat.connection.execute('SELECT * FROM health_observations WHERE accession=%s ORDER BY id', (accession,)))
        result['repairs'] = list(cat.connection.execute('''SELECT a.* FROM repair_attempts a WHERE
            (a.stage='download' AND a.target=%s) OR (a.stage='parse' AND a.target IN
            (SELECT id::text FROM parse_results WHERE accession=%s)) ORDER BY a.id''', (accession, accession)))
    result['extraction'] = inspect_ownership(cfg, accession, result_id=result_id)
    return result


def _restore(cfg, cat, stage, row, client, offline):
    if stage == 'download':
        snapshot = dict(row, source_sha256=row['sha256'], source_snapshot_path=
                        f"raw/submission_versions/{row['sha256'][:2]}/{row['sha256']}.zip")
        try:
            _, archive = read_archive(cfg, snapshot, snapshot=True)
        except CONTENT_ERRORS:
            if offline:
                raise ValueError('No verified local copy; retry repair with SEC access')
            # Validate URL against the accession before making any request.
            url = row['source_url']
            if not re.fullmatch(r'https://www\.sec\.gov/Archives/edgar/data/[0-9]{1,10}/' + re.escape(row['accession']) + r'\.txt', url):
                raise ValueError('Stored source URL is not a matching SEC submission URL')
            body = client.get(url, max_bytes=cfg.max_submission_bytes).body
            with tempfile.TemporaryDirectory(dir=cfg.data_root / 'staging') as directory:
                staged_cfg = replace(cfg, data_root=Path(directory))
                digest, relative, _, _ = _publish_zip(staged_cfg, row['accession'], body)
                if digest != row['sha256']:
                    raise ValueError('SEC content differs from registered version; requires reviewed source replacement')
                archive = (staged_cfg.data_root / relative).read_bytes()
        _replace_preserving_damage(cfg, _safe_path(cfg.data_root, row['archive_path']), archive)
    else:
        if offline:
            raise ValueError('Index restoration requires SEC access')
        source = cat.connection.execute('SELECT kind,period FROM sources WHERE id=%s', (row['source_id'],)).fetchone()
        body = client.get(source_for(source['kind'], source['period']).url).body
        if hashlib.sha256(body).hexdigest() != row['sha256']:
            raise ValueError('SEC index differs from retained version; use discover to review a new revision')
        parse_master_index(body)
        _replace_preserving_damage(cfg, _safe_path(cfg.data_root, row['relative_path']), gzip.compress(body, mtime=0))


def _replace_preserving_damage(cfg, destination, content):
    """Keep damaged bytes for investigation before atomically restoring a file."""
    if destination.exists():
        quarantine = cfg.data_root / 'raw' / 'repair_quarantine' / (uuid.uuid4().hex + '-' + destination.name)
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(destination, quarantine)
        atomic_write(quarantine.with_suffix(quarantine.suffix + '.json'),
                     json.dumps(dict(original_path=str(destination.relative_to(cfg.data_root)),
                                                  preserved_at=utc_now())).encode())
    atomic_write(destination, content)


def _repair_source_snapshot(cfg, cat, row, client, offline):
    try:
        return read_archive(cfg, row, snapshot=True)
    except CONTENT_ERRORS:
        registered = cat.connection.execute('SELECT * FROM submissions WHERE accession=%s AND sha256=%s',
                                            (row['accession'], row['source_sha256'])).fetchone()
        if registered is None:
            raise ValueError('Preserved source is damaged and no matching registered version exists') from None
        try:
            body, archive = read_archive(cfg, registered)
        except CONTENT_ERRORS:
            _restore(cfg, cat, 'download', registered, client, offline)
            body, archive = read_archive(cfg, registered)
        _replace_preserving_damage(cfg, _safe_path(cfg.data_root, row['source_snapshot_path']), archive)
        return body, archive


def _repair_index(cfg, cat, row, client, offline, audit):
    path = _safe_path(cfg.data_root, row['relative_path'])
    try:
        with gzip.open(path, 'rb') as stream:
            body = stream.read(cfg.max_index_bytes + 1)
        if len(body) > cfg.max_index_bytes or hashlib.sha256(body).hexdigest() != row['sha256']:
            raise ValueError('Index bytes do not match registered hash')
    except CONTENT_ERRORS:
        _restore(cfg, cat, 'discover', row, client, offline)
        with gzip.open(path, 'rb') as stream:
            body = stream.read(cfg.max_index_bytes + 1)
    parsed = parse_master_index(body)
    old_rows = list(cat.connection.execute('SELECT * FROM associations WHERE artifact_id=%s ORDER BY row_number', (row['id'],)))
    # Preserve the pre-repair database projection as well as damaged files.
    evidence = cfg.data_root / 'reports' / 'repairs' / f'{audit}-index-before.json.gz'
    atomic_write(evidence, gzip.compress(json.dumps(dict(artifact=row, associations=old_rows), default=str).encode(), mtime=0))
    with cat.connection.transaction(), cat.connection.cursor() as cursor:
        cursor.executemany('INSERT INTO filings VALUES(%s,now(),now()) ON CONFLICT DO NOTHING',
                           [(r.accession,) for r in parsed.rows])
        cursor.execute('DELETE FROM associations WHERE artifact_id=%s', (row['id'],))
        cursor.executemany('INSERT INTO associations VALUES(%s,%s,%s,%s,%s,%s,%s,%s)',
                           [(row['id'],r.row_number,r.accession,r.cik,r.company_name,r.form_type,r.filed_date,r.archive_path) for r in parsed.rows])
        cursor.execute('UPDATE artifacts SET bytes=%s,total_rows=%s,ownership_rows=%s,min_filed_date=%s,max_filed_date=%s WHERE id=%s',
                       (len(body),parsed.total_rows,len(parsed.rows),parsed.min_date,parsed.max_date,row['id']))
        updated = cursor.execute('SELECT * FROM artifacts WHERE id=%s', (row['id'],)).fetchone()
        messages = verify(cfg, cat, 'discover', updated)
        if messages:
            raise ValueError('; '.join(messages))
    return updated


def repair(cfg, *, stage=None, accession=None, limit=100, dry_run=False, offline=False, client=None, progress=None):
    if not 1 <= limit <= 10000:
        raise ValueError('Limit must be between 1 and 10000')
    outcomes = []
    with Catalog(cfg) as cat, publisher_lock(cfg.data_root, cat):
        selection = f'''FROM health_receipts h
            WHERE NOT h.valid AND {ACTIVE} AND (%s::text IS NULL OR h.stage=%s)
            AND (%s::text IS NULL OR h.accession=%s)'''
        parameters = (stage, stage, accession, accession)
        total = cat.connection.execute('SELECT count(*) AS n ' + selection, parameters).fetchone()['n']
        findings = list(cat.connection.execute('SELECT h.* ' + selection + ' ORDER BY h.stage,h.target LIMIT %s',
                                               (*parameters, limit)))
        if dry_run:
            return dict(dry_run=True, total_findings=total, selected=len(findings), actions=[dict(stage=r['stage'], target=r['target'],
                version=r['version'], action='Recheck, then restore exact source bytes or publish a new extraction revision') for r in findings])
        if progress:
            progress(dict(event='start', total=len(findings)))
        if not findings:
            return dict(status='succeeded', selected=0, remaining=0, outcomes=[], next='insider issues')
        if client is None and not offline:
            client = HttpClient(cfg)
        run = cat.start_run('repair', cfg)
        check_id = cat.connection.execute("INSERT INTO health_checks(scope,status) VALUES(%s,'running') RETURNING id",
            (Jsonb(dict(command='repair', stage=stage, accession=accession, limit=limit)),)).fetchone()['id']
        try:
            for finding in findings:
                selected = finding['stage']
                table, key = ('submissions', 'accession') if selected == 'download' else ('parse_results', 'id') if selected == 'parse' else ('artifacts', 'id')
                row = cat.connection.execute(f'SELECT * FROM {table} WHERE {key}=%s', (finding['target'],)).fetchone()
                audit = cat.connection.execute('''INSERT INTO repair_attempts(stage,target,version,status)
                    VALUES(%s,%s,%s,'running') RETURNING id''', (selected, finding['target'], finding['version'])).fetchone()['id']
                replacement_id = None
                outcome, message = 'already_valid', 'Verified without changes'
                try:
                    messages = verify(cfg, cat, selected, row)
                    if messages:
                        if selected == 'parse':
                            if row['parser_version'] != PARSER_VERSION:
                                raise ValueError('Use parse to upgrade this extraction to the current parser')
                            body, archive = _repair_source_snapshot(cfg, cat, row, client, offline)
                            if hashlib.sha256(body).hexdigest() != row['source_text_sha256']:
                                raise ValueError('Preserved text checksum mismatch')
                            extracted = parse_submission(body, row['accession'])
                            observations = cat.connection.execute('SELECT DISTINCT form_type,filed_date FROM current_associations WHERE accession=%s', (row['accession'],))
                            if any(o['form_type'] != extracted.report['form_type'] or o['filed_date'] != extracted.header['filed_date'] for o in observations):
                                extracted.issue('index_document_mismatch', 'SEC-HEADER', 'Current index form/date differs from submission')
                            # Entire replacement, validation, and selection are atomic.
                            with cat.connection.transaction():
                                attempt = cat.connection.execute('''INSERT INTO parse_attempts
                                    (run_id,accession,source_sha256,parser_version,started_at,status,source_snapshot_path,source_text_sha256)
                                    VALUES(%s,%s,%s,%s,%s,'running',%s,%s) RETURNING id''',
                                    (run,row['accession'],row['source_sha256'],PARSER_VERSION,utc_now(),row['source_snapshot_path'],row['source_text_sha256'])).fetchone()['id']
                                replacement_id = publish(cat, dict(row, sha256=row['source_sha256']), attempt, extracted,
                                                         row['source_snapshot_path'], row['source_text_sha256'])
                                cat.connection.execute('UPDATE parse_results SET supersedes_result_id=%s WHERE id=%s', (row['id'],replacement_id))
                                replacement = cat.connection.execute('SELECT * FROM parse_results WHERE id=%s', (replacement_id,)).fetchone()
                                errors = verify(cfg, cat, selected, replacement)
                                if errors:
                                    raise ValueError('; '.join(errors))
                                record(cat, check_id, selected, replacement, [])
                        elif selected == 'discover':
                            row = _repair_index(cfg, cat, row, client, offline, audit)
                            record(cat, check_id, selected, row, [])
                        else:
                            _restore(cfg, cat, selected, row, client, offline)
                            messages = verify(cfg, cat, selected, row)
                            if messages:
                                raise ValueError('; '.join(messages))
                            record(cat, check_id, selected, row, [])
                        outcome, message = 'repaired', 'Replacement verified; previous extraction history retained'
                    else:
                        record(cat, check_id, selected, row, [])
                except (*CONTENT_ERRORS, FetchError) as exc:
                    replacement_id = None
                    outcome, message = 'failed', str(exc)
                    record(cat, check_id, selected, row, verify(cfg, cat, selected, row))
                    blocked = getattr(exc, 'status', None) in (403,429)
                else:
                    blocked = False
                cat.connection.execute('''UPDATE repair_attempts SET status=%s,message=%s,
                    replacement_result_id=%s,finished_at=now() WHERE id=%s''', (outcome,message,replacement_id,audit))
                outcomes.append(dict(stage=selected,target=finding['target'],status=outcome,message=message,replacement_result_id=replacement_id))
                if progress:
                    progress(dict(event='progress', current=len(outcomes), accession=finding.get('accession') or finding['target'],
                                  status=outcome, message=message, problems=sum(r['status']=='failed' for r in outcomes)))
                if blocked:
                    break
            failed = sum(r['status']=='failed' for r in outcomes)
            result = dict(status='incomplete' if failed else 'succeeded', selected=len(findings),
                          processed=len(outcomes), not_attempted=len(findings)-len(outcomes),
                          remaining=total-len(outcomes)+failed, outcomes=outcomes)
            cat.connection.execute('UPDATE health_checks SET status=%s,finished_at=now(),checked=%s,problems=%s WHERE id=%s',
                ('invalid' if failed else 'valid',len(outcomes),failed,check_id))
            cat.finish_run(run,result['status'],result)
            return result
        except BaseException:
            cat.connection.execute("UPDATE repair_attempts SET status='interrupted',finished_at=now() WHERE status='running'")
            cat.connection.execute("UPDATE health_checks SET status='interrupted',finished_at=now() WHERE id=%s", (check_id,))
            cat.finish_run(run,'interrupted',{},'Repair interrupted; rerun check to reconcile')
            raise

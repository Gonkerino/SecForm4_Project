"""Versioned local extraction, atomic publication, inspection and validation."""
from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
import uuid
import zipfile
import zlib

from psycopg import sql
from psycopg.types.json import Jsonb

from .catalog import Catalog, utc_now
from .locking import publisher_lock
from .ownership import PARSER_VERSION, UnsupportedSubmission, parse_submission

TABLES = {
    "header_fields": "submission_header_fields",
    "documents": "submission_documents", "owners": "ownership_reporting_owners",
    "transactions": "ownership_transactions", "holdings": "ownership_holdings",
    "footnotes": "ownership_footnotes", "fields": "ownership_fields",
    "field_notes": "ownership_field_notes", "signatures": "ownership_signatures",
    "issues": "parse_issues",
}
JSON_COLUMNS = {"attributes", "counts_json"}


def _safe_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Archive path escapes data root")
    return path


def read_archive(cfg, row, *, snapshot=False):
    """Bounded verification and source-version preservation before extraction."""
    relative = row["source_snapshot_path"] if snapshot else row["archive_path"]
    path = _safe_path(cfg.data_root, relative)
    maximum = min(cfg.max_submission_bytes, 128 * 1024 * 1024)
    with path.open("rb") as stream:
        archive_bytes = stream.read(maximum * 2 + 1024 * 1024 + 1)
    if len(archive_bytes) > maximum * 2 + 1024 * 1024:
        raise ValueError("ZIP exceeds bounded archive size")
    expected_hash = row["source_sha256"] if snapshot else row["sha256"]
    if hashlib.sha256(archive_bytes).hexdigest() != expected_hash:
        raise ValueError("ZIP checksum differs from catalogue")
    if not snapshot and len(archive_bytes) != row["zip_bytes"]:
        raise ValueError("ZIP length differs from catalogue")
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        if archive.namelist() != [row["accession"] + ".txt"]:
            raise ValueError("ZIP must contain exactly the expected submission text")
        info = archive.infolist()[0]
        if info.file_size > maximum:
            raise ValueError("Submission exceeds configured extraction size")
        with archive.open(info) as source:
            body = source.read(maximum + 1)
        if len(body) > maximum:
            raise ValueError("Decompressed submission exceeds configured extraction size")
        if not body:
            raise ValueError("Empty submission")
    if not snapshot and len(body) != row["source_bytes"]:
        raise ValueError("Submission length differs from catalogue")
    return body, archive_bytes


def _snapshot(cfg, digest, archive_bytes):
    relative = f"raw/submission_versions/{digest[:2]}/{digest}.zip"
    path = _safe_path(cfg.data_root, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                raise ValueError("Existing immutable source snapshot is damaged")
        return relative
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(archive_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return relative


def _insert(connection, table, rows):
    if not rows:
        return
    keys = list(rows[0])
    query = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
        sql.Identifier(table), sql.SQL(",").join(map(sql.Identifier, keys)),
        sql.SQL(",").join(sql.Placeholder() for _ in keys))
    with connection.cursor() as cursor:
        cursor.executemany(query, [[Jsonb(row[k]) if k in JSON_COLUMNS else row[k] for k in keys] for row in rows])


def _counts(extracted):
    return {name: len(getattr(extracted, name)) for name in TABLES}


def publish(catalog, row, attempt_id, extracted, snapshot_path, source_hash):
    """All result tables and selection publish together; older results stay intact."""
    connection = catalog.connection
    with connection.transaction():
        result_id = connection.execute("""INSERT INTO parse_results
            (accession,attempt_id,source_sha256,source_text_sha256,source_snapshot_path,
             parser_version,parsed_at,status,counts_json)
            VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
            (row["accession"], attempt_id, row["sha256"], source_hash, snapshot_path,
             PARSER_VERSION, utc_now(), extracted.status, Jsonb(_counts(extracted)))).fetchone()["id"]
        _insert(connection, "submission_headers", [dict(result_id=result_id, **extracted.header)])
        _insert(connection, "submission_documents", [dict(result_id=result_id, **r) for r in extracted.documents])
        _insert(connection, "ownership_reports", [dict(result_id=result_id, **extracted.report)])
        for name, table in TABLES.items():
            if name == "documents":
                continue
            rows = [dict(result_id=result_id, **r) for r in getattr(extracted, name)]
            if name == "issues":
                rows = [dict(issue_number=i, **r) for i, r in enumerate(rows, 1)]
            _insert(connection, table, rows)
        connection.execute("""INSERT INTO ownership_current_results VALUES(%s,%s)
            ON CONFLICT(accession) DO UPDATE SET result_id=excluded.result_id""", (row["accession"], result_id))
        connection.execute("UPDATE parse_attempts SET status=%s,finished_at=%s WHERE id=%s",
                           (extracted.status, utc_now(), attempt_id))
    return result_id


def _selection(start=None, end=None, form=None, accession=None):
    if start and end and start > end:
        raise ValueError("Start date must be on or before end date")
    parts, params = [], []
    if accession:
        parts.append("s.accession=%s")
        params.append(accession)
    association_parts = ["a.accession=s.accession"]
    for expression, value in (("a.filed_date >= %s", start), ("a.filed_date <= %s", end), ("a.form_type=%s", form)):
        if value is not None:
            association_parts.append(expression)
            params.append(value)
    if start is not None or end is not None or form is not None:
        parts.append("EXISTS(SELECT 1 FROM associations a WHERE " + " AND ".join(association_parts) + ")")
    return " AND ".join(parts) if parts else "TRUE", params


def _eligibility(retry_failed):
    where = """NOT EXISTS(SELECT 1 FROM parse_results r JOIN ownership_current_results c ON c.result_id=r.id WHERE r.accession=s.accession
        AND r.source_sha256=s.sha256 AND r.parser_version=%s)"""
    if not retry_failed:
        where += """ AND NOT EXISTS(SELECT 1 FROM parse_attempts p WHERE p.id=(
            SELECT max(p2.id) FROM parse_attempts p2 WHERE p2.accession=s.accession
            AND p2.source_sha256=s.sha256 AND p2.parser_version=%s)
            AND p.status IN ('failed','unsupported'))"""
    return where, [PARSER_VERSION] * (1 if retry_failed else 2)


def plan_parse(cfg, *, retry_failed=False, **filters):
    where, params = _selection(**filters)
    eligible, versions = _eligibility(retry_failed)
    with Catalog(cfg) as catalog:
        row = catalog.connection.execute(f"""SELECT count(*) AS downloaded_selected,
            count(*) FILTER(WHERE {eligible}) AS eligible,
            count(*) FILTER(WHERE EXISTS(SELECT 1 FROM parse_results r JOIN ownership_current_results c ON c.result_id=r.id WHERE r.accession=s.accession
                AND r.source_sha256=s.sha256 AND r.parser_version=%s)) AS already_extracted
            FROM submissions s WHERE {where}""", [*versions, PARSER_VERSION, *params]).fetchone()
    return dict(row, parser_version=PARSER_VERSION, previous_failures_excluded=row["downloaded_selected"] - row["eligible"] - row["already_extracted"],
                note="Local archives only. Date/form filters use historical index observations. No default date cutoff.")


def parse_ownership(cfg, *, limit=None, retry_failed=False, progress=None, **filters):
    if limit is not None and not 1 <= limit <= 10000:
        raise ValueError("Limit must be between 1 and 10000")
    where, params = _selection(**filters)
    eligible, versions = _eligibility(retry_failed)
    summary = dict(parser_version=PARSER_VERSION, processed=0, complete=0, complete_with_warnings=0,
                   needs_review=0, unsupported=0, failed=0, errors=[])
    with Catalog(cfg) as catalog, publisher_lock(cfg.data_root, catalog):
        # Lock excludes acquisition/publication changes while the keyset pages run.
        eligible_count = catalog.connection.execute(f"SELECT count(*) AS n FROM submissions s WHERE {where} AND {eligible}",
                                                  [*params, *versions]).fetchone()["n"]
        total = min(eligible_count, limit) if limit is not None else eligible_count
        summary.update(eligible_before=eligible_count, queued=total)
        if progress:
            progress({"event": "start", "total": total})
        run_id = catalog.start_run("parse-ownership", cfg)
        summary["run_id"] = run_id
        after = ""
        try:
            while summary["processed"] < total:
                rows = list(catalog.connection.execute(f"""SELECT s.* FROM submissions s
                    WHERE {where} AND {eligible} AND s.accession>%s ORDER BY s.accession LIMIT %s""",
                    [*params, *versions, after, min(100, total - summary["processed"])]))
                if not rows:
                    break
                for row in rows:
                    after = row["accession"]
                    attempt = catalog.connection.execute("""INSERT INTO parse_attempts
                        (run_id,accession,source_sha256,parser_version,started_at,status)
                        VALUES(%s,%s,%s,%s,%s,'running') RETURNING id""",
                        (run_id, after, row["sha256"], PARSER_VERSION, utc_now())).fetchone()["id"]
                    message = None
                    try:
                        body, archive_bytes = read_archive(cfg, row)
                        snapshot_path = _snapshot(cfg, row["sha256"], archive_bytes)
                        source_hash = hashlib.sha256(body).hexdigest()
                        catalog.connection.execute("UPDATE parse_attempts SET source_snapshot_path=%s,source_text_sha256=%s WHERE id=%s",
                                                   (snapshot_path, source_hash, attempt))
                        extracted = parse_submission(body, after)
                        observations = list(catalog.connection.execute("SELECT DISTINCT form_type,filed_date FROM current_associations WHERE accession=%s", (after,)))
                        if any(o["form_type"] != extracted.report["form_type"] or o["filed_date"] != extracted.header["filed_date"] for o in observations):
                            extracted.issue("index_document_mismatch", "SEC-HEADER", "Current index form/date differs from submission")
                        publish(catalog, row, attempt, extracted, snapshot_path, source_hash)
                        status = extracted.status
                    except (ValueError, OSError, zipfile.BadZipFile, EOFError, RuntimeError, zlib.error) as exc:
                        status = "unsupported" if isinstance(exc, UnsupportedSubmission) else "failed"
                        message = str(exc)[:2000]
                        catalog.connection.execute("UPDATE parse_attempts SET status=%s,finished_at=%s,error=%s WHERE id=%s",
                                                   (status, utc_now(), message, attempt))
                        # Detailed failures are retained in the DB; keep CLI output bounded.
                        if len(summary["errors"]) < 100:
                            summary["errors"].append(dict(accession=after, status=status, message=message))
                    summary[status] += 1
                    summary["processed"] += 1
                    if progress:
                        progress(dict(event="progress", current=summary["processed"], accession=after,
                                      status="failed" if status in ("unsupported", "failed") else status,
                                      message=message, problems=summary["needs_review"] + summary["unsupported"] + summary["failed"]))
        except BaseException as exc:
            catalog.connection.execute("UPDATE parse_attempts SET status='interrupted',finished_at=%s,error=%s WHERE run_id=%s AND status='running'",
                                       (utc_now(), "Parser run interrupted", run_id))
            catalog.finish_run(run_id, "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", summary,
                               "Parser run interrupted" if isinstance(exc, KeyboardInterrupt) else "Parser run failed; see attempt records")
            raise
        summary["unattempted_eligible"] = eligible_count - summary["processed"]
        summary["status"] = "incomplete" if summary["failed"] or summary["unsupported"] or summary["needs_review"] else "succeeded"
        catalog.finish_run(run_id, summary["status"], summary)
    return summary


def parsing_status(cfg):
    with Catalog(cfg) as catalog, catalog.connection.transaction():
        catalog.connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
        counts = catalog.connection.execute("""SELECT count(*) AS downloaded,
            count(*) FILTER(WHERE r.id IS NOT NULL) AS selected_extractions,
            count(*) FILTER(WHERE r.source_sha256=s.sha256 AND r.parser_version=%s) AS current_source_and_parser,
            count(*) FILTER(WHERE r.status='complete') AS complete,
            count(*) FILTER(WHERE r.status='complete_with_warnings') AS complete_with_warnings,
            count(*) FILTER(WHERE r.status='needs_review') AS needs_review
            FROM submissions s LEFT JOIN ownership_current_results c USING(accession)
            LEFT JOIN parse_results r ON r.id=c.result_id""", (PARSER_VERSION,)).fetchone()
        attempts = list(catalog.connection.execute("""SELECT p.status,count(*) AS count FROM submissions s
            JOIN LATERAL(SELECT status FROM parse_attempts p WHERE p.accession=s.accession
                AND p.source_sha256=s.sha256 AND p.parser_version=%s ORDER BY id DESC LIMIT 1) p ON TRUE
            GROUP BY p.status ORDER BY p.status""", (PARSER_VERSION,)))
        issues = list(catalog.connection.execute("""SELECT i.severity,i.code,count(*) AS count FROM parse_issues i
            JOIN ownership_current_results c ON c.result_id=i.result_id GROUP BY i.severity,i.code ORDER BY count(*) DESC"""))
    return dict(parser_version=PARSER_VERSION, **counts, latest_attempts=attempts, issues=issues,
                note="Selected results may include review cases or older parser/source versions. No features or analysis.")


def inspect_ownership(cfg, accession, *, result_id=None):
    with Catalog(cfg) as catalog, catalog.connection.transaction():
        catalog.connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
        if result_id is None:
            selected = catalog.connection.execute("SELECT result_id FROM ownership_current_results WHERE accession=%s", (accession,)).fetchone()
            result_id = selected["result_id"] if selected else None
        result = catalog.connection.execute("SELECT * FROM parse_results WHERE accession=%s AND id=%s", (accession, result_id)).fetchone()
        attempts = list(catalog.connection.execute("SELECT * FROM parse_attempts WHERE accession=%s ORDER BY id", (accession,)))
        if result is None:
            return dict(accession=accession, result=None, attempts=attempts)
        output = dict(result=result, attempts=attempts)
        for name, table in {"header": "submission_headers", "report": "ownership_reports", **TABLES}.items():
            order = "source_path" if name in ("fields", "field_notes") else "line_number" if name == "header_fields" else "document_number" if name == "documents" else "issue_number" if name == "issues" else "row_number" if name not in ("header", "report") else "result_id"
            output[name] = list(catalog.connection.execute(sql.SQL("SELECT * FROM {} WHERE result_id=%s ORDER BY {}").format(sql.Identifier(table), sql.Identifier(order)), (result_id,)))
        return output


def validate_result(cfg, catalog, row):
    messages = []
    try:
        body, _ = read_archive(cfg, row, snapshot=True)
        if hashlib.sha256(body).hexdigest() != row["source_text_sha256"]:
            raise ValueError("Submission text checksum mismatch")
        if row["parser_version"] != PARSER_VERSION:
            raise ValueError("Result uses a different parser version; source verified, replay unavailable")
        extracted = parse_submission(body, row["accession"])
        if set(row["counts_json"]) != set(TABLES):
            messages.append("Result manifest does not account for every extraction table")
        for name, table in {"header": "submission_headers", "report": "ownership_reports", **TABLES}.items():
            expected = [extracted.header] if name == "header" else [extracted.report] if name == "report" else getattr(extracted, name)
            actual = [{k: v for k, v in r.items() if k != "result_id"} for r in catalog.connection.execute(
                sql.SQL("SELECT * FROM {} WHERE result_id=%s").format(sql.Identifier(table)), (row["id"],))]
            def canonical(records):
                return sorted(records, key=lambda r: (r.get("source_path", ""),
                              r.get("row_number", 0), r.get("document_number", 0),
                              r.get("line_number", 0),
                              r.get("code", ""), r.get("severity", ""), r.get("message", "")))
            if name in row["counts_json"] and len(actual) != row["counts_json"][name]:
                messages.append(f"Stored {name} count differs from manifest")
            if name == "issues":
                # Index warnings depend on the observation at parse time.
                issue_status = "needs_review" if any(i["severity"] == "review" for i in actual) else "complete_with_warnings" if actual else "complete"
                if row["status"] != issue_status:
                    messages.append("Result status differs from recorded issues")
                actual = [{k: v for k, v in i.items() if k != "issue_number"} for i in actual if i["code"] != "index_document_mismatch"]
            if canonical(expected) != canonical(actual):
                messages.append(f"Stored {name} differs from source replay")
        attempt = catalog.connection.execute("SELECT status,source_snapshot_path,source_text_sha256 FROM parse_attempts WHERE id=%s", (row["attempt_id"],)).fetchone()
        if (attempt["status"] != row["status"] or attempt["source_snapshot_path"] != row["source_snapshot_path"]
                or attempt["source_text_sha256"] != row["source_text_sha256"]):
            messages.append("Attempt receipt differs from published result")
    except (ValueError, OSError, zipfile.BadZipFile, EOFError, RuntimeError, zlib.error) as exc:
        messages.append(str(exc))
    return messages


def validate_ownership(cfg, *, progress=None, on_result=None, **filters):
    """Read-only source replay and full typed/evidence comparison, all retained versions."""
    problems, checked, problem_count = [], 0, 0
    with Catalog(cfg) as catalog, catalog.connection.transaction():
        catalog.connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
        where, params = _selection(**filters)
        where = where.replace("s.accession", "r.accession")
        total = catalog.connection.execute(f"SELECT count(*) AS n FROM parse_results r WHERE {where}", params).fetchone()["n"]
        if progress:
            progress({"event": "start", "total": total})
        after = 0
        while True:
            rows = list(catalog.connection.execute(f"SELECT r.* FROM parse_results r WHERE {where} AND id>%s ORDER BY id LIMIT 100", [*params, after]))
            if not rows:
                break
            for row in rows:
                after = row["id"]
                messages = validate_result(cfg, catalog, row)
                if on_result:
                    on_result(row, messages)
                checked += 1
                if messages:
                    problem_count += 1
                    if len(problems) < 100:
                        problems.append(dict(result_id=row["id"], accession=row["accession"], messages=messages))
                if progress:
                    progress(dict(event="progress", current=checked, problems=problem_count))
    return dict(status="valid" if problem_count == 0 else "invalid", results_checked=checked,
                problem_count=problem_count, problems=problems,
                note="Checks extraction fidelity, not source truth or full SEC coverage. Use parsing-status for unresolved cases.")

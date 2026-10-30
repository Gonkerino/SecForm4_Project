"""Retryable SEC full-submission downloads published as verified ZIP files."""

from __future__ import annotations

import hashlib
import io
import os
import re
import uuid
import zipfile
import zlib
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from .catalog import Catalog
from .config import Config, FORMS, RESEARCH_START
from .http import FetchError, HttpClient
from .locking import publisher_lock
from .sources.indexes import ACCESSION


def _verify_zip(path: Path, accession: str, source_sha256: str | None = None,
                *, max_source_bytes: int | None = None) -> tuple[str, int, int]:
    # Small archives are read once from disk. Large ones retain bounded-memory
    # streaming; hashing and ZIP checks use the same open file descriptor.
    with path.open("rb") as stream:
        zip_size = os.fstat(stream.fileno()).st_size
        if max_source_bytes is not None and zip_size > max_source_bytes * 2 + 1024 * 1024:
            raise ValueError("ZIP exceeds bounded archive size")
        if zip_size <= 8 * 1024 * 1024:
            body = stream.read(zip_size + 1)
            if len(body) != zip_size:
                raise ValueError("ZIP changed size while being read")
            digest = hashlib.sha256(body).hexdigest()
            return _verify_zip_contents(io.BytesIO(body), accession, source_sha256, digest, zip_size,
                                        max_source_bytes=max_source_bytes)
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        stream.seek(0)
        return _verify_zip_contents(stream, accession, source_sha256, digest, zip_size,
                                    max_source_bytes=max_source_bytes)


def _verify_zip_contents(stream, accession, source_sha256, digest, zip_size, *, max_source_bytes=None):
    with zipfile.ZipFile(stream) as archive:
        names = archive.namelist()
        expected = f"{accession}.txt"
        if names != [expected]:
            raise ValueError("Submission ZIP has an unexpected file list")
        info = archive.infolist()[0]
        if max_source_bytes is not None and info.file_size > max_source_bytes:
            raise ValueError("Submission exceeds configured extraction size")
        source_hash = hashlib.sha256() if source_sha256 is not None else None
        source_size = 0
        with archive.open(expected) as source:
            while chunk := source.read(min(1024 * 1024, info.file_size + 1)):
                if source_hash is not None:
                    source_hash.update(chunk)
                source_size += len(chunk)
                if max_source_bytes is not None and source_size > max_source_bytes:
                    raise ValueError("Decompressed submission exceeds configured extraction size")
        if source_size == 0:
            raise ValueError("Submission file is empty")
        if source_hash is not None and source_hash.hexdigest() != source_sha256:
            raise ValueError("Submission ZIP source checksum mismatch")
    return digest, zip_size, source_size


def _publish_zip(cfg: Config, accession: str, body: bytes) -> tuple[str, str, int, int]:
    if not body or b"\x00" in body:
        raise ValueError("SEC submission is empty or contains binary null bytes")
    # Full submission files are SGML documents. Requiring the identifying header
    # rejects HTML block pages and truncated/error responses before publication.
    prefix = body[:65536].upper()
    if b"<SEC-DOCUMENT>" not in prefix or b"<SEC-HEADER>" not in prefix:
        raise ValueError("Response is not a complete SEC full-submission document")
    if accession.encode("ascii") not in prefix:
        raise ValueError("SEC submission header does not match the requested accession")
    source_hash = hashlib.sha256(body).hexdigest()
    relative = Path("raw") / "submissions" / f"accession={accession[:10]}" / f"{accession}.zip"
    destination = cfg.data_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".{uuid.uuid4().hex}.tmp")
    try:
        with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            info = zipfile.ZipInfo(f"{accession}.txt", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, body)
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        _verify_zip(temporary, accession, source_hash)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    zip_hash, zip_size, source_size = _verify_zip(destination, accession, source_hash)
    return zip_hash, relative.as_posix(), zip_size, source_size


def _verify_registered_submission(root, row, *, max_source_bytes=None):
    try:
        path = (root / row["archive_path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Submission archive path escapes data root")
        digest, zip_size, source_size = _verify_zip(path, row["accession"], max_source_bytes=max_source_bytes)
        if (digest != row["sha256"] or zip_size != row["zip_bytes"]
                or source_size != row["source_bytes"]):
            raise ValueError("ZIP checksum or recorded length mismatch")
    except (OSError, ValueError, zipfile.BadZipFile, EOFError, zlib.error) as exc:
        return {"accession": row["accession"], "message": str(exc)}
    return None


def verify_submissions(cfg: Config, *, progress=None, workers=1) -> dict:
    if not 1 <= workers <= 20:
        raise ValueError("Workers must be between 1 and 20")
    problems, checked = [], 0
    if progress:
        progress({"event": "loading"})
    with Catalog(cfg) as catalog, catalog.connection.transaction():
        catalog.connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        total = catalog.connection.execute("SELECT count(*) AS n FROM submissions").fetchone()["n"]
        if progress:
            progress({"event": "start", "total": total})

        def completed(row, problem):
            nonlocal checked
            checked += 1
            if problem is not None:
                problems.append(problem)
            if progress:
                progress({"event": "progress", "current": checked, "total": total,
                          "accession": row["accession"], "problems": len(problems)})

        # Stream one stable catalogue snapshot instead of materializing millions
        # of rows. Only the coordinator touches PostgreSQL and progress output.
        with catalog.connection.cursor(name="submission_validation") as cursor:
            cursor.itersize = 256
            cursor.execute("SELECT accession, archive_path, sha256, zip_bytes, source_bytes "
                           "FROM submissions ORDER BY accession")
            if workers == 1:
                for row in cursor:
                    completed(row, _verify_registered_submission(cfg.data_root, row))
            else:
                with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="zip-verify") as pool:
                    pending = {}
                    rows = iter(cursor)
                    exhausted = False
                    try:
                        while pending or not exhausted:
                            while not exhausted and len(pending) < workers * 2:
                                row = next(rows, None)
                                if row is None:
                                    exhausted = True
                                    break
                                pending[pool.submit(_verify_registered_submission, cfg.data_root, row)] = row
                            if not pending:
                                break
                            done, _ = wait(pending, return_when=FIRST_COMPLETED)
                            for future in done:
                                row = pending.pop(future)
                                completed(row, future.result())
                    finally:
                        for future in pending:
                            future.cancel()
    problems.sort(key=lambda problem: problem["accession"])
    return {"status": "valid" if not problems else "incomplete",
            "submissions_checked": checked, "problems": problems}


def download_submissions(cfg: Config, *, start=RESEARCH_START, end=None, form=None, limit=None,
                         retry_failed=False, workers=10, client=None, progress=None, accession=None) -> dict:
    if limit is not None and not 1 <= limit <= 10000:
        raise ValueError("Limit must be between 1 and 10000 when supplied")
    if not 1 <= workers <= 20:
        raise ValueError("Workers must be between 1 and 20")
    if start is not None and end is not None and start > end:
        raise ValueError("Start date must be on or before end date")
    if form is not None and form not in FORMS:
        raise ValueError("Unsupported ownership form")
    if client is None:
        client = HttpClient(cfg)
    selected_accession = accession
    summary = {"submissions_requested": 0, "submissions_downloaded": 0,
               "submissions_failed": 0, "errors": []}
    with Catalog(cfg) as catalog, publisher_lock(cfg.data_root, catalog, lane="download"):
        counts = catalog.submission_counts(start=start.isoformat() if start else None,
            end=end.isoformat() if end else None, form=form, accession=selected_accession)
        selection = dict(start=start.isoformat() if start else None,
            end=end.isoformat() if end else None, form=form, accession=selected_accession, retry_failed=retry_failed)
        eligible = catalog.submission_candidate_count(**selection)
        requested = min(eligible, limit) if limit else eligible
        candidates = catalog.submission_candidates(**selection, limit=limit, stream=True)
        summary["submissions_requested"] = requested
        summary["total_indexed"] = counts["total_indexed"]
        summary["already_downloaded"] = counts["already_downloaded"]
        summary["latest_failed_before_run"] = counts["latest_failed"]
        summary["pending_before_run"] = counts["total_indexed"] - counts["already_downloaded"]
        if progress:
            progress({"event": "start", "total": requested, **summary})
        run_id = catalog.start_run("download-submissions", cfg)
        summary["run_id"] = run_id
        state = "succeeded"

        def fetch_and_archive(accession, url):
            response = client.get(url, max_bytes=cfg.max_submission_bytes)
            return (*_publish_zip(cfg, accession, response.body), url)

        try:
            completed = 0
            next_candidate = 0
            blocked = False
            pending = {}
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="sec-download") as pool:
                while (not blocked and next_candidate < requested) or pending:
                    while (not blocked and next_candidate < requested
                           and len(pending) < workers):
                        filing = next(candidates, None)
                        if filing is None:
                            next_candidate = requested
                            break
                        accession, archive_path = filing["accession"], filing["archive_path"]
                        if (not ACCESSION.fullmatch(accession)
                                or not re.fullmatch(rf"edgar/data/[0-9]{{1,10}}/{re.escape(accession)}\.txt", archive_path)):
                            raise ValueError(f"Unsafe or inconsistent archive path for accession {accession}")
                        attempt_id = catalog.begin_submission_attempt(run_id, accession)
                        url = "https://www.sec.gov/Archives/" + archive_path
                        future = pool.submit(fetch_and_archive, accession, url)
                        pending[future] = (accession, attempt_id)
                        next_candidate += 1

                    if not pending:
                        break
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        accession, attempt_id = pending.pop(future)
                        try:
                            zip_hash, relative, zip_size, source_size, url = future.result()
                            catalog.register_submission(accession, relative, zip_hash, zip_size,
                                                        source_size, url, attempt_id)
                            summary["submissions_downloaded"] += 1
                            progress_status, error_message = "downloaded", None
                        except (FetchError, ValueError, OSError, EOFError, zipfile.BadZipFile, zlib.error) as exc:
                            status = getattr(exc, "status", None)
                            catalog.finish_submission_attempt(attempt_id, status="failed", error=str(exc), http_status=status)
                            summary["submissions_failed"] += 1
                            if len(summary["errors"]) < 100:
                                summary["errors"].append({"accession": accession, "message": str(exc)})
                            progress_status, error_message = "failed", str(exc)
                            if status in {403, 429}:
                                blocked = True
                        completed += 1
                        if progress:
                            progress({"event": "progress", "current": completed,
                                      "total": requested, "accession": accession,
                                      "status": progress_status, "message": error_message})

            if blocked:
                summary["submissions_not_attempted"] = requested - completed
            after = catalog.submission_counts(start=start.isoformat() if start else None,
                end=end.isoformat() if end else None, form=form, accession=selected_accession)
            summary["downloaded_after_run"] = after["already_downloaded"]
            summary["remaining_after_run"] = after["total_indexed"] - after["already_downloaded"]
            summary["latest_failed_after_run"] = after["latest_failed"]
            summary["coverage_complete"] = (after["already_downloaded"] == after["total_indexed"]
                                             and after["latest_failed"] == 0)
            if summary["submissions_failed"]:
                state = "incomplete"
            summary["status"] = state
            catalog.finish_run(run_id, state, summary)
        except BaseException as exc:
            catalog.connection.execute("UPDATE submission_attempts SET status='interrupted',finished_at=now(),error='Download run interrupted' WHERE run_id=%s AND status='running'", (run_id,))
            catalog.finish_run(run_id, "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                               summary, type(exc).__name__)
            raise
        finally:
            candidates.close()
    return summary

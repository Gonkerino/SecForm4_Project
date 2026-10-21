"""Preserved source artifacts and transactional catalogue discovery."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import uuid
from pathlib import Path

from .catalog import Catalog, utc_now
from .config import Config
from .http import FetchError, HttpClient, Response
from .locking import publisher_lock
from .sources.indexes import IndexSource, parse_master_index


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def preserve_index(cfg: Config, source: IndexSource, body: bytes) -> tuple[str, str]:
    digest = hashlib.sha256(body).hexdigest()
    relative = Path("raw") / "indexes" / f"source={source.kind}" / f"period={source.period}" / f"{digest}.idx.gz"
    path = cfg.data_root / relative
    if path.exists():
        if hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest() != digest:
            raise ValueError("Preserved index failed its content checksum")
    else:
        atomic_write(path, gzip.compress(body, compresslevel=6, mtime=0))
    return digest, relative.as_posix()


def cached_response(catalog: Catalog, cfg: Config, source: IndexSource) -> Response:
    row = catalog.connection.execute("""SELECT r.* FROM sources s JOIN artifacts r
        ON r.id=s.current_artifact_id WHERE s.kind=? AND s.period=?""", (source.kind, source.period)).fetchone()
    if row is None:
        raise ValueError(f"No validated cached index for {source.period}; offline coverage is unresolved")
    path = (cfg.data_root / row["relative_path"]).resolve()
    if not path.is_relative_to(cfg.data_root):
        raise ValueError("Cached index path escapes the data root")
    with gzip.open(path, "rb") as stream:
        body = stream.read(cfg.max_index_bytes + 1)
    if len(body) > cfg.max_index_bytes or hashlib.sha256(body).hexdigest() != row["sha256"]:
        raise ValueError("Cached index is oversized or failed its checksum")
    return Response(body, {"offline": True, "url": source.url})


def discover(cfg: Config, sources: list[IndexSource], *, local_file: Path | None = None,
             offline: bool = False, accept_removals: bool = False,
             client: HttpClient | None = None, progress=None) -> dict:
    if not cfg.database.is_file():
        raise ValueError("Catalogue is not initialized; run 'insider init' first")
    if local_file and len(sources) != 1:
        raise ValueError("Local import requires exactly one source period")
    if not local_file and not offline and client is None:
        client = HttpClient(cfg)
    summary = {"sources_requested": len(sources), "sources_validated": 0,
               "sources_unresolved": 0, "ownership_associations_observed": 0,
               "removed_accessions": 0, "errors": []}
    with publisher_lock(cfg.data_root), Catalog(cfg.database) as catalog:
        run_id = catalog.start_run("ingest-index" if local_file else "discover", cfg)
        summary["run_id"] = run_id
        active_attempt = None
        try:
            catalog.register_sources(sources)
            atomic_write(cfg.data_root / "reports" / run_id / "plan.json",
                         json.dumps({"sources": [s.as_dict() for s in sources]}, indent=2).encode())
            for source in sources:
                source_id, active_attempt = catalog.begin_attempt(run_id, source.kind, source.period, source.url)
                try:
                    if local_file:
                        with local_file.open("rb") as stream:
                            body = stream.read(cfg.max_index_bytes + 1)
                        if len(body) > cfg.max_index_bytes:
                            raise ValueError("Local index exceeds the configured byte limit")
                        response = Response(body, {"local_file": str(local_file.resolve()), "offline": True})
                    elif offline:
                        response = cached_response(catalog, cfg, source)
                    else:
                        response = client.get(source.url)
                    digest, relative = preserve_index(cfg, source, response.body)
                    # Failed content remains preserved, but is never activated as validated coverage.
                    receipt = {"source": source.as_dict(), "sha256": digest,
                               "relative_path": relative, "observed_at": utc_now(),
                               "response": response.metadata}
                    atomic_write(cfg.data_root / "reports" / run_id / f"attempt-{active_attempt}.json",
                                 json.dumps(receipt, indent=2).encode())
                    parsed = parse_master_index(response.body)
                    old = {r[0] for r in catalog.connection.execute("""SELECT DISTINCT a.accession
                        FROM associations a JOIN sources s ON s.current_artifact_id=a.artifact_id
                        WHERE s.id=?""", (source_id,))}
                    removed = old - {r.accession for r in parsed.rows}
                    if removed and not accept_removals:
                        raise ValueError(f"Revision removes {len(removed)} previously indexed accessions; inspect the receipt and use --accept-removals to activate")
                    catalog.commit_index(source_id=source_id, attempt_id=active_attempt,
                                         digest=digest, relative_path=relative, byte_count=len(response.body),
                                         parsed=parsed, response=response.metadata)
                    summary["sources_validated"] += 1
                    summary["removed_accessions"] += len(removed)
                    summary["ownership_associations_observed"] += len(parsed.rows)
                    if progress:
                        progress(f"{source.period}: validated {len(parsed.rows)} ownership associations")
                except (FetchError, ValueError, OSError, EOFError) as exc:
                    status = getattr(exc, "status", None)
                    catalog.fail_attempt(active_attempt, str(exc), "unavailable" if status == 404 else "failed", status)
                    summary["sources_unresolved"] += 1
                    summary["errors"].append({"period": source.period, "message": str(exc)})
                    if progress:
                        progress(f"{source.period}: unresolved ({exc})")
                    if status in {403, 429}:
                        # Stop on access blocking; do not hammer remaining source periods.
                        break
                active_attempt = None
            summary["sources_not_attempted"] = len(sources) - summary["sources_validated"] - summary["sources_unresolved"]
            state = "succeeded" if summary["sources_validated"] == len(sources) else "incomplete"
            summary["status"] = state
            catalog.finish_run(run_id, state, summary)
        except BaseException as exc:
            state = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            if active_attempt is not None:
                catalog.fail_attempt(active_attempt, type(exc).__name__, "interrupted" if state == "interrupted" else "failed")
            catalog.finish_run(run_id, state, summary, type(exc).__name__)
            raise
        atomic_write(cfg.data_root / "reports" / run_id / "summary.json", json.dumps(summary, indent=2).encode())
    return summary

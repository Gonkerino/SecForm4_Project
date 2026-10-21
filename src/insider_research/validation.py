"""Read-only logical checks for PostgreSQL catalogue and preserved index bytes."""

from __future__ import annotations

import gzip
import hashlib

from .catalog import Catalog
from .config import Config


def validate(cfg: Config, *, progress=None) -> dict:
    problems = []
    checked = 0
    if progress:
        progress({"event": "loading"})
    with Catalog(cfg) as catalog, catalog.connection.transaction():
        catalog.connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        unvalidated = list(catalog.connection.execute("""SELECT c.conname AS name
            FROM pg_constraint c JOIN pg_namespace n ON n.oid=c.connamespace
            WHERE n.nspname=%s AND NOT c.convalidated""", (cfg.database_schema,)))
        if unvalidated:
            problems.append({"type": "unvalidated_constraints", "details": unvalidated})
        total = catalog.connection.execute("SELECT count(*) AS n FROM artifacts").fetchone()["n"]
        if progress:
            progress({"event": "start", "total": total})
        for row in catalog.connection.execute("""SELECT r.*,count(a.row_number) AS actual_rows
            FROM artifacts r LEFT JOIN associations a ON a.artifact_id=r.id GROUP BY r.id"""):
            checked += 1
            try:
                path = (cfg.data_root / row["relative_path"]).resolve()
                if not path.is_relative_to(cfg.data_root):
                    raise ValueError("Artifact path escapes data root")
                digest = hashlib.sha256()
                size = 0
                with gzip.open(path, "rb") as stream:
                    while chunk := stream.read(1024 * 1024):
                        digest.update(chunk)
                        size += len(chunk)
                        if size > cfg.max_index_bytes:
                            raise ValueError("Artifact exceeds configured byte limit")
                if digest.hexdigest() != row["sha256"] or size != row["bytes"]:
                    raise ValueError("Content checksum/size mismatch")
                if row["actual_rows"] != row["ownership_rows"]:
                    raise ValueError("Catalogue association count differs from parsed ownership count")
            except (ValueError, OSError, EOFError) as exc:
                problems.append({"type": "artifact", "artifact_id": row["id"], "message": str(exc)})
            if progress:
                progress({"event": "progress", "current": checked, "total": total,
                          "problems": len(problems)})
        unresolved = [r for r in catalog.coverage() if r["latest_attempt_status"] != "validated"]
    return {"status": "valid" if not problems and not unresolved else "incomplete",
            "database_backend": "postgresql", "artifacts_checked": checked, "problems": problems,
            "unresolved_sources": len(unresolved),
            "note": "Checks constraints, index hashes and catalogue consistency; not PostgreSQL physical integrity or submission coverage."}

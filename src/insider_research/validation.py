"""Read-only integrity checks for committed index discovery artifacts."""

from __future__ import annotations

import gzip
import hashlib
import sqlite3

from .catalog import Catalog
from .config import Config


def validate(cfg: Config) -> dict:
    problems = []
    checked = 0
    with Catalog(cfg.database) as catalog:
        # Hold a consistent read transaction while checking catalogue references.
        catalog.connection.execute("BEGIN")
        integrity = [r[0] for r in catalog.connection.execute("PRAGMA quick_check")]
        if integrity != ["ok"]:
            problems.append({"type": "sqlite_integrity", "details": integrity})
        foreign = [tuple(r) for r in catalog.connection.execute("PRAGMA foreign_key_check")]
        if foreign:
            problems.append({"type": "foreign_keys", "details": foreign})
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
        unresolved = [r for r in catalog.coverage() if r["latest_attempt_status"] != "validated"]
    return {"status": "valid" if not problems and not unresolved else "incomplete",
            "artifacts_checked": checked, "problems": problems,
            "unresolved_sources": len(unresolved),
            "note": "Checks index storage and catalogue consistency, not submission or transaction coverage."}

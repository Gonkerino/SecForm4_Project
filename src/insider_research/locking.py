"""Local artifact lock and PostgreSQL session lock for catalogue publishers."""

from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from pathlib import Path

from .catalog import lock_key


@contextmanager
def publisher_lock(root: Path, catalog, *, lane=None):
    # Shared database gate permits download + parse, while exclusive maintenance
    # (including recovery and initialization) still excludes every publisher.
    if lane not in (None, "download", "parse"):
        raise ValueError("Unknown publisher lane")
    path = root / "catalog" / (f"publisher-{lane}.lock" if lane else "publisher.lock")
    with path.open("a+b") as stream:
        if path.stat().st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError("Another catalogue writer is active; retry after it finishes") from exc
        database_locked = False
        lane_locked = False
        gate = "pg_try_advisory_lock_shared" if lane else "pg_try_advisory_lock"
        unlock = "pg_advisory_unlock_shared" if lane else "pg_advisory_unlock"
        lane_key = None
        try:
            database_locked = catalog.connection.execute(
                f"SELECT {gate}(%s) AS acquired", (lock_key(catalog.cfg),)
            ).fetchone()["acquired"]
            if not database_locked:
                raise ValueError("Another catalogue writer is active in PostgreSQL; retry after it finishes")
            if lane:
                lane_key = int.from_bytes(hashlib.sha256(
                    f"insider-research:{catalog.cfg.database_schema}:{lane}".encode()
                ).digest()[:8], signed=True)
                lane_locked = catalog.connection.execute(
                    "SELECT pg_try_advisory_lock(%s) AS acquired", (lane_key,)
                ).fetchone()["acquired"]
                if not lane_locked:
                    raise ValueError(f"Another {lane} writer is active in PostgreSQL; retry after it finishes")
            yield
        finally:
            try:
                if lane_locked and not catalog.connection.closed:
                    catalog.connection.execute("SELECT pg_advisory_unlock(%s)", (lane_key,))
                if database_locked and not catalog.connection.closed:
                    catalog.connection.execute(f"SELECT {unlock}(%s)", (lock_key(catalog.cfg),))
            finally:
                stream.seek(0)
                if os.name == "nt":
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

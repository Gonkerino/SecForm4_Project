"""Local artifact lock and PostgreSQL session lock for catalogue publishers."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

from .catalog import lock_key


@contextmanager
def publisher_lock(root: Path, catalog):
    path = root / "catalog" / "publisher.lock"
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
        try:
            database_locked = catalog.connection.execute(
                "SELECT pg_try_advisory_lock(%s) AS acquired", (lock_key(catalog.cfg),)
            ).fetchone()["acquired"]
            if not database_locked:
                raise ValueError("Another catalogue writer is active in PostgreSQL; retry after it finishes")
            yield
        finally:
            try:
                if database_locked and not catalog.connection.closed:
                    catalog.connection.execute("SELECT pg_advisory_unlock(%s)", (lock_key(catalog.cfg),))
            finally:
                stream.seek(0)
                if os.name == "nt":
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

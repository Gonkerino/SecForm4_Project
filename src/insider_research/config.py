"""Validated settings with no import-time filesystem or network effects."""

from __future__ import annotations

import math
import os
import re
import tomllib
from dataclasses import dataclass, field, fields
from datetime import date
from pathlib import Path

from psycopg.conninfo import conninfo_to_dict

FORMS = frozenset({"3", "3/A", "4", "4/A", "5", "5/A"})
# First full month after mandatory electronic filing began on 30 June 2003.
RESEARCH_START = date(2003, 7, 1)


def default_config_path() -> Path | None:
    """Find this source checkout's config, independent of the working directory."""
    project = Path(__file__).resolve().parents[2]
    candidate = project / "config" / "default.toml"
    if (project / "pyproject.toml").is_file() and candidate.is_file():
        return candidate
    return None


@dataclass(frozen=True)
class Config:
    data_root: Path
    database_url: str = field(
        default="host=127.0.0.1 port=5432 dbname=insider_research user=insider",
        repr=False,
    )
    database_schema: str = "insider_research"
    user_agent: str = ""
    max_rps: float = 10.0
    timeout_seconds: float = 45.0
    max_attempts: int = 4
    max_index_bytes: int = 128 * 1024 * 1024
    max_submission_bytes: int = 64 * 1024 * 1024

    def __post_init__(self) -> None:
        root = Path(self.data_root).expanduser().resolve()
        if root == Path(root.anchor):
            raise ValueError("The data root must be a dedicated directory, not a drive root")
        object.__setattr__(self, "data_root", root)
        if not isinstance(self.database_url, str) or not self.database_url.strip():
            raise ValueError("database_url must be a PostgreSQL URI or libpq connection string")
        try:
            conninfo_to_dict(self.database_url)
        except Exception:
            # libpq's parser error can contain the supplied password/URI.
            raise ValueError("Invalid PostgreSQL database_url; check the connection settings") from None
        if (not isinstance(self.database_schema, str)
                or not re.fullmatch(r"[a-z][a-z0-9_]{0,62}", self.database_schema)
                or self.database_schema in {"public", "information_schema"}
                or self.database_schema.startswith("pg_")):
            raise ValueError("database_schema must be a dedicated lowercase PostgreSQL schema name")
        for name, lower, upper in (("max_rps", 0.01, 10),
                                   ("timeout_seconds", 0.1, 600)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)):
                raise ValueError(f"{name} must be numeric")
            if not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"{name} must be between {lower} and {upper}")
        for name, upper in (("max_attempts", 10), ("max_index_bytes", 1024**3),
                            ("max_submission_bytes", 1024**3)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= upper:
                raise ValueError(f"{name} must be a positive integer no greater than {upper}")
        if not isinstance(self.user_agent, str) or any(ord(c) < 32 or ord(c) == 127 for c in self.user_agent):
            raise ValueError("user_agent must be a single-line string without control characters")

    def require_identity(self) -> None:
        parts = self.user_agent.strip().rsplit(None, 1)
        if len(parts) != 2 or not re.fullmatch(r"[^\s@<>()[\]]+@[^\s@<>()[\]]+\.[^\s@<>()[\]]+", parts[-1]):
            raise ValueError(
                "Online discovery requires 'Name contact@email' in user_agent configuration "
                "or the SEC_USER_AGENT environment variable"
            )

    def public_settings(self) -> dict:
        return {"data_root": str(self.data_root), "max_rps": self.max_rps,
                "timeout_seconds": self.timeout_seconds, "max_attempts": self.max_attempts,
                "max_index_bytes": self.max_index_bytes,
                "max_submission_bytes": self.max_submission_bytes,
                "contact_configured": bool(self.user_agent),
                "database_backend": "postgresql", "database_schema": self.database_schema}


def load_config(path: Path | None = None, data_root: Path | None = None) -> Config:
    if path is None:
        path = default_config_path()
    values = {}
    if path is not None:
        with path.open("rb") as stream:
            values = tomllib.load(stream)
        unknown = set(values) - {f.name for f in fields(Config)}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {', '.join(sorted(unknown))}")
        if "data_root" in values:
            if not isinstance(values["data_root"], str):
                raise ValueError("Configured data_root must be a directory path string")
            configured = Path(values["data_root"]).expanduser()
            values["data_root"] = configured if configured.is_absolute() else path.resolve().parent / configured
    if "SEC_USER_AGENT" in os.environ:
        values["user_agent"] = os.environ["SEC_USER_AGENT"]
    if "INSIDER_DATABASE_URL" in os.environ:
        values["database_url"] = os.environ["INSIDER_DATABASE_URL"]
    if "INSIDER_DATABASE_SCHEMA" in os.environ:
        values["database_schema"] = os.environ["INSIDER_DATABASE_SCHEMA"]
    root = data_root or os.environ.get("INSIDER_DATA_ROOT") or values.get("data_root")
    values["data_root"] = Path(root) if root else (
        Path("D:/Data/InsiderResearch") if os.name == "nt"
        else Path.home() / ".local" / "share" / "insider-research")
    return Config(**values)

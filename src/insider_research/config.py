"""Validated settings with no import-time filesystem or network effects."""

from __future__ import annotations

import math
import os
import re
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

FORMS = frozenset({"3", "3/A", "4", "4/A", "5", "5/A"})


@dataclass(frozen=True)
class Config:
    data_root: Path
    user_agent: str = ""
    max_rps: float = 3.0
    timeout_seconds: float = 45.0
    max_attempts: int = 4
    max_index_bytes: int = 128 * 1024 * 1024

    def __post_init__(self) -> None:
        root = Path(self.data_root).expanduser().resolve()
        if root == Path(root.anchor):
            raise ValueError("The data root must be a dedicated directory, not a drive root")
        object.__setattr__(self, "data_root", root)
        for name, lower, upper in (("max_rps", 0.01, 10),
                                   ("timeout_seconds", 0.1, 600)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)):
                raise ValueError(f"{name} must be numeric")
            if not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"{name} must be between {lower} and {upper}")
        for name, upper in (("max_attempts", 10), ("max_index_bytes", 1024**3)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= upper:
                raise ValueError(f"{name} must be a positive integer no greater than {upper}")
        if not isinstance(self.user_agent, str) or any(c in self.user_agent for c in "\r\n"):
            raise ValueError("user_agent must be a single-line string")

    @property
    def database(self) -> Path:
        return self.data_root / "catalog" / "catalog.sqlite"

    def require_identity(self) -> None:
        if not re.search(r"[^\s@]+@[^\s@]+\.[^\s@()]+", self.user_agent):
            raise ValueError("Online discovery requires SEC_USER_AGENT with a real contact email")

    def public_settings(self) -> dict:
        return {"data_root": str(self.data_root), "max_rps": self.max_rps,
                "timeout_seconds": self.timeout_seconds, "max_attempts": self.max_attempts,
                "max_index_bytes": self.max_index_bytes,
                "contact_configured": bool(self.user_agent)}


def load_config(path: Path | None = None, data_root: Path | None = None) -> Config:
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
    root = data_root or os.environ.get("INSIDER_DATA_ROOT") or values.get("data_root")
    values["data_root"] = Path(root) if root else (
        Path("D:/Data/InsiderResearch") if os.name == "nt"
        else Path.home() / ".local" / "share" / "insider-research")
    return Config(**values)

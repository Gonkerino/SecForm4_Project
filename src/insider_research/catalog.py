"""Transactional filing identities, source revisions and discovery receipts."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path

from .config import Config

SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class Catalog:
    def __init__(self, path: Path):
        if not path.is_file():
            raise ValueError("Catalogue is not initialized; run 'insider init' first")
        self.connection = sqlite3.connect(path, timeout=30)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA synchronous=FULL")
        version = self.connection.execute("PRAGMA user_version").fetchone()[0]
        if version != SCHEMA_VERSION:
            self.close()
            raise ValueError(f"Unsupported catalogue schema {version}; expected {SCHEMA_VERSION}")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self.connection.close()

    @classmethod
    def initialize(cls, cfg: Config) -> None:
        # A marker prevents accidentally initializing inside an old/unrelated dataset.
        marker = cfg.data_root / ".insider-research.json"
        if cfg.data_root.exists() and any(cfg.data_root.iterdir()) and not marker.is_file():
            raise ValueError("Data root is nonempty and has no Insider Research marker; choose a fresh directory")
        cfg.data_root.mkdir(parents=True, exist_ok=True)
        if marker.exists():
            if json.loads(marker.read_text(encoding="utf-8")).get("format") != "insider-research-v1":
                raise ValueError("Unrecognized data root marker")
        else:
            marker.write_text(json.dumps({"format": "insider-research-v1",
                                          "created_at": utc_now()}, indent=2), encoding="utf-8")
        for directory in ("catalog", "raw/indexes", "raw/submissions", "raw/sec_bulk",
                          "canonical", "snapshots", "staging", "logs", "reports"):
            (cfg.data_root / directory).mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(cfg.database, timeout=30)
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version == 0:
                schema = files("insider_research").joinpath("migrations/001_catalog.sql").read_text()
                connection.executescript("BEGIN IMMEDIATE;\n" + schema + "\nCOMMIT;")
            elif version != SCHEMA_VERSION:
                raise ValueError(f"Unsupported catalogue schema {version}")
        finally:
            connection.close()

    def start_run(self, command: str, cfg: Config) -> str:
        run_id = uuid.uuid4().hex
        with self.connection:
            self.connection.execute("INSERT INTO runs(id,command,started_at,status,settings_json) VALUES(?,?,?,'running',?)",
                                    (run_id, command, utc_now(), json.dumps(cfg.public_settings())))
        return run_id

    def finish_run(self, run_id: str, status: str, summary: dict, error: str | None = None):
        with self.connection:
            self.connection.execute("UPDATE runs SET finished_at=?,status=?,summary_json=?,error=? WHERE id=?",
                                    (utc_now(), status, json.dumps(summary), error, run_id))

    def begin_attempt(self, run_id: str, kind: str, period: str, url: str) -> tuple[int, int]:
        with self.connection:
            self.connection.execute("INSERT INTO sources(kind,period,url) VALUES(?,?,?) ON CONFLICT(kind,period) DO UPDATE SET url=excluded.url",
                                    (kind, period, url))
            source_id = self.connection.execute("SELECT id FROM sources WHERE kind=? AND period=?", (kind, period)).fetchone()[0]
            cursor = self.connection.execute("INSERT INTO attempts(run_id,source_id,started_at,status) VALUES(?,?,?,'running')",
                                             (run_id, source_id, utc_now()))
        return source_id, cursor.lastrowid

    def fail_attempt(self, attempt_id: int, error: str, status: str = "failed", http_status: int | None = None):
        with self.connection:
            self.connection.execute("UPDATE attempts SET status=?,finished_at=?,error=?,http_status=? WHERE id=?",
                                    (status, utc_now(), error, http_status, attempt_id))

    def register_sources(self, sources) -> None:
        with self.connection:
            self.connection.executemany("""INSERT INTO sources(kind,period,url) VALUES(?,?,?)
                ON CONFLICT(kind,period) DO NOTHING""", [(s.kind, s.period, s.url) for s in sources])

    def commit_index(self, *, source_id: int, attempt_id: int, digest: str,
                     relative_path: str, byte_count: int, parsed, response: dict) -> int:
        now = utc_now()
        # The whole revision and its associations become visible together.
        with self.connection:
            cursor = self.connection.execute("""INSERT INTO artifacts
                (source_id,sha256,relative_path,bytes,first_observed_at,last_observed_at,
                 total_rows,ownership_rows,min_filed_date,max_filed_date)
                VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(source_id,sha256)
                DO UPDATE SET last_observed_at=excluded.last_observed_at RETURNING id""",
                (source_id, digest, relative_path, byte_count, now, now, parsed.total_rows,
                 len(parsed.rows), parsed.min_date, parsed.max_date))
            artifact_id = cursor.fetchone()[0]
            for row in parsed.rows:
                self.connection.execute("INSERT INTO filings VALUES(?,?,?) ON CONFLICT(accession) DO UPDATE SET last_seen_at=excluded.last_seen_at",
                                        (row.accession, now, now))
                self.connection.execute("INSERT INTO associations VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(artifact_id,row_number) DO NOTHING",
                                        (artifact_id, row.row_number, row.accession, row.cik,
                                         row.company_name, row.form_type, row.filed_date, row.archive_path))
            self.connection.execute("UPDATE sources SET current_artifact_id=? WHERE id=?", (artifact_id, source_id))
            self.connection.execute("UPDATE attempts SET status='validated',finished_at=?,artifact_id=?,http_status=?,response_json=? WHERE id=?",
                                    (now, artifact_id, response.get("status"), json.dumps(response), attempt_id))
        return artifact_id

    def status(self) -> dict:
        def count(sql):
            return self.connection.execute(sql).fetchone()[0]
        return {"schema_version": SCHEMA_VERSION,
                "unique_filings_ever_observed": count("SELECT count(*) FROM filings"),
                "unique_filings_in_current_sources": count("SELECT count(DISTINCT accession) FROM current_associations"),
                "current_index_associations": count("SELECT count(*) FROM current_associations"),
                "source_revisions": count("SELECT count(*) FROM artifacts"),
                "sources": count("SELECT count(*) FROM sources"),
                "failed_latest_attempts": count("SELECT count(*) FROM attempts a WHERE a.id=(SELECT max(id) FROM attempts b WHERE b.source_id=a.source_id) AND a.status!='validated'"),
                "running_runs": count("SELECT count(*) FROM runs WHERE status='running'"),
                "filing_submissions_downloaded": 0,
                "note": "Index discovery only. Submission downloading and ownership parsing are not implemented yet."}

    def coverage(self) -> list[dict]:
        rows = [dict(row) for row in self.connection.execute("""SELECT s.kind,s.period,
            a.status AS latest_attempt_status,a.error,a.http_status,
            a.response_json,
            r.sha256 AS latest_validated_sha256,r.total_rows,r.ownership_rows,
            r.last_observed_at AS last_validated_at
            FROM sources s LEFT JOIN attempts a ON a.id=(SELECT max(id) FROM attempts WHERE source_id=s.id)
            LEFT JOIN artifacts r ON r.id=s.current_artifact_id ORDER BY s.kind,s.period""")]
        for row in rows:
            response = json.loads(row.pop("response_json") or "{}")
            row["latest_validation_offline"] = response.get("offline", False) if row["latest_attempt_status"] == "validated" else None
        return rows

    def list_filings(self, *, start: str | None = None, end: str | None = None,
                     form: str | None = None, cik: str | None = None, limit: int = 100) -> list[dict]:
        if not 1 <= limit <= 10000:
            raise ValueError("Limit must be between 1 and 10000")
        conditions, parameters = [], []
        for expression, value in (("a.filed_date>=?", start), ("a.filed_date<=?", end),
                                  ("a.form_type=?", form), ("a.indexed_cik=?", cik)):
            if value is not None:
                conditions.append(expression)
                parameters.append(value)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        sql = """SELECT a.accession,min(a.filed_date) AS first_indexed_filing_date,
            count(DISTINCT a.archive_path) AS download_path_count,
            count(DISTINCT a.indexed_cik) AS indexed_entity_count,
            group_concat(DISTINCT a.form_type) AS indexed_forms
            FROM current_associations a""" + where + " GROUP BY a.accession ORDER BY first_indexed_filing_date,a.accession LIMIT ?"
        return [dict(r) for r in self.connection.execute(sql, [*parameters, limit])]

    def recover(self) -> dict:
        # Caller must own the publisher lock, so running records belong to dead writers.
        with self.connection:
            attempts = self.connection.execute("""UPDATE attempts SET status='interrupted',finished_at=?,
                error='Previous writer exited before finishing this attempt' WHERE status='running'""", (utc_now(),)).rowcount
            runs = self.connection.execute("""UPDATE runs SET status='interrupted',finished_at=?,
                error='Previous writer exited before finishing this run' WHERE status='running'""", (utc_now(),)).rowcount
        return {"runs_reconciled": runs, "attempts_reconciled": attempts,
                "note": "Committed revisions are preserved. Repeat the original discovery command to resume unresolved sources."}

    def inspect(self, accession: str) -> dict:
        filing = self.connection.execute("SELECT * FROM filings WHERE accession=?", (accession,)).fetchone()
        if filing is None:
            raise ValueError(f"Accession not found: {accession}")
        rows = [dict(r) for r in self.connection.execute("""SELECT a.*,s.kind,s.period,
            r.sha256,r.relative_path,(s.current_artifact_id=a.artifact_id) AS in_current_source
            FROM associations a JOIN artifacts r ON r.id=a.artifact_id
            JOIN sources s ON s.id=r.source_id WHERE a.accession=? ORDER BY s.period,a.artifact_id,a.row_number""", (accession,))]
        return {**dict(filing), "associations": rows,
                "metadata_conflict": len({(r['form_type'], r['filed_date']) for r in rows if r['in_current_source']}) > 1,
                "note": "Indexed CIK is a discovery association, not a verified issuer or owner identity."}

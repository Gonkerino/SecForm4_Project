"""PostgreSQL filing identities, source revisions and discovery receipts."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from importlib.resources import files

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .config import Config

SCHEMA_VERSION = 2


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def connect(cfg: Config):
    connection = None
    try:
        connection = psycopg.connect(cfg.database_url, autocommit=True, row_factory=dict_row,
                                     connect_timeout=5, application_name="insider_research")
        connection.execute("SET TIME ZONE 'UTC'")
        connection.execute("SET synchronous_commit = on")
        connection.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(cfg.database_schema)))
        return connection
    except psycopg.Error:
        if connection is not None:
            connection.close()
        # Connection diagnostics may contain a supplied URI/password.
        raise ValueError("Cannot connect to PostgreSQL. Check the server, database, role and password "
                         "settings (INSIDER_DATABASE_URL); see README.md for setup.") from None


def lock_key(cfg: Config) -> int:
    return int.from_bytes(hashlib.sha256(
        f"insider-research:{cfg.database_schema}".encode()).digest()[:8], signed=True)


def check_data_root(cfg: Config) -> None:
    marker = cfg.data_root / ".insider-research.json"
    if cfg.data_root.exists() and any(cfg.data_root.iterdir()) and not marker.is_file():
        raise ValueError("Data root is nonempty and has no Insider Research marker; choose a fresh directory")
    if marker.exists():
        if json.loads(marker.read_text(encoding="utf-8")).get("format") != "insider-research-v1":
            raise ValueError("Unrecognized data root marker")


class Catalog:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.connection = connect(cfg)
        try:
            self._check_schema()
            check_data_root(cfg)
            if not (cfg.data_root / ".insider-research.json").is_file():
                raise ValueError("Data root is not initialized; run 'insider init' first")
        except BaseException:
            self.close()
            raise

    def _check_schema(self):
        exists = self.connection.execute("SELECT to_regclass(%s) AS relation",
            (f'{self.cfg.database_schema}.metadata',)).fetchone()["relation"]
        if exists is None:
            raise ValueError("PostgreSQL catalogue is not initialized; run 'insider init' first")
        metadata = {r["key"]: r["value"] for r in self.connection.execute("SELECT * FROM metadata")}
        if metadata.get("schema_version") != str(SCHEMA_VERSION):
            raise ValueError(f"Unsupported catalogue schema {metadata.get('schema_version')}; expected {SCHEMA_VERSION}")
        if metadata.get("data_root") != str(self.cfg.data_root):
            raise ValueError("PostgreSQL catalogue belongs to a different data_root; use its configured directory")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self.connection.close()

    @classmethod
    def initialize(cls, cfg: Config) -> None:
        check_data_root(cfg)
        with connect(cfg) as connection, connection.transaction():
            # Serializes initialization with publishers, including other data-root paths.
            acquired = connection.execute("SELECT pg_try_advisory_xact_lock(%s) AS acquired",
                                          (lock_key(cfg),)).fetchone()["acquired"]
            if not acquired:
                raise ValueError("Another catalogue writer is active; retry initialization after it finishes")
            connection.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(cfg.database_schema)))
            exists = connection.execute("SELECT to_regclass(%s) AS relation",
                (f'{cfg.database_schema}.metadata',)).fetchone()["relation"]
            if exists is None:
                occupied = connection.execute("SELECT EXISTS(SELECT 1 FROM pg_class c "
                    "JOIN pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname=%s) AS occupied",
                    (cfg.database_schema,)).fetchone()["occupied"]
                if occupied:
                    raise ValueError("Application schema is nonempty and unrecognized; choose a fresh schema")
                schema = files("insider_research").joinpath("migrations/001_catalog.sql").read_text(encoding="utf-8-sig")
                connection.execute(schema)
                connection.execute("INSERT INTO metadata(key,value) VALUES ('data_root',%s)", (str(cfg.data_root),))
            metadata = {r["key"]: r["value"] for r in connection.execute("SELECT * FROM metadata")}
            version = metadata.get("schema_version")
            if version == "1":
                migration = files("insider_research").joinpath("migrations/002_submissions.sql").read_text(encoding="utf-8-sig")
                connection.execute(migration)
            elif version != str(SCHEMA_VERSION):
                raise ValueError(f"Unsupported catalogue schema {version}")
            if metadata.get("data_root") not in (None, str(cfg.data_root)):
                raise ValueError("PostgreSQL catalogue belongs to a different data_root")
            cfg.data_root.mkdir(parents=True, exist_ok=True)
            marker = cfg.data_root / ".insider-research.json"
            if not marker.exists():
                marker.write_text(json.dumps({"format": "insider-research-v1", "created_at": utc_now()}, indent=2), encoding="utf-8")
            for directory in ("catalog", "raw/indexes", "raw/submissions", "raw/sec_bulk",
                              "canonical", "snapshots", "staging", "logs", "reports"):
                (cfg.data_root / directory).mkdir(parents=True, exist_ok=True)

    def start_run(self, command: str, cfg: Config) -> str:
        run_id = uuid.uuid4().hex
        with self.connection.transaction():
            self.connection.execute("INSERT INTO runs(id,command,started_at,status,settings_json) VALUES(%s,%s,%s,'running',%s)",
                                    (run_id, command, utc_now(), Jsonb(cfg.public_settings())))
        return run_id

    def finish_run(self, run_id: str, status: str, summary: dict, error: str | None = None):
        with self.connection.transaction():
            self.connection.execute("UPDATE runs SET finished_at=%s,status=%s,summary_json=%s,error=%s WHERE id=%s",
                                    (utc_now(), status, Jsonb(summary), error, run_id))

    def begin_attempt(self, run_id: str, kind: str, period: str, url: str) -> tuple[int, int]:
        with self.connection.transaction():
            source_id = self.connection.execute("""INSERT INTO sources(kind,period,url) VALUES(%s,%s,%s)
                ON CONFLICT(kind,period) DO UPDATE SET url=excluded.url RETURNING id""",
                (kind, period, url)).fetchone()["id"]
            attempt_id = self.connection.execute("INSERT INTO attempts(run_id,source_id,started_at,status) VALUES(%s,%s,%s,'running') RETURNING id",
                (run_id, source_id, utc_now())).fetchone()["id"]
        return source_id, attempt_id

    def fail_attempt(self, attempt_id: int, error: str, status: str = "failed", http_status: int | None = None):
        with self.connection.transaction():
            self.connection.execute("UPDATE attempts SET status=%s,finished_at=%s,error=%s,http_status=%s WHERE id=%s",
                                    (status, utc_now(), error, http_status, attempt_id))

    def register_sources(self, sources) -> None:
        with self.connection.transaction(), self.connection.cursor() as cursor:
            cursor.executemany("""INSERT INTO sources(kind,period,url) VALUES(%s,%s,%s)
                ON CONFLICT(kind,period) DO NOTHING""", [(s.kind, s.period, s.url) for s in sources])

    def current_accessions(self, source_id: int) -> set[str]:
        return {r["accession"] for r in self.connection.execute("""SELECT DISTINCT a.accession
            FROM associations a JOIN sources s ON s.current_artifact_id=a.artifact_id
            WHERE s.id=%s""", (source_id,))}

    def cached_artifact(self, kind: str, period: str):
        return self.connection.execute("""SELECT r.* FROM sources s JOIN artifacts r
            ON r.id=s.current_artifact_id WHERE s.kind=%s AND s.period=%s""", (kind, period)).fetchone()

    def commit_index(self, *, source_id: int, attempt_id: int, digest: str,
                     relative_path: str, byte_count: int, parsed, response: dict) -> int:
        now = utc_now()
        with self.connection.transaction(), self.connection.cursor() as cursor:
            artifact_id = cursor.execute("""INSERT INTO artifacts
                (source_id,sha256,relative_path,bytes,first_observed_at,last_observed_at,
                 total_rows,ownership_rows,min_filed_date,max_filed_date)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(source_id,sha256)
                DO UPDATE SET last_observed_at=excluded.last_observed_at RETURNING id""",
                (source_id, digest, relative_path, byte_count, now, now, parsed.total_rows,
                 len(parsed.rows), parsed.min_date, parsed.max_date)).fetchone()["id"]
            cursor.executemany("""INSERT INTO filings VALUES(%s,%s,%s) ON CONFLICT(accession)
                DO UPDATE SET last_seen_at=excluded.last_seen_at""",
                [(row.accession, now, now) for row in parsed.rows])
            cursor.executemany("INSERT INTO associations VALUES(%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(artifact_id,row_number) DO NOTHING",
                [(artifact_id, row.row_number, row.accession, row.cik, row.company_name,
                  row.form_type, row.filed_date, row.archive_path) for row in parsed.rows])
            cursor.execute("UPDATE sources SET current_artifact_id=%s WHERE id=%s", (artifact_id, source_id))
            cursor.execute("UPDATE attempts SET status='validated',finished_at=%s,artifact_id=%s,http_status=%s,response_json=%s WHERE id=%s",
                           (now, artifact_id, response.get("status"), Jsonb(response), attempt_id))
        return artifact_id

    def status(self) -> dict:
        # One statement gives all counts the same MVCC snapshot.
        result = self.connection.execute("""SELECT
            (SELECT count(*) FROM filings) AS unique_filings_ever_observed,
            (SELECT count(DISTINCT accession) FROM current_associations) AS unique_filings_in_current_sources,
            (SELECT count(*) FROM current_associations) AS current_index_associations,
            (SELECT count(*) FROM artifacts) AS source_revisions,
            (SELECT count(*) FROM sources) AS sources,
            (SELECT count(*) FROM submissions) AS submissions_downloaded,
            (SELECT count(*) FROM attempts a WHERE a.id=(SELECT max(id) FROM attempts b
                WHERE b.source_id=a.source_id) AND a.status!='validated') AS failed_latest_attempts,
            (SELECT count(*) FROM runs WHERE status='running') AS running_runs""").fetchone()
        return {"database_backend": "postgresql", "schema_version": SCHEMA_VERSION, **result,
                "note": "Ownership parsing, analytical snapshots, market prices and signals are not implemented yet."}

    def coverage(self) -> list[dict]:
        rows = list(self.connection.execute("""SELECT s.kind,s.period,
            a.status AS latest_attempt_status,a.error,a.http_status,a.response_json,
            r.sha256 AS latest_validated_sha256,r.total_rows,r.ownership_rows,
            r.last_observed_at AS last_validated_at
            FROM sources s LEFT JOIN attempts a ON a.id=(SELECT max(id) FROM attempts WHERE source_id=s.id)
            LEFT JOIN artifacts r ON r.id=s.current_artifact_id ORDER BY s.kind,s.period"""))
        for row in rows:
            response = row.pop("response_json") or {}
            row["latest_validation_offline"] = response.get("offline", False) if row["latest_attempt_status"] == "validated" else None
        return rows

    def list_filings(self, *, start: str | None = None, end: str | None = None,
                     form: str | None = None, cik: str | None = None, limit: int = 100) -> list[dict]:
        if not 1 <= limit <= 10000:
            raise ValueError("Limit must be between 1 and 10000")
        conditions, parameters = [], []
        for expression, value in (("a.filed_date>=%s", start), ("a.filed_date<=%s", end),
                                  ("a.form_type=%s", form), ("a.indexed_cik=%s", cik)):
            if value is not None:
                conditions.append(expression)
                parameters.append(value)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        query = """SELECT a.accession,min(a.filed_date) AS first_indexed_filing_date,
            count(DISTINCT a.archive_path) AS download_path_count,
            count(DISTINCT a.indexed_cik) AS indexed_entity_count,
            string_agg(DISTINCT a.form_type, ',' ORDER BY a.form_type) AS indexed_forms
            FROM current_associations a""" + where + " GROUP BY a.accession ORDER BY first_indexed_filing_date,a.accession LIMIT %s"
        return list(self.connection.execute(query, [*parameters, limit]))

    def recover(self) -> dict:
        # Caller owns both the local and database publisher locks.
        with self.connection.transaction():
            attempts = self.connection.execute("""UPDATE attempts SET status='interrupted',finished_at=%s,
                error='Previous writer exited before finishing this attempt' WHERE status='running'""", (utc_now(),)).rowcount
            runs = self.connection.execute("""UPDATE runs SET status='interrupted',finished_at=%s,
                error='Previous writer exited before finishing this run' WHERE status='running'""", (utc_now(),)).rowcount
            submissions = self.connection.execute("""UPDATE submission_attempts SET status='interrupted',finished_at=%s,
                error='Previous writer exited before finishing this attempt' WHERE status='running'""", (utc_now(),)).rowcount
        return {"runs_reconciled": runs, "attempts_reconciled": attempts,
                "submission_attempts_reconciled": submissions,
                "note": "Committed revisions are preserved. Repeat the original discovery command to resume unresolved sources."}

    def _submission_filter(self, start=None, end=None, form=None):
        conditions, parameters = [], []
        if start is not None:
            conditions.append("a.filed_date >= %s")
            parameters.append(start)
        if end is not None:
            conditions.append("a.filed_date <= %s")
            parameters.append(end)
        if form is not None:
            conditions.append("a.form_type = %s")
            parameters.append(form)
        return (" AND ".join(conditions) if conditions else "TRUE"), parameters

    def submission_counts(self, *, start=None, end=None, form=None):
        where, parameters = self._submission_filter(start, end, form)
        return self.connection.execute(f"""SELECT count(DISTINCT a.accession) AS total_indexed,
            count(DISTINCT a.accession) FILTER (WHERE s.accession IS NOT NULL) AS already_downloaded,
            count(DISTINCT a.accession) FILTER (WHERE latest.status='failed') AS latest_failed
            FROM current_associations a
            LEFT JOIN submissions s ON s.accession=a.accession
            LEFT JOIN LATERAL (SELECT sa.status FROM submission_attempts sa
                WHERE sa.accession=a.accession ORDER BY sa.id DESC LIMIT 1) latest ON TRUE
            WHERE {where}""", parameters).fetchone()

    def submission_candidates(self, *, start=None, end=None, form=None, limit=None, retry_failed=False):
        where, parameters = self._submission_filter(start, end, form)
        if not retry_failed:
            where += """ AND NOT EXISTS (SELECT 1 FROM submission_attempts sa WHERE sa.accession=f.accession
                AND sa.status='failed' AND sa.id=(SELECT max(last_sa.id) FROM submission_attempts last_sa
                WHERE last_sa.accession=f.accession))"""
        query = f"""SELECT f.accession,min(a.archive_path) AS archive_path
            FROM current_associations a JOIN filings f USING(accession)
            WHERE {where} AND NOT EXISTS (SELECT 1 FROM submissions s WHERE s.accession=f.accession)
            GROUP BY f.accession ORDER BY min(a.filed_date),f.accession"""
        if limit is not None:
            query += " LIMIT %s"
            parameters.append(limit)
        return list(self.connection.execute(query, parameters))

    def begin_submission_attempt(self, run_id: str, accession: str) -> int:
        with self.connection.transaction():
            return self.connection.execute("""INSERT INTO submission_attempts
                (run_id,accession,started_at,status) VALUES(%s,%s,%s,'running') RETURNING id""",
                (run_id, accession, utc_now())).fetchone()["id"]

    def finish_submission_attempt(self, attempt_id: int, *, status: str, error=None, http_status=None):
        with self.connection.transaction():
            self.connection.execute("""UPDATE submission_attempts SET status=%s,finished_at=%s,error=%s,http_status=%s
                WHERE id=%s""", (status, utc_now(), error, http_status, attempt_id))

    def register_submission(self, accession: str, archive_path: str, sha256: str,
                            zip_bytes: int, source_bytes: int, source_url: str, attempt_id: int):
        now = utc_now()
        with self.connection.transaction():
            self.connection.execute("""INSERT INTO submissions
                (accession,archive_path,sha256,zip_bytes,source_bytes,source_url,downloaded_at,last_verified_at)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT(accession) DO UPDATE SET archive_path=excluded.archive_path,sha256=excluded.sha256,
                zip_bytes=excluded.zip_bytes,source_bytes=excluded.source_bytes,source_url=excluded.source_url,
                last_verified_at=excluded.last_verified_at""",
                (accession, archive_path, sha256, zip_bytes, source_bytes, source_url, now, now))
            self.connection.execute("UPDATE submission_attempts SET status='downloaded',finished_at=%s WHERE id=%s",
                                    (now, attempt_id))

    def inspect(self, accession: str) -> dict:
        filing = self.connection.execute("SELECT * FROM filings WHERE accession=%s", (accession,)).fetchone()
        if filing is None:
            raise ValueError(f"Accession not found: {accession}")
        rows = list(self.connection.execute("""SELECT a.*,s.kind,s.period,
            r.sha256,r.relative_path,(s.current_artifact_id=a.artifact_id) AS in_current_source
            FROM associations a JOIN artifacts r ON r.id=a.artifact_id
            JOIN sources s ON s.id=r.source_id WHERE a.accession=%s ORDER BY s.period,a.artifact_id,a.row_number""", (accession,)))
        return {**filing, "associations": rows,
                "metadata_conflict": len({(r['form_type'], r['filed_date']) for r in rows if r['in_current_source']}) > 1,
                "note": "Indexed CIK is a discovery association, not a verified issuer or owner identity."}

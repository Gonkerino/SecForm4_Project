-- Submission archive tracking is additive; index catalogue records are preserved.
CREATE TABLE submission_attempts (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id),
    accession TEXT NOT NULL REFERENCES filings(accession),
    started_at TIMESTAMPTZ NOT NULL,
    finished_at TIMESTAMPTZ,
    status TEXT NOT NULL CHECK(status IN ('running','downloaded','failed','interrupted')),
    http_status INTEGER,
    error TEXT
);
CREATE INDEX submission_attempts_accession ON submission_attempts(accession, id);
CREATE TABLE submissions (
    accession TEXT PRIMARY KEY REFERENCES filings(accession),
    archive_path TEXT NOT NULL UNIQUE,
    sha256 TEXT NOT NULL CHECK(sha256 ~ '^[0-9a-f]{64}$'),
    zip_bytes BIGINT NOT NULL CHECK(zip_bytes > 0),
    source_bytes BIGINT NOT NULL CHECK(source_bytes > 0),
    source_url TEXT NOT NULL,
    downloaded_at TIMESTAMPTZ NOT NULL,
    last_verified_at TIMESTAMPTZ NOT NULL
);
UPDATE metadata SET value='2' WHERE key='schema_version';

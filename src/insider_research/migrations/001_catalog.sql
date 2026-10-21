-- Executed transactionally in the dedicated application schema.
CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE runs (
    id TEXT PRIMARY KEY, command TEXT NOT NULL, started_at TIMESTAMPTZ NOT NULL,
    finished_at TIMESTAMPTZ, status TEXT NOT NULL CHECK(status IN
        ('running','succeeded','incomplete','interrupted','failed')),
    settings_json JSONB NOT NULL, summary_json JSONB, error TEXT
);
CREATE TABLE sources (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    kind TEXT NOT NULL CHECK(kind IN ('quarterly','daily')),
    period TEXT NOT NULL, url TEXT NOT NULL, current_artifact_id BIGINT,
    UNIQUE(kind, period)
);
CREATE TABLE artifacts (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_id BIGINT NOT NULL REFERENCES sources(id),
    sha256 TEXT NOT NULL CHECK(sha256 ~ '^[0-9a-f]{64}$'),
    relative_path TEXT NOT NULL, bytes BIGINT NOT NULL CHECK(bytes >= 0),
    first_observed_at TIMESTAMPTZ NOT NULL, last_observed_at TIMESTAMPTZ NOT NULL,
    total_rows BIGINT NOT NULL CHECK(total_rows >= 0),
    ownership_rows BIGINT NOT NULL CHECK(ownership_rows >= 0 AND ownership_rows <= total_rows),
    min_filed_date DATE, max_filed_date DATE,
    CHECK(min_filed_date <= max_filed_date),
    UNIQUE(source_id, sha256), UNIQUE(source_id, id)
);
-- A source can only activate one of its own artifacts.
ALTER TABLE sources ADD CONSTRAINT sources_current_artifact_fk
    FOREIGN KEY(id, current_artifact_id) REFERENCES artifacts(source_id, id);
CREATE TABLE attempts (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id),
    source_id BIGINT NOT NULL REFERENCES sources(id), started_at TIMESTAMPTZ NOT NULL,
    finished_at TIMESTAMPTZ, status TEXT NOT NULL CHECK(status IN
        ('running','validated','failed','unavailable','interrupted')),
    artifact_id BIGINT, http_status INTEGER,
    response_json JSONB, error TEXT,
    FOREIGN KEY(source_id, artifact_id) REFERENCES artifacts(source_id, id)
);
CREATE TABLE filings (
    accession TEXT PRIMARY KEY, first_seen_at TIMESTAMPTZ NOT NULL, last_seen_at TIMESTAMPTZ NOT NULL
);
CREATE TABLE associations (
    artifact_id BIGINT NOT NULL REFERENCES artifacts(id), row_number BIGINT NOT NULL CHECK(row_number > 0),
    accession TEXT NOT NULL REFERENCES filings(accession), indexed_cik TEXT NOT NULL,
    company_name TEXT NOT NULL, form_type TEXT NOT NULL CHECK(form_type IN
        ('3','3/A','4','4/A','5','5/A')), filed_date DATE NOT NULL,
    archive_path TEXT NOT NULL,
    PRIMARY KEY(artifact_id, row_number)
);
CREATE INDEX association_accession ON associations(accession);
CREATE INDEX association_date ON associations(filed_date);
CREATE INDEX association_cik ON associations(indexed_cik);
CREATE INDEX attempts_source ON attempts(source_id, id);
CREATE VIEW current_associations AS
    SELECT a.* FROM associations a JOIN sources s ON s.current_artifact_id = a.artifact_id;
INSERT INTO metadata(key,value) VALUES ('schema_version','1');

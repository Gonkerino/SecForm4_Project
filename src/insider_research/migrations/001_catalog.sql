CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE runs (
    id TEXT PRIMARY KEY, command TEXT NOT NULL, started_at TEXT NOT NULL,
    finished_at TEXT, status TEXT NOT NULL CHECK(status IN
        ('running','succeeded','incomplete','interrupted','failed')),
    settings_json TEXT NOT NULL, summary_json TEXT, error TEXT
);
CREATE TABLE sources (
    id INTEGER PRIMARY KEY, kind TEXT NOT NULL CHECK(kind IN ('quarterly','daily')),
    period TEXT NOT NULL, url TEXT NOT NULL, current_artifact_id INTEGER,
    UNIQUE(kind, period),
    FOREIGN KEY(current_artifact_id) REFERENCES artifacts(id)
);
CREATE TABLE artifacts (
    id INTEGER PRIMARY KEY, source_id INTEGER NOT NULL REFERENCES sources(id),
    sha256 TEXT NOT NULL, relative_path TEXT NOT NULL, bytes INTEGER NOT NULL,
    first_observed_at TEXT NOT NULL, last_observed_at TEXT NOT NULL,
    total_rows INTEGER NOT NULL, ownership_rows INTEGER NOT NULL,
    min_filed_date TEXT, max_filed_date TEXT,
    UNIQUE(source_id, sha256)
);
CREATE TABLE attempts (
    id INTEGER PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id),
    source_id INTEGER NOT NULL REFERENCES sources(id), started_at TEXT NOT NULL,
    finished_at TEXT, status TEXT NOT NULL CHECK(status IN
        ('running','validated','failed','unavailable','interrupted')),
    artifact_id INTEGER REFERENCES artifacts(id), http_status INTEGER,
    response_json TEXT, error TEXT
);
CREATE TABLE filings (
    accession TEXT PRIMARY KEY, first_seen_at TEXT NOT NULL, last_seen_at TEXT NOT NULL
);
CREATE TABLE associations (
    artifact_id INTEGER NOT NULL REFERENCES artifacts(id), row_number INTEGER NOT NULL,
    accession TEXT NOT NULL REFERENCES filings(accession), indexed_cik TEXT NOT NULL,
    company_name TEXT NOT NULL, form_type TEXT NOT NULL CHECK(form_type IN
        ('3','3/A','4','4/A','5','5/A')), filed_date TEXT NOT NULL,
    archive_path TEXT NOT NULL,
    PRIMARY KEY(artifact_id, row_number)
);
CREATE INDEX association_accession ON associations(accession);
CREATE INDEX association_date ON associations(filed_date);
CREATE INDEX association_cik ON associations(indexed_cik);
CREATE INDEX attempts_source ON attempts(source_id, id);
CREATE VIEW current_associations AS
    SELECT a.* FROM associations a JOIN sources s ON s.current_artifact_id = a.artifact_id;
PRAGMA user_version = 1;

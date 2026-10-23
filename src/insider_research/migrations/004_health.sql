-- Durable verification receipts, repair audit, and retained extraction revisions.
ALTER TABLE parse_results DROP CONSTRAINT parse_results_accession_source_sha256_parser_version_key;
ALTER TABLE parse_results ADD COLUMN supersedes_result_id BIGINT REFERENCES parse_results(id);
CREATE INDEX parse_results_identity ON parse_results(accession,source_sha256,parser_version);
CREATE TABLE health_checks (
 id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
 started_at TIMESTAMPTZ NOT NULL DEFAULT now(), finished_at TIMESTAMPTZ,
 scope JSONB NOT NULL, status TEXT NOT NULL CHECK(status IN ('running','valid','invalid','interrupted')),
 checked BIGINT NOT NULL DEFAULT 0, problems BIGINT NOT NULL DEFAULT 0
);
CREATE TABLE health_receipts (
 stage TEXT NOT NULL CHECK(stage IN ('discover','download','parse')),
 target TEXT NOT NULL, version TEXT NOT NULL, accession TEXT,
 check_id BIGINT NOT NULL REFERENCES health_checks(id), checked_at TIMESTAMPTZ NOT NULL DEFAULT now(),
 valid BOOLEAN NOT NULL, messages JSONB NOT NULL,
 PRIMARY KEY(stage,target,version)
);
CREATE INDEX health_receipts_failed ON health_receipts(stage,accession) WHERE NOT valid;
CREATE TABLE health_observations (
 id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
 stage TEXT NOT NULL, target TEXT NOT NULL, version TEXT NOT NULL, accession TEXT,
 check_id BIGINT NOT NULL REFERENCES health_checks(id), checked_at TIMESTAMPTZ NOT NULL DEFAULT now(),
 valid BOOLEAN NOT NULL, messages JSONB NOT NULL
);
CREATE INDEX health_observations_identity ON health_observations(stage,target,version,id);
CREATE TABLE repair_attempts (
 id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
 stage TEXT NOT NULL, target TEXT NOT NULL, version TEXT NOT NULL,
 started_at TIMESTAMPTZ NOT NULL DEFAULT now(), finished_at TIMESTAMPTZ,
 status TEXT NOT NULL CHECK(status IN ('running','repaired','already_valid','superseded','failed','needs_review','interrupted')),
 message TEXT, replacement_result_id BIGINT REFERENCES parse_results(id)
);
UPDATE metadata SET value='4' WHERE key='schema_version';

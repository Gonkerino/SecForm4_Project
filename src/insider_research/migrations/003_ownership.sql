-- Local ownership extraction; original catalogue and source files are preserved.
CREATE TABLE parse_attempts (
 id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
 run_id TEXT NOT NULL REFERENCES runs(id), accession TEXT NOT NULL REFERENCES filings(accession),
 source_sha256 TEXT NOT NULL CHECK(source_sha256 ~ '^[0-9a-f]{64}$'), parser_version TEXT NOT NULL,
 started_at TIMESTAMPTZ NOT NULL, finished_at TIMESTAMPTZ,
 status TEXT NOT NULL CHECK(status IN ('running','complete','complete_with_warnings','needs_review','unsupported','failed','interrupted')),
 error TEXT, source_snapshot_path TEXT, source_text_sha256 TEXT,
 UNIQUE(id,accession,source_sha256,parser_version)
);
CREATE INDEX parse_attempts_identity ON parse_attempts(accession,source_sha256,parser_version,id DESC);
CREATE TABLE parse_results (
 id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
 accession TEXT NOT NULL REFERENCES filings(accession), attempt_id BIGINT NOT NULL UNIQUE REFERENCES parse_attempts(id),
 source_sha256 TEXT NOT NULL CHECK(source_sha256 ~ '^[0-9a-f]{64}$'),
 source_text_sha256 TEXT NOT NULL CHECK(source_text_sha256 ~ '^[0-9a-f]{64}$'),
 source_snapshot_path TEXT NOT NULL, parser_version TEXT NOT NULL, parsed_at TIMESTAMPTZ NOT NULL,
 status TEXT NOT NULL CHECK(status IN ('complete','complete_with_warnings','needs_review')),
 counts_json JSONB NOT NULL, UNIQUE(accession,source_sha256,parser_version), UNIQUE(accession,id),
 FOREIGN KEY(attempt_id,accession,source_sha256,parser_version)
 REFERENCES parse_attempts(id,accession,source_sha256,parser_version)
);
CREATE TABLE ownership_current_results (
 accession TEXT PRIMARY KEY REFERENCES filings(accession), result_id BIGINT NOT NULL UNIQUE,
 FOREIGN KEY(accession,result_id) REFERENCES parse_results(accession,id)
);
CREATE TABLE submission_headers (
 result_id BIGINT PRIMARY KEY REFERENCES parse_results(id), accession TEXT NOT NULL,
 raw_header TEXT NOT NULL, form_type TEXT, acceptance_raw TEXT, acceptance_at TIMESTAMPTZ,
 acceptance_method TEXT NOT NULL, filed_date DATE, period_of_report DATE, public_document_count INTEGER
);
CREATE TABLE submission_documents (
 result_id BIGINT NOT NULL REFERENCES parse_results(id), document_number INTEGER NOT NULL,
 sequence TEXT, document_type TEXT, filename TEXT, description TEXT, raw_metadata TEXT NOT NULL,
 byte_offset BIGINT NOT NULL CHECK(byte_offset>=0), byte_length BIGINT NOT NULL CHECK(byte_length>=0),
 sha256 TEXT NOT NULL CHECK(sha256 ~ '^[0-9a-f]{64}$'), PRIMARY KEY(result_id,document_number)
);
CREATE TABLE submission_header_fields (
 result_id BIGINT NOT NULL REFERENCES parse_results(id), line_number INTEGER NOT NULL,
 section_path TEXT NOT NULL, field_name TEXT, value TEXT NOT NULL, raw_line TEXT NOT NULL,
 PRIMARY KEY(result_id,line_number)
);
CREATE TABLE ownership_reports (
 result_id BIGINT PRIMARY KEY REFERENCES parse_results(id), document_number INTEGER NOT NULL,
 form_type TEXT,
 schema_version TEXT,
 period_of_report DATE,
 date_of_original_submission DATE,
 issuer_cik TEXT,
 issuer_name TEXT,
 issuer_trading_symbol TEXT,
 issuer_foreign_trading_symbol TEXT,
 no_securities_owned BOOLEAN,
 not_subject_to_section16 BOOLEAN,
 form3_holdings_reported BOOLEAN,
 form4_transactions_reported BOOLEAN,
 aff10b5_one BOOLEAN,
 remarks TEXT,
 CHECK(form_type IN ('3','3/A','4','4/A','5','5/A')),
 FOREIGN KEY(result_id,document_number) REFERENCES submission_documents(result_id,document_number)
);
CREATE TABLE ownership_reporting_owners (
 result_id BIGINT NOT NULL REFERENCES ownership_reports(result_id), row_number INTEGER NOT NULL CHECK(row_number>0),
 source_path TEXT NOT NULL,
 owner_cik TEXT,
 owner_name TEXT,
 street1 TEXT,
 street2 TEXT,
 city TEXT,
 state TEXT,
 zip_code TEXT,
 state_description TEXT,
 non_us_address BOOLEAN,
 non_us_state_territory TEXT,
 country TEXT,
 is_director BOOLEAN,
 is_officer BOOLEAN,
 is_ten_percent_owner BOOLEAN,
 is_other BOOLEAN,
 officer_title TEXT,
 other_text TEXT,
 PRIMARY KEY(result_id,row_number), UNIQUE(result_id,source_path)
);
CREATE TABLE ownership_transactions (
 result_id BIGINT NOT NULL REFERENCES ownership_reports(result_id), row_number INTEGER NOT NULL CHECK(row_number>0),
 source_path TEXT NOT NULL,
 is_derivative BOOLEAN NOT NULL,
 security_title TEXT,
 transaction_form_type TEXT,
 shares_owned_following NUMERIC,
 value_owned_following NUMERIC,
 direct_or_indirect TEXT,
 nature_of_ownership TEXT,
 conversion_or_exercise_price NUMERIC,
 exercise_date DATE,
 expiration_date DATE,
 underlying_security_title TEXT,
 underlying_security_shares NUMERIC,
 underlying_security_value NUMERIC,
 transaction_date DATE,
 deemed_execution_date DATE,
 transaction_code TEXT,
 equity_swap_involved BOOLEAN,
 transaction_timeliness TEXT,
 transaction_shares NUMERIC,
 transaction_price_per_share NUMERIC,
 transaction_total_value NUMERIC,
 acquired_disposed_code TEXT,
 PRIMARY KEY(result_id,row_number), UNIQUE(result_id,source_path)
);
CREATE TABLE ownership_holdings (
 result_id BIGINT NOT NULL REFERENCES ownership_reports(result_id), row_number INTEGER NOT NULL CHECK(row_number>0),
 source_path TEXT NOT NULL,
 is_derivative BOOLEAN NOT NULL,
 security_title TEXT,
 transaction_form_type TEXT,
 shares_owned_following NUMERIC,
 value_owned_following NUMERIC,
 direct_or_indirect TEXT,
 nature_of_ownership TEXT,
 conversion_or_exercise_price NUMERIC,
 exercise_date DATE,
 expiration_date DATE,
 underlying_security_title TEXT,
 underlying_security_shares NUMERIC,
 underlying_security_value NUMERIC,
 PRIMARY KEY(result_id,row_number), UNIQUE(result_id,source_path)
);
CREATE TABLE ownership_signatures (
 result_id BIGINT NOT NULL REFERENCES ownership_reports(result_id), row_number INTEGER NOT NULL CHECK(row_number>0),
 source_path TEXT NOT NULL,
 signature_name TEXT,
 signature_date DATE,
 PRIMARY KEY(result_id,row_number), UNIQUE(result_id,source_path)
);
CREATE TABLE ownership_footnotes (
 result_id BIGINT NOT NULL REFERENCES ownership_reports(result_id), row_number INTEGER NOT NULL,
 source_path TEXT NOT NULL, footnote_id TEXT, text TEXT,
 PRIMARY KEY(result_id,row_number), UNIQUE(result_id,source_path)
);
CREATE TABLE ownership_fields (
 result_id BIGINT NOT NULL REFERENCES ownership_reports(result_id), source_path TEXT NOT NULL,
 tag TEXT NOT NULL, text TEXT, tail TEXT, attributes JSONB NOT NULL,
 value_state TEXT NOT NULL CHECK(value_state IN ('value','empty','container','footnote_only')),
 mapped BOOLEAN NOT NULL, PRIMARY KEY(result_id,source_path)
);
CREATE TABLE ownership_field_notes (
 result_id BIGINT NOT NULL REFERENCES ownership_reports(result_id), source_path TEXT NOT NULL,
 field_path TEXT NOT NULL, footnote_id TEXT, PRIMARY KEY(result_id,source_path),
 FOREIGN KEY(result_id,source_path) REFERENCES ownership_fields(result_id,source_path),
 FOREIGN KEY(result_id,field_path) REFERENCES ownership_fields(result_id,source_path)
);
CREATE TABLE parse_issues (
 result_id BIGINT NOT NULL REFERENCES parse_results(id), issue_number INTEGER NOT NULL,
 severity TEXT NOT NULL CHECK(severity IN ('warning','review')), code TEXT NOT NULL,
 source_path TEXT NOT NULL, message TEXT NOT NULL, PRIMARY KEY(result_id,issue_number)
);
CREATE INDEX ownership_issuer ON ownership_reports(issuer_cik,period_of_report);
CREATE INDEX ownership_owner ON ownership_reporting_owners(owner_cik);
CREATE INDEX ownership_transaction_date ON ownership_transactions(transaction_date);
CREATE INDEX parse_issue_code ON parse_issues(code,severity);
CREATE VIEW current_ownership_reports AS SELECT r.accession,t.*,p.status AS extraction_status,p.parser_version AS extraction_parser_version FROM ownership_reports t JOIN ownership_current_results r ON r.result_id=t.result_id JOIN parse_results p ON p.id=t.result_id;
CREATE VIEW current_ownership_reporting_owners AS SELECT r.accession,t.*,p.status AS extraction_status,p.parser_version AS extraction_parser_version FROM ownership_reporting_owners t JOIN ownership_current_results r ON r.result_id=t.result_id JOIN parse_results p ON p.id=t.result_id;
CREATE VIEW current_ownership_transactions AS SELECT r.accession,t.*,p.status AS extraction_status,p.parser_version AS extraction_parser_version FROM ownership_transactions t JOIN ownership_current_results r ON r.result_id=t.result_id JOIN parse_results p ON p.id=t.result_id;
CREATE VIEW current_ownership_holdings AS SELECT r.accession,t.*,p.status AS extraction_status,p.parser_version AS extraction_parser_version FROM ownership_holdings t JOIN ownership_current_results r ON r.result_id=t.result_id JOIN parse_results p ON p.id=t.result_id;
CREATE VIEW current_ownership_footnotes AS SELECT r.accession,t.*,p.status AS extraction_status,p.parser_version AS extraction_parser_version FROM ownership_footnotes t JOIN ownership_current_results r ON r.result_id=t.result_id JOIN parse_results p ON p.id=t.result_id;
CREATE VIEW current_ownership_field_notes AS SELECT r.accession,t.*,p.status AS extraction_status,p.parser_version AS extraction_parser_version FROM ownership_field_notes t JOIN ownership_current_results r ON r.result_id=t.result_id JOIN parse_results p ON p.id=t.result_id;
CREATE VIEW current_ownership_signatures AS SELECT r.accession,t.*,p.status AS extraction_status,p.parser_version AS extraction_parser_version FROM ownership_signatures t JOIN ownership_current_results r ON r.result_id=t.result_id JOIN parse_results p ON p.id=t.result_id;
UPDATE metadata SET value='3' WHERE key='schema_version';

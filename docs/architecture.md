# Filing catalogue — PostgreSQL design

The catalogue tracks Forms 3, 3/A, 4, 4/A, 5 and 5/A from original full/daily EDGAR master indexes. Its records describe filing discovery and source provenance. PostgreSQL is the sole operational backend; migration does not import previous datasets.

## Storage and schema

Configuration defaults to `config/default.toml` located relative to the installed source checkout, independent of the current working directory. An explicit `--config` replaces that selection; environment overrides still apply. Without a checkout default, built-in settings apply.

Psycopg 3 connects to an existing PostgreSQL database using `database_url` (environment override: `INSIDER_DATABASE_URL`). Passwords may come from libpq's password file or `PGPASSWORD`. Connection strings are excluded from configuration repr/public settings/reports; CLI database errors are redacted.

All application objects live in a dedicated `database_schema`, default `insider_research`. Initialization obtains a database advisory transaction lock, creates an empty schema transactionally, and records `schema_version` and the bound `data_root` in `metadata`. It rejects unknown versions, unrelated nonempty schemas and conflicting data-root bindings. The current schema version is 2. `init` upgrades version 1 transactionally using `002_submissions.sql`, adding submission tables while preserving existing index records; ordinary catalogue connections require version 2. Repeated initialization preserves records. Server/role/database provisioning is separate in `scripts/create_catalogue.sql`.

- `filings`: one accession identity and first/last observation times; no guessed issuer.
- `sources`: kind/period, URL and current validated artifact pointer.
- `artifacts`: source/content-hash revisions, lengths, locations, parsed counts and observation times.
- `associations`: source row, accession, indexed CIK/name, form, filing date and archive path.
- `runs`: command execution, safe settings, status and summary.
- `attempts`: source-level attempts, HTTP status and response metadata.
- `submissions`: one verified full-submission ZIP per accession, including archive checksum, lengths and source URL.
- `submission_attempts`: per-filing download history and recoverable running/failed states.
- `current_associations`: selects active source revisions; queries deduplicate daily/quarterly overlap.

Identifiers use generated BIGINT identities where appropriate; dates use DATE, observation times use TIMESTAMPTZ, and structured settings/receipts use JSONB. Primary/foreign keys, uniqueness and CHECK constraints reject invalid writes. Composite foreign keys ensure a source's active artifact and an attempt's artifact belong to that source. Source revisions, associations and activation commit together only after their artifact exists. Typed dates are serialized as ISO strings at the CLI boundary.

Connections use autocommit for standalone statements and explicit transaction blocks for compound writes. Sessions set UTC and synchronous commit. Status counts share one statement snapshot. Validation holds a repeatable-read, read-only snapshot. PostgreSQL durability also depends on normal server durability settings, storage and backups.

## Publication, locking and recovery

An OS file lock protects the local artifact directory. A PostgreSQL session advisory lock, keyed by application schema and scoped to the database, prevents another cooperating publisher even if it uses a different local lock path. Initialization uses the same key. OS/session termination releases locks. Discovery and submission commands share the publisher locks, so only one cooperating publishing command runs at a time. Within a submission run, worker threads fetch and publish separate accession ZIPs; the coordinating thread owns the catalogue connection and commits attempt/archive records.

`recover` holds both locks before marking abandoned discovery and submission records interrupted. Previously committed revisions remain available. Filesystem publication and PostgreSQL commit are separate: a failure between them may leave an unregistered artifact, which remains invisible to readers. Submission retries replace deterministic per-accession ZIP paths only after validating their contents; index orphan adoption remains future work.

`validate` checks source artifacts and association counts and reports unvalidated PostgreSQL constraints and unresolved source periods. Database constraints enforce referential integrity on writes; validation is not a physical PostgreSQL integrity scan.

Preserved source bytes remain SHA-256-addressed gzip files beneath `raw/indexes/`. Full SEC submission text is retained byte-for-byte inside per-accession ZIPs beneath `raw/submissions/`; the database stores the ZIP SHA-256 and ZIP/source lengths. A source SHA-256 is calculated during publication to verify the ZIP contents against the fetched bytes, but is not stored separately. Failed diagnostic responses never activate index coverage, and invalid submission responses never publish a ZIP. A revision removing accessions requires explicit `--accept-removals`; a missing row does not itself prove legal withdrawal. Historical observations remain preserved.

## Submission selection and verification

Submission candidates come from `current_associations`, deduplicated by accession and ordered by earliest matching filing date, then accession. Optional start/end/form filters apply to current index observations. `--limit` accepts 1–10,000; omitting it materializes all eligible candidates for the run. Already registered submissions are excluded. Latest failed attempts are excluded unless `--retry-failed` is supplied; that flag also retains other eligible candidates.

The default worker count is 10, configurable from 1 to 20. Workers share one HTTP client and request-start limiter, including retries and redirects. HTTP 403 or a final 429 failure prevents further dispatch; in-flight workers finish. The coordinator records each success or failure independently. A successful limited batch can leave `coverage_complete` false; coverage describes matching entries in the current catalogue, not all SEC filings.

Before writing a ZIP, publication rejects empty/null-containing responses and requires SEC document/header markers plus the requested accession in the first 64 KiB. This is a response sanity check rather than full SGML parsing or proof of document completeness. ZIP publication uses fixed entry metadata, DEFLATE compression, a unique temporary file, file fsync, content verification and atomic replacement. Registration and the successful attempt state then commit together.

`validate-submissions` checks registered archive paths stay inside the data root, verifies the single expected entry, reads it fully, and compares ZIP SHA-256 and ZIP/source lengths with PostgreSQL. It does not establish download coverage, adopt orphan ZIPs or update `last_verified_at`. Downloads skip registered archives without a fresh disk check; damaged archives require explicit repair outside the current automatic retry flow.

## Source behavior

Quarterly planning includes full intersecting quarters. Daily planning skips weekends, not inferred holidays. All requested periods are registered before fetching, so access blocking does not hide unattempted periods. Missing/failed sources remain unresolved.

Requests use the configured `User-Agent: Malcolm MacLeod s2247907@ed.ac.uk`, one thread-safe request-start scheduler, bounded retries/bytes/time and HTTPS SEC-only redirects. Submission downloads use ten workers to hide network latency while sharing the SEC's 10 requests/second ceiling. That rate limit is aggregate across processes and machines, so other SEC clients must be counted too. The header follows the SEC's [declared-header example](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data). `SEC_USER_AGENT` overrides the TOML value. HTTP-200 block pages fail parsing. Detailed intermediate HTTP retry history remains future work.

`plan` needs neither database nor SEC access and writes nothing. `--offline` prohibits SEC requests; cache import/replay still connects to PostgreSQL. No command silently creates a SQLite catalogue or falls back if PostgreSQL is unavailable.

## Operation and boundaries

Code and the ignored data root live under `D:/Projects/SecForm4`; files default to `D:/Projects/SecForm4/data`. PostgreSQL stores the catalogue separately. The data-root `catalog/` directory now holds the local publisher lock, not a database file. Back up PostgreSQL with its database backup tools and preserve the artifact directory too. Automatic data-root relocation and backup orchestration are not implemented.

This Windows account has a per-user PostgreSQL 18.6 installation at `%LOCALAPPDATA%/Programs/PostgreSQL/18`, with a cluster at `D:/Projects/SecForm4/data/postgres`. It listens on `127.0.0.1:5432` with SCRAM authentication. A user Startup shortcut starts it in the background at login; `scripts/postgres.py` controls it manually. Passwords are stored in the account's protected libpq password file, never in repository configuration. The application and test databases have separate non-superuser owner logins; the test login cannot connect to the working database.

Integration tests use a separately configured PostgreSQL test database, randomly named schemas and temporary artifact roots. Integration fixtures are isolated from the working application schema. Ownership parsing, analytical snapshots, market data, models, signals and scheduling remain outside current functionality.

## Local dashboard

`insider dashboard` runs a read-only standard-library `ThreadingHTTPServer` on `127.0.0.1` (port 8765 by default). The bundled HTML contains a single monthly coverage strip. GET `/api/coverage` returns every month from July 2003 through the present in SEC/New York time. POST requests are rejected; the previous action, inspection and listing routes are removed. CLI operations remain available independently.

One repeatable-read, read-only PostgreSQL transaction aggregates current associations by accession/month and joins registered submissions. Distinct accessions avoid double-counting entities and daily/quarterly overlaps. The active quarterly artifact's most recent validated online attempt (or first observation for an imported artifact) establishes whether a month has a known index denominator. Offline cache replay cannot advance this date. Only months whose index was observed after month end can be complete; daily-only or absent coverage and the open month retain an unknown expected count. Zero downloaded filings always produce an empty/red state, including a known zero-filing index; only a positive, fully downloaded count can be green. Registered archive counts do not revalidate disk contents.

The browser refreshes every ten seconds while visible and exposes monthly counts on hover, focus or tap. Arrow keys, Home and End navigate the strip. There are no background jobs or operation controls in the dashboard. The server checks Host headers, uses bundled assets with a restrictive content policy, and requires no new dependencies or schema migrations. It is intended for local account use, not remote hosting.

The main research dataset starts on **1 July 2003**, the first full month after electronic ownership filing became mandatory on 30 June 2003 ([SEC rule](https://www.sec.gov/files/rules/final/33-8230.htm)). `config.RESEARCH_START` is the shared boundary for the dashboard, default CLI filing listings/downloads and the Python submission-download entry point. Explicit earlier CLI dates remain supported. Status, validation and low-level catalogue queries cover all retained records. No historical observations or files are deleted.

The general index planner retains its July 1994 lower bound and requires explicit dates; use a start of 2003-07-01 for principal research discovery. Voluntary electronic ownership filing began on 18 December 1995 ([SEC announcement](https://www.sec.gov/news/digest/1995/dig111495.pdf)); pre-mandate electronic records omit paper reports and remain supplementary archival data. Future forecasting datasets must use public information-availability timing to avoid look-ahead bias. Transaction parsing, publication-time research snapshots, market-price alignment and statistical evaluation remain outside current functionality.

Unindexed months have a separate grey state: no monthly filing observations or applicable active index source. A validated daily source or a quarterly observation reaching the month establishes index presence even for zero ownership rows; it does not necessarily establish full-month coverage. The coverage endpoint also returns the latest validated index attempt timestamp/source, labelled when offline. Failed attempts and submission downloads do not advance this timestamp. The UI separately timestamps its last successful fetch and renders both times in Europe/London.

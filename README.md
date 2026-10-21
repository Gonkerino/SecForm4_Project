# Insider Research

SEC ownership research infrastructure for Forms **3, 3/A, 4, 4/A, 5 and 5/A**. The current milestone is a **PostgreSQL filing catalogue**: discover filings from original EDGAR master indexes, preserve source bytes, and track accession identities separately from indexed entity/path associations.

**Implemented:** initialization, quarterly/daily discovery, offline index import/replay, revision history, filing-level full submission downloads in verified ZIP archives, retry/recovery tracking, coverage, queries and integrity validation. **Not yet implemented:** ownership parsing, analytical snapshots, prices, models, signals or scheduling. Discovery downloads index metadata only; submission downloads run as a separate explicit command.

The application uses PostgreSQL exclusively. There is no SQLite fallback or automatic import of old data. Initialization creates an empty schema and never downloads filings; repeating it preserves existing records.

## Installed on this Windows account

PostgreSQL **18.6** is installed and configured for this account. The working catalogue has been initialized; use `status` and `coverage` to check its current contents. No further installation or password entry is needed for normal use. Sections 1–5 below describe setting up a fresh machine.

- Server: `127.0.0.1:5432`, accepting local connections with SCRAM password authentication.
- Binaries: `%LOCALAPPDATA%/Programs/PostgreSQL/18`.
- PostgreSQL data directory: `D:/Projects/SecForm4/data/postgres`.
- Working database: `insider_research`, login `insider`.
- Separate test database: `insider_research_test`, login `insider_test`. That login cannot connect to the working database.
- Passwords: `%APPDATA%/postgresql/pgpass.conf`, outside the repository, in a directory restricted to this account and SYSTEM.
- Automatic startup: `Insider Research PostgreSQL.lnk` in this account's Windows Startup folder. The server starts in the background at login; it runs as this user.

This account's user PATH includes `D:/Projects/SecForm4/.venv/Scripts`, so `insider` is available without activating the environment. Existing terminals may need their PATH refreshed, or the terminal application restarted. To refresh the current PowerShell session:

```powershell
$env:Path = 'D:\Projects\SecForm4\.venv\Scripts;' + $env:Path
```

On a fresh machine, activate the environment with `.\.venv\Scripts\Activate.ps1` to make `insider` available by name for that session, or invoke `.venv/Scripts/insider.exe` directly.

Check the catalogue from any directory:

```powershell
insider status
insider validate
```

If needed, manage the server:

```powershell
.venv/Scripts/python.exe scripts/postgres.py status
.venv/Scripts/python.exe scripts/postgres.py start
.venv/Scripts/python.exe scripts/postgres.py restart
.venv/Scripts/python.exe scripts/postgres.py stop
```

`start` is safe to repeat. Stop/restart disconnect active database sessions. Server messages are in `D:/Projects/SecForm4/data/postgres/server.log`. The helper reads the installation paths from `%LOCALAPPDATA%/Programs/PostgreSQL/18/insider-research-install.json`; it does not contain passwords.

For an administrator SQL session using the saved password:

```powershell
& "$env:LOCALAPPDATA/Programs/PostgreSQL/18/bin/psql.exe" -h 127.0.0.1 -p 5432 -U postgres -d postgres
```

## 1. Install Python dependencies

Python 3.12+ and PostgreSQL 16+ are required. The migration was tested against PostgreSQL 18.6. This checkout already has a Python environment; update it from PowerShell:

```powershell
Set-Location D:/Projects/SecForm4
.venv/Scripts/python.exe -m pip install --no-build-isolation -e .
```

To recreate the environment:

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-build.txt
.venv/Scripts/python.exe -m pip install --no-build-isolation -e .
```

Psycopg's pinned binary distribution includes its PostgreSQL client library. The PostgreSQL server is installed separately.

## 2. Install PostgreSQL on Windows

Use the installer linked from the official [PostgreSQL Windows download page](https://www.postgresql.org/download/windows/). Choose PostgreSQL 18, install the server and command-line tools, and keep port **5432** unless it is already in use. Record the administrator password you choose for the `postgres` account. pgAdmin is optional; Stack Builder is not required for this application.

The installer normally creates a Windows service. Ensure it is running before using catalogue commands. The application can use a local server; cloud hosting and Docker are optional.

## 3. Create the application role and database once

From the project folder, run the following, adjusting `18` if your installation uses another supported version:

```powershell
& 'C:/Program Files/PostgreSQL/18/bin/psql.exe' -h 127.0.0.1 -p 5432 -U postgres -d postgres -W -f scripts/create_catalogue.sql
```

You will enter the administrator password, then choose and confirm a separate password for the application role **insider**. The script creates that login and the database **insider_research**, owned by it. The application role is not a superuser and cannot create roles or databases.

This is a one-time provisioning script. It stops if the role or database already exists and does not reset anything. If interrupted, use psql/pgAdmin to inspect what was created and complete the remaining statement; do not drop a database to rerun the script.

The script provisions the database only. The application's tables are created by `insider init`.

## 4. Supply the application password

For the current PowerShell session:

```powershell
$dbCredential = Get-Credential -UserName insider -Message 'Enter the insider database password'
$env:PGPASSWORD = $dbCredential.GetNetworkCredential().Password
Remove-Variable dbCredential
```

For persistent local use, PostgreSQL also supports `%APPDATA%/postgresql/pgpass.conf`. A matching line has this format:

```text
127.0.0.1:5432:insider_research:insider:YOUR_DATABASE_PASSWORD
```

Keep this file outside the repository and follow PostgreSQL's [password-file rules](https://www.postgresql.org/docs/current/libpq-pgpass.html), including escaping colons and backslashes in passwords. `PGPASSWORD`, if set, takes precedence; remove it when switching to the password file:

```powershell
Remove-Item Env:PGPASSWORD -ErrorAction SilentlyContinue
```

## 5. Initialize and confirm the empty catalogue

The committed `config/default.toml` contains a password-free connection string:

```toml
data_root = "D:/Projects/SecForm4/data"
database_url = "host=127.0.0.1 port=5432 dbname=insider_research user=insider"
database_schema = "insider_research"
user_agent = "Malcolm MacLeod s2247907@ed.ac.uk"
```

Then run:

```powershell
insider init
insider status
insider coverage
insider validate
```

A fresh catalogue reports zero filings, associations, source revisions and sources; `coverage` is empty and `validate` is valid with zero artifacts checked. Fresh initialization inserts schema/data-root metadata only. **Stop here to keep it empty.** The examples below import or discover data.

The schema is dedicated to this application and bound to its data-root path. Initialization refuses unrelated nonempty schemas, unrecognized nonempty directories, and using the same catalogue with another data root. It never creates the PostgreSQL server, login or database itself.

## Ownership filing history and research period

The project's main research dataset starts on **1 July 2003**. Electronic filing of Forms 3, 4 and 5 became mandatory on **30 June 2003**, making July the first full month under that requirement. This avoids using the earlier voluntary electronic sample as the main forecasting dataset. See the [SEC's mandatory electronic filing rule](https://www.sec.gov/files/rules/final/33-8230.htm).

The dashboard begins in July 2003. `filings` and `download-submissions` default to `--start 2003-07-01`; the Python submission-download function uses the same default. Explicit earlier start dates remain supported for archival investigations. `status` and integrity checks cover the whole catalogue, including retained earlier records. The dashboard's green means registered downloads match the loaded index.

Discovery requires explicit dates and still accepts the general index range from July 1994. Use `--start 2003-07-01` for the principal research period. Whole-quarter discovery may include dates outside a partial-quarter selection. Earlier catalogue records and downloaded files are retained.

Voluntary electronic ownership filing became effective on 18 December 1995 ([SEC announcement](https://www.sec.gov/news/digest/1995/dig111495.pdf)). Pre-mandate electronic records omit paper reports and remain supplementary archival data. Forecasting must use when information became publicly available, rather than treating a transaction date as the date the market knew about it. Transaction parsing, publication-time research snapshots, market prices and statistical evaluation remain future work.

## Local dashboard

Run `insider dashboard` to open a read-only monthly coverage strip at `http://127.0.0.1:8765`. Keep the terminal running; Ctrl+C stops it. Use `--port 8766` for another port or `--no-browser` to serve without opening a tab. Configuration is shared with the CLI.

One bar represents each month from July 2003 (the first full month of mandatory electronic ownership filing) through the current SEC/New York month. Hover, tap or use arrow keys to see unique indexed filings, registered submission downloads, the expected total and remaining count when known. All six ownership forms are included; duplicate indexed entities and overlapping sources count once per accession/month. Counts refresh every ten seconds while visible. The strip opens at the newest months and scrolls horizontally on narrower screens. Bar height scales linearly with the number of unique indexed filings in that month, relative to the largest month across the entire timeline. Heights stay comparable while scrolling. A four-pixel minimum keeps empty or very small months available for inspection; green fill still represents the downloaded fraction.

Grey means no index has been loaded for that month. Red means an index is loaded but no submissions are downloaded; partial months fill green from the bottom in proportion to downloaded/expected filings, with red above; amber indicates partial downloads with an unknown expected total; green means all expected submissions in a closed month are registered. An expected total is available only with a quarterly index observed after that month ended. Offline cache replay does not advance the index freshness date. Missing/daily-only/stale indexes leave the total unknown; the current month remains provisional. A month with zero downloaded filings stays red, even when its loaded index contains zero ownership filings. Completeness is relative to the loaded index, not an independent SEC census or fresh disk-integrity check. Use the CLI for downloading, discovery, validation and recovery; the dashboard exposes no write actions.

The Last indexed panel shows the most recent successful index load and its source period, including local imports/cache replays labelled as such. Failed attempts and submission downloads do not advance it. View refreshed shows the last successful dashboard fetch; both timestamps display London time.

The server binds only to loopback, checks request hosts, serves bundled assets, and requires no additional dependencies. Restart an already-running dashboard after code updates.

## Upgrading an existing catalogue

The current catalogue schema is **version 2**. After updating the Python package, run `insider --config config/default.toml init` again. This applies `002_submissions.sql` to a version-1 PostgreSQL catalogue, adding submission archives and attempt tracking while preserving index records. Other commands require version 2 and do not migrate automatically. Unknown schema versions are rejected. Back up the database and artifacts before an upgrade.

## Configuration and identity

The installed editable checkout automatically loads its own `config/default.toml`, even when invoked from another directory. Use `--config C:/path/to/custom.toml` to select a different file. If the source checkout has no default file (for example, a standalone wheel installation), built-in settings and environment variables apply. Global options (`--config`, `--data-root`, `--offline`) go before the subcommand.

- Data-root precedence: command line, `INSIDER_DATA_ROOT`, selected TOML (explicit or automatic), platform default.
- `INSIDER_DATABASE_URL` overrides `database_url`; PostgreSQL URIs and libpq connection strings are accepted.
- `INSIDER_DATABASE_SCHEMA` overrides `database_schema`.
- `SEC_USER_AGENT` overrides `user_agent`.
- Relative TOML data-root paths resolve against the configuration file directory.
- Default HTTP settings are 10 request starts/second, a 45-second socket timeout, four attempts, a 128 MiB index limit and a 64 MiB submission limit. `max_rps` may be lowered; it cannot exceed 10.
- A `plan` command needs no database connection and writes no files.
- `--offline` prevents **SEC requests**; PostgreSQL must still be available, including for imports and cache replay.

Connection strings and the contact identity are excluded from routine reports/public settings. Keep passwords outside committed configuration. Database errors are redacted at the CLI boundary.

The configured request header is `User-Agent: Malcolm MacLeod s2247907@ed.ac.uk`. This follows the identifying-name-and-contact-email example in the SEC's [access guidance](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data). Local validation checks the pattern and rejects control characters; it does not establish email deliverability or guarantee access. The downloader is paced at the SEC's published ceiling of 10 request starts per second. The SEC counts requests across machines and processes, so do not run other SEC download tools at the same time; if 429 responses occur, the client backs off and stops the batch.

## Discovery, import and queries

Preview a selection without contacting PostgreSQL or SEC:

```powershell
insider plan --source quarterly --start 2024-01-01 --end 2024-03-31
```

These commands populate the catalogue:

```powershell
insider discover --source quarterly --start 2024-01-01 --end 2024-03-31
insider discover --source daily --start 2024-01-02 --end 2024-01-02
insider ingest-index --file C:/path/to/master.idx --source quarterly --period 2024Q1
insider --offline discover --source quarterly --start 2024-01-01 --end 2024-03-31
```

Quarterly discovery imports whole intersecting source quarters, including dates outside a partial-quarter selection. Current-quarter coverage reflects the source available when retrieved. Daily planning skips weekends; absent weekday/holiday indexes remain unresolved. Access blocking stops further requests in a run. No scheduler is installed.

Imports accept original plain master-index bytes, not CSV/pickle/company indexes. Supply the correct source period. Offline replay verifies the committed cache and never establishes fresh SEC coverage.

```powershell
insider filings --start 2024-01-01 --end 2024-01-31 --form 4 --limit 20
insider filings --cik 320193 --limit 20
insider inspect 0001234567-24-000001
```

Replace the example accession with an actual result. Output is JSON; dates/timestamps use ISO format, and progress/errors go to stderr. Indexed CIK is not a verified issuer/owner. `filings` returns unique accessions, limited to 100 by default; `inspect` shows historical associations and conflicting current form/date observations.

`status` distinguishes unique filings ever observed, filings in current sources and association counts. These are not transaction counts. `coverage` covers requested periods only. Revised indexes removing accessions require review and an explicit rerun with `--accept-removals`; prior observations remain preserved.

## Recovery, validation and backups

```powershell
insider recover
insider validate
```

`recover` obtains the publisher locks, marks abandoned discovery and submission attempts/runs interrupted, then releases the locks. Repeat the discovery command to retry unresolved sources. Submission downloads skip registered archives and resume unattempted or interrupted filings; use `--retry-failed` to include filings whose latest attempt failed. A ZIP published immediately before a process interruption may be safely replaced on retry; the catalogue records it only after checksum verification.

`validate` uses a repeatable-read, read-only PostgreSQL transaction to check unvalidated constraints, association counts, index hashes/lengths and unresolved periods. `validate-submissions` checks each ZIP checksum, internal file name, decompression and source length against the catalogue. PostgreSQL constraints enforce relationships when writing. These are logical/application checks, not a physical database corruption scan.

## Download full filings

Download complete SEC submission documents for filings already present in the current index catalogue. By default, the command counts matching ownership filings and processes those without registered archives, excluding filings whose latest attempt failed. It uses ten concurrent workers behind the shared 10-request-per-second limiter, and prints the indexed total, downloaded count, remaining count and a live progress bar with ETA in an interactive terminal. Use `--workers` to choose 1–20 concurrent downloads (default 10) and `--limit` to cap a batch at 1–10,000 filings:

```powershell
insider download-submissions --start 2024-01-01 --end 2024-03-31 --form 4 --limit 100
insider download-submissions --start 2024-01-01 --end 2024-03-31
insider download-submissions --retry-failed --limit 100
insider validate-submissions
```

Each archive is stored at `raw/submissions/accession=<first 10 accession digits>/<accession>.zip`, with the original SEC text at the ZIP root as `<accession>.txt`. The raw submission bytes are retained unchanged inside the ZIP. The catalogue records the ZIP SHA-256, ZIP/source lengths, source URL and download time. Downloads use the configured SEC identity, pacing, HTTPS-only SEC redirects, bounded response sizes, timeouts and retries. Before publication, responses must be nonempty, contain no null bytes, and include SEC document/header markers and the requested accession in the first 64 KiB. These checks reject common error responses; they do not fully parse or establish completeness of the SGML document. Files are written to a temporary ZIP, reopened and checksum-verified, then atomically renamed; PostgreSQL marks the filing downloaded only after publication. The publisher lock prevents overlapping writers.

Failed filings remain visible in attempt history and do not count as downloaded. HTTP 403 or a final HTTP 429 failure stops scheduling new downloads; already dispatched workers finish. Rerunning skips verified catalogue entries, resumes unattempted/interrupted filings, and leaves latest failures for an explicit `--retry-failed`. Files published just before a crash but not registered in PostgreSQL are safely replaced on retry. `recover` marks their in-progress attempts interrupted. Downloading the full historical set can take substantial time and disk space; `--limit` and filters let you build coverage in smaller batches. The final JSON reports the indexed total, requested/downloaded/failed counts, remaining count and `coverage_complete` for the selected current catalogue entries. Exit status 0 means the queued batch succeeded; excluded failures or a batch limit can still leave `coverage_complete` false. Registered archives are skipped without rechecking disk contents on every run; use `validate-submissions` to detect missing or damaged ZIPs. This step does not parse ownership transactions or download issuer/owner exhibits separately.

Back up **both** PostgreSQL and the data root: preserved source bytes remain on disk. Use PostgreSQL backup tools such as `pg_dump` for a consistent database backup. Copying `D:/Projects/SecForm4/data` alone does not back up the catalogue. Keep the data-root path when restoring; automatic relocation is not implemented. The project-local `data/` directory is ignored by Git. See [architecture](docs/architecture.md).

Exit codes: 0 success, 1 operational error, 2 incomplete discovery/download batch or failed validation, 64 invalid arguments, 130 interruption. Header-only indexes are unresolved; valid indexes containing other forms but no ownership forms are valid zero-ownership sources.

## Tests

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

HTTP/parser/config tests need no server or SEC access. PostgreSQL integration tests are explicitly skipped unless `INSIDER_TEST_DATABASE_URL` is set. The local installation already has a separate `insider_research_test` database owned by `insider_test`, with its password configured. To run all tests:

```powershell
$env:INSIDER_TEST_DATABASE_URL = 'host=127.0.0.1 port=5432 dbname=insider_research_test user=insider_test'
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

Each integration test uses a random `test_insider_*` schema and a temporary artifact directory, then drops only that schema. Tests never download SEC data or insert fixtures into the application schema. On a fresh machine, provision a separate test database/login with permission to create schemas there and supply its credentials. No skipped integration tests should be treated as PostgreSQL validation.

## CLI progress

Submission validation uses one worker by default with the optimized archive checks. Use
`insider validate-submissions --workers 8` to change concurrency (1–20), or
`--workers 1` for sequential checking. Every run still checks every registered
archive, fully decompresses its expected entry (including ZIP CRC checking),
and compares the archive SHA-256 and compressed/uncompressed lengths with the
catalogue. It does not skip previously checked files or repair damaged files.
Small ZIPs (up to 8 MiB) are read once into memory; larger ZIPs are streamed.
The decompressed source SHA-256 is calculated only during publication, when an
expected source hash is available. Validation streams a stable read-only database
snapshot and bounds queued work instead of loading all submission records into
memory. More workers may not help when storage is the bottleneck.

Discovery/import, submission downloads, index validation and submission validation use
consistent tqdm progress bars in interactive terminals, with counts, elapsed time,
estimated remaining time and processing rate. Validation/download bars also show a
problem count. Bars close on completion, errors and interruption; a stopped batch
retains its actual completed count. Preparation messages appear before database work.
Progress goes to stderr; stdout remains JSON. Redirected stderr uses periodic plain
text updates instead of terminal animation. Commands and filters are unchanged.

# Insider Research

A fresh SEC ownership research platform. The implemented first milestone is a filing catalogue for Forms **3, 3/A, 4, 4/A, 5 and 5/A**. It discovers filings from original EDGAR master indexes, preserves exact source bytes, and records accession identities separately from their indexed entity/path associations.

**Implemented:** initialization, quarterly/daily discovery, offline index import/replay, revision history, coverage, filtering/inspection, integrity validation and abandoned-run recovery. **Not yet implemented:** submission downloads, ownership parsing, analytical snapshots, market prices, models, signals or scheduling. Discovery downloads index metadata only.

See [the current architecture](docs/architecture.md) for the catalogue design. The active application is in `src/insider_research/`; tests and configuration are kept separately.

## Setup

Python 3.12+ is required. This checkout has a Python 3.12 environment at `.venv`. To recreate it:

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-build.txt
.venv/Scripts/python.exe -m pip install --no-build-isolation -e .
```

The package uses Python's standard library plus pinned timezone data on Windows. Build tools and the timezone dependency are pinned. No pandas, database server or API subscription is needed yet. The data root defaults to `D:/Data/InsiderResearch` on Windows and `~/.local/share/insider-research` elsewhere. Use a suitable config/override on other platforms.

```powershell
.venv/Scripts/insider.exe --config config/default.toml init
.venv/Scripts/insider.exe --config config/default.toml status
```

Initialization refuses an unrelated nonempty directory. It can be repeated safely on an initialized root. Code and data are separate.

Online collection requires your real contact identity in the process environment. Replace the example before running; it is not saved in routine reports or the catalogue:

```powershell
$env:SEC_USER_AGENT = 'Your name your-real-contact@example.org'
.venv/Scripts/insider.exe --config config/default.toml plan --source quarterly --start 2024-01-01 --end 2024-03-31
.venv/Scripts/insider.exe --config config/default.toml discover --source quarterly --start 2024-01-01 --end 2024-03-31
```

Data-root precedence: command line, `INSIDER_DATA_ROOT`, explicit TOML, platform default. TOML is read only with `--config`; relative roots resolve against its directory. `SEC_USER_AGENT` overrides a configured identity. Global options (`--config`, `--data-root`, `--offline`) go before the subcommand.

Quarterly discovery imports the **whole source quarter**, including when your selection covers only part of it. Query date filters are separate. A current-quarter index is only as complete as the source available at retrieval time.

## Daily discovery and offline use

```powershell
.venv/Scripts/insider.exe --config config/default.toml discover --source daily --start 2024-01-02 --end 2024-01-02
.venv/Scripts/insider.exe --config config/default.toml coverage
.venv/Scripts/insider.exe --config config/default.toml validate
```

Daily planning skips weekends. Unavailable weekday/holiday indexes remain unresolved. Failed requests leave earlier validated revisions intact, while the latest failure stays visible. Access blocking stops further requests in that run. No scheduler is installed automatically.

Import an original master index without a network call, then replay its committed cache:

```powershell
.venv/Scripts/insider.exe --config config/default.toml ingest-index --file C:/path/to/master.idx --source quarterly --period 2024Q1
.venv/Scripts/insider.exe --config config/default.toml --offline discover --source quarterly --start 2024-01-01 --end 2024-03-31
```

Imports accept plain original master-index bytes, not company indexes, CSV or pickle. The caller supplies the source period; do not label an unrelated index as another period. Offline replay never requests the network and verifies content hashes; missing/corrupt cache means incomplete coverage.

## Query and inspect

```powershell
.venv/Scripts/insider.exe --config config/default.toml filings --start 2024-01-01 --end 2024-01-31 --form 4 --limit 20
.venv/Scripts/insider.exe --config config/default.toml filings --cik 320193 --limit 20
.venv/Scripts/insider.exe --config config/default.toml inspect 0001234567-24-000001
```

Replace the example accession with one returned by `filings`. Results are JSON on stdout; progress/errors go to stderr. `filings` returns unique accessions matching current associations, limited to 100 by default. Indexed CIK is **not a verified issuer or owner**. `inspect` shows source revisions and flags conflicting current form/date observations.

`status` distinguishes unique filings ever observed, filings in current source revisions and association counts. These are not transaction counts. `coverage` lists planned periods, including those left unattempted after an access block, but does not establish coverage for periods never requested.

`latest_validation_offline` identifies cache/import checks. Revalidating cached bytes does not establish current SEC freshness.

A revised source removing accessions requires review. The candidate source and receipt are preserved while the old revision stays current. Rerun with `--accept-removals` to activate reviewed changes. Historical observations remain preserved.

## Integrity and recovery

Source bytes are SHA-256-addressed gzip artifacts. Failed content is retained diagnostically but never activated. One publisher lock prevents concurrent writers. Source associations and activation commit together after the artifact exists. Unchanged reruns do not duplicate records.

```powershell
.venv/Scripts/insider.exe --config config/default.toml recover
.venv/Scripts/insider.exe --config config/default.toml validate
```

Use `recover` after abnormal termination, then repeat the desired discovery command. It marks abandoned runs interrupted under the writer lock. `validate` verifies SQLite integrity/references, artifact hashes/lengths, association counts and unresolved periods. Full orphan recovery and backup commands remain future work. Preserve the whole data root; copying only a live SQLite main file omits important WAL/source state.

Exit codes: 0 success, 1 operational error, 2 incomplete discovery/validation, 64 invalid arguments, 130 interruption. Header-only indexes are unresolved. An index with valid other forms and no ownership forms is a valid zero-ownership result.

## Tests

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

Tests use temporary roots and no network. They cover forms, associations, revisions, removals, malformed inputs, offline behavior, failure boundaries, rate scheduling, locking and CLI operations.

# Operating Insider

The normal workflow is **discover -> download -> parse**. Each stage is explicit;
commands never launch the next stage automatically. Forms 3, 4, 5 and their
amendments are included unless `--form` narrows the selection.

```powershell
insider admin init
insider discover --source quarterly --start 2003-07-01 --end 2026-10-23
insider download --dry-run
insider download
insider parse --dry-run
insider parse
insider status
```

`admin init` upgrades existing schemas transactionally to version 4 without
discarding data. It refuses to run while another publisher owns the database
lock. Finish or interrupt that operation normally before upgrading. Restart
long-lived dashboard processes after upgrading the installed code/schema.

## Commands

| Command | Purpose |
| --- | --- |
| `discover` | Fetch SEC indexes and register filing identities |
| `download` | Fetch pending full submissions into verified ZIPs |
| `parse` | Extract local ownership documents into relational tables |
| `status` | Combine acquisition, extraction, integrity checks and outstanding work |
| `check` | Verify registered data and persist findings |
| `issues` | Browse integrity failures, failed attempts and source review cases |
| `repair` | Recheck and repair recorded integrity failures |
| `show ACCESSION` | Inspect one filing's source, extraction and history |
| `dashboard` | Display monthly download and parsing coverage |
| `admin` | Rare setup, imports, listings, coverage detail and recovery |

Main commands print readable text by default. Add `--json` for machine-readable
output. Progress goes to stderr, leaving stdout suitable for JSON pipelines.
`show --details` exposes all extracted tables and preserved XML/header evidence;
`show --json` always includes that complete representation. `--result-id` selects
a retained extraction revision. The dashboard shows acquisition and current
extraction coverage; integrity verification remains a separate CLI operation.

The dashboard opens on the current year. Select **2016 onward** or **Full
history** (July 2003 onward) for broader coverage. Download and parsing strips
share the same months and scroll position; left/right arrows explore months,
and up/down arrows switch strips. Parsing height represents downloaded filings.
Its segments distinguish completed extractions, extracted cases needing review,
failed/unsupported attempts, recorded in-progress work and waiting files.
The summary cards show totals across every month in the selected range: extracted
count and percentage of downloads, completed without issues, completed with
warnings, needs review, not yet extracted, failed/unsupported, in progress and
waiting. An aggregate progress bar and downloaded/indexed totals provide context.
Hovering or focusing a month changes only its tooltip; the range totals stay fixed.
Changing the coverage period recalculates the totals.

Only selected extractions matching the registered ZIP and current or explicitly
compatible parser count. Unaffected ownership-1/2 results remain compatible with
v3; old date/plan-warning or missing-code review cases need an upgrade. Other
source/parser versions remain in the backlog; an older failure does not
override a selected current extraction. Interrupted work is waiting, and running
counts describe recorded attempts, which may need `admin recover` after a crash.
Counts deduplicate each accession per month across indexed entities and sources.
They use current index associations, so retained removed filings remain available
through the CLI. Parsing completeness is relative to downloaded files, independently
of whether the SEC index is complete.

The current-year and 2016 views check for updates every 30 seconds,
and full history every 60 seconds while visible. Results are shared in a bounded
server cache so multiple tabs do not multiply database scans. All queries use a
read-only, repeatable-read snapshot, bounded transaction-local working memory,
and a 60-second statement timeout; PostgreSQL defaults and writer settings are
unchanged. **Data refreshed** names the snapshot time, including cached results.
Restart only the dashboard process after updating; no schema migration is needed.

## Selection, previews and retries

`download`, `parse` and `check` support `--start`, `--end`, `--form` and
`--accession`. Dates are inclusive **indexed filing dates**, not transaction dates.
The new commands have no implicit date cutoff. Download selection uses current
index associations; parsing/checking filters also recognize historical index
observations so removed filings remain inspectable. For discovery-stage checks,
filters select whole retained index artifacts containing matching observations.

`discover` requires dates. Quarterly sources include whole intersecting quarters;
daily planning skips weekends but does not infer holidays. Repeating discovery
fetches the requested sources again, allowing revised indexes to be observed.
Accession removals still require explicit `--accept-removals`.

`discover`, `download`, `parse` and `repair` accept `--dry-run`. Discovery preview
needs no database; other previews read the catalogue. No preview sends SEC
requests or changes application records. `download` and `parse` accept
`--limit 1..10000`; omitted means all eligible work. `repair` defaults to 100
recorded findings per invocation; repeat as needed. All three report their scope.

Successful registered downloads and current source/parser extractions are
skipped. Interrupted work is eligible to resume. Previously failed work requires:

```powershell
insider download --retry-failed
insider parse --retry-failed
```

Unsupported XML may need a parser update rather than another retry. `parse`
does not silently replace an extraction because it has warnings. Amendments
remain separate filings, and transactions are not multiplied by joint owners.

`parse --workers 1..20` defaults to four local preparation workers. They verify
archives, preserve durable source snapshots and extract XML without database
access. `--executor auto` uses spawned processes for at least 4,096 queued filings,
threads for smaller runs, and serial preparation for `--workers 1`. Process workers
prepare chunks of up to 16 filings per IPC message, rather than sending one
message per filing; the existing 64-filing/8-MiB publication bounds still apply.
This uses multiple CPU cores without sharing database connections or locks with
children. `--executor thread|process` explicitly selects a preparation mode.

When `--write-workers` is omitted, runs with at least 4,096 queued filings and
more than one preparation worker use **two database publishers**. Smaller runs
and serial preparation use one. Thus the measured fast combination is automatic:

```powershell
insider parse --start 2020-01-01
```

`--dry-run` reports the resolved executor, preparation workers and database
publishers using the same policy as execution, after applying `--limit`.
Explicit options always override their corresponding defaults:

```powershell
insider parse --start 2020-01-01 --executor thread --write-workers 1
```

`--write-workers 1..4` selects independent connections that publish separate
atomic batches concurrently. Four preparation workers remain the default;
benchmark the filing mix before increasing writer concurrency beyond two.

Process children ignore Ctrl+C; the coordinator stops scheduling and performs
receipt reconciliation while holding the publisher locks. Worker failure leaves
unpublished attempts resumable. `--executor thread` is the supported fallback.
Python callers using process or auto mode should invoke the parser under an
`if __name__ == "__main__":` guard; the CLI entrypoint already does so.

The coordinator retains both publisher locks until all internal writers finish,
so other discovery/download/check/repair commands remain excluded. At most
`write-workers + 1` groups are retained, each capped at 64 filings and 8 MiB of
recorded source text (a larger individual filing forms a one-filing group).
Publishers use bulk `COPY` for larger typed/evidence tables. Each group's data,
selection and attempt receipts commit atomically; no connection/transaction is
shared across publisher threads. Evidence rows are streamed with their result
identifiers, SQL templates are cached without row values, and empty JSON objects
reuse an exact serialization. Text COPY, column types, constraints, source
snapshots and durable commit settings are retained.

Progress counts only confirmed commits. Ctrl+C stops scheduling, cancels queued
writes, waits for running transactions and counts any batches committed during
the drain. Only then are remaining attempts marked interrupted and the publisher
locks released. A failed batch rolls back without discarding successful sibling
batches. Rerun the same command to resume. These changes require no schema or
parser-semantics upgrade.

To apply current parser corrections only to affected published older results,
without starting the unparsed backlog:

```powershell
insider parse --upgrade-only --dry-run
insider parse --upgrade-only
```

Unaffected v1/v2 results remain compatible and are skipped. Affected results get
new revisions linked to their predecessors; historical results and source
snapshots remain intact. The upgrade can be resumed with the same command.
Ownership-3 treats an omitted optional transaction-coding block as a source-data
warning, preserving NULL codes; a supplied coding block missing its required
code still needs review. It does not infer transaction codes or repair conflicting
source dates.

For the downloaded 2016-onward collection, use:

```powershell
insider parse --start 2016-01-01 --dry-run
insider parse --start 2016-01-01
```

Larger worker counts are not necessarily faster: XML processing, durable writes
and database publication share machine resources. Use `--workers 1` for serial
preparation when comparing throughput; batching and bulk loading remain enabled.

## Checking and repairing

Discovery status distinguishes `unresolved_sources` (registered sources without
confirmed alternative coverage), `covered_daily_sources` (unresolved daily
requests covered by an active quarterly index), and `unresolved_source_requests`
(the original request-level total). Counts describe registered sources, not an
exhaustive audit of all possible SEC dates.

`issues` omits covered daily requests by default. Use
`insider issues --stage discover --include-covered` to include them as informational
entries with the covering quarter/artifact and original error. Add `--severity info`
to show only these entries. The request history is preserved.

A quarterly index covers earlier dates evidenced by its latest indexed filing
date; its final indexed date requires an online observation on a later New York
day. A quarter fetched online after quarter-end covers the whole quarter, including
days with no filings. Offline replay cannot advance that observation date. Only
the active revision counts, and a recorded integrity failure disqualifies it until
successfully rechecked/repaired. This does not certify submission downloads.

```powershell
insider check
insider issues
insider repair --dry-run
insider repair
insider status
```

Use `--stage discover|download|parse` to narrow checks, issues and repairs.
Issues/repairs also accept `--accession` and `--limit`; issues can be filtered by
`--severity error|review|warning`. Repair works from integrity findings generated
by `check`. Acquisition failures remain retryable through their respective stage.

Checks cover all retained index artifacts, registered ZIPs, and **selected**
extractions. Each index is replayed against its stored associations. ZIP checks
verify checksum, sizes, entry name and decompression. Extraction checks replay
the preserved source and compare every typed/evidence table and attempt receipt.
Historical superseded extractions remain available through `show --result-id`;
the compatibility command `validate-ownership` additionally checks all retained
extraction versions and can continue reporting intentionally preserved damage.

Checks record scope, timing and completion in `health_checks`. Immutable
`health_observations` retain every checked outcome; `health_receipts` hold the
latest outcome for each object/version. A replacement extraction or changed ZIP
does not inherit its predecessor's verification. An interrupted check certifies
only objects it actually finished checking. A zero-object check does not establish
coverage. Verification describes the recorded time; it cannot detect later disk
damage without another check.

Download checks fully hash every ZIP and decompress its expected entry, including
CRC and recorded-length checks. Small ZIPs are read with file-sized buffers;
large ZIPs are streamed instead of retaining the decompressed submission in
memory. Configured archive/source limits and data-root path checks still apply.
`check --workers 1..20` controls concurrent archive verification (default 8).
The CLI's `--executor auto` uses process workers for selections of at least
4,096 downloads and grouped thread workers for smaller selections. Processes
allow ZIP/path-processing work to run on multiple Python interpreters. Use
`--executor thread` or `--executor process` to select a mode explicitly;
`--workers 1` always verifies serially. Discovery and extraction checks remain
sequential, and only the coordinator accesses PostgreSQL.

Workers check small groups (up to 16 files per thread task or 64 per process
task), avoiding a separate scheduling/IPC operation for every tiny ZIP. Groups
are capped at 8 MiB of recorded source text, with larger individual filings in
one-file groups. At most two groups per worker are queued. Full SHA-256, entry,
decompression/CRC and size checks still run on every selected file; every file
still receives its own audit observation and receipt. The final report names
the selected download executor and worker count.

Python callers of `health.check()` retain thread mode by default. Explicit
`executor='auto'` or `'process'` calls should use a guarded application entry
point (`if __name__ == '__main__':`) for portable process startup. No additional
packages or schema migration are required.

Date/form-filtered download checks advance both the submission scan and the
matching historical-index scan past the last accession on every page. This
prevents late pages in a large catalogue from rescanning earlier index rows.
Only archive-verification columns are fetched. Parsing uses the same bounded
selection rule. Filter meanings and the set of selected filings are unchanged.

Download verification outcomes are committed atomically in batches of up to 256,
with an immutable observation and latest receipt for every checked file. Progress
counts committed outcomes. Interruption retains those batches; an unfinished
batch is not certified. Running the command again checks the selected files
again, including previously checked files. No schema upgrade is needed for these
throughput improvements.

The running check's checked/problem counters are updated in the same atomic
statement as each receipt batch. They remain accurate after an interruption
immediately following a commit, rather than remaining zero until completion.

Repair first rechecks each finding under the publisher lock:

| Fault | Action |
| --- | --- |
| Damaged/missing submission ZIP | Restore a verified local source copy, otherwise fetch the exact registered SEC version |
| Damaged/missing index file | Fetch bytes matching the retained hash |
| Incorrect index associations/counts | Rebuild from the verified index, preserving the old database projection in `reports/repairs/` |
| Damaged source snapshot | Restore from the matching registered ZIP, with exact-version network recovery if necessary |
| Incorrect parsed values/evidence | Publish a new extraction revision, validate it in the same transaction, then select it |
| Already fixed object | Record successful verification without replacing it |
| Changed SEC content or unrecoverable historical version | Leave unresolved with an explanation; never silently substitute a different source |
| Source contradictions, unknown fields, unsupported documents | Preserve evidence and leave for review/parser development |

Old extractions remain immutable. Repaired extractions identify the result they
supersede. `repair_attempts` records outcomes; damaged files are copied into
`raw/repair_quarantine/` before restoration, with original-path sidecars.
If extraction verification fails, its replacement transaction rolls back.
File restoration uses atomic replacement; a crash between restoration and its
receipt is safe to reconcile by rerunning the check/repair.

Checks and repairs take the same publisher lock as discovery/download/parsing,
so they refuse to overlap an active writer. Status, issues, show and dashboard
remain readable. Handled interruptions update attempt state immediately. After
a killed process or machine restart, `insider admin recover` reconciles abandoned
runs/checks/repairs only after acquiring that lock. It does not delete data.

Global options go **before** the command: `--config PATH`, `--data-root PATH`,
`--offline`, `--version`. Offline mode permits database access, local parsing,
checks and repair from local copies, but no SEC requests. Network repair and
download share the bounded SEC HTTP client; HTTP 403/final 429 stops further
dispatch. `download --workers 1..20` defaults to 10; candidate queues stream
through a server cursor instead of loading the entire catalogue into Python.

Exit codes: `0` successful command (including a successful limited batch), `2`
incomplete processing or failed integrity check, `1` operational/configuration
error, `64` invalid command syntax, `130` interruption. `issues` succeeds even
when findings exist. Read counts/remaining scope rather than treating exit 0
as complete SEC coverage.

## Compatibility

Old names remain callable but are hidden from top-level help: `init`, `plan`,
`coverage`, `filings`, `inspect`, `ingest-index`, `download-submissions`,
`plan-parse`, `parse-ownership`, `parsing-status`, `inspect-ownership`, `validate`,
`validate-submissions`, `validate-ownership`, `recover`. These retain their JSON
output and legacy validation behavior; legacy validators do not write health
receipts. Use `check` for findings that feed `issues` and `repair`.

`download-submissions` retains its historical default start date of 2003-07-01;
new `download` defaults to all indexed dates. `status` and `discover` now use the
public interface and accept `--json`. Administration is available as
`admin init`, `admin import-index`, `admin recover`, `admin filings`, and
`admin coverage` (JSON output for compatibility).

This is extraction and validation infrastructure only. Feature generation,
market data, statistical tests, models and scheduling are outside this pipeline.


Download and parse commands may run concurrently. Each holds a shared PostgreSQL
maintenance gate and its own exclusive local/database lane lock; a second command
in the same lane is rejected. Discovery, check, repair, recover and initialization
retain the exclusive gate and cannot overlap either command. Terminating a process
releases its locks; recover can mark abandoned attempts only after active publishers
have exited. Parsing freezes eligible committed submission identities at run start
in a temporary database table; later downloads are eligible on the next parse run.
Archives are validated and atomically published before their database registration.

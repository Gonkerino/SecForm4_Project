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
| `dashboard` | Display monthly acquisition coverage |
| `admin` | Rare setup, imports, listings, coverage detail and recovery |

Main commands print readable text by default. Add `--json` for machine-readable
output. Progress goes to stderr, leaving stdout suitable for JSON pipelines.
`show --details` exposes all extracted tables and preserved XML/header evidence;
`show --json` always includes that complete representation. `--result-id` selects
a retained extraction revision. The dashboard remains an acquisition view, not
proof that data has been parsed or checked.

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

`parse --workers 1..20` defaults to four local preparation threads. They verify
archives, preserve durable source snapshots and extract XML while the coordinator
publishes the preceding group. Only the coordinator accesses PostgreSQL.
Preparation queues at most two groups, each capped at 32 filings and 8 MiB of
recorded source text (a larger individual filing forms a one-filing group).
The coordinator uses bulk `COPY` for larger typed/evidence tables and commits
each group atomically, including selections and per-filing attempt receipts.
Progress counts committed filings; Ctrl+C retains completed groups and makes
unfinished attempts eligible on the next run. No schema or parser-version
upgrade is required: extracted values and evidence have unchanged semantics.

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

# Filing catalogue — implemented design

The filing catalogue tracks all six ownership forms using original full/daily EDGAR master indexes. Its records describe filing discovery and source provenance.

## Tables and publication

- `filings`: one accession identity and first/last observation times; no guessed issuer.
- `sources`: kind/period, URL and current validated artifact pointer.
- `artifacts`: unique source/content-hash revisions, lengths, locations, parsed counts and times.
- `associations`: source row, accession, indexed CIK/name, form, filing date and archive path.
- `runs`: command execution, safe settings, status and summary.
- `attempts`: source-level attempts, HTTP status and receipt metadata. Successful HTTP retry counts are included; detailed intermediate retry history is future work.
- `current_associations`: selects active source revisions. Queries deduplicate accessions across daily/quarterly overlap.

Schema version is `PRAGMA user_version`. Newer/unknown versions are refused. Migration 001 is packaged with the distribution. Foreign keys, WAL and full synchronous writes are enabled. Keep the live database on a local disk.

Bytes are preserved before parsing. Parsing rejects bad headers, malformed rows, invalid identifiers and invalid paths. Revision insertion, associations and source activation share one SQLite transaction. Failed content cannot replace a validated revision. Removals require explicit activation.

Filesystem publication and SQLite commit are separate. Failure between them can leave an unregistered artifact; readers use committed references so it remains invisible. Failed diagnostic responses are also outside the validated artifact set. Full orphan management is planned alongside raw-store maintenance. `recover` reconciles stale run state under the OS-backed publisher lock.

A missing index row is a source discrepancy, not proof of legal withdrawal. Prior observations are retained. `inspect` shows conflicting current form/date values; original submissions will resolve verified identities in the next milestone.

## Source behavior

Quarterly planning includes full intersecting quarters. Daily planning skips weekends, not inferred holidays. Failures/absence remain unresolved. All requested periods are registered before requests, so blocking cannot make later periods invisible.

Requests require contact identity, use one request-start scheduler, bound retries/bytes/time and restrict redirects to HTTPS SEC hosts. HTTP-200 block pages fail parsing. Redirects are paced too. Reports contain source plans, receipts and summaries without contact strings.

## Current boundaries

Implemented: package/configuration, first migration, discovery, provenance, query commands and index integrity checks. Transaction extraction, bulk-release import, filing ZIP storage, Parquet snapshots, scheduling and prices are outside the current functionality.

Code lives at `D:/Projects/SecForm4`. The default data directory is independent at `D:/Data/InsiderResearch`; it can be initialized empty with `insider init`.

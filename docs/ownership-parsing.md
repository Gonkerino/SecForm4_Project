# Ownership extraction (introduced in schema 3)

This stage extracts downloaded Forms 3, 3/A, 4, 4/A, 5 and 5/A. It performs no
network requests, executive-role classification, feature engineering, economic
position reconstruction, amendment reconciliation, market-data lookup or analysis.

## Upgrade and run

The shorter public commands are documented in [the CLI guide](cli.md). The compatibility commands below retain their JSON output. Schema 4 adds integrity receipts, audited repairs and retained extraction revisions.

From PowerShell in the checkout:

```powershell
insider init
insider plan-parse
insider --offline parse-ownership --limit 100
insider parsing-status
insider validate-ownership
insider --offline parse-ownership
```

`admin init` applies additive migrations in order from schemas 1, 2 or 3 to schema 4. It
does not reset the catalogue or download files. It requires the existing publisher
lock, so **wait for any running discovery/download command to finish**. Do not
interrupt a writer or bypass the lock to migrate. Ordinary catalogue commands
require the current schema; restart the dashboard after upgrading.

Parsing reads registered `submissions`, not arbitrary/unregistered ZIPs. With no
filters, all registered downloads are eligible, including retained records before
July 2003 and records no longer present in active index revisions. `--start`,
`--end` and `--form` filter historical index observations, not transaction dates.
`--accession` selects one accession. `--limit` bounds a batch to 1â€“10,000 records;
omitting it processes all eligible records at the start of the locked run.

One downloader and one parser may run concurrently, with separate exclusive
command locks. Discovery and maintenance exclude both. Each parser run freezes
its queue at startup; later downloads are picked up next time. The parser uses bounded
keyset pages and four preparation workers (`--workers 1..20`). Runs with 4,096
or more queued filings default to spawned process preparation and two database
publishers. Smaller runs use threads and one publisher; one preparation worker
defaults to serial preparation and one publisher.
`--executor thread|process` overrides auto selection. Process IPC groups at most
16 filings per task inside the existing publication bounds; no child accesses
PostgreSQL or inherits the coordinator's connection. The CLI has a guarded
spawn entrypoint, and Python callers should supply their own main guard. Explicit `--write-workers 1..4` overrides the writer default; benchmark the
filing mix before increasing beyond two. Each publisher owns its
connection and commits a complete group atomically. Preparation never accesses
PostgreSQL. At most `write-workers + 1` groups of 64 filings/8 MiB recorded source
text are retained (a larger filing forms a one-filing group). Each group's typed
data, evidence, selected results and attempt receipts commit together. Ctrl+C
cancels queued publications and drains active transactions before reconciling
attempts or releasing the publisher locks; all successful commits are retained
and counted, including sibling batches committed during failure cleanup.
Unfinished attempts remain retryable. After a hard process crash,
`insider recover` marks abandoned running attempts/runs interrupted under the lock.

```powershell
insider parse-ownership --accession 0000001750-26-000002
insider parse-ownership --form 5 --start 2020-01-01 --end 2020-12-31
insider parse-ownership --retry-failed --limit 100
insider inspect-ownership 0000001750-26-000002
insider inspect-ownership 0000001750-26-000002 --result-id 123
```

Failed/unsupported attempts are excluded on subsequent runs unless
`--retry-failed` is supplied. A changed source checksum or incompatible parser
version is a new input and becomes eligible. Published `needs_review` results
are preserved and skipped for the same parser/source. Extraction changes bump
`PARSER_VERSION`; older results and their attempt receipts remain immutable.

`ownership-2` accepts XML dates with valid `Z` or signed timezone suffixes,
retaining the stated calendar date and the full raw field text. It also removes
the keyword-based `plan_evidence_conflict` warning: a 10b5-1 mention alone cannot
establish a conflict with the checkbox. Checkbox and footnotes remain independent
evidence. This includes positive mentions as well as negations and amendments;
the parser makes no automatic plan-status determination from prose.

Unaffected `ownership-1` results remain compatible. Only v1 results with the old
plan warning or invalid-date issues need upgrading. Run the bounded upgrade
without starting the unparsed backlog:

```powershell
insider parse --upgrade-only --dry-run
insider parse --upgrade-only
```

Date/form/accession filters, limits and workers also apply to this operation.
It verifies local sources, publishes new v2 results atomically, links them to the
results they supersede, and leaves the previous results available for inspection.
It is resumable and idempotent. Other/unknown parser versions are not implicitly
compatible.

`ownership-3` aligns transaction-code validation with the SEC v5.4 and v5.5
ownership schemas: the complete `transactionCoding` block is optional on
non-derivative and derivative transactions in Forms 4/4A/5/5A. An omitted block
leaves all coding values NULL and emits an `unreported_transaction_code` warning,
not a blocking review issue. If the block is supplied but its required code is
absent, `missing_transaction_field` remains a review issue. Other required
transaction fields retain their existing validation. No code is inferred from
prices, acquisition/disposal indicators, footnotes or similar transactions.

Unaffected v1 and v2 results remain compatible; `parse --upgrade-only` also selects
older results with missing-code review issues for v3 replay. Prior v1/v2 results
remain immutable and can be replayed with their recorded semantics. Missing codes
still matter for research classification even when extraction is complete.

Exit 0 means the selected batch completed without review/failure cases. It does
not mean the entire archive has been parsed: a limit or excluded earlier failure
can leave work. Exit 2 means a batch contains review/unsupported/failed cases, or
validation failed. Use `parsing-status` and `plan-parse --retry-failed` for the
full picture. Progress is stderr, JSON is stdout. Decimal quantities/prices are
JSON strings to avoid binary floating-point rounding.

## Source reader and preservation

1. Restrict archive paths to the configured data root.
2. Verify the registered ZIP hash and compressed/uncompressed lengths, expected
   single `<accession>.txt` entry, decompression and CRC.
3. Bound compressed/decompressed input sizes using the configured submission
   limit (and the parser's 128 MiB hard ceiling).
4. Preserve a content-addressed ZIP copy at
   `raw/submission_versions/<hash-prefix>/<ZIP-SHA256>.zip` before interpreting
   verified bytes. The additional copies use roughly the original ZIP storage
   again; failed/unsupported verified sources are retained too.
5. Read the SEC header, embedded document inventory and primary ownership XML.

Each embedded document records its byte offset/length within the uncompressed
submission plus a content hash and its raw metadata header. Additional SEC header
information (such as SIC, file numbers and former names) remains queryable as
section-scoped line records, without silently flattening repeated issuer/owner
sections into one record. Exhibits are inventoried and remain in the source;
their prose is not interpreted into legal facts. The reader never extracts ZIP
paths onto disk or follows XML references.

Only one embedded ownership primary document is supported. Ambiguous primary
documents and legacy non-XML sources are explicit unsupported cases, not empty
successful reports. DTD/entity declarations are rejected. XML node/depth limits
are enforced. Namespaced ownership XML is supported; elements in a different
namespace are preserved and flagged for review.

## Database layout

All tables remain in the existing configured PostgreSQL schema. Existing
acquisition/index observations are not overwritten.

| Table | Row meaning |
| --- | --- |
| `parse_attempts` | Processing attempt, run identity, checksum/version, source snapshot, outcome and error |
| `parse_results` | Immutable published extraction, source hashes/path, counts and parser version |
| `ownership_current_results` | Explicit selected result per accession |
| `submission_headers` | Raw header and typed header fields for a result |
| `submission_header_fields` | Every nonempty header line, section/occurrence path, name, value and original line |
| `submission_documents` | Embedded document inventory, hashes and byte locations |
| `ownership_reports` | Typed report/issuer fields and filing indicators |
| `ownership_reporting_owners` | Each owner occurrence, raw title, relationships and address |
| `ownership_transactions` | Each reported transaction across both tables |
| `ownership_holdings` | Each standalone holding across both tables |
| `ownership_footnotes` | Each footnote occurrence and full text |
| `ownership_fields` | Every XML element: indexed source path, qualified tag, text, tail, attributes, state and mapping coverage |
| `ownership_field_notes` | Field-specific references to footnote IDs |
| `ownership_signatures` | Each signature occurrence |
| `parse_issues` | Source-location-specific warning/review evidence |

The parser does not attach transactions independently to every joint owner.
Transactions and holdings have separate row identities and `is_derivative` flags.
No standalone holding is manufactured from a transaction's balance afterward.
Amendments remain separate accession records; original-submission dates are kept
without guessing the original accession or superseded economic rows.

Current views provide accession, extraction status and parser version:

```sql
SELECT accession, issuer_cik, issuer_name, form_type, period_of_report,
       extraction_status
FROM current_ownership_reports;

SELECT accession, is_derivative, transaction_date, security_title,
       transaction_code, transaction_shares, transaction_price_per_share,
       shares_owned_following
FROM current_ownership_transactions;

SELECT accession, owner_cik, owner_name, officer_title, is_ten_percent_owner
FROM current_ownership_reporting_owners;
```

Equivalent current views exist for holdings, footnotes, field notes and signatures.
Current views can contain `needs_review` records: filter `extraction_status` when
appropriate. Join source locations through `result_id`; never join owners to
transactions and then sum the duplicated transaction quantities.

## Scalar and time conventions

- CIKs/accessions are text. Dates use `DATE`; prices/quantities use unconstrained
  PostgreSQL `NUMERIC` and Python `Decimal` (no two-decimal-place rounding).
- Invalid numeric/date/boolean text remains in `ownership_fields`, with the typed
  column NULL and a review issue. No comma stripping or guessed conversions.
  XML date timezone offsets are validated up to ±14:00; the calendar date is
  preserved without converting it to a different UTC day.
- Absent fields have no XML-node record. Present empty nodes have state `empty`;
  `footnote_only` marks wrappers carrying notes without values. Explicit zero is
  a value, not missing. Structural nodes have state `container`.
- Raw titles, names, descriptions, codes and notes remain as reported. The typed
  text projection trims outer whitespace; field records preserve original XML
  text/tails. Exact original markup, comments and encoding remain in the ZIP.
- Acceptance raw text is retained. The typed timestamp interprets the SEC header
  in America/New_York, including daylight-saving rules, and converts to UTC. This
  is labelled as an interpretation, not a measured dissemination timestamp.
- Report/filing/transaction/deemed-execution dates, acquisition time and parse time
  stay separate. No trading decision time is calculated.

Unknown elements/attributes, repeated scalar values, invalid values and missing
required structure produce review issues. Known containers do not make unknown
descendants disappear. Footnote references remain even when missing or ambiguous;
duplicate footnote IDs are retained as occurrences and flagged. Reported plan
flags and plan text remain separate evidence; keyword mentions do not generate
a conflict warning.

## Publication, recovery and validation

One transaction inserts the report and all child tables, sets the selected result
and finishes its attempt. An exception rolls back the whole result. A previous
selection survives unsuccessful reparsing. A source/parser combination publishes
only once, while all attempts remain in history.

Outcomes are `complete`, `complete_with_warnings`, `needs_review`, `unsupported`,
`failed`, or `interrupted`. Review results preserve extracted facts but make no
claim of fully understood content. `parsing-status` separates current source/parser
matches from older selected results and reports issue categories and latest attempts.

`validate-ownership` opens a repeatable-read read-only database snapshot, checks
every retained result's source ZIP/text hash and replays its source using the
recorded v1, v2 or v3 semantics, comparing typed tables, document inventory, all field evidence,
notes, signatures, issue status/counts and attempt receipts. Index-comparison
warnings retain their original observation rather than being recomputed from a
possibly revised index. Unknown parser versions are reported as replay-unavailable
(source still checked). Ownership-1, ownership-2 and ownership-3 replay are supported. Validation does not repair records or establish
the truth of the filing, complete SEC coverage or research eligibility.

Back up PostgreSQL **and** the data root, including `raw/submission_versions`.
The monthly dashboard shows both downloads and current extraction coverage,
including review cases, failed/unsupported attempts and remaining work. This
coverage does not replace integrity verification. See the CLI guide for scope,
refresh intervals and the distinction between extracted and review-ready data.

## Extension boundaries

`ownership.py` is a pure ownership extractor with typed field maps; it has no
database or filesystem effects. `ownership_store.py` handles archive verification,
versioned storage, local batch execution, inspection and replay validation.
`submission_documents`, source provenance and attempt/result infrastructure are
reusable for other form families; future domain tables should not be forced into
ownership tables. No feature bank, market data or analysis is added in schema 3.

Tests use reviewed unchanged public sources under `tests/fixtures/ownership/`,
explicit synthetic transformations and the existing isolated PostgreSQL test
database with temporary schemas. There is no SEC access during tests.

Batch selection evaluates filters and retry eligibility once when constructing the
indexed temporary queue. Each page seeks past the last scheduled accession and
limits queue identities before joining submission metadata. Completed prefixes
and eligibility tables are not rescanned per batch. Queue statistics are analyzed
at startup. The parser lane, maintenance gate, atomic batch commits and interruption
reconciliation remain in force; a resumed command rebuilds eligibility from durable
results and attempts, including newly downloaded filings.

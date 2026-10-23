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
`--accession` selects one accession. `--limit` bounds a batch to 1–10,000 records;
omitting it processes all eligible records at the start of the locked run.

Discovery, downloads and parsing currently share the publisher lock. Only one
cooperating writer runs at a time. The parser uses bounded keyset pages and one
submission at a time. A batch may be interrupted with Ctrl+C; committed results
survive and the interrupted attempt is retryable. After a hard process crash,
`insider recover` marks abandoned running attempts/runs interrupted under the lock.

```powershell
insider parse-ownership --accession 0000001750-26-000002
insider parse-ownership --form 5 --start 2020-01-01 --end 2020-12-31
insider parse-ownership --retry-failed --limit 100
insider inspect-ownership 0000001750-26-000002
insider inspect-ownership 0000001750-26-000002 --result-id 123
```

Failed/unsupported attempts are excluded on subsequent runs unless
`--retry-failed` is supplied. A changed source checksum or parser version is a new
input and becomes eligible. Published `needs_review` results are preserved and
skipped for the same parser/source; resolve their cause and release a new parser
version to re-extract them. Do not silently change extraction semantics without
bumping `PARSER_VERSION`.

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
flags and contradictory plan text remain separate evidence.

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
every retained result's source ZIP/text hash and replays its source through the
current parser, comparing typed tables, document inventory, all field evidence,
notes, signatures, issue status/counts and attempt receipts. Index-comparison
warnings retain their original observation rather than being recomputed from a
possibly revised index. A retained result using a different parser version is
reported as replay-unavailable (source still checked); keep old code if exact
historical replay is required. Validation does not repair records or establish
the truth of the filing, complete SEC coverage or research eligibility.

Back up PostgreSQL **and** the data root, including `raw/submission_versions`.
The monthly dashboard still reports downloaded submissions, not parsing readiness.

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

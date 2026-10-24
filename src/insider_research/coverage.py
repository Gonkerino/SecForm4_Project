"""Read-only source coverage classification shared by status and issues.

A failed request is not necessarily a coverage gap. Only the active quarterly
artifact can supply alternative coverage; attempt history is never rewritten.
"""

SOURCE_COVERAGE_SQL = """
WITH quarters AS (
    SELECT s.period, r.id AS artifact_id, r.max_filed_date,
           (SELECT max(t.finished_at) AT TIME ZONE 'America/New_York'
            FROM attempts t WHERE t.artifact_id=r.id AND t.status='validated'
            AND t.response_json->>'offline' IS DISTINCT FROM 'true')::date AS observed_date,
           (make_date(left(s.period,4)::int, right(s.period,1)::int*3-2, 1)
             + interval '3 months')::date AS next_quarter
    FROM sources s JOIN artifacts r ON r.id=s.current_artifact_id
    WHERE s.kind='quarterly'
      AND NOT EXISTS(SELECT 1 FROM health_receipts h WHERE h.stage='discover'
          AND h.target=r.id::text AND h.version=r.sha256 AND NOT h.valid)
), classified AS (
    SELECT s.kind,s.period,a.status AS latest_attempt_status,a.error,
           q.period AS covering_quarter,q.artifact_id AS covering_artifact_id,
           CASE WHEN a.status='validated' THEN 'validated'
                WHEN s.kind='daily' AND q.artifact_id IS NOT NULL AND (
                    q.observed_date >= q.next_quarter
                    OR q.max_filed_date > CASE WHEN s.kind='daily' THEN s.period::date END
                    OR (q.max_filed_date >= CASE WHEN s.kind='daily' THEN s.period::date END AND q.observed_date > CASE WHEN s.kind='daily' THEN s.period::date END))
                THEN 'covered_by_quarter'
                ELSE 'unresolved' END AS coverage_status
    FROM sources s
    LEFT JOIN LATERAL(SELECT status,error FROM attempts a WHERE a.source_id=s.id ORDER BY id DESC LIMIT 1) a ON TRUE
    LEFT JOIN quarters q ON s.kind='daily' AND q.period=
        CASE WHEN s.kind='daily' THEN left(s.period,4) || 'Q' || extract(quarter FROM s.period::date)::int END
)
SELECT * FROM classified
"""


def coverage_counts(catalog):
    return catalog.connection.execute('''SELECT
        count(*) FILTER(WHERE coverage_status='unresolved') AS unresolved_sources,
        count(*) FILTER(WHERE coverage_status='covered_by_quarter') AS covered_daily_sources,
        count(*) FILTER(WHERE latest_attempt_status IS DISTINCT FROM 'validated') AS unresolved_source_requests
        FROM (''' + SOURCE_COVERAGE_SQL + ') coverage').fetchone()

import unittest
from datetime import date

from insider_research.sources.indexes import parse_master_index, plan_sources, source_for
from tests.helpers import HEADER, record


class IndexTests(unittest.TestCase):
    def test_all_six_forms_with_nonownership_counted(self):
        forms = ["3", "3/A", "4", "4/A", "5", "5/A", "10-K"]
        parsed = parse_master_index(HEADER + b"".join(record(f, f"0000000123-24-{i:06}") for i, f in enumerate(forms)))
        self.assertEqual(parsed.total_rows, 7)
        self.assertEqual([r.form_type for r in parsed.rows], forms[:-1])

    def test_joint_index_associations_are_preserved(self):
        parsed = parse_master_index(HEADER + record(cik="123") + record(cik="456"))
        self.assertEqual(len(parsed.rows), 2)
        self.assertEqual(parsed.rows[0].accession, parsed.rows[1].accession)
        self.assertNotEqual(parsed.rows[0].archive_path, parsed.rows[1].archive_path)

    def test_embedded_pipe_in_company(self):
        self.assertEqual(parse_master_index(HEADER + record(company="Example | Holdings")).rows[0].company_name, "Example | Holdings")

    def test_legacy_path_and_compact_date(self):
        parsed = parse_master_index(HEADER + record(cik="0000000123", filed="20240102").replace(b"edgar/data", b"data"))
        self.assertEqual(parsed.rows[0].cik, "123")
        self.assertEqual(parsed.rows[0].filed_date, "2024-01-02")
        self.assertTrue(parsed.rows[0].archive_path.startswith("edgar/data/"))

    def test_legacy_encoding_preserves_name(self):
        content = HEADER + record().replace(b"Example Corporation", "Société".encode("latin-1"))
        self.assertEqual(parse_master_index(content).rows[0].company_name, "Société")

    def test_alternate_filename_header(self):
        body = HEADER.replace(b"File Name", b"Filename") + record()
        self.assertEqual(len(parse_master_index(body).rows), 1)

    def test_invalid_documents_rejected(self):
        for content in (b"", b"<html>Access denied</html>", HEADER, HEADER + b"truncated",
                        HEADER + record(filed="2024-02-30"), HEADER + record().replace(b"edgar/data/123/", b"../../"),
                        HEADER + record(cik="0"), HEADER + b"\x00"):
            with self.subTest(content=content), self.assertRaises(ValueError):
                parse_master_index(content)

    def test_no_ownership_is_valid_when_other_forms_exist(self):
        self.assertEqual(parse_master_index(HEADER + record(form="10-K")).rows, [])

    def test_blank_nonownership_name_does_not_reject_quarter(self):
        row = b"1593547||485BPOS|2016-02-26|edgar/data/1593547/0001135428-16-001124.txt\n"
        parsed = parse_master_index(HEADER + row + record())
        self.assertEqual(parsed.total_rows, 2)
        self.assertEqual(len(parsed.rows), 1)
        self.assertEqual(parsed.min_date, "2016-02-26")
        for invalid in (record(company=""), record(form="485BPOS", company="", cik="0"),
                        record(form="485BPOS", company="", filed="2024-02-30"),
                        row.replace(b"edgar/data/1593547/", b"../../")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_master_index(HEADER + invalid)

    def test_partial_quarter_plan_has_one_source(self):
        result = plan_sources(date(2024, 2, 5), date(2024, 2, 6), "quarterly")
        self.assertEqual([s.period for s in result], ["2024Q1"])

    def test_quarter_rollover(self):
        result = plan_sources(date(2023, 12, 31), date(2024, 4, 1), "quarterly")
        self.assertEqual([s.period for s in result], ["2023Q4", "2024Q1", "2024Q2"])

    def test_daily_weekends_skipped(self):
        result = plan_sources(date(2024, 1, 5), date(2024, 1, 8), "daily")
        self.assertEqual([s.period for s in result], ["2024-01-05", "2024-01-08"])

    def test_invalid_periods_and_ranges(self):
        for kind, period in [("quarterly", "2024Q5"), ("quarterly", "1994Q1"), ("daily", "2024-02-30"), ("bad", "2024Q1")]:
            with self.subTest(period=period), self.assertRaises(ValueError):
                source_for(kind, period)
        for start, end in [(date(2024, 1, 2), date(2024, 1, 1)),
                           (date(1994, 1, 1), date(1994, 8, 1)),
                           (date(2024, 1, 1), date(2025, 1, 1))]:
            with self.subTest(start=start), self.assertRaises(ValueError):
                plan_sources(start, end, "quarterly", today=date(2024, 12, 31))

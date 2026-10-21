"""Strict EDGAR master-index parsing and bounded source-period planning."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import date, timedelta

from ..config import FORMS

ACCESSION = re.compile(r"\d{10}-\d{2}-\d{6}", re.ASCII)
ARCHIVE_PATH = re.compile(r"edgar/data/[0-9]{1,10}/([0-9]{10}-[0-9]{2}-[0-9]{6})\.txt")


@dataclass(frozen=True)
class IndexRow:
    row_number: int
    accession: str
    cik: str
    company_name: str
    form_type: str
    filed_date: str
    archive_path: str


@dataclass(frozen=True)
class ParsedIndex:
    rows: list[IndexRow]
    total_rows: int
    min_date: str | None
    max_date: str | None


@dataclass(frozen=True)
class IndexSource:
    kind: str
    period: str
    url: str

    def as_dict(self) -> dict:
        return asdict(self)


def source_for(kind: str, period: str) -> IndexSource:
    if kind == "quarterly":
        match = re.fullmatch(r"([0-9]{4})Q([1-4])", period)
        if not match:
            raise ValueError("Quarter period must look like 2024Q1")
        year, quarter = int(match[1]), int(match[2])
        if year < 1994 or (year == 1994 and quarter < 3):
            raise ValueError("EDGAR index coverage begins in 1994Q3")
        url = f"https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{quarter}/master.idx"
    elif kind == "daily":
        day = date.fromisoformat(period)
        if day < date(1994, 7, 1):
            raise ValueError("EDGAR index coverage begins in 1994Q3")
        quarter = (day.month - 1) // 3 + 1
        url = f"https://www.sec.gov/Archives/edgar/daily-index/{day.year}/QTR{quarter}/master.{day:%Y%m%d}.idx"
    else:
        raise ValueError("Source must be quarterly or daily")
    return IndexSource(kind, period, url)


def plan_sources(start: date, end: date, kind: str, *, today: date | None = None) -> list[IndexSource]:
    today = today or date.today()
    if start > end:
        raise ValueError("Start date must be on or before end date")
    if start < date(1994, 7, 1):
        raise ValueError("EDGAR index coverage begins in 1994Q3")
    if end > today:
        raise ValueError("End date cannot be in the future")
    sources = []
    current = start
    while current <= end:
        if kind == "quarterly":
            quarter = (current.month - 1) // 3 + 1
            sources.append(source_for(kind, f"{current.year}Q{quarter}"))
            current = date(current.year + 1, 1, 1) if quarter == 4 else date(current.year, quarter * 3 + 1, 1)
        elif kind == "daily":
            # Skip weekends; weekday holidays remain unresolved unless verified.
            if current.weekday() < 5:
                sources.append(source_for(kind, current.isoformat()))
            current += timedelta(days=1)
        else:
            raise ValueError("Source must be quarterly or daily")
    return sources


def parse_master_index(content: bytes) -> ParsedIndex:
    if not content or b"\x00" in content:
        raise ValueError("Index is empty or contains binary null bytes")
    # Names in older SEC indexes are not uniformly UTF-8. Retain original bytes separately.
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = content.decode("latin-1")
    lines = text.splitlines()
    header = None
    data_start = None
    for index, line in enumerate(lines[:100]):
        if line.strip().casefold().replace("file name", "filename") == "cik|company name|form type|date filed|filename":
            header = index
        if header is not None and index > header and re.fullmatch(r"-{5,}", line.strip()):
            data_start = index + 1
            break
    if data_start is None:
        raise ValueError("Not a recognized EDGAR master index: header or separator missing")
    rows = []
    dates = []
    total = 0
    for line_number, line in enumerate(lines[data_start:], start=data_start + 1):
        if not line.strip():
            continue
        try:
            cik, rest = line.split("|", 1)
            company, form, filed, path = rest.rsplit("|", 3)
            cik, company, form, filed, path = (v.strip() for v in (cik, company, form, filed, path))
            if not re.fullmatch(r"[0-9]{1,10}", cik) or int(cik) == 0 or not company or not form:
                raise ValueError("invalid CIK, company or form")
            if re.fullmatch(r"[0-9]{8}", filed):
                filed = f"{filed[:4]}-{filed[4:6]}-{filed[6:]}"
            filed = date.fromisoformat(filed).isoformat()
            if path.startswith("data/"):
                path = "edgar/" + path
            match = ARCHIVE_PATH.fullmatch(path)
            if not match:
                raise ValueError("invalid submission archive path")
        except ValueError as exc:
            raise ValueError(f"Malformed index data at line {line_number}: {exc}") from exc
        total += 1
        dates.append(filed)
        if form in FORMS:
            rows.append(IndexRow(line_number, match[1], cik.lstrip("0"), company, form, filed, path))
    if not total:
        raise ValueError("Index has a header but no data rows; completeness cannot be established")
    return ParsedIndex(rows, total, min(dates), max(dates))

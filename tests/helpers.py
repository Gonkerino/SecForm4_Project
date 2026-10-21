from pathlib import Path

from insider_research.config import Config

HEADER = b"Description: Master Index of EDGAR Dissemination Feed\nCIK|Company Name|Form Type|Date Filed|File Name\n------------------------------\n"


def record(form="4", accession="0000000123-24-000001", cik="123", filed="2024-01-02", company="Example Corporation"):
    return f"{cik}|{company}|{form}|{filed}|edgar/data/{cik}/{accession}.txt\n".encode()


def config(root: Path, **kwargs) -> Config:
    return Config(data_root=root / "fresh-data", **kwargs)

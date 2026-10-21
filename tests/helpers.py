import os
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import psycopg
from psycopg import sql

from insider_research.config import Config

HEADER = b"Description: Master Index of EDGAR Dissemination Feed\nCIK|Company Name|Form Type|Date Filed|File Name\n------------------------------\n"


def record(form="4", accession="0000000123-24-000001", cik="123", filed="2024-01-02", company="Example Corporation"):
    return f"{cik}|{company}|{form}|{filed}|edgar/data/{cik}/{accession}.txt\n".encode()


@contextmanager
def postgres_config(root: Path):
    """Each integration test owns a random schema; never use the application schema."""
    dsn = os.environ.get("INSIDER_TEST_DATABASE_URL")
    if not dsn:
        raise unittest.SkipTest("Set INSIDER_TEST_DATABASE_URL to run PostgreSQL integration tests")
    schema = "test_insider_" + uuid.uuid4().hex
    cfg = Config(data_root=root, database_url=dsn, database_schema=schema)
    try:
        with patch.dict(os.environ, {"INSIDER_DATABASE_URL": dsn, "INSIDER_DATABASE_SCHEMA": schema}):
            yield cfg
    finally:
        with psycopg.connect(dsn, autocommit=True, connect_timeout=5) as connection:
            connection.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema)))

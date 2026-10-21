import contextlib
import io
import os
import tempfile
import unittest
import urllib.error
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import patch

from insider_research.cli import main, parser
from insider_research.config import Config, load_config
from insider_research.http import HttpClient, RateLimiter, retry_delay
from tests.helpers import HEADER, postgres_config, record


class Clock:
    now = 0.0

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class HttpAndCliTests(unittest.TestCase):
    def test_research_date_defaults_and_explicit_archival_override(self):
        for command in ("filings", "download-submissions"):
            with self.subTest(command=command):
                self.assertEqual(parser().parse_args([command]).start, date(2003, 7, 1))
                self.assertEqual(parser().parse_args([command, "--start", "1996-01-01"]).start,
                                 date(1996, 1, 1))

    def test_default_config_is_loaded_outside_project_directory(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {}, clear=True):
            path = Path(temp) / "default.toml"
            path.write_text('data_root = "dataset"\nmax_rps = 3\n', encoding="utf-8")
            with patch("insider_research.config.default_config_path", return_value=path):
                cfg = load_config()
            self.assertEqual(cfg.max_rps, 3)
            self.assertEqual(cfg.data_root, (path.parent / "dataset").resolve())

    def test_explicit_config_overrides_default(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {}, clear=True):
            path = Path(temp) / "custom.toml"
            path.write_text('max_rps = 2\n', encoding="utf-8")
            with patch("insider_research.config.default_config_path", side_effect=AssertionError("default lookup")):
                self.assertEqual(load_config(path).max_rps, 2)

    def test_missing_default_config_uses_builtin_settings(self):
        with patch.dict(os.environ, {}, clear=True), patch("insider_research.config.default_config_path", return_value=None):
            self.assertEqual(load_config().user_agent, "")

    def test_rate_spacing_including_sub_one_rates(self):
        for rate in (0.5, 3, 8, 10):
            clock = Clock()
            limiter = RateLimiter(rate, clock=clock.time, sleep=clock.sleep)
            starts = []
            for _ in range(20):
                limiter.acquire()
                starts.append(clock.now)
            for a, b in zip(starts, starts[1:]):
                self.assertGreaterEqual(b - a + 1e-12, 1 / rate)

    def test_retry_after_seconds_and_date(self):
        now = datetime(2024, 1, 1, tzinfo=timezone.utc)
        self.assertEqual(retry_delay("12", 1), 12)
        self.assertEqual(retry_delay("Mon, 01 Jan 2024 00:00:30 GMT", 1, now=now), 30)
        self.assertEqual(retry_delay("invalid", 2), 2)
        self.assertEqual(retry_delay("NaN", 2), 2)

    def test_config_rejects_bad_limits(self):
        for name, value in [("max_rps", 11), ("max_rps", float("nan")), ("max_attempts", 0),
                            ("max_rps", True), ("timeout_seconds", -1), ("max_index_bytes", 0),
                            ("max_submission_bytes", 0),
                            ("user_agent", "bad\nheader"), ("user_agent", "bad\x00header"),
                            ("user_agent", "bad\x7fheader")]:
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                Config(data_root=Path("local-data"), **{name: value})

    def test_contact_required_only_for_online(self):
        cfg = Config(data_root=Path("local-data"))
        with self.assertRaises(ValueError):
            HttpClient(cfg)

    def test_online_identity_requires_identifying_prefix_and_email(self):
        for value in ("contact@example.org", "Research Operator", "Research Operator invalid-email"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Config(data_root=Path("local-data"), user_agent=value).require_identity()
        Config(data_root=Path("local-data"), user_agent="Research Operator contact@example.org").require_identity()

    def test_configured_and_overridden_identity_reaches_http_header(self):
        class Body(io.BytesIO):
            status = 200
            url = "https://www.sec.gov/sample.idx"
            headers = {"Content-Length": "3"}

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.toml"
            configured = "Research Operator contact@example.org"
            path.write_text(f'user_agent = "{configured}"\n', encoding="utf-8")
            for override in (None, "Other Operator other@example.org"):
                environment = {} if override is None else {"SEC_USER_AGENT": override}
                with self.subTest(override=override), patch.dict(os.environ, environment, clear=True):
                    cfg = load_config(path)
                    client = HttpClient(cfg)
                    with patch.object(client.opener, "open", return_value=Body(b"abc")) as opening:
                        client.get("https://www.sec.gov/sample.idx")
                    sent = opening.call_args.args[0].get_header("User-agent")
                    self.assertEqual(sent, override or configured)
                    self.assertNotIn("user_agent", cfg.public_settings())

    def test_http_retries_are_paced_and_preserve_response(self):
        class Body(io.BytesIO):
            status = 200
            url = "https://www.sec.gov/sample.idx"
            headers = {"Content-Length": "3"}

        cfg = Config(data_root=Path("local-data"), user_agent="Test test@example.org")
        client = HttpClient(cfg)
        error = urllib.error.HTTPError("https://www.sec.gov/sample.idx", 503, "retry", {"Retry-After": "0"}, None)
        with patch.object(client.opener, "open", side_effect=[error, Body(b"abc")]) as opening, \
             patch.object(client.limiter, "acquire") as paced, \
             patch("insider_research.http.time.sleep"):
            result = client.get("https://www.sec.gov/sample.idx")
        self.assertEqual(result.body, b"abc")
        self.assertEqual(result.metadata["network_attempts"], 2)
        self.assertEqual(paced.call_count, 2)
        self.assertEqual(opening.call_count, 2)

    def test_http_403_is_not_retried(self):
        client = HttpClient(Config(data_root=Path("local-data"), user_agent="Test test@example.org"))
        error = urllib.error.HTTPError("https://www.sec.gov/sample.idx", 403, "blocked", {}, None)
        with patch.object(client.opener, "open", side_effect=error) as opening:
            with self.assertRaises(Exception) as raised:
                client.get("https://www.sec.gov/sample.idx")
        self.assertEqual(raised.exception.status, 403)
        self.assertEqual(opening.call_count, 1)

    def test_cli_init_import_query_and_offline_rerun(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root = base / "dataset"
            self.enterContext(postgres_config(root))
            fixture = base / "sample.idx"
            fixture.write_bytes(HEADER + record())
            options = ["--data-root", str(root)]
            commands = [options + ["init"],
                        options + ["ingest-index", "--file", str(fixture), "--source", "quarterly", "--period", "2024Q1"],
                        options + ["status"], options + ["coverage"],
                        options + ["inspect", "0000000123-24-000001"],
                        options + ["--offline", "discover", "--start", "2024-01-01", "--end", "2024-01-31"]]
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                for args in commands:
                    self.assertEqual(main(args), 0, args)

    def test_plan_has_no_filesystem_or_network_effect(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "never-created"
            with contextlib.redirect_stdout(io.StringIO()), patch("insider_research.discovery.HttpClient", side_effect=AssertionError("network")):
                code = main(["--data-root", str(path), "plan", "--start", "2024-01-01", "--end", "2024-01-31"])
            self.assertEqual(code, 0)
            self.assertFalse(path.exists())

    def test_offline_submission_download_is_refused_before_network_or_database(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
                patch("insider_research.submissions.download_submissions", side_effect=AssertionError("network")):
            self.assertEqual(main(["--offline", "download-submissions"]), 1)

    def test_exit_codes(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.enterContext(postgres_config(Path(temp) / "data"))
            options = ["--data-root", str(Path(temp) / "data")]
            self.assertEqual(main(["unknown"]), 64)
            self.assertEqual(main(options + ["status"]), 1)
            self.assertEqual(main(options + ["init"]), 0)
            self.assertEqual(main(options + ["--offline", "discover", "--source", "daily", "--start", "2024-01-02", "--end", "2024-01-02"]), 2)

    def test_unknown_configuration_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.toml"
            path.write_text('max_rsp = 5\n')
            with self.assertRaises(ValueError):
                load_config(path)

    def test_database_configuration_and_environment_override(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.toml"
            path.write_text('database_url = "dbname=configured user=insider"\n')
            with patch.dict(os.environ, {"INSIDER_DATABASE_URL": "dbname=override user=insider password=secret",
                                         "INSIDER_DATABASE_SCHEMA": "custom_schema"}):
                cfg = load_config(path)
            self.assertIn("dbname=override", cfg.database_url)
            self.assertEqual(cfg.database_schema, "custom_schema")
            self.assertNotIn("secret", repr(cfg))
            self.assertNotIn("secret", str(cfg.public_settings()))

    def test_bad_database_settings_do_not_disclose_credentials(self):
        for dsn in ("postgresql://insider:secret@host/db?badoption=secret", "password=secret invalid"):
            with self.assertRaises(ValueError) as raised:
                Config(data_root=Path("local-data"), database_url=dsn)
            self.assertNotIn("secret", str(raised.exception))
        for schema in ("public", "pg_catalog", "information_schema", "bad;sql", "UpperCase"):
            with self.subTest(schema=schema), self.assertRaises(ValueError):
                Config(data_root=Path("local-data"), database_schema=schema)

    def test_connection_error_is_actionable_and_redacted(self):
        import psycopg
        output = io.StringIO()
        with contextlib.redirect_stderr(output), patch("insider_research.catalog.psycopg.connect",
                side_effect=psycopg.OperationalError("password=secret")):
            self.assertEqual(main(["status"]), 1)
        self.assertIn("Cannot connect to PostgreSQL", output.getvalue())
        self.assertNotIn("secret", output.getvalue())

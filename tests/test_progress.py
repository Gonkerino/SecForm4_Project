import contextlib
import io
import json
import unittest
from unittest.mock import patch

from insider_research.cli import main
from insider_research.progress import Progress


class Terminal(io.StringIO):
    def isatty(self):
        return True


class ProgressTests(unittest.TestCase):
    def test_terminal_partial_failure_and_interrupt_close(self):
        output = Terminal()
        with contextlib.redirect_stderr(output):
            with self.assertRaises(KeyboardInterrupt):
                with Progress("Downloads", "filing") as progress:
                    progress({"event": "start", "total": 3})
                    progress({"event": "progress", "current": 1, "status": "failed",
                              "accession": "example", "message": "blocked"})
                    raise KeyboardInterrupt
        self.assertTrue(progress.bar.disable)
        self.assertEqual(progress.bar.n, 1)
        self.assertIn("1/3", output.getvalue())
        self.assertIn("Failed example: blocked", output.getvalue())
        self.assertNotIn("100%", output.getvalue())

    def test_redirected_output_and_empty_selection(self):
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            with Progress("Validation", "archive") as progress:
                progress({"event": "start", "total": 0})
            with Progress("Discovery", "source", 1) as progress:
                progress("2024Q1: validated")
        self.assertNotIn("\r", output.getvalue())
        self.assertIn("0 archives queued", output.getvalue())
        self.assertIn("1/1 sources", output.getvalue())

    def test_validation_cli_preserves_json_and_exit_status(self):
        def verify(cfg, *, progress, workers):
            progress({"event": "loading"})
            progress({"event": "start", "total": 1})
            progress({"event": "progress", "current": 1, "problems": 1})
            return {"status": "incomplete", "problems": ["corrupt"]}

        output, errors = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors), \
                patch("insider_research.cli.verify_submissions", side_effect=verify):
            self.assertEqual(main(["validate-submissions"]), 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "incomplete")
        self.assertIn("problems=1", errors.getvalue())

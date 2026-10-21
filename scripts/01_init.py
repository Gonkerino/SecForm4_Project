"""Thin wrapper; install the package first. Global options may follow this script."""

import sys

from insider_research.cli import main

raise SystemExit(main([*sys.argv[1:], "init"]))

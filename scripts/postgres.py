"""Control this user's local PostgreSQL installation without administrator access."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "stop", "restart", "status", "reload"))
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("This helper manages the per-user Windows installation")
    manifest = Path(os.environ["LOCALAPPDATA"]) / "Programs/PostgreSQL/18/insider-research-install.json"
    if not manifest.is_file():
        parser.error("Local PostgreSQL installation was not found; see README.md")
    settings = json.loads(manifest.read_text(encoding="utf-8"))
    command = [str(Path(settings["program_root"]) / "bin/pg_ctl.exe"),
               "-D", settings["data_directory"]]
    hidden = {"creationflags": subprocess.CREATE_NO_WINDOW}
    if args.action == "start":
        status = subprocess.run([*command, "status"], capture_output=True, text=True, **hidden)
        if status.returncode == 0:
            print("PostgreSQL is already running.")
            return 0
        command += ["-l", settings["server_log"], "-w", "-t", "30", "start"]
    elif args.action in {"stop", "restart"}:
        if args.action == "restart":
            command += ["-l", settings["server_log"]]
        command += ["-m", "fast", "-w", "-t", "30", args.action]
    else:
        command += [args.action]
    # A Windows server child can inherit pipe handles and keep communicate()
    # waiting after pg_ctl exits. A temporary file avoids that EOF dependency.
    with tempfile.TemporaryFile() as output:
        result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=output,
                                stderr=subprocess.STDOUT, **hidden)
        output.seek(0)
        message = output.read().decode(errors="replace").rstrip()
    if message:
        print(message)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())

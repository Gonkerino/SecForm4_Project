"""Consistent CLI progress; research functions remain independent of tqdm."""

import sys
import time

from tqdm import tqdm


class Progress:
    def __init__(self, description, unit, total=None):
        self.description, self.unit = description, unit
        self.total = total
        self.bar = None
        self.current = self.failures = 0
        self.last_log = 0

    def __enter__(self):
        self.stream = sys.stderr
        self.terminal = self.stream.isatty()
        self.write(f"{self.description}: preparing...")
        if self.total is not None:
            self.start(self.total)
        return self

    def write(self, message):
        if self.bar is not None and self.terminal:
            self.bar.write(message, file=self.stream)
        else:
            print(message, file=self.stream, flush=True)

    def start(self, total):
        self.total = total
        self.bar = tqdm(total=total, desc=self.description, unit=self.unit,
                        file=self.stream, disable=not self.terminal,
                        dynamic_ncols=True, mininterval=0.2, miniters=1,
                        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                                   "[{elapsed}<{remaining}, {rate_fmt}{postfix}]")
        if not self.terminal or total == 0:
            self.write(f"{self.description}: {total:,} {self.unit}s queued")

    def __call__(self, event):
        if isinstance(event, str):
            self.write(event)
            self.advance(self.current + 1, "")
            return
        kind = event["event"]
        if kind == "loading":
            return
        if kind == "start":
            if "total_indexed" in event:
                self.write("Indexed: {total_indexed:,}; downloaded: {already_downloaded:,}; "
                           "remaining: {pending_before_run:,}; previous failures: "
                           "{latest_failed_before_run:,}".format(**event))
            self.start(event["total"])
            return
        if event.get("status") == "failed":
            self.failures += 1
            self.write(f"Failed {event['accession']}: {event.get('message', '')}")
        problems = event.get("problems", self.failures)
        self.advance(event["current"], f"problems={problems:,}")

    def advance(self, current, detail):
        delta = current - self.current
        self.current = current
        self.bar.set_postfix_str(detail, refresh=False)
        self.bar.update(delta)
        now = time.monotonic()
        if not self.terminal and (current == 1 or current == self.total or now - self.last_log >= 10):
            self.write(f"{self.description}: {current:,}/{self.total:,} {self.unit}s {detail}".rstrip())
            self.last_log = now

    def __exit__(self, exc_type, exc, traceback):
        if self.bar is not None:
            self.bar.close()

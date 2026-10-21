"""One bounded, paced HTTP implementation for index discovery."""

from __future__ import annotations

import email.utils
import http.client
import random
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone

from .config import Config


class FetchError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class RateLimiter:
    """Space every request start, without initial bursts or double-counted sleeps."""

    def __init__(self, rps: float, *, clock=time.monotonic, sleep=time.sleep):
        self.period = 1 / rps
        self.clock, self.sleep = clock, sleep
        self.next_start = 0.0
        self.lock = threading.Lock()

    def acquire(self):
        with self.lock:
            wait = max(0.0, self.next_start - self.clock())
            if wait:
                self.sleep(wait)
            self.next_start = self.clock() + self.period


def retry_delay(value: str | None, fallback: float, *, now: datetime | None = None) -> float:
    if value:
        try:
            delay = float(value)
        except ValueError:
            try:
                stamp = email.utils.parsedate_to_datetime(value)
                if stamp.tzinfo is None:
                    stamp = stamp.replace(tzinfo=timezone.utc)
                delay = (stamp - (now or datetime.now(timezone.utc))).total_seconds()
            except (TypeError, ValueError, OverflowError):
                delay = fallback
        if delay != delay or delay == float("inf"):
            delay = fallback
        return max(0.0, delay)
    return fallback


class SecOnlyRedirect(urllib.request.HTTPRedirectHandler):
    def __init__(self, limiter):
        super().__init__()
        self.limiter = limiter

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlsplit(newurl).hostname not in {"www.sec.gov", "sec.gov"}:
            raise FetchError("SEC response redirected outside the permitted source hosts")
        if urllib.parse.urlsplit(newurl).scheme != "https":
            raise FetchError("SEC response redirected to a non-HTTPS source")
        self.limiter.acquire()
        return super().redirect_request(req, fp, code, msg, headers, newurl)


@dataclass(frozen=True)
class Response:
    body: bytes
    metadata: dict


class HttpClient:
    def __init__(self, cfg: Config):
        cfg.require_identity()
        self.cfg = cfg
        self.limiter = RateLimiter(cfg.max_rps)
        self.opener = urllib.request.build_opener(SecOnlyRedirect(self.limiter))

    def get(self, url: str, *, max_bytes: int | None = None) -> Response:
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != "https" or parsed.hostname != "www.sec.gov":
            raise FetchError("Only HTTPS www.sec.gov source URLs are supported")
        request = urllib.request.Request(url, headers={"User-Agent": self.cfg.user_agent,
                                                      "Accept-Encoding": "identity"})
        max_bytes = self.cfg.max_index_bytes if max_bytes is None else max_bytes
        deadline = time.monotonic() + (900 if max_bytes > self.cfg.max_index_bytes else 180)
        for attempt in range(self.cfg.max_attempts):
            if time.monotonic() >= deadline:
                raise FetchError("Request elapsed-time budget exhausted; retry this incomplete run later")
            self.limiter.acquire()
            retry_after = None
            status = None
            try:
                remaining = max(0.1, deadline - time.monotonic())
                with self.opener.open(request, timeout=min(self.cfg.timeout_seconds, remaining)) as response:
                    if response.status != 200:
                        raise FetchError(f"Unexpected HTTP {response.status}; complete index required", response.status)
                    parts, byte_count = [], 0
                    while True:
                        if time.monotonic() >= deadline:
                            raise FetchError("Request elapsed-time budget exhausted while reading index")
                        chunk = response.read(min(65536, max_bytes + 1 - byte_count))
                        if not chunk:
                            break
                        parts.append(chunk)
                        byte_count += len(chunk)
                        if byte_count > max_bytes:
                            raise FetchError("Response exceeds the configured byte limit")
                    body = b"".join(parts)
                    declared = response.headers.get("Content-Length")
                    if declared is not None and int(declared) != len(body):
                        raise FetchError("Index transfer ended before the declared content length")
                    return Response(body, {"status": response.status, "url": response.url,
                                           "etag": response.headers.get("ETag"),
                                           "last_modified": response.headers.get("Last-Modified"),
                                           "content_type": response.headers.get("Content-Type"),
                                           "network_attempts": attempt + 1})
            except urllib.error.HTTPError as exc:
                status = exc.code
                retry_after = exc.headers.get("Retry-After")
                exc.close()
                if status not in {429, 500, 502, 503, 504}:
                    raise FetchError(f"SEC returned HTTP {status}; coverage remains unresolved", status) from exc
                message = f"SEC returned HTTP {status} after {attempt + 1} attempts"
            except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
                # Avoid logging request headers or contact details.
                message = f"Network request failed ({type(exc).__name__}) after {attempt + 1} attempts"
            if attempt + 1 == self.cfg.max_attempts:
                raise FetchError(message, status)
            delay = retry_delay(retry_after, min(2**attempt, 30) + random.uniform(0, 0.5))
            if delay > 60 or time.monotonic() + delay >= deadline:
                raise FetchError("Source requested a retry delay over 60 seconds; retry this incomplete run later", status)
            time.sleep(delay)
        raise AssertionError("Unreachable request loop")

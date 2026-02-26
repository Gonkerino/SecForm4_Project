from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm


# --- Defaults ---
ARCHIVES_BASE_URL = "https://www.sec.gov/Archives/"
DEFAULT_USER_AGENT = "Malcolm MacLeod (M.MacLeod-14@sms.ed.ac.uk)"


# --- Thread-local session (requests.Session is not thread-safe) ---
_thread_local = threading.local()


def _make_session(user_agent: str, retries: int) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
    retry = Retry(
        total=retries,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=64, pool_maxsize=64)
    s.mount("https://", adapter)
    return s


def get_session(user_agent: str, retries: int) -> requests.Session:
    s = getattr(_thread_local, "session", None)
    if s is None:
        s = _make_session(user_agent=user_agent, retries=retries)
        _thread_local.session = s
    return s


# --- Rate limiter (global across all threads) ---
class RateLimiter:
    def __init__(self, rps: float):
        self.period = 1.0 / max(rps, 0.1)
        self._next_t = 0.0
        self._lock = threading.Lock()

    def acquire(self):
        with self._lock:
            now = time.monotonic()
            if now < self._next_t:
                time.sleep(self._next_t - now)
                now = time.monotonic()
            self._next_t = max(self._next_t + self.period, now)


# --- Path helpers ---
def normalize_archive_path(p: str) -> str:
    # index paths usually look like: "edgar/data/....."
    # sometimes you might have a leading "/"
    p = str(p).strip()
    while p.startswith("/"):
        p = p[1:]
    return p


def year_from_accession(accession: str) -> int | None:
    """
    Handles:
      - dashed: 0000320193-23-000119
      - undashed: 000032019323000119
    Returns full year as int, or None if not parseable.
    """
    acc = accession
    if acc.endswith(".txt") or acc.endswith(".htm") or acc.endswith(".html") or acc.endswith(".xml"):
        acc = acc.rsplit(".", 1)[0]

    yy = None
    if "-" in acc:
        parts = acc.split("-")
        if len(parts) >= 2 and parts[1].isdigit():
            yy = int(parts[1])
    else:
        # undashed accession: ##########YY######
        if len(acc) >= 12 and acc[10:12].isdigit():
            yy = int(acc[10:12])

    if yy is None:
        return None

    # SEC has 1994+ in your index. This keeps 94-99 in 1900s, else 2000s.
    return (1900 + yy) if yy >= 80 else (2000 + yy)


def local_path_from_archive(archive_path: str, out_dir: Path, layout: str = "year_cik_accession") -> Path:
    """
    layout options:
      - "mirror": mirrors the SEC archive path under out_dir
      - "year_cik_accession": out_dir / YEAR / CIK / ACCESSION / filename
    """
    archive_path = normalize_archive_path(archive_path)
    if layout == "mirror":
        return out_dir / archive_path

    parts = archive_path.split("/")
    # expected: edgar/data/{cik}/{accession_or_folder}/{filename?}
    # daily index paths are usually: edgar/data/{cik}/{accession}.txt
    cik = parts[2] if len(parts) >= 3 else "unknown_cik"

    # If path ends with .txt, that's the file, and the "accession" is the filename stem
    filename = parts[-1]
    accession_guess = filename.rsplit(".", 1)[0]

    yr = year_from_accession(accession_guess)
    yr_folder = str(yr) if yr is not None else "unknown_year"

    return out_dir / yr_folder / cik / accession_guess / filename


# --- Core download logic ---
def download_one(
    archive_path: str,
    out_dir: Path,
    limiter: RateLimiter,
    base_url: str,
    user_agent: str,
    retries: int,
    timeout: int,
    layout: str,
) -> tuple[str, str | None]:
    """
    Returns (archive_path, error). error=None on success or skip.
    """
    archive_path = normalize_archive_path(archive_path)
    out_path = local_path_from_archive(archive_path, out_dir=out_dir, layout=layout)

    if out_path.exists():
        return archive_path, None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    url = base_url + archive_path

    s = get_session(user_agent=user_agent, retries=retries)

    try:
        limiter.acquire()
        r = s.get(url, timeout=timeout)

        if r.status_code == 200 and r.content:
            out_path.write_bytes(r.content)
            return archive_path, None

        return archive_path, f"HTTP {r.status_code}"

    except Exception as e:
        return archive_path, f"EXC {type(e).__name__}: {e}"


def _to_int_yyyymmdd(series: pd.Series) -> pd.Series:
    # Handles "YYYYMMDD", "YYYY-MM-DD", int-like, etc.
    s = series.astype("string").str.replace("-", "", regex=False).str.slice(0, 8)
    return pd.to_numeric(s, errors="coerce").astype("Int64")


@dataclass
class DownloadConfig:
    out_dir: Path = Path("./form4_filings")
    index_pkl: str = "all_form4_filings.pkl"
    base_url: str = ARCHIVES_BASE_URL
    user_agent: str = DEFAULT_USER_AGENT

    max_workers: int = 12
    target_rps: float = 5.0   # stay comfortably below SEC 10 rps cap :contentReference[oaicite:1]{index=1}

    timeout: int = 30
    retries: int = 5

    min_date: int | None = None  # YYYYMMDD
    max_date: int | None = None  # YYYYMMDD
    layout: str = "year_cik_accession"  # or "mirror"

    max_items: int | None = None  # for quick testing


def load_index(config: DownloadConfig, df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None:
        df = pd.read_pickle(config.index_pkl)

    # Expect columns: Company, Form, CIK, Date, Path
    if "Form" in df.columns:
        df = df[df["Form"].isin(["4", "4/A"])]

    if "Date" in df.columns and (config.min_date or config.max_date):
        d = _to_int_yyyymmdd(df["Date"])
        if config.min_date is not None:
            df = df[d >= config.min_date]
        if config.max_date is not None:
            df = df[d <= config.max_date]

    df = df.dropna(subset=["Path"]).copy()
    return df.reset_index(drop=True)


def download_from_index(df: pd.DataFrame, config: DownloadConfig) -> dict:
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # unique paths to avoid re-downloading duplicates
    paths = pd.unique(df["Path"].astype(str))
    paths = [normalize_archive_path(p) for p in paths]

    if config.max_items is not None:
        paths = paths[: config.max_items]

    # pre-filter missing (fast)
    jobs = []
    for p in paths:
        if not local_path_from_archive(p, out_dir=out_dir, layout=config.layout).exists():
            jobs.append(p)

    stats = {"ok": 0, "fail": 0, "skipped": len(paths) - len(jobs)}
    failures = defaultdict(list)

    if not jobs:
        print("✅ All filings already present for this selection.")
        return stats

    limiter = RateLimiter(config.target_rps)

    with ThreadPoolExecutor(max_workers=config.max_workers) as ex:
        futs = [
            ex.submit(
                download_one,
                archive_path=p,
                out_dir=out_dir,
                limiter=limiter,
                base_url=config.base_url,
                user_agent=config.user_agent,
                retries=config.retries,
                timeout=config.timeout,
                layout=config.layout,
            )
            for p in jobs
        ]

        for fut in tqdm(as_completed(futs), total=len(futs), desc="📥 Downloading Form 4 / 4A"):
            path, err = fut.result()
            if err is None:
                stats["ok"] += 1
            else:
                stats["fail"] += 1
                failures[err].append(path)

    # write failure log
    if failures:
        fail_rows = []
        for reason, items in failures.items():
            for p in items:
                fail_rows.append({"Path": p, "Reason": reason})
        pd.DataFrame(fail_rows).to_csv(out_dir / "failed_downloads.csv", index=False)

    print(f"\n✅ Done. ok={stats['ok']}  fail={stats['fail']}  skipped={stats['skipped']}")
    if failures:
        print(f"⚠️ Wrote failure log to: {out_dir / 'failed_downloads.csv'}")
        # brief summary
        print("Failures by reason:")
        for reason, items in failures.items():
            print(f"  - {reason}: {len(items)}")

    return stats


def download_form4_filings(config: DownloadConfig, df: pd.DataFrame | None = None) -> dict:
    """
    Convenience wrapper:
      - loads index (pkl or passed df)
      - downloads filings
    """
    df2 = load_index(config, df=df)
    return download_from_index(df2, config)

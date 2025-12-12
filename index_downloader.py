import os
import requests
from bs4 import BeautifulSoup
from time import sleep
import pandas as pd
from glob import glob
from tqdm import tqdm

BASE_URL = "https://www.sec.gov/Archives/edgar/daily-index/"
USER_AGENT = "Malcolm MacLeod (M.MacLeod-14@sms.ed.ac.uk)"
DOWNLOAD_DIR = "./sec_idx_files"
HEADERS = {"User-Agent": USER_AGENT}

def fetch_directory_listing(url: str) -> list[str]:
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        res.raise_for_status()
    except Exception as e:
        print(f"❌ Failed to access {url}: {e}")
        return []

    soup = BeautifulSoup(res.text, "html.parser")
    return [
        a["href"]
        for a in soup.find_all("a")
        if a.get("href", "").endswith(".idx") and "company" in a["href"]
    ]

def download_file_if_missing(url: str, local_path: str) -> bool:
    if os.path.exists(local_path):
        return False  # just skip silently

    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        res.raise_for_status()
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(res.content)
        # If you *want* a message, keep only this one:
        # print(f"⬇️ Downloaded: {local_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {url}: {e}")
        return False


    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        res.raise_for_status()
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(res.content)
        print(f"⬇️ Downloaded: {local_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {url}: {e}")
        return False

def parse_company_idx(file_path: str) -> list[list[str]]:
    with open(file_path, "r", encoding="latin1") as f:
        lines = f.readlines()

    try:
        data_start = next(
            i for i, line in enumerate(lines) if line.startswith("Company Name")
        ) + 1
    except StopIteration:
        return []

    rows = []
    for line in lines[data_start:]:
        if not line.strip():
            continue
        try:
            parts = line.rsplit(maxsplit=4)
            if len(parts) < 5:
                continue
            company, form, cik, date, path = parts
            if form in ("4", "4/A"):
                rows.append([company, form, cik, date, path])
        except Exception as e:
            print(f"⚠️ Line parsing error: {e}\nLine: {line}")
    return rows

from concurrent.futures import ThreadPoolExecutor

def aggregate_all_form4(idx_root: str = DOWNLOAD_DIR) -> pd.DataFrame:
    idx_files = glob(f"{idx_root}/**/company.*.idx", recursive=True)
    print(f"📁 Found {len(idx_files)} idx files to process...")

    all_rows: list[list[str]] = []
    if not idx_files:
        return pd.DataFrame(columns=["Company", "Form", "CIK", "Date", "Path"])

    # Use a thread pool to parse files in parallel
    max_workers = min(8, os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for rows in tqdm(ex.map(parse_company_idx, idx_files),
                         total=len(idx_files),
                         desc="🔍 Processing .idx files"):
            if rows:
                all_rows.extend(rows)

    print("✅ Finished processing all idx files.")
    return pd.DataFrame(all_rows, columns=["Company", "Form", "CIK", "Date", "Path"])


def download_all_idx(
    start_year: int = 1994,
    end_year: int = 2025,
    verbose: bool = False,
) -> None:
    """
    Download all company*.idx files between start_year and end_year.

    If verbose=False (default), prints only one summary line per year.
    """

    for year in range(start_year, end_year + 1):
        year_downloaded = 0
        year_skipped = 0
        downloaded_by_quarter = []

        for q in range(1, 5):
            quarter = f"QTR{q}"
            quarter_url = f"{BASE_URL}{year}/{quarter}/"

            if verbose:
                print(f"\n🔍 Scanning {quarter_url}")

            idx_files = fetch_directory_listing(quarter_url)
            if not idx_files:
                if verbose:
                    print(f"⏩ No .idx files found for {quarter_url}")
                continue

            quarter_dir = os.path.join(DOWNLOAD_DIR, str(year), quarter)
            os.makedirs(quarter_dir, exist_ok=True)

            # One directory listing → cheap existence check
            existing = set(os.listdir(quarter_dir))

            downloaded = skipped = 0
            for fname in idx_files:
                if fname in existing:
                    skipped += 1
                    continue

                file_url = f"{quarter_url}{fname}"
                local_path = os.path.join(quarter_dir, fname)
                if download_file_if_missing(file_url, local_path):
                    downloaded += 1
                    sleep(0.11)

            year_downloaded += downloaded
            year_skipped += skipped
            if downloaded > 0:
                downloaded_by_quarter.append(f"{quarter}:{downloaded}")

            # In verbose mode you still see the per-quarter summary
            if verbose:
                print(f"📂 {year} {quarter}: downloaded={downloaded}, skipped={skipped}")

        # --- Year summary (always shown) ---
        if downloaded_by_quarter:
            detail = ", ".join(downloaded_by_quarter)
            print(f"📆 {year}: downloaded={year_downloaded} new files ({detail}), "
                  f"skipped={year_skipped}")
        else:
            # Nothing new this year – single, compact line
            print(f"📆 {year}: no new files (all {year_skipped} already present)")

def build_form4_index() -> pd.DataFrame:
    df = aggregate_all_form4(DOWNLOAD_DIR)
    df.to_csv("all_form4_filings.csv", index=False)
    df.to_pickle("all_form4_filings.pkl")
    print(f"📦 Exported Form 4 data: {df.shape[0]} rows")
    return df

if __name__ == "__main__":
    download_all_idx(1994, 2025)
    build_form4_index()
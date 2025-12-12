import os
import asyncio
import aiohttp
import aiofiles
from bs4 import BeautifulSoup
import pandas as pd
from glob import glob
from tqdm import tqdm
import concurrent.futures
from datetime import date
from pathlib import Path

# --- Constants and Configuration ---
MASTER_PKL = "all_form4_filings.pkl"
MASTER_CSV = "all_form4_filings.csv"
BASE_URL = "https://www.sec.gov/Archives/edgar/daily-index/"
USER_AGENT = "Malcolm MacLeod (M.MacLeod-14@sms.ed.ac.uk)"
DOWNLOAD_DIR = Path("./sec_idx_files")
HEADERS = {"User-Agent": USER_AGENT}
TARGET_FORMS = {"4", "4/A"}  # Use a set for fast O(1) lookups

# --- Asynchronous Networking Functions ---
async def fetch_directory_listing(session: aiohttp.ClientSession, url: str) -> list[str]:
    """Asynchronously fetches all .idx file names from a directory URL."""
    try:
        async with session.get(url, headers=HEADERS) as response:
            response.raise_for_status()
            text = await response.text()
            soup = BeautifulSoup(text, "html.parser")
            
            # --- CORRECTED CODE ---
            # This version safely handles cases where 'href' might be missing or not a string.
            return [
                href
                for a in soup.find_all("a")
                if (href := a.get("href"))      # 1. Get href and ensure it's not None.
                and isinstance(href, str)       # 2. Ensure it's a string.
                and href.endswith(".idx")       # 3. Now safely perform string checks.
                and "company" in href
            ]
            # --- END CORRECTION ---

    except aiohttp.ClientError as e:
        print(f"❌ Failed to fetch directory {url}: {e}")
        return []

async def download_file(session: aiohttp.ClientSession, url: str, local_path: Path, semaphore: asyncio.Semaphore):
    """Asynchronously downloads a single file if it doesn't exist locally, respecting the semaphore."""
    if local_path.exists():
        return "skipped"

    async with semaphore:
        try:
            async with session.get(url, headers=HEADERS) as response:
                response.raise_for_status()
                content = await response.read()
                local_path.parent.mkdir(parents=True, exist_ok=True)
                async with aiofiles.open(local_path, "wb") as f:
                    await f.write(content)
                # Adhere to SEC rate limits (10 requests/sec)
                await asyncio.sleep(0.1)
                return "downloaded"
        except aiohttp.ClientError as e:
            print(f"❌ Failed to download {url}: {e}")
            return "failed"

# --- CPU-Bound Parsing Function (Unchanged) ---
def parse_company_idx(file_path: str) -> list[tuple]:
    """
    Parses a single .idx file.
    Optimized for speed: uses a file iterator, fast string operations,
    and returns tuples for a lower memory footprint.
    """
    rows = []
    try:
        with open(file_path, "r", encoding="latin1") as f:
            for line in f:
                if line.startswith("---"):
                    break
            for line in f:
                if len(line) < 20:
                    continue
                parts = line.strip().rsplit(None, 4)
                if len(parts) == 5 and parts[1] in TARGET_FORMS:
                    rows.append(tuple(parts))
    except IOError as e:
        print(f"⚠️  Error reading {file_path}: {e}")
    return rows

# --- Data Aggregation (Unchanged, already optimized) ---
def aggregate_all_form4(idx_root: Path = DOWNLOAD_DIR) -> pd.DataFrame:
    """Aggregates all Form 4 filings from local .idx files using a process pool."""
    idx_files = list(idx_root.rglob("company.*.idx"))
    print(f"📁 Found {len(idx_files)} .idx files to process...")

    if not idx_files:
        return pd.DataFrame(columns=["Company", "Form", "CIK", "Date", "Path"])

    all_rows = []
    max_workers = os.cpu_count()
    print(f"🚀 Starting parallel parsing with {max_workers} processes...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(parse_company_idx, str(file)) for file in idx_files]
        
        progress_bar = tqdm(
            concurrent.futures.as_completed(futures),
            total=len(idx_files),
            desc="🔍 Processing .idx files"
        )
        
        for future in progress_bar:
            try:
                rows = future.result()
                if rows:
                    all_rows.extend(rows)
            except Exception as e:
                print(f"A parsing process failed: {e}")

    print("✅ Finished processing. Converting to DataFrame...")
    return pd.DataFrame(all_rows, columns=["Company", "Form", "CIK", "Date", "Path"])

# --- Main Orchestration Logic ---
async def download_all_idx_async(
    start_year: int = 1994,
    end_year: int = date.today().year,
    update_mode: bool = False,
):
    """
    Asynchronously downloads all company.idx files.
    If update_mode=True, it only scans the current year for new files.
    """
    print("🚀 Initializing asynchronous download process...")
    
    years_to_scan = range(start_year, end_year + 1)
    if update_mode:
        scan_year = date.today().year
        years_to_scan = range(scan_year, scan_year + 1)
        print(f"🚀 Running in fast update mode. Checking only for year {scan_year}...")
    else:
        print(f"🔍 Running in full scan mode ({start_year}-{end_year}). This may take a while on the first run.")

    # Create a list of all quarterly index URLs to scan
    quarter_urls = []
    for year in years_to_scan:
        start_quarter = 3 if year == 1994 else 1
        for q in range(start_quarter, 5):
            quarter_urls.append(f"{BASE_URL}{year}/QTR{q}/")
    
    # Use a single session for all requests for efficiency
    async with aiohttp.ClientSession(headers=HEADERS) as session:
        # 1. Concurrently fetch all directory listings
        dir_listing_tasks = [fetch_directory_listing(session, url) for url in quarter_urls]
        all_remote_files = await asyncio.gather(*dir_listing_tasks)
        
        # 2. Create download tasks for all files that need to be downloaded
        download_tasks = []
        for url, remote_files in zip(quarter_urls, all_remote_files):
            if not remote_files:
                continue
            
            # Extract year/quarter from URL
            parts = url.strip("/").split("/")
            year, quarter = parts[-2], parts[-1]
            quarter_dir = DOWNLOAD_DIR / year / quarter
            
            for fname in remote_files:
                file_url = f"{url}{fname}"
                local_path = quarter_dir / fname
                download_tasks.append(
                    (file_url, local_path)
                )
        
        print(f"🔎 Found {len(download_tasks)} total files to check.")
        if not download_tasks:
            print("✅ All files are already up-to-date.")
            return

        # 3. Execute downloads concurrently with a rate limit
        # The semaphore limits concurrent requests to 10
        semaphore = asyncio.Semaphore(10)
        
        tasks_to_run = [
            download_file(session, url, path, semaphore) for url, path in download_tasks
        ]
        
        results = await asyncio.gather(*tasks_to_run)
        
        downloaded_count = results.count("downloaded")
        skipped_count = results.count("skipped")
        failed_count = results.count("failed")

        print("\n--- Download Summary ---")
        print(f"✅ Downloaded: {downloaded_count} new files.")
        print(f"⏩ Skipped: {skipped_count} existing files.")
        if failed_count > 0:
            print(f"❌ Failed: {failed_count} files.")
        print("------------------------\n")

def build_form4_index() -> pd.DataFrame:
    """Builds the final index from downloaded files and saves it."""
    df = aggregate_all_form4(DOWNLOAD_DIR)
    if not df.empty:
        df.to_csv(MASTER_CSV, index=False)
        df.to_pickle(MASTER_PKL)
        print(f"📦 Exported Form 4 data: {df.shape[0]} rows to {MASTER_CSV} and {MASTER_PKL}")
    else:
        print("⚠️ No Form 4 data found. No files were exported.")
    return df

# --- Main execution block ---
if __name__ == "__main__":
    # To run the async function from a .py file
    # Set desired parameters here
    asyncio.run(download_all_idx_async(start_year=1994, end_year=2025, update_mode=False))
    build_form4_index()
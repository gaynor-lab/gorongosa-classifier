#!/usr/bin/env python3
import os
import io
import re
import time
import argparse
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, get_context
from PIL import Image, UnidentifiedImageError

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# === Constants (edit as needed) ===
CSV_PATH = "full_df_filtered.csv"
SERVICE_ACCOUNT_FILE = "../../credentials.json"
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
SAVE_DIR = "/mnt/sharedstorage/sabdelazim/images/all_species_images"
LOG_PATH = "download_log.csv"
NUM_WORKERS = 4
MAX_RETRIES = 3

os.makedirs(SAVE_DIR, exist_ok=True)

# ---------- Helpers ----------
def extract_drive_id(url: str | None) -> str | None:
    """Handle common Drive URL forms."""
    if not url:
        return None
    try:
        if "/d/" in url:
            return url.split("/d/")[1].split("/")[0]
        # handle uc?id= style
        m = re.search(r"[?&]id=([A-Za-z0-9_\-]+)", url)
        if m:
            return m.group(1)
    except Exception:
        pass
    return None

def sanitize(text):
    return re.sub(r"\W+", "", str(text)).strip().lower() or "unknown"

def load_download_log():
    if os.path.exists(LOG_PATH):
        return pd.read_csv(LOG_PATH)
    return pd.DataFrame(columns=["filename", "status"])

def expected_filename(row) -> str:
    original_filename = os.path.splitext(row.get("filename", "unnamed.jpg"))[0]
    site = sanitize(row.get("site", "unknown"))
    species = sanitize(row.get("species", "unknown"))
    return f"{original_filename}_{site}_{species}.jpg"

def is_image_bad(path: str) -> bool:
    """Zero-byte OR not decodable by PIL."""
    try:
        if not os.path.exists(path):
            return True
        if os.path.getsize(path) == 0:
            return True
        # decode test
        with Image.open(path) as im:
            im.verify()
        return False
    except (UnidentifiedImageError, OSError, ValueError):
        return True
    except Exception:
        # conservative: treat as bad
        return True

# ---------- Drive client per process ----------
def _build_drive():
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)

# ---------- Download (single row) ----------
def download_file(row_tuple):
    index, row = row_tuple
    drive = _build_drive()
    file_id = extract_drive_id(row.get("filepath", ""))
    new_filename = expected_filename(row)
    final_path = os.path.join(SAVE_DIR, new_filename)
    tmp_path = final_path + ".part"

    if not file_id:
        return (new_filename, "⚠️ Invalid ID")

    # Ensure parent exists
    os.makedirs(os.path.dirname(final_path), exist_ok=True)

    # Try download into .part, then move
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

            request = drive.files().get_media(fileId=file_id)
            with io.FileIO(tmp_path, "wb") as fh:
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()

            # Validate temp file
            if is_image_bad(tmp_path):
                raise RuntimeError("Downloaded file failed validation (empty or unreadable).")

            # Replace target atomically
            if os.path.exists(final_path):
                try:
                    os.remove(final_path)
                except Exception:
                    pass
            os.replace(tmp_path, final_path)
            return (new_filename, "🔁 Redownloaded")
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(2 * attempt)  # backoff
                continue
            # cleanup tmp
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            return (new_filename, f"⚠️ Failed: {e}")

# ---------- Build worklist ----------
def find_bad_rows(df: pd.DataFrame, include_failed_from_log: bool = True) -> list[tuple[int, pd.Series]]:
    """
    Returns list of (index, row) to redownload:
      - any existing local file that is 'bad' (0B or undecodable)
      - any row with a previously 'Failed' status in the log (optional)
    """
    log_df = load_download_log()
    failed_set = set()
    if include_failed_from_log and not log_df.empty:
        failed_set = set(
            log_df.loc[log_df["status"].fillna("").str.startswith("⚠️"), "filename"].tolist()
        )

    work = []
    for index, row in df.iterrows():
        fname = expected_filename(row)
        fpath = os.path.join(SAVE_DIR, fname)

        need = False
        if os.path.exists(fpath):
            if is_image_bad(fpath):
                need = True
        else:
            # if missing entirely, you *could* also re-queue; but we focus on "bad only"
            pass

        if fname in failed_set:
            need = True

        if need:
            work.append((index, row))
    return work

def download_all_images(df: pd.DataFrame, mode: str = "bad_only"):
    """
    mode:
      - 'bad_only' (default): re-download only bad or previously failed
      - 'all': process all rows (will skip/keep already-good files)
    """
    log_df = load_download_log()
    downloaded_set = set(log_df.loc[log_df["status"].str.startswith("✅"), "filename"])

    if mode == "all":
        rows_to_download = [(i, r) for i, r in df.iterrows()]
    else:
        rows_to_download = find_bad_rows(df, include_failed_from_log=True)

    if not rows_to_download:
        print("No files to (re)download. All good 🎉")
        return

    new_log_entries = []
    with get_context("spawn").Pool(processes=NUM_WORKERS) as pool:
        for filename, status in tqdm(
            pool.imap_unordered(download_file, rows_to_download),
            total=len(rows_to_download),
            desc="Re-downloading" if mode != "all" else "Downloading",
        ):
            print(f"{status}: {filename}")
            new_log_entries.append({"filename": filename, "status": status})

    if new_log_entries:
        new_df = pd.DataFrame(new_log_entries)
        updated_log = pd.concat([log_df, new_df], ignore_index=True)
        updated_log = updated_log.drop_duplicates(subset=["filename"], keep="last")
        updated_log.to_csv(LOG_PATH, index=False)

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="Re-download bad Google Drive images only.")
    parser.add_argument("--csv", default=CSV_PATH, help="Path to CSV with columns: filename, site, species, filepath")
    parser.add_argument("--mode", choices=["bad_only", "all"], default="bad_only",
                        help="Redownload only bad/failed files (default) or all rows")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    download_all_images(df, mode=args.mode)

if __name__ == "__main__":
    main()
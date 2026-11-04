#!/usr/bin/env python3
"""
Build a manifest CSV of ghost (blank/background) camera trap crops.

Expected structure:
  ghost_photos/
    August_2016/
      A06/
        Ghost/
          1/            <- sometimes present, sometimes not, sometimes 2+ (1, 2, ...)
            img.jpg
          img2.jpg       <- images can also sit directly in Ghost/
    November_2016/
      ...

Walks each month folder -> each site folder -> the "Ghost" folder, recursively,
so it picks up images whether or not there's a numbered subfolder in between.

Usage:
  python3 build_ghost_manifest.py

Edit BASE_DIR and OUTPUT_CSV below if your paths differ.
"""

import csv
import os

BASE_DIR = "/arc/project/st-kgaynor-1/gorongosa_classifier/ghost_photos"
MONTH_FOLDERS = ["August_2016", "November_2016"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
OUTPUT_CSV = "ghost_manifest.csv"

rows = []
skipped_no_ghost = []

for month in MONTH_FOLDERS:
    month_dir = os.path.join(BASE_DIR, month)
    if not os.path.isdir(month_dir):
        print(f"WARNING: month folder not found, skipping: {month_dir}")
        continue

    site_names = sorted(
        d for d in os.listdir(month_dir)
        if os.path.isdir(os.path.join(month_dir, d))
    )

    for site in site_names:
        ghost_dir = os.path.join(month_dir, site, "Ghost")
        if not os.path.isdir(ghost_dir):
            skipped_no_ghost.append(os.path.join(month, site))
            continue

        for root, _dirs, files in os.walk(ghost_dir):
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext in IMAGE_EXTENSIONS:
                    full_path = os.path.abspath(os.path.join(root, fname))
                    rows.append({
                        "crop_path": full_path,
                        "species": "background",
                        "site": site,
                        "source": "ghost",
                    })

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["crop_path", "species", "site", "source"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")

if skipped_no_ghost:
    print(f"\nNote: {len(skipped_no_ghost)} site folder(s) had no 'Ghost' subfolder and were skipped:")
    for s in skipped_no_ghost:
        print(f"  - {s}")

# quick per-month / per-site sanity summary
from collections import Counter
month_counts = Counter()
for r in rows:
    for m in MONTH_FOLDERS:
        if f"/{m}/" in r["crop_path"]:
            month_counts[m] += 1
            break

print("\nRows per month:")
for m in MONTH_FOLDERS:
    print(f"  {m}: {month_counts.get(m, 0)}")

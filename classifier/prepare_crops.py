#!/usr/bin/env python3
"""
prepare_crops.py
- scans raw images
- runs MegaDetector incrementally 
- saves full_df_filtered.csv
- saves megadetector_dropped.csv

"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

from utilities import filter_bad_files
from detector import filter_df_with_megadetector_and_crop


# ------------------------------------------------------------------------------
# Project root & sys.path configuration
# ------------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ------------------------------------------------------------------------------
# Imports that depend on sys.path
# ------------------------------------------------------------------------------
from splitting import build_df_from_folder


# ------------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------------
BASE = Path("/scratch/st-kgaynor-1/gorongosa_classifier")

run_dir = BASE / "processing" / "resnet_training"
run_dir.mkdir(parents=True, exist_ok=True)

image_dir = Path("/arc/project/st-kgaynor-1/gorongosa_classifier/training_data")

filtered_all_path = run_dir / "full_df_filtered.csv"
dropped_all_path = run_dir / "megadetector_dropped.csv"

if not image_dir.exists():
    raise FileNotFoundError(f"image_dir not found: {image_dir}")

run_dir = str(run_dir)
image_dir = str(image_dir)
filtered_all_path = str(filtered_all_path)
dropped_all_path = str(dropped_all_path)


# ------------------------------------------------------------------------------
# MegaDetector config
# ------------------------------------------------------------------------------
config = {
    "cache_filtered_all": True,
    "use_megadetector_if_missing": True,
    "megadetector_model": "MDV5A",
    "megadetector_conf": 0.2,
    "megadetector_device": "cuda",
    "animals_only": True,
    "pad_frac": 0.10,
    "save_format": "jpg",
    "jpeg_quality": 95,
}


# ------------------------------------------------------------------------------
# Build CURRENT df from raw images
# ------------------------------------------------------------------------------
print("[info] Scanning image_dir to find current images...")
current_df = build_df_from_folder(image_dir)

if current_df is None or len(current_df) == 0:
    raise RuntimeError("build_df_from_folder() returned an empty dataframe. Check filename pattern / image_dir.")

for col in ("species", "site"):
    if col not in current_df.columns:
        raise ValueError(f"Expected column '{col}' in current_df but it is missing. Columns: {list(current_df.columns)}")
    current_df[col] = current_df[col].astype(str).str.strip().str.lower()

if "filename" not in current_df.columns:
    raise ValueError(f"Expected column 'filename' in current_df. Columns: {list(current_df.columns)}")

current_df = filter_bad_files(current_df, image_dir)


# ------------------------------------------------------------------------------
# Incremental MegaDetector filtering:
# skip images already in KEPT or DROPPED caches
# ------------------------------------------------------------------------------
if os.path.exists(filtered_all_path):
    print(f"[info] Using cached filtered CSV: {filtered_all_path}")
    filtered_all_df = pd.read_csv(filtered_all_path)

    for col in ("species", "site"):
        if col in filtered_all_df.columns:
            filtered_all_df[col] = filtered_all_df[col].astype(str).str.strip().str.lower()

    if "filename" not in filtered_all_df.columns:
        raise ValueError(f"Cached filtered CSV is missing 'filename' column: {filtered_all_path}")

    # load dropped cache if present
    if os.path.exists(dropped_all_path):
        dropped_all_df = pd.read_csv(dropped_all_path)
        if "filename" in dropped_all_df.columns:
            dropped_seen = set(dropped_all_df["filename"].astype(str).str.strip().str.lower())
        else:
            dropped_seen = set()
    else:
        dropped_all_df = pd.DataFrame()
        dropped_seen = set()

    kept_seen = set(filtered_all_df["filename"].astype(str).str.strip().str.lower())
    current_names = current_df["filename"].astype(str).str.strip().str.lower()

    seen = kept_seen | dropped_seen

    print(f"[debug] current_df rows: {len(current_df)}")
    print(f"[debug] kept_seen: {len(kept_seen)}")
    print(f"[debug] dropped_seen: {len(dropped_seen)}")
    print(f"[debug] total seen: {len(seen)}")

    is_new = ~current_names.isin(seen)
    new_df = current_df[is_new].reset_index(drop=True)

    print(f"[info] Cached kept images: {len(filtered_all_df)}")
    print(f"[info] Cached dropped images: {len(dropped_seen)}")
    print(f"[info] New images found: {len(new_df)}")

    if len(new_df) > 0 and bool(config["use_megadetector_if_missing"]):
        new_kept, new_dropped = filter_df_with_megadetector_and_crop(
            df=new_df,
            image_dir=image_dir,
            out_dir=run_dir,
            conf_thresh=float(config["megadetector_conf"]),
            model_name_or_path=str(config["megadetector_model"]),
            device=str(config["megadetector_device"]),
            animals_only=bool(config["animals_only"]),
            pad_frac=float(config["pad_frac"]),
            save_format=str(config["save_format"]),
            jpeg_quality=int(config["jpeg_quality"]),
            save_crops=False,
        )

        # append new kept
        filtered_all_df = pd.concat([filtered_all_df, new_kept], ignore_index=True)
        filtered_all_df = filtered_all_df.drop_duplicates(subset=["filename"]).reset_index(drop=True)

        # append new dropped
        if len(new_dropped) > 0:
            if os.path.exists(dropped_all_path):
                old_dropped = pd.read_csv(dropped_all_path)
                dropped_all_df = pd.concat([old_dropped, new_dropped], ignore_index=True)
            else:
                dropped_all_df = new_dropped.copy()

            if "filename" in dropped_all_df.columns:
                dropped_all_df["filename"] = dropped_all_df["filename"].astype(str).str.strip().str.lower()
                dropped_all_df = dropped_all_df.drop_duplicates(subset=["filename"]).reset_index(drop=True)

            print(f"[info] Updating cached dropped CSV -> {dropped_all_path}")
            dropped_all_df.to_csv(dropped_all_path, index=False)

        if bool(config["cache_filtered_all"]):
            print(f"[info] Updating cached filtered CSV -> {filtered_all_path}")
            filtered_all_df.to_csv(filtered_all_path, index=False)
    else:
        print("[info] No new images to run MegaDetector on (or MegaDetector disabled).")

else:
    print(f"[info] No cached filtered CSV. Running MegaDetector on ALL images ({len(current_df)}).")

    if bool(config["use_megadetector_if_missing"]):
        filtered_all_df, dropped_df = filter_df_with_megadetector_and_crop(
            df=current_df,
            image_dir=image_dir,
            out_dir=run_dir,
            conf_thresh=float(config["megadetector_conf"]),
            model_name_or_path=str(config["megadetector_model"]),
            device=str(config["megadetector_device"]),
            animals_only=bool(config["animals_only"]),
            pad_frac=float(config["pad_frac"]),
            save_format=str(config["save_format"]),
            jpeg_quality=int(config["jpeg_quality"]),
            save_crops=False,
        )
    else:
        print("[warn] MegaDetector disabled; using current_df after file sanity filtering.")
        filtered_all_df = current_df.copy()
        dropped_df = pd.DataFrame()

    if bool(config["cache_filtered_all"]):
        print(f"[info] Saving filtered CSV -> {filtered_all_path}")
        filtered_all_df.to_csv(filtered_all_path, index=False)

        print(f"[info] Saving dropped CSV -> {dropped_all_path}")
        dropped_df.to_csv(dropped_all_path, index=False)


# ------------------------------------------------------------------------------
# Final normalization
# ------------------------------------------------------------------------------
for col in ("species", "site"):
    if col in filtered_all_df.columns:
        filtered_all_df[col] = filtered_all_df[col].astype(str).str.strip().str.lower()

print("[done] Prepared MegaDetector crops and CSVs.")
print(f"[done] Filtered CSV: {filtered_all_path}")
print(f"[done] Dropped CSV: {dropped_all_path}")
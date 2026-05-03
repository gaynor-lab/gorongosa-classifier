"""
review_crops.py
===============
Two-step manual review pipeline for animal training crops.
 
STEP 1 — generate_review_crops
    Takes the crops already produced by detector.py (via training.py)
    and adds 100px of original-image context around each one for easier
    human review. Saves these padded crops into a review folder organised
    by species.
 
    Human then opens the review folder, deletes any crop that does not
    contain a real animal, and runs Step 2.
 
STEP 2 — recrop_after_review
    Looks at what crops survived human review.
    Goes back to the original source images and re-crops them using the
    original bounding box + pad_frac (default 0.10, matching training.py).
    Saves the final crops into a output folder that matches exactly what
    training.py expects: a flat folder of .jpg files named {stem}_crop{j}.jpg
 
    The full_df_filtered.csv is also updated to only include rows whose
    crops survived review.
 
Usage:
 
    # Step 1 — add 100px padding for review
    python review_crops.py generate \\
        --filtered_csv   /path/to/full_df_filtered.csv \\
        --image_dir      /path/to/all_species_images \\
        --review_dir     /path/to/review_crops
 
    # (Human deletes bad crops from review_dir)
 
    # Step 2 — re-crop survivors back to training format
    python review_crops.py recrop \\
        --filtered_csv   /path/to/full_df_filtered.csv \\
        --image_dir      /path/to/all_species_images \\
        --review_dir     /path/to/review_crops \\
        --output_dir     /path/to/crops \\
        --output_csv     /path/to/full_df_filtered_reviewed.csv
 
Dependencies:
    pip install pandas Pillow tqdm
"""
 
from __future__ import annotations
 
import argparse
import json
import os
from pathlib import Path
 
import pandas as pd
from PIL import Image
from tqdm import tqdm
 
 
# ──────────────────────────────────────────────────────────────
# Geometry helpers (mirrors detector.py exactly)
# ──────────────────────────────────────────────────────────────
 
def _clip(v: int, lo: int, hi: int) -> int:
    return max(lo, min(v, hi))
 
 
def _bbox_to_pixels(img: Image.Image, bbox: list) -> tuple:
    """
    Convert MD relative bbox [x, y, w, h] to pixel coords (x1, y1, x2, y2).
    Does NOT apply any padding.
    """
    W, H = img.size
    x, y, w, h = bbox
    x1 = int(round(x * W))
    y1 = int(round(y * H))
    x2 = int(round((x + w) * W))
    y2 = int(round((y + h) * H))
    return x1, y1, x2, y2
 
 
def _crop_with_pad_frac(img: Image.Image, bbox: list,
                         pad_frac: float = 0.10) -> Image.Image:
    """
    Identical to _crop_with_padding in detector.py.
    Padding is pad_frac * box_width and pad_frac * box_height.
    This is what training.py uses (pad_frac=0.10).
    """
    W, H = img.size
    x1, y1, x2, y2 = _bbox_to_pixels(img, bbox)
 
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad_x = int(round(bw * pad_frac))
    pad_y = int(round(bh * pad_frac))
 
    x1 = _clip(x1 - pad_x, 0, W)
    y1 = _clip(y1 - pad_y, 0, H)
    x2 = _clip(x2 + pad_x, 0, W)
    y2 = _clip(y2 + pad_y, 0, H)
 
    if x2 <= x1 or y2 <= y1:
        return img
 
    return img.crop((x1, y1, x2, y2))
 
 
def _crop_with_px_padding(img: Image.Image, bbox: list,
                           padding_px: int = 100) -> Image.Image:
    """
    Add a fixed number of pixels from the ORIGINAL IMAGE around the MD box.
    Used only for the human review step — not for training.
    """
    W, H = img.size
    x1, y1, x2, y2 = _bbox_to_pixels(img, bbox)
 
    x1 = _clip(x1 - padding_px, 0, W)
    y1 = _clip(y1 - padding_px, 0, H)
    x2 = _clip(x2 + padding_px, 0, W)
    y2 = _clip(y2 + padding_px, 0, H)
 
    if x2 <= x1 or y2 <= y1:
        return img
 
    return img.crop((x1, y1, x2, y2))
 
 
# ──────────────────────────────────────────────────────────────
# Step 1 — Generate review crops (100px padding)
# ──────────────────────────────────────────────────────────────
 
def generate_review_crops(
    filtered_csv: str,
    image_dir: str,
    review_dir: str,
    padding_px: int = 100,
    jpeg_quality: int = 95,
):
    """
    For every crop in full_df_filtered.csv, go back to the original image,
    add padding_px pixels of context around the MD bounding box, and save
    into review_dir/<species>/<crop_name>.jpg
 
    Organised by species so the human can focus on one species at a time
    and spot misidentifications more easily.
 
    After running this, open review_dir, delete any crop that does not
    contain a real animal, then run Step 2 (recrop_after_review).
    """
    review_path = Path(review_dir)
    review_path.mkdir(parents=True, exist_ok=True)
 
    df = pd.read_csv(filtered_csv)
    print(f"[step1] Loaded {len(df)} rows from {filtered_csv}")
 
    required = {"filename", "bboxes", "filename_crops"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )
 
    saved   = 0
    failed  = 0
    skipped = 0
 
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating review crops"):
        src_path = Path(image_dir) / str(row["filename"])
 
        if not src_path.exists():
            print(f"  Warning: source image not found: {src_path}")
            failed += 1
            continue
 
        try:
            bboxes      = json.loads(row["bboxes"])
            crop_names  = json.loads(row["filename_crops"])
            species     = str(row.get("species", "unknown")).strip().lower()
        except Exception as e:
            print(f"  Warning: could not parse row for {row['filename']}: {e}")
            failed += 1
            continue
 
        try:
            img = Image.open(src_path).convert("RGB")
        except Exception as e:
            print(f"  Warning: could not open {src_path}: {e}")
            failed += 1
            continue
 
        species_dir = review_path / species
        species_dir.mkdir(parents=True, exist_ok=True)
 
        for bbox, crop_name in zip(bboxes, crop_names):
            try:
                review_crop = _crop_with_px_padding(img, bbox, padding_px=padding_px)
                out_path    = species_dir / crop_name
                review_crop.save(out_path, quality=jpeg_quality)
                saved += 1
            except Exception as e:
                print(f"  Warning: failed to save {crop_name}: {e}")
                skipped += 1
 
    print(f"\n[step1] Done.")
    print(f"  Saved  : {saved}")
    print(f"  Failed : {failed}")
    print(f"  Skipped: {skipped}")
    print(f"\nReview folder: {review_dir}")
    print("Delete any crop that does not contain a real animal.")
    print("Then run:  python review_crops.py recrop ...")
 
 
# ──────────────────────────────────────────────────────────────
# Step 2 — Re-crop survivors back to training format
# ──────────────────────────────────────────────────────────────
 
def recrop_after_review(
    filtered_csv: str,
    image_dir: str,
    review_dir: str ,
    output_dir: str,
    output_csv: str,
    pad_frac: float = 0.10,
    jpeg_quality: int = 95,
):
    """
    After human review:
      1. Collect all crop filenames still present in review_dir (the survivors).
      2. For each surviving crop, go back to the ORIGINAL image and re-crop
         using the same pad_frac as training.py (default 0.10).
         This removes the 100px review padding and produces the same format
         as detector.py originally would have made.
      3. Save final crops to output_dir (flat folder, same as training.py expects).
      4. Update full_df_filtered.csv to only include rows where at least one
         crop survived, with filename_crops/bboxes/det_confs updated to
         reflect only the surviving detections.
      5. Save the updated CSV to output_csv.
 
    output_dir should be the same path as cropped_image_dir in training.py.
    output_csv should replace full_df_filtered.csv in training.py.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
 
    df = pd.read_csv(filtered_csv)
    print(f"[step2] Loaded {len(df)} rows from {filtered_csv}")
 
    # Collect all crop names that survived human review
    # review_dir/<species>/<crop_name>.jpg
    survivors: set[str] = set()
    review_path = Path(review_dir)
    for f in review_path.rglob("*"):
        if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            survivors.add(f.name)
 
    print(f"[step2] Crops surviving review: {len(survivors)}")
 
    if len(survivors) == 0:
        print("  No survivors found. Did you run Step 1 and delete bad crops?")
        print(f"  Looking in: {review_dir}")
        return
 
    kept_rows = []
    saved     = 0
    failed    = 0
 
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Re-cropping survivors"):
        src_path = Path(image_dir) / str(row["filename"])
 
        try:
            bboxes      = json.loads(row["bboxes"])
            crop_names  = json.loads(row["filename_crops"])
            det_confs   = json.loads(row.get("det_confs", "[]") or "[]")
            det_cats    = json.loads(row.get("det_categories", "[]") or "[]")
        except Exception as e:
            print(f"  Warning: could not parse row for {row['filename']}: {e}")
            failed += 1
            continue
 
        # Filter to only the crops that survived review
        surviving = [
            (bbox, name, conf, cat)
            for bbox, name, conf, cat in zip(
                bboxes,
                crop_names,
                det_confs   if det_confs else [None] * len(crop_names),
                det_cats    if det_cats  else [None] * len(crop_names),
            )
            if name in survivors
        ]
 
        if not surviving:
            # All crops for this image were deleted — skip the whole row
            continue
 
        if not src_path.exists():
            print(f"  Warning: source image not found: {src_path}")
            failed += 1
            continue
 
        try:
            img = Image.open(src_path).convert("RGB")
        except Exception as e:
            print(f"  Warning: could not open {src_path}: {e}")
            failed += 1
            continue
 
        surviving_crop_names = []
        surviving_bboxes     = []
        surviving_confs      = []
        surviving_cats       = []
 
        for bbox, crop_name, conf, cat in surviving:
            try:
                # Re-crop using the original pad_frac (matching training.py)
                final_crop = _crop_with_pad_frac(img, bbox, pad_frac=pad_frac)
                out_path   = output_path / crop_name
                final_crop.save(out_path, quality=jpeg_quality)
 
                surviving_crop_names.append(crop_name)
                surviving_bboxes.append(bbox)
                surviving_confs.append(conf)
                surviving_cats.append(cat)
                saved += 1
 
            except Exception as e:
                print(f"  Warning: failed to save {crop_name}: {e}")
 
        if not surviving_crop_names:
            continue
 
        # Rebuild the row with updated crop lists
        updated_row = row.to_dict()
        updated_row["n_animals"]       = len(surviving_crop_names)
        updated_row["filename_crops"]  = json.dumps(surviving_crop_names)
        updated_row["bboxes"]          = json.dumps(surviving_bboxes)
        updated_row["det_confs"]       = json.dumps(surviving_confs)
        updated_row["det_categories"]  = json.dumps(surviving_cats)
 
        # Update the primary (first) crop fields for backward compatibility
        updated_row["filename_crop"]   = surviving_crop_names[0]
        updated_row["bbox"]            = surviving_bboxes[0]
        updated_row["det_conf"]        = surviving_confs[0]
        updated_row["det_category"]    = surviving_cats[0]
 
        kept_rows.append(updated_row)
 
    output_df = pd.DataFrame(kept_rows).reset_index(drop=True)
    output_df.to_csv(output_csv, index=False)
 
    print(f"\n[step2] Done.")
    print(f"  Rows in original CSV       : {len(df)}")
    print(f"  Rows after review          : {len(output_df)}")
    print(f"  Crops saved to output_dir  : {saved}")
    print(f"  Images failed              : {failed}")
    print(f"\nFinal crops folder : {output_dir}")
    print(f"Updated CSV        : {output_csv}")
    print(f"\nIn training.py, point:")
    print(f"  cropped_image_dir -> {output_dir}")
    print(f"  filtered_all_path -> {output_csv}")
 
 
# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────
 
def main():
    parser = argparse.ArgumentParser(
        description="Two-step manual review pipeline for animal training crops.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
 
    # ── Step 1 ──────────────────────────────────────────────
    p1 = subparsers.add_parser(
        "generate",
        help="Add 100px padding to crops for manual review."
    )
    p1.add_argument("--filtered_csv", required=True,
        help="Path to full_df_filtered.csv from training.py.")
    p1.add_argument("--image_dir", required=True,
        help="Root folder of original source images (all_species_images).")
    p1.add_argument("--review_dir", required=True,
        help="Where to save the padded review crops (organised by species).")
    p1.add_argument("--padding_px", type=int, default=100,
        help="Pixels of original-image context to add around each MD box. "
             "(default: 100)")
    p1.add_argument("--jpeg_quality", type=int, default=95,
        help="JPEG quality for saved crops. (default: 95)")
 
    # ── Step 2 ──────────────────────────────────────────────
    p2 = subparsers.add_parser(
        "recrop",
        help="Re-crop surviving images back to training format after human review."
    )
    p2.add_argument("--filtered_csv", required=True,
        help="Path to full_df_filtered.csv from training.py.")
    p2.add_argument("--image_dir", required=True,
        help="Root folder of original source images (all_species_images).")
    p2.add_argument("--review_dir", required=True,
        help="The review folder from Step 1 (after human has deleted bad crops).")
    p2.add_argument("--output_dir", required=True,
        help="Where to save the final training crops. "
             "Set this to the same path as cropped_image_dir in training.py.")
    p2.add_argument("--output_csv", required=True,
        help="Where to save the updated full_df_filtered.csv. "
             "Use this as filtered_all_path in training.py.")
    p2.add_argument("--pad_frac", type=float, default=0.10,
        help="Fractional padding matching training.py config. (default: 0.10)")
    p2.add_argument("--jpeg_quality", type=int, default=95,
        help="JPEG quality for saved crops. (default: 95)")
 
    args = parser.parse_args()
 
    if args.command == "generate":
        generate_review_crops(
            filtered_csv=args.filtered_csv,
            image_dir=args.image_dir,
            review_dir=args.review_dir,
            padding_px=args.padding_px,
            jpeg_quality=args.jpeg_quality,
        )
    elif args.command == "recrop":
        recrop_after_review(
            filtered_csv=args.filtered_csv,
            image_dir=args.image_dir,
            review_dir=args.review_dir,
            output_dir=args.output_dir,
            output_csv=args.output_csv,
            pad_frac=args.pad_frac,
            jpeg_quality=args.jpeg_quality,
        )
 
 
if __name__ == "__main__":
    main()
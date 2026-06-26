"""
make_test_data.py
=================

Creates a tiny synthetic dataset for smoke-testing the review_crops.py
pipeline end-to-end before running it on the full ~400k-row CSV.

Picks N source images from a folder, fabricates plausible MegaDetector
bboxes for each, and writes a `test_filtered.csv` that review_crops.py
can consume directly (same column schema as `training.py`'s output).

Why bother:
    The real pipeline takes hours. This lets you walk through
    generate -> human delete -> recrop in a few minutes against a
    handful of crops to verify everything wires up before committing
    to a long run.

Usage:
    python make_test_data.py \\
        --image_dir   /path/to/some_images \\
        --output_csv  /tmp/test_filtered.csv \\
        --n_images    5

Then:
    python review_crops.py generate \\
        --filtered_csv /tmp/test_filtered.csv \\
        --image_dir   /path/to/some_images \\
        --review_dir  /tmp/test_review

    # delete a few crops from /tmp/test_review/<species>/

    python review_crops.py recrop \\
        --filtered_csv /tmp/test_filtered.csv \\
        --image_dir   /path/to/some_images \\
        --review_dir  /tmp/test_review \\
        --output_dir  /tmp/test_cleaned \\
        --output_csv  /tmp/test_filtered_reviewed.csv

Dependencies: pandas (no Pillow needed — bbox values aren't validated).
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def fake_bbox():
    """Plausible MD bbox [x, y, w, h] in 0-1 relative coords."""
    x = random.uniform(0.1, 0.6)
    y = random.uniform(0.1, 0.6)
    w = random.uniform(0.1, min(0.3, 0.95 - x))
    h = random.uniform(0.1, min(0.3, 0.95 - y))
    return [round(x, 4), round(y, 4), round(w, 4), round(h, 4)]


def main():
    ap = argparse.ArgumentParser(
        description="Build a synthetic CSV for testing review_crops.py.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--image_dir", required=True,
        help="Folder containing real source images to point the synthetic "
             "CSV at. Can be any folder of JPEGs — they don't need labels.")
    ap.add_argument("--output_csv", required=True,
        help="Where to write the synthetic CSV.")
    ap.add_argument("--n_images", type=int, default=5,
        help="How many images to include. (default: 5)")
    ap.add_argument("--max_boxes_per_image", type=int, default=3,
        help="Maximum synthetic detections per image. (default: 3)")
    ap.add_argument("--species_list", nargs="+",
        default=["warthog", "impala", "lion"],
        help="Species labels to randomly assign across the picked images.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    root = Path(args.image_dir).resolve()
    if not root.exists():
        raise SystemExit(f"image_dir does not exist: {root}")

    candidates = [
        p.relative_to(root)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not candidates:
        raise SystemExit(f"No images found under {root}")

    picks = random.sample(candidates, min(args.n_images, len(candidates)))

    rows = []
    for img_rel in picks:
        n_boxes    = random.randint(1, args.max_boxes_per_image)
        species    = random.choice(args.species_list)
        stem       = Path(img_rel).stem
        bboxes     = [fake_bbox() for _ in range(n_boxes)]
        crop_names = [f"{stem}_crop{j}.jpg" for j in range(n_boxes)]
        confs      = [round(random.uniform(0.2, 0.95), 3) for _ in range(n_boxes)]
        cats       = ["1"] * n_boxes  # 1 = animal in MD categories
        rows.append({
            "filename":       str(img_rel),
            "species":        species,
            "site":           "test_site",
            "n_animals":      n_boxes,
            "bboxes":         json.dumps(bboxes),
            "filename_crops": json.dumps(crop_names),
            "det_confs":      json.dumps(confs),
            "det_categories": json.dumps(cats),
        })

    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)

    print(f"\nWrote {len(df)} rows -> {args.output_csv}")
    print(f"Total synthetic crops: {int(df['n_animals'].sum())}")
    print(f"Species distribution : {df['species'].value_counts().to_dict()}")
    print(f"\nNext steps:")
    print(f"  python review_crops.py generate \\")
    print(f"      --filtered_csv {args.output_csv} \\")
    print(f"      --image_dir    {root} \\")
    print(f"      --review_dir   /tmp/test_review")


if __name__ == "__main__":
    main()

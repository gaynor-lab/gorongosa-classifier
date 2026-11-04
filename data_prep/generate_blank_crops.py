"""
generate_blank_crops.py
=======================
Script 1 of 2.

Runs MegaDetector (Dan Morris's `megadetector` pip package) on all images
inside Ghost / Ghost2 folders and saves every detection crop — animal,
person, or vehicle — to a per-site output folder.

These crops are the raw material for training a "blank" class in a species
classifier. The idea: MegaDetector fires on these known-empty images, so
whatever it draws a box on is exactly the kind of false positive your
classifier will see in production. Training on these crops teaches the
classifier to say "blank" instead of guessing a species.

Run Script 2 (sample_blank_crops.py) afterwards to subsample proportionally
across sites to hit a training target.

Install:
    pip install megadetector Pillow tqdm

Usage:
    python generate_blank_crops.py \\
        --input_dir  /path/to/survey/root \\
        --output_dir /path/to/all_blank_crops \\
        --threshold  0.01

Directory structure expected:
    input_dir/
        SiteA/
            Ghost/
                img001.jpg
                1/
                    img002.jpg
        SiteB/
            Ghost2/
                img003.jpg

Site name is inferred from the folder immediately above Ghost / Ghost2.
Images can be at any depth inside the Ghost folder.

# Why we use a very low threshold (0.01) here rather than the default 0.05:
#
# If we use a high threshold, we only keep high-confidence detections, which
# are more likely to be real animals that slipped into the Ghost folder by
# mistake. That's the opposite of what we want.
#
# At very low thresholds (0.01-0.05), we capture the faint, uncertain
# detections — the swaying grass, the dappled shadows, the rocks at dusk —
# that MD is not sure about but will still pass on to the classifier in
# production (since production typically runs at 0.1-0.2). Training the
# classifier on these teaches it to say "blank" for exactly the inputs it
# will actually encounter, rather than only learning from clean, obvious
# background patches that would never reach it in real use.
#
# The tradeoff: a lower threshold means more crops to manually review,
# since a few will turn out to be real animals MD was uncertain about.
# This is why the confidence-first filename scheme matters — sort by name
# and review the highest-confidence crops first, as those are most likely
# to contain something real.
"""

import os
import sys
import json
import argparse
from pathlib import Path

from PIL import Image
from tqdm import tqdm


GHOST_FOLDER_NAMES  = {"ghost", "ghost2"}
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
MD_CATEGORY_MAP     = {"1": "animal", "2": "person", "3": "vehicle"}


# ──────────────────────────────────────────────────────────────
# Step 1 — Discover images
# ──────────────────────────────────────────────────────────────

def discover_images(root_dir, debug=False):
    """
    Walk root_dir and find all images inside Ghost / Ghost2 subfolders
    at any depth. Site name is the folder immediately above Ghost/Ghost2.

    Returns list of {"path": Path, "site": str, "camera": str}.
    """
    root  = Path(root_dir).resolve()
    found = []

    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            if debug:
                print(f"  skip (ext): {path.name}")
            continue

        parts     = path.relative_to(root).parts
        ghost_idx = None
        for i, part in enumerate(parts[:-1]):
            if part.lower() in GHOST_FOLDER_NAMES:
                ghost_idx = i
                break

        if ghost_idx is None:
            if debug:
                print(f"  skip (no Ghost ancestor): {parts}")
            continue

        camera = parts[ghost_idx]
        site   = parts[ghost_idx - 1] if ghost_idx > 0 else "unknown"

        if debug:
            print(f"  found: site={site!r} camera={camera!r} file={path.name}")

        found.append({"path": path, "site": site, "camera": camera})

    return found


# ──────────────────────────────────────────────────────────────
# Step 2 — Run MegaDetector (Dan Morris pip package)
# ──────────────────────────────────────────────────────────────

def run_megadetector(image_records, model_name="MDV5A", threshold=0.05):
    """
    Runs MegaDetector using the `megadetector` pip package by Dan Morris.
    (pip install megadetector)

    Model is downloaded automatically on first run.
    Valid model_name values: MDV5A (recommended), MDV5B.

    We run at a low threshold to capture everything — the caller's
    --threshold filters afterwards so you don't need to re-run MD to try
    a different cutoff.

    Returns:
        {"/abs/path/img.jpg": [{"bbox": [x,y,w,h], "conf": float,
                                 "category": "1"}, ...]}
        bbox is relative coords [x_min, y_min, width, height] in 0-1 range.
    """
    try:
        from megadetector.detection.run_detector_batch import (
            load_and_run_detector_batch,
        )
    except ImportError:
        raise RuntimeError(
            "megadetector package not found.\n"
            "Install:  pip install megadetector\n"
            "Or supply a pre-run results JSON with --md_json"
        )

    print(f"  Model    : {model_name} (auto-downloads if not cached)")
    all_paths = [str(r["path"]) for r in image_records]
    print(f"  Images   : {len(all_paths)}")

    results_list = load_and_run_detector_batch(model_name, all_paths, quiet=True)

    # results_list is a list of:
    #   {"file": path_str,
    #    "detections": [{"category": "1", "conf": 0.9, "bbox": [x,y,w,h]}],
    #    "failure": None}
    results = {}
    for entry in results_list:
        key  = entry.get("file", "")
        dets = entry.get("detections") or []
        results[key] = [
            {
                "bbox":     d["bbox"],
                "conf":     d["conf"],
                "category": str(d.get("category", "1")),
            }
            for d in dets
            if d["conf"] >= threshold
        ]

    for r in image_records:
        if str(r["path"]) not in results:
            results[str(r["path"])] = []

    return results


def load_md_json(json_path, threshold):
    """Load a pre-run MegaDetector results JSON, filtering by threshold."""
    print(f"  Loading MD results from {json_path} ...")
    with open(json_path) as f:
        data = json.load(f)
    results = {}
    for entry in data.get("images", []):
        key = entry["file"]
        results[key] = [
            {
                "bbox":     d["bbox"],
                "conf":     d["conf"],
                "category": str(d.get("category", "1")),
            }
            for d in (entry.get("detections") or [])
            if d["conf"] >= threshold
        ]
    return results


def save_md_json(results, output_dir):
    """Save MD results JSON so MD doesn't need to re-run next time."""
    out = Path(output_dir) / "md_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved MD results -> {out}")


# ──────────────────────────────────────────────────────────────
# Step 3 — Filter detections
# ──────────────────────────────────────────────────────────────

def build_detection_list(image_records, md_results, threshold,
                          categories, sites):
    """
    Merge image metadata with MD results and apply filters.
    Returns flat list of detection dicts ready for cropping.
    """
    allowed_categories = {c.lower() for c in categories} if categories else None
    allowed_sites      = {s.lower() for s in sites}      if sites      else None

    detections = []
    for record in image_records:
        path = record["path"]
        site = record["site"]

        if allowed_sites and site.lower() not in allowed_sites:
            continue

        for d in md_results.get(str(path), []):
            conf     = d["conf"]
            cat_name = MD_CATEGORY_MAP.get(d["category"], "unknown")

            if conf < threshold:
                continue
            if allowed_categories and cat_name not in allowed_categories:
                continue

            detections.append({
                "path":     path,
                "site":     site,
                "camera":   record["camera"],
                "bbox":     d["bbox"],
                "conf":     conf,
                "category": cat_name,
            })

    return detections


# ──────────────────────────────────────────────────────────────
# Step 4 — Crop and save (all detections, organised by site)
# ──────────────────────────────────────────────────────────────

def bbox_to_pixels(bbox, img_w, img_h):
    """MD relative [x, y, w, h] -> pixel (left, top, right, bottom)."""
    x, y, w, h = bbox
    left   = max(0, int(x * img_w))
    top    = max(0, int(y * img_h))
    right  = min(img_w, int((x + w) * img_w))
    bottom = min(img_h, int((y + h) * img_h))
    return left, top, right, bottom


def make_filename(detection, crop_idx):
    """
    Confidence-first filename so file viewers sort highest-conf first.
    Example: blank_conf_0p813_site_SiteA_camera_Ghost_animal_crop_0001.jpg
    """
    conf_str   = f"{detection['conf']:.3f}".replace(".", "p")
    site_str   = detection["site"].replace(" ", "_")
    camera_str = detection["camera"].replace(" ", "_")
    cat_str    = detection["category"]
    return (
        f"blank_conf_{conf_str}"
        f"_site_{site_str}"
        f"_camera_{camera_str}"
        f"_{cat_str}"
        f"_crop_{crop_idx:04d}.jpg"
    )


def crop_and_save(detection, crop_idx, output_dir, padding):
    """Crop detection box + padding and save to a per-site subfolder."""
    img  = Image.open(detection["path"]).convert("RGB")
    w, h = img.size

    left, top, right, bottom = bbox_to_pixels(detection["bbox"], w, h)
    left   = max(0, left   - padding)
    top    = max(0, top    - padding)
    right  = min(w, right  + padding)
    bottom = min(h, bottom + padding)

    site_dir = output_dir / detection["site"]
    site_dir.mkdir(parents=True, exist_ok=True)

    out_path = site_dir / make_filename(detection, crop_idx)
    img.crop((left, top, right, bottom)).save(out_path, "JPEG", quality=10)
    return out_path


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate ALL blank crops from Ghost folders using MegaDetector "
            "(Dan Morris pip package). One subfolder per site in output. "
            "Run sample_blank_crops.py next to subsample to a training target."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--input_dir",  required=True,
        help="Root folder containing site subfolders with Ghost/Ghost2 directories.")
    parser.add_argument("--output_dir", required=True,
        help="Where to write crops. One subfolder per site is created here.")

    md_group = parser.add_mutually_exclusive_group(required=True)
    md_group.add_argument("--md_model", default=None,
        help="MegaDetector model name: MDV5A (default, recommended) or MDV5B. "
             "Downloads automatically on first run.")
    md_group.add_argument("--md_json",
        help="Path to pre-run MD results JSON. Skips running MD.")

    parser.add_argument("--threshold", type=float, default=0.05,
        help="Minimum MD confidence to save a crop. Keep low (0.05) to "
             "capture all plausible false positives. (default: 0.05)")
    parser.add_argument("--categories", nargs="+",
        choices=["animal", "person", "vehicle"], default=None,
        help="Only save crops of these MD categories. Default: all three.")
    parser.add_argument("--sites", nargs="+", default=None,
        help="Only process these site names. Default: all sites.")
    parser.add_argument("--padding", type=int, default=32,
        help="Pixels of padding around each detection box. (default: 32)")
    parser.add_argument("--save_md_json", action="store_true",
        help="Save MD results JSON to output_dir for reuse later.")
    parser.add_argument("--debug", action="store_true",
        help="Print every file the scanner considers, to diagnose path issues.")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover images
    print(f"\n[1/4] Scanning for Ghost/Ghost2 images in: {args.input_dir}")
    image_records = discover_images(args.input_dir, debug=args.debug)

    if not image_records:
        print("  No images found. Check --input_dir and Ghost folder names.")
        sys.exit(1)

    sites_found = sorted({r["site"] for r in image_records})
    print(f"  Found {len(image_records)} images across "
          f"{len(sites_found)} sites: {sites_found}")

    # 2. Run MD or load JSON
    print(f"\n[2/4] Getting MegaDetector results ...")
    if args.md_json:
        md_results = load_md_json(args.md_json, threshold=args.threshold)
    else:
        md_results = run_megadetector(
            image_records,
            model_name=args.md_model or "MDV5A",
            threshold=args.threshold,
        )
        if args.save_md_json:
            save_md_json(md_results, args.output_dir)

    # 3. Filter
    print(f"\n[3/4] Filtering detections ...")
    print(f"  Confidence threshold : {args.threshold}")
    print(f"  Sites filter         : {args.sites or 'all'}")
    print(f"  Category filter      : {args.categories or 'all'}")

    detections = build_detection_list(
        image_records, md_results,
        threshold=args.threshold,
        categories=args.categories,
        sites=args.sites,
    )
    print(f"  Total detections     : {len(detections)}")

    if not detections:
        print("  No detections found. Try lowering --threshold.")
        sys.exit(0)

    # 4. Crop and save all, organised by site
    print(f"\n[4/4] Saving crops to: {output_dir}/<site>/")
    failed          = 0
    per_site        = {}
    crop_idx_by_site = {}

    for det in tqdm(detections):
        site = det["site"]
        crop_idx_by_site[site] = crop_idx_by_site.get(site, 0) + 1
        try:
            crop_and_save(det, crop_idx=crop_idx_by_site[site],
                          output_dir=output_dir, padding=args.padding)
            per_site[site] = per_site.get(site, 0) + 1
        except Exception as e:
            print(f"  Warning: failed on {det['path'].name}: {e}")
            failed += 1

    total_saved = sum(per_site.values())
    print(f"\nDone. Saved {total_saved} crops ({failed} failed).")
    print(f"\nCrops per site:")
    for site, count in sorted(per_site.items()):
        print(f"  {site:30s}  {count:5d}")
    print(f"\nNext: run sample_blank_crops.py --crops_dir {args.output_dir} "
          f"--target_total <N> --output_dir /path/to/training/blank")


if __name__ == "__main__":
    main()
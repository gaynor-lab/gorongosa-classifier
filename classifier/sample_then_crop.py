"""
sample_then_crop.py
===================

Subsample-first variant of the blank-crops pipeline.

Reads an md_results.json produced by `generate_blank_crops.py --no_crops`,
runs the same proportional-allocation algorithm as randomize_blank_crops.py
(but on detections, not files on disk), randomly samples within each site,
and crops ONLY the keepers.

Why this exists:
    Cropping is the expensive step. At threshold=0.01, MegaDetector fires
    many detections per image, but you only want to keep ~N total crops for
    training. The original two-script flow crops everything first and then
    subsamples — which wastes most of the cropping I/O. This script flips
    the order: subsample first, crop second.

Usage:
    # 1. Run MD-only to produce the recipe
    python generate_blank_crops.py \\
        --input_dir  /arc/.../ghost_photos \\
        --output_dir /scratch/.../md_run \\
        --md_model   MDV5A \\
        --threshold  0.01 \\
        --no_crops

    # 2. Sample to a target and crop the keepers
    python sample_then_crop.py \\
        --md_json    /scratch/.../md_run/md_results.json \\
        --input_dir  /arc/.../ghost_photos \\
        --output_dir /scratch/.../classifier_blank_class_training \\
        --target_total 2000

Output structure matches generate_blank_crops.py: one subfolder per site.

Dependencies: same as the other two scripts (Pillow, tqdm).
"""

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from generate_blank_crops import (
    discover_images,
    load_md_json,
    build_detection_list,
    crop_and_save,
)
from randomize_blank_crops import proportional_allocation


def group_by_site(detections):
    """Detections -> {site: [detection, ...]}."""
    by_site = defaultdict(list)
    for d in detections:
        by_site[d["site"]].append(d)
    return dict(by_site)


def sample_detections(by_site, allocation, seed):
    """Random-sample allocation[site] detections from each site's pool."""
    random.seed(seed)
    selected = []
    for site, dets in by_site.items():
        quota = allocation.get(site, 0)
        if quota == 0:
            continue
        selected.extend(random.sample(dets, min(quota, len(dets))))
    return selected


def print_plan(by_site, allocation):
    total_pool   = sum(len(v) for v in by_site.values())
    total_picked = sum(allocation.values())
    print(f"\n  {'Site':<30}  {'Available':>9}  {'Selected':>8}  {'Share':>6}")
    print(f"  {'-'*30}  {'-'*9}  {'-'*8}  {'-'*6}")
    for site in sorted(by_site):
        avail = len(by_site[site])
        sel   = allocation[site]
        pct   = 100 * sel / total_picked if total_picked else 0
        note  = "  <- all available" if sel == avail else ""
        print(f"  {site:<30}  {avail:>9}  {sel:>8}  {pct:>5.1f}%{note}")
    print(f"  {'TOTAL':<30}  {total_pool:>9}  {total_picked:>8}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Subsample MD detections proportionally across sites, then crop "
            "only the keepers. Use after `generate_blank_crops.py --no_crops`."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--md_json", required=True,
        help="Path to md_results.json produced by generate_blank_crops.py.")
    parser.add_argument("--input_dir", required=True,
        help="Root folder containing site subfolders with Ghost/Ghost2 "
             "directories — same as you passed to generate_blank_crops.py. "
             "Needed to map MD's image paths back to site/camera.")
    parser.add_argument("--output_dir", required=True,
        help="Where to write the sampled crops. One subfolder per site.")
    parser.add_argument("--target_total", type=int, required=True,
        help="Total number of crops to keep across all sites.")

    parser.add_argument("--threshold", type=float, default=0.05,
        help="Minimum MD confidence to consider. (default: 0.05)")
    parser.add_argument("--categories", nargs="+",
        choices=["animal", "person", "vehicle"], default=None,
        help="Only sample these MD categories. Default: all three.")
    parser.add_argument("--sites", nargs="+", default=None,
        help="Only process these site names. Default: all sites.")
    parser.add_argument("--padding", type=int, default=32,
        help="Pixels of padding around each detection box. (default: 32)")
    parser.add_argument("--seed", type=int, default=42,
        help="Random seed for reproducible sampling. (default: 42)")
    parser.add_argument("--dry_run", action="store_true",
        help="Print the allocation plan without cropping anything.")
    parser.add_argument("--debug", action="store_true",
        help="Pass through to image discovery.")

    args = parser.parse_args()

    # 1. Discover images so we can attach site/camera to each MD entry.
    print(f"\n[1/4] Scanning images in: {args.input_dir}")
    image_records = discover_images(args.input_dir, debug=args.debug)
    if not image_records:
        print("  No images found.")
        sys.exit(1)
    print(f"  Found {len(image_records)} images.")

    # 2. Load MD recipe and build the candidate detection pool.
    print(f"\n[2/4] Loading MD JSON: {args.md_json}")
    md_results = load_md_json(args.md_json, threshold=args.threshold)

    detections = build_detection_list(
        image_records, md_results,
        threshold=args.threshold,
        categories=args.categories,
        sites=args.sites,
    )
    print(f"  Candidate detections after filters: {len(detections)}")
    if not detections:
        print("  Nothing to sample. Lower --threshold or widen filters.")
        sys.exit(0)

    # 3. Allocate proportionally, then random-sample within each site.
    print(f"\n[3/4] Allocating {args.target_total} crops across sites ...")
    by_site    = group_by_site(detections)
    allocation = proportional_allocation(by_site, args.target_total)
    print_plan(by_site, allocation)

    if args.dry_run:
        print("\nDry run — no crops written.")
        return

    selected = sample_detections(by_site, allocation, seed=args.seed)
    print(f"\n  Sampled {len(selected)} detections.")

    # 4. Crop only the keepers.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[4/4] Cropping to: {output_dir}/<site>/")
    failed           = 0
    per_site         = defaultdict(int)
    crop_idx_by_site = defaultdict(int)

    for det in tqdm(selected):
        site = det["site"]
        crop_idx_by_site[site] += 1
        try:
            crop_and_save(det,
                          crop_idx=crop_idx_by_site[site],
                          output_dir=output_dir,
                          padding=args.padding)
            per_site[site] += 1
        except Exception as e:
            print(f"  Warning: failed on {det['path'].name}: {e}")
            failed += 1

    total_saved = sum(per_site.values())
    print(f"\nDone. Wrote {total_saved} crops ({failed} failed).")
    print(f"\nCrops per site:")
    for site, n in sorted(per_site.items()):
        print(f"  {site:30s}  {n:5d}")
    print(f"\nOutput: {output_dir}")
    print("This is your blank class — drop it alongside species folders for training.")


if __name__ == "__main__":
    main()

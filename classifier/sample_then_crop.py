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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from generate_blank_crops import (
    discover_images,
    load_md_json,
    build_detection_list,
    bbox_to_pixels,
    make_filename,
)
from randomize_blank_crops import proportional_allocation


def crop_image_group(image_path, items, output_dir, padding):
    """Process all sampled crops from a single source image.

    items: list of (detection_dict, crop_idx) tuples — all share the same
    source image path, so we open the file once and crop multiple times.

    Skips crops whose output files already exist (resume support).

    Returns (per_site_on_disk: dict, failed: int).
    `per_site_on_disk` counts both pre-existing files and newly written ones,
    so the caller's final tally matches what's actually on disk.
    """
    per_site_on_disk = defaultdict(int)
    failed = 0

    todo = []
    for det, crop_idx in items:
        site_dir = output_dir / det["site"]
        out_path = site_dir / make_filename(det, crop_idx)
        if out_path.exists():
            per_site_on_disk[det["site"]] += 1
        else:
            todo.append((det, site_dir, out_path))

    if not todo:
        return per_site_on_disk, 0

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return per_site_on_disk, len(todo)

    w, h = img.size
    for det, site_dir, out_path in todo:
        try:
            left, top, right, bottom = bbox_to_pixels(det["bbox"], w, h)
            left   = max(0, left   - padding)
            top    = max(0, top    - padding)
            right  = min(w, right  + padding)
            bottom = min(h, bottom + padding)
            site_dir.mkdir(parents=True, exist_ok=True)
            img.crop((left, top, right, bottom)).save(out_path, "JPEG", quality=10)
            per_site_on_disk[det["site"]] += 1
        except Exception:
            failed += 1

    return per_site_on_disk, failed


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
    parser.add_argument("--workers", type=int, default=8,
        help="Number of parallel cropping threads. PIL releases the GIL "
             "during JPEG ops, so this scales well for I/O-bound runs. "
             "(default: 8)")
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

    # 4. Crop only the keepers — grouped per source image, threaded, resumable.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Walk `selected` serially to assign deterministic per-site crop indices
    # (so resume picks the same filenames), then bucket by source image path
    # so each image only gets opened once even when it has many detections.
    crop_idx_by_site = defaultdict(int)
    groups = defaultdict(list)
    for det in selected:
        crop_idx_by_site[det["site"]] += 1
        groups[str(det["path"])].append((det, crop_idx_by_site[det["site"]]))

    total_planned   = len(selected)
    n_unique_images = len(groups)
    print(f"\n[4/4] Cropping {total_planned} detections from "
          f"{n_unique_images} source images to: {output_dir}/<site>/")
    print(f"  Threads: {args.workers}. Existing crops are skipped (resumable).")

    per_site = defaultdict(int)
    failed   = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(crop_image_group, Path(path), items,
                      output_dir, args.padding): len(items)
            for path, items in groups.items()
        }
        with tqdm(total=total_planned) as pbar:
            for fut in as_completed(futures):
                try:
                    per_site_on_disk, f = fut.result()
                    for site, n in per_site_on_disk.items():
                        per_site[site] += n
                    failed += f
                except Exception:
                    pass
                pbar.update(futures[fut])

    total_on_disk = sum(per_site.values())
    print(f"\nDone. {total_on_disk} crops on disk ({failed} failed).")
    print(f"\nCrops per site:")
    for site, n in sorted(per_site.items()):
        print(f"  {site:30s}  {n:5d}")
    print(f"\nOutput: {output_dir}")
    print("This is your blank class — drop it alongside species folders for training.")


if __name__ == "__main__":
    main()

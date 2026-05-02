"""
sample_blank_crops.py
=====================

Takes the output of generate_blank_crops.py (a folder with one subfolder
per site, each containing all MD blank crops for that site) and produces
a classifier_blank_class_training folder sampled proportionally across
sites to hit a target total.

How the sampling works:
  1. Count how many crops each site has (their natural total).
  2. Each site's share of the target is proportional to its natural total.
     e.g. A06 has 600 crops, A10 has 400 (1000 total), target is 500:
          A06 gets 300, A10 gets 200.
  3. If a site has fewer crops than its proportional share, take all of
     them and redistribute the leftover quota to the remaining sites
     proportionally. Repeats until the full target is allocated or all
     sites are exhausted.
  4. Within each site, selection is random.

Output structure:
    classifier_blank_class_training/
        A06/
            blank_conf_0p012_site_A06_...jpg
        A10/
            blank_conf_0p031_site_A10_...jpg

Usage:
    # Preview allocation without copying anything
    python sample_blank_crops.py \\
        --crops_dir  /path/to/all_blank_crops \\
        --output_dir /path/to/classifier_blank_class_training \\
        --target_total 2000 \\
        --dry_run

    # Run for real
    python sample_blank_crops.py \\
        --crops_dir  /path/to/all_blank_crops \\
        --output_dir /path/to/classifier_blank_class_training \\
        --target_total 2000

Dependencies:
    pip install tqdm
"""

import argparse
import random
import shutil
import sys
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# ──────────────────────────────────────────────────────────────
# Step 1 — Inventory crops by site
# ──────────────────────────────────────────────────────────────

def inventory_crops(crops_dir):
    """
    Read the output folder from generate_blank_crops.py.
    Expected structure: crops_dir / <site> / *.jpg

    Returns:
        {site_name: [Path, ...]}
    """
    crops_dir = Path(crops_dir)
    by_site   = defaultdict(list)

    for site_dir in sorted(crops_dir.iterdir()):
        if not site_dir.is_dir():
            continue
        for f in sorted(site_dir.iterdir()):
            if f.suffix.lower() in SUPPORTED_EXTENSIONS:
                by_site[site_dir.name].append(f)

    return dict(by_site)


def print_inventory(by_site):
    total = sum(len(v) for v in by_site.values())
    print(f"  {'Site':<30}  {'Crops':>6}  {'Share':>6}")
    print(f"  {'-'*30}  {'-'*6}  {'-'*6}")
    for site, paths in sorted(by_site.items()):
        pct = 100 * len(paths) / total if total else 0
        print(f"  {site:<30}  {len(paths):>6}  {pct:>5.1f}%")
    print(f"  {'TOTAL':<30}  {total:>6}")
    return total


# ──────────────────────────────────────────────────────────────
# Step 2 — Proportional allocation with redistribution
# ──────────────────────────────────────────────────────────────

def proportional_allocation(by_site, target_total):
    """
    Allocate target_total crops across sites proportionally.

    If a site has fewer crops than its share, take all of them and
    redistribute the remainder to other sites that still have capacity.
    Repeats until the full target is allocated or all sites are exhausted.

    Returns:
        {site_name: int}  — number of crops to take from each site
    """
    sites      = sorted(by_site.keys())
    available  = {s: len(by_site[s]) for s in sites}
    allocation = {s: 0 for s in sites}
    remaining  = target_total

    while remaining > 0:
        # Sites that still have unallocated crops
        eligible = {s: available[s] - allocation[s]
                    for s in sites
                    if available[s] - allocation[s] > 0}

        if not eligible:
            break  # No crops left anywhere

        eligible_total = sum(eligible.values())

        # Proportional share of remaining quota for each eligible site
        raw     = {s: remaining * eligible[s] / eligible_total
                   for s in eligible}
        floored = {s: int(raw[s]) for s in eligible}

        # Distribute rounding remainder to sites with largest fractional parts
        rounding_gap = remaining - sum(floored.values())
        by_fraction  = sorted(eligible, key=lambda s: -(raw[s] - floored[s]))
        for s in by_fraction[:rounding_gap]:
            floored[s] += 1

        # Apply, clamping each site to what it actually has left
        added = 0
        for s, quota in floored.items():
            can_take      = available[s] - allocation[s]
            actually_take = min(quota, can_take)
            allocation[s] += actually_take
            added         += actually_take

        remaining -= added

        if added == 0:
            break  # Stuck — shouldn't happen

    return allocation


# ──────────────────────────────────────────────────────────────
# Step 3 — Sample and copy into per-site subfolders
# ──────────────────────────────────────────────────────────────

def sample_and_copy(by_site, allocation, output_dir, seed=42):
    """
    Randomly sample allocation[site] crops from each site and copy
    them into output_dir/<site>/ subfolders.
    """
    output_dir = Path(output_dir)
    random.seed(seed)

    copied   = 0
    per_site = {}

    for site in sorted(by_site.keys()):
        quota = allocation.get(site, 0)
        if quota == 0:
            per_site[site] = 0
            continue

        pool     = list(by_site[site])
        selected = random.sample(pool, min(quota, len(pool)))

        site_out = output_dir / site
        site_out.mkdir(parents=True, exist_ok=True)

        for src in tqdm(selected, desc=f"  {site}", leave=False):
            shutil.copy2(src, site_out / src.name)
            copied += 1

        per_site[site] = len(selected)

    return per_site, copied


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Proportionally sample blank crops across sites to hit a "
            "training target. Output goes into one subfolder per site."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--crops_dir", required=True,
        help="Output folder from generate_blank_crops.py "
             "(contains one subfolder per site).")
    parser.add_argument("--output_dir", required=True,
        help="Where to write sampled crops. Name this "
             "classifier_blank_class_training.")
    parser.add_argument("--target_total", type=int, required=True,
        help="Total number of blank crops to produce across all sites.")
    parser.add_argument("--seed", type=int, default=42,
        help="Random seed for reproducibility. (default: 42)")
    parser.add_argument("--dry_run", action="store_true",
        help="Print the allocation plan without copying any files.")

    args = parser.parse_args()

    # 1. Inventory
    print(f"\n[1/3] Reading crops from: {args.crops_dir}")
    by_site = inventory_crops(args.crops_dir)

    if not by_site:
        print("  No site subfolders found. Did generate_blank_crops.py run?")
        sys.exit(1)

    total_available = print_inventory(by_site)
    print()

    if args.target_total > total_available:
        print(f"  Warning: target_total ({args.target_total}) exceeds total "
              f"available ({total_available}). Will take everything.")

    # 2. Compute allocation
    print(f"[2/3] Computing proportional allocation "
          f"(target={args.target_total}) ...")

    allocation      = proportional_allocation(by_site, args.target_total)
    total_allocated = sum(allocation.values())

    print(f"\n  {'Site':<30}  {'Available':>9}  {'Selected':>8}  {'Share':>6}")
    print(f"  {'-'*30}  {'-'*9}  {'-'*8}  {'-'*6}")
    for site in sorted(by_site.keys()):
        avail = len(by_site[site])
        sel   = allocation[site]
        pct   = 100 * sel / total_allocated if total_allocated else 0
        note  = "  <- all available" if sel == avail else ""
        print(f"  {site:<30}  {avail:>9}  {sel:>8}  {pct:>5.1f}%{note}")
    print(f"  {'TOTAL':<30}  {total_available:>9}  {total_allocated:>8}")

    if args.dry_run:
        print("\nDry run — no files copied.")
        return

    # 3. Sample and copy
    print(f"\n[3/3] Copying to: {args.output_dir}")
    per_site, total_copied = sample_and_copy(
        by_site, allocation, args.output_dir, seed=args.seed
    )

    print(f"\nDone. Copied {total_copied} crops.")
    print(f"\nFinal crops per site:")
    for site, count in sorted(per_site.items()):
        print(f"  {site:<30}  {count:>5}")
    print(f"\nOutput folder: {args.output_dir}")
    print("This is your blank class — drop it alongside your species "
          "folders for classifier training.")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# splitting.py

from __future__ import annotations
import os
import re
import zlib
from typing import List, Tuple, Literal

import numpy as np
import pandas as pd
from PIL import Image

# Optional: detector-based filtering support (only if you use it from training)
# from speciesnet.detector import SpeciesNetDetector


# -------- Filename parsing & folder scan (optional helpers) --------

def extract_species_and_site_from_filename(filename: str) -> tuple[str, str]:
    """
    Parse a filename like: IMG_0001_A06_bushbuck.JPG  -> ('bushbuck', 'a06')
    We look for a token that matches letter + 2 digits (e.g., A06, d07) as the site,
    and everything after that underscore as the species.
    """
    base = os.path.splitext(filename)[0]
    parts = base.split("_")
    site_pat = re.compile(r"^[a-zA-Z]\d{2}$")
    for i, part in enumerate(parts):
        if site_pat.match(part):
            site = part.lower()
            species = "_".join(parts[i + 1 :]).lower()
            return species, site
    return "unknown", "unknown"


def build_df_from_folder(image_dir: str) -> pd.DataFrame:
    """
    Scan a directory and return a DataFrame with columns:
    ['filename', 'species', 'site'] for recognized images.
    """
    image_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp", ".gif"}
    rows = []
    for fname in os.listdir(image_dir):
        if os.path.splitext(fname)[1].lower() in image_exts:
            species, site = extract_species_and_site_from_filename(fname)
            if species != "unknown" and site != "unknown":
                rows.append({"filename": fname, "species": species, "site": site})
    return pd.DataFrame(rows)

# -------- Core splitter with two modes --------

SplitMode = Literal["instance", "sitewise"]

def _normalize_cols(df: pd.DataFrame, site_col: str, species_col: str) -> pd.DataFrame:
    df = df.copy()
    df[site_col]    = df[site_col].astype(str).str.strip().str.lower()
    df[species_col] = df[species_col].astype(str).str.strip().str.lower()
    return df


def split_train_val_holdout(
    df: pd.DataFrame,
    site_col: str = "site",
    species_col: str = "species",
    holdout_sites: List[str] | None = None,
    test_size: float = 0.30,
    random_state: int = 42,
    min_in_each: int = 1,
    mode: SplitMode = "instance",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create TRAIN, VALID, and HOLDOUT splits.

    1) HOLDOUT: all rows with site ∈ holdout_sites (case-insensitive).
    2) TRAIN/VALID on the remaining rows using one of two strategies:

       mode="instance"  (NEW):
         • Split within EACH (site, species) group ~ (1 - test_size)/(test_size).
         • Singletons (n==1) go to TRAIN.
         • Deterministic per-group shuffle using zlib.crc32 for reproducibility.
         • 'min_in_each' enforces at least that many instances on both sides when feasible.

       mode="sitewise"   (OLD):
         • For EACH species (after excluding holdout sites), split SITES ~ 70/30.
         • All rows from the chosen train_sites go to TRAIN; val_sites to VALID.
         • Deterministic per-species site shuffle.

    Returns: (train_df, val_df, holdout_df)
    """
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be in (0,1).")

    df = _normalize_cols(df, site_col, species_col)

    holdout_norm = set(s.strip().lower() for s in (holdout_sites or []))

    mask_holdout = df[site_col].isin(holdout_norm)
    holdout_df = df[mask_holdout].reset_index(drop=True)
    rest_df    = df[~mask_holdout].reset_index(drop=True)

    if mode == "instance":
        # --- per-(site,species) instance split ---
        train_idx, val_idx = [], []
        for (site, species), g in rest_df.groupby([site_col, species_col], sort=False):
            idx = g.index.to_numpy()
            n   = len(idx)

            if n == 1:
                train_idx.extend(idx)
                continue

            n_val = int(round(test_size * n))

            # enforce min_in_each when feasible
            if min_in_each > 0:
                n_val = max(min_in_each, n_val)
                if n_val > n - min_in_each:
                    n_val = max(0, n - min_in_each)

            if n_val == 0:
                train_idx.extend(idx);  continue
            if n_val >= n:
                val_idx.extend(idx);    continue

            # deterministic per-group permutation
            key  = f"{site}|{species}"
            seed = (zlib.crc32(key.encode("utf-8")) ^ random_state) & 0xFFFFFFFF
            rng  = np.random.RandomState(seed)
            perm = rng.permutation(idx)

            val_take   = perm[:n_val]
            train_take = perm[n_val:]

            train_idx.extend(train_take)
            val_idx.extend(val_take)

        train_df = rest_df.loc[train_idx].reset_index(drop=True)
        val_df   = rest_df.loc[val_idx].reset_index(drop=True)
        return train_df, val_df, holdout_df

    elif mode == "sitewise":
        # --- per-species SITE split (old behavior) ---
        parts = []
        for species, g in rest_df.groupby(species_col, sort=False):
            sites = sorted(g[site_col].unique().tolist())

            # deterministic site shuffle per species
            seed = (zlib.crc32(species.encode("utf-8")) ^ random_state) & 0xFFFFFFFF
            rng  = np.random.RandomState(seed)
            perm_sites = rng.permutation(sites)

            n_sites   = len(perm_sites)
            n_val     = max(1, int(round(test_size * n_sites))) if n_sites > 1 else 0
            val_sites = set(perm_sites[:n_val])
            train_sites = set(perm_sites[n_val:])

            g_train = g[g[site_col].isin(train_sites)]
            g_val   = g[g[site_col].isin(val_sites)]
            parts.append((g_train, g_val))

        train_df = pd.concat([p[0] for p in parts], ignore_index=True)
        val_df   = pd.concat([p[1] for p in parts], ignore_index=True)
        return train_df, val_df, holdout_df

    else:
        raise ValueError("mode must be 'instance' or 'sitewise'")

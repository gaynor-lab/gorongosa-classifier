#!/usr/bin/env python3
"""
training.py

ResNet18-only training script with:
- incremental MegaDetector processing
- persistent kept + dropped caches
- crop-based training
- early stopping on validation loss

Behavior:
- Scan raw image_dir to build current_df
- If kept/dropped caches exist:
    - skip images already present in either cache
    - run MegaDetector only on truly new images
    - append new kept rows to full_df_filtered.csv
    - append new dropped rows to megadetector_dropped.csv
- If caches do not exist:
    - run MegaDetector on all images
    - save kept/dropped caches

Assumes:
- build_df_from_folder(image_dir) -> columns: filename, species, site
- detector.filter_df_with_megadetector_and_crop(...) returns (kept_df, dropped_df)
- kept_df includes at least: filename, species, site, filename_crops and/or filename_crop
- SpeciesImageDataset handles filename_crops by exploding to one row per crop
"""

from __future__ import annotations

import os
import sys
import gc
import importlib.util
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader
import wandb

from dataloader import SpeciesImageDataset, collate_keep_good
from utilities import (
    compute_allowed_species,
    plot_confmat,
    save_predictions_json,
    filter_bad_files,
    expand_to_crop_level,
)
from detector import filter_df_with_megadetector_and_crop


# ------------------------------------------------------------------------------
# Performance toggles
# ------------------------------------------------------------------------------
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


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
from splitting import build_df_from_folder, split_train_val_holdout

try:
    from ct_classifier.model import CustomResNet18
except Exception:
    CT_MODEL_PATH = PROJECT_ROOT / "ct_classifier" / "model.py"
    if not CT_MODEL_PATH.exists():
        raise FileNotFoundError(f"Could not find ct_classifier/model.py at {CT_MODEL_PATH}")
    spec = importlib.util.spec_from_file_location("ct_model", CT_MODEL_PATH)
    ct_model = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(ct_model)
    CustomResNet18 = ct_model.CustomResNet18


# ------------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------------
BASE = Path("/mnt/sharedstorage/sabdelazim")

run_dir = BASE / "Desktop" / "kaitlyn_catalyst" / "resnet_training"
run_dir.mkdir(parents=True, exist_ok=True)

image_dir = BASE / "images" / "all_species_images"

cropped_image_dir = run_dir / "crops"
cropped_image_dir.mkdir(parents=True, exist_ok=True)

filtered_all_path = run_dir / "full_df_filtered.csv"
dropped_all_path = run_dir / "megadetector_dropped.csv"

if not image_dir.exists():
    raise FileNotFoundError(f"image_dir not found: {image_dir}")

run_dir = str(run_dir)
image_dir = str(image_dir)
cropped_image_dir = str(cropped_image_dir)
filtered_all_path = str(filtered_all_path)
dropped_all_path = str(dropped_all_path)


# ------------------------------------------------------------------------------
# W&B config
# ------------------------------------------------------------------------------
wandb.init(
    project="Species-Classification",
    config={
        # Core
        "epochs": 10,
        "lr": 1e-4,
        "batch_size": 256,
        "label_smoothing": 0.0,
        "weight_decay": 0.0,

        # Class selection
        "num_classes": "all",  # int or "all"
        "include_species": [],

        # Splitting
        "holdout_sites": ["d05", "d03", "g02", "e02", "e06", "f05", "i10", "i04", "i08", "d07", "b05", "g08"],
        "split_mode": "instance",  # "instance" or "sitewise"
        "min_in_each": 1,
        "test_size": 0.30,

        # MegaDetector / crop controls
        "cache_filtered_all": True,
        "use_megadetector_if_missing": True,
        "megadetector_model": "MDV5A",
        "megadetector_conf": 0.2,
        "megadetector_device": "cuda",
        "animals_only": True,
        "pad_frac": 0.10,
        "save_format": "jpg",
        "jpeg_quality": 95,

        # Early stopping
        "early_stop": True,
        "early_stop_patience": 3,
        "early_stop_min_delta": 1e-4,

        # Prediction jsons
        "save_probs_json": True,
        "upload_probs_artifact": True,
    },
)

wandb.define_metric("epoch")
wandb.define_metric("*", step_metric="epoch")
config = wandb.config
wandb.run.name = f"resnet18_{config.split_mode}_N{config.num_classes}_ls{config.label_smoothing}"


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

    if len(new_df) > 0 and bool(config.use_megadetector_if_missing):
        new_kept, new_dropped = filter_df_with_megadetector_and_crop(
            df=new_df,
            image_dir=image_dir,
            out_dir=cropped_image_dir,
            conf_thresh=float(config.megadetector_conf),
            model_name_or_path=str(config.megadetector_model),
            device=str(config.megadetector_device),
            animals_only=bool(config.animals_only),
            pad_frac=float(config.pad_frac),
            save_format=str(config.save_format),
            jpeg_quality=int(config.jpeg_quality),
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

        if bool(config.cache_filtered_all):
            print(f"[info] Updating cached filtered CSV -> {filtered_all_path}")
            filtered_all_df.to_csv(filtered_all_path, index=False)
    else:
        print("[info] No new images to run MegaDetector on (or MegaDetector disabled).")

else:
    print(f"[info] No cached filtered CSV. Running MegaDetector on ALL images ({len(current_df)}).")

    if bool(config.use_megadetector_if_missing):
        filtered_all_df, dropped_df = filter_df_with_megadetector_and_crop(
            df=current_df,
            image_dir=image_dir,
            out_dir=cropped_image_dir,
            conf_thresh=float(config.megadetector_conf),
            model_name_or_path=str(config.megadetector_model),
            device=str(config.megadetector_device),
            animals_only=bool(config.animals_only),
            pad_frac=float(config.pad_frac),
            save_format=str(config.save_format),
            jpeg_quality=int(config.jpeg_quality),
        )
    else:
        print("[warn] MegaDetector disabled; using current_df after file sanity filtering.")
        filtered_all_df = current_df.copy()
        dropped_df = pd.DataFrame()

    if bool(config.cache_filtered_all):
        print(f"[info] Saving filtered CSV -> {filtered_all_path}")
        filtered_all_df.to_csv(filtered_all_path, index=False)

        print(f"[info] Saving dropped CSV -> {dropped_all_path}")
        dropped_df.to_csv(dropped_all_path, index=False)

# Final normalization
for col in ("species", "site"):
    if col in filtered_all_df.columns:
        filtered_all_df[col] = filtered_all_df[col].astype(str).str.strip().str.lower()


# ------------------------------------------------------------------------------
# Train on crops
# ------------------------------------------------------------------------------
train_image_root = cropped_image_dir

expanded_df = expand_to_crop_level(filtered_all_df)

print(f"[info] Original kept rows: {len(filtered_all_df)}")
print(f"[info] Expanded crop rows: {len(expanded_df)}")

# ------------------------------------------------------------------------------
# Split
# ------------------------------------------------------------------------------
train_df, val_df, holdout_df = split_train_val_holdout(
    expanded_df,
    site_col="site",
    species_col="species",
    holdout_sites=list(config.holdout_sites),
    test_size=float(config.test_size),
    random_state=42,
    min_in_each=int(config.min_in_each),
    mode=str(config.split_mode),
)

for _df in (train_df, val_df, holdout_df):
    _df["species"] = _df["species"].astype(str).str.strip().str.lower()
    _df["site"] = _df["site"].astype(str).str.strip().str.lower()


# ------------------------------------------------------------------------------
# Class list & datasets
# ------------------------------------------------------------------------------
include_list = None
if not (isinstance(config.num_classes, str) and config.num_classes.lower() == "all"):
    include_list = list(config.include_species) if hasattr(config, "include_species") else None

allowed_species = compute_allowed_species(
    train_df,
    config.num_classes,
    include_list=include_list,
)

# ------------------------------------------------------------------------------
# Debug: count crop images in folder
# ------------------------------------------------------------------------------
print("\n[info] Counting images in crops directory...")
num_crop_images = len(list(Path(cropped_image_dir).glob("*.jpg")))
print(f"[info] Total crop images in folder: {num_crop_images}")


num_classes_eff = len(allowed_species)
print("Classes:", allowed_species)
print("Effective num_classes:", num_classes_eff)

model = CustomResNet18(num_classes=num_classes_eff).to(
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

train_dataset = SpeciesImageDataset(
    train_df,
    train_image_root,
    classifier=None,
    backbone="resnet18",
    top_n_species=None,
    include_species=allowed_species,
)

val_dataset = SpeciesImageDataset(
    val_df,
    train_image_root,
    classifier=None,
    backbone="resnet18",
    top_n_species=None,
    include_species=allowed_species,
)

hold_dataset = SpeciesImageDataset(
    holdout_df,
    train_image_root,
    classifier=None,
    backbone="resnet18",
    top_n_species=None,
    include_species=allowed_species,
)
# ------------------------------------------------------------------------------
# Debug: dataset sizes
# ------------------------------------------------------------------------------
print("\n[dataset stats]")
print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Holdout dataset size: {len(hold_dataset)}")

total_training_samples = len(train_dataset) + len(val_dataset) + len(hold_dataset)
print(f"Total samples used for training pipeline: {total_training_samples}")

# ------------------------------------------------------------------------------
# Loss / optimizer / loaders
# ------------------------------------------------------------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

criterion = nn.CrossEntropyLoss(label_smoothing=float(config.label_smoothing)).to(device)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=float(config.lr),
    weight_decay=float(config.weight_decay),
)

common_loader_kw = dict(
    batch_size=int(config.batch_size),
    pin_memory=True,
    num_workers=8,
    persistent_workers=True,
    prefetch_factor=4,
)

train_loader = DataLoader(train_dataset, shuffle=True, collate_fn=collate_keep_good, **common_loader_kw)
val_loader = DataLoader(val_dataset, shuffle=False, collate_fn=collate_keep_good, **common_loader_kw)
hold_loader = DataLoader(hold_dataset, shuffle=False, collate_fn=collate_keep_good, **common_loader_kw)

with torch.no_grad():
    dummy = torch.randn(2, 3, 224, 224).to(device)
    out = model(dummy)
    assert out.shape[1] == len(allowed_species), f"Model head {out.shape[1]} != classes {len(allowed_species)}"


# ------------------------------------------------------------------------------
# Early stopping state
# ------------------------------------------------------------------------------
best_val_loss = float("inf")
bad_epochs = 0
best_ckpt_path = Path(run_dir) / "best_model_state_resnet18.pkl"


# ------------------------------------------------------------------------------
# Train
# ------------------------------------------------------------------------------
last_train_preds, last_train_trues = [], []
last_val_preds, last_val_trues = [], []
last_hold_preds, last_hold_trues = [], []

train_prob_rows, val_prob_rows, hold_prob_rows = [], [], []

num_epochs = int(config.epochs)
ran_epochs = 0

for epoch in range(num_epochs):
    ran_epochs = epoch + 1
    print(f"Epoch {epoch + 1}/{num_epochs}")

    # ---------------- TRAIN ----------------
    model.train()
    train_loss = 0.0
    train_preds, train_trues = [], []

    for x_batch, y_batch, names in tqdm(train_loader, desc="Training"):
        if x_batch.numel() == 0:
            continue

        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)

        train_preds.extend(preds.detach().cpu().numpy())
        train_trues.extend(y_batch.detach().cpu().numpy())

        del x_batch, y_batch, outputs, preds

    avg_train_loss = train_loss / max(1, len(train_loader))
    train_acc = accuracy_score(train_trues, train_preds)

    # ---------------- VALID ----------------
    model.eval()
    val_loss = 0.0
    val_preds, val_trues = [], []

    with torch.no_grad():
        for x_batch, y_batch, names in tqdm(val_loader, desc="Validating"):
            if x_batch.numel() == 0:
                continue

            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            val_preds.extend(preds.detach().cpu().numpy())
            val_trues.extend(y_batch.detach().cpu().numpy())

            del x_batch, y_batch, outputs, preds

    avg_val_loss = val_loss / max(1, len(val_loader))
    val_acc = accuracy_score(val_trues, val_preds)

    # ---------------- HOLDOUT ----------------
    hold_loss = 0.0
    hold_preds, hold_trues = [], []

    with torch.no_grad():
        for x_batch, y_batch, names in tqdm(hold_loader, desc="Holdout"):
            if x_batch.numel() == 0:
                continue

            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            hold_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            hold_preds.extend(preds.detach().cpu().numpy())
            hold_trues.extend(y_batch.detach().cpu().numpy())

            del x_batch, y_batch, outputs, preds

    avg_hold_loss = hold_loss / max(1, len(hold_loader))
    hold_acc = accuracy_score(hold_trues, hold_preds)

    # Keep current epoch predictions
    last_train_preds, last_train_trues = train_preds, train_trues
    last_val_preds, last_val_trues = val_preds, val_trues
    last_hold_preds, last_hold_trues = hold_preds, hold_trues

    train_report = classification_report(
        train_trues, train_preds,
        target_names=allowed_species,
        labels=list(range(len(allowed_species))),
        output_dict=True,
        zero_division=0,
    )
    val_report = classification_report(
        val_trues, val_preds,
        target_names=allowed_species,
        labels=list(range(len(allowed_species))),
        output_dict=True,
        zero_division=0,
    )
    hold_report = classification_report(
        hold_trues, hold_preds,
        target_names=allowed_species,
        labels=list(range(len(allowed_species))),
        output_dict=True,
        zero_division=0,
    )

    wandb.log({
        "epoch": epoch + 1,
        "train/avg_loss": avg_train_loss,
        "train/accuracy": train_acc,
        "val/avg_loss": avg_val_loss,
        "val/accuracy": val_acc,
        "holdout/avg_loss": avg_hold_loss,
        "holdout/accuracy": hold_acc,
    })

    for split_name, report in [("train", train_report), ("val", val_report), ("holdout", hold_report)]:
        for cname in allowed_species:
            if cname in report:
                wandb.log({
                    f"{cname}/{split_name}_precision": report[cname]["precision"],
                    f"{cname}/{split_name}_recall": report[cname]["recall"],
                    f"{cname}/{split_name}_f1": report[cname]["f1-score"],
                }, step=epoch + 1)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[Epoch {epoch+1}] train_acc={train_acc:.4f} val_acc={val_acc:.4f} holdout_acc={hold_acc:.4f}")

    # ---------------- EARLY STOPPING ----------------
    if bool(config.early_stop):
        improved = (best_val_loss - avg_val_loss) > float(config.early_stop_min_delta)

        if improved:
            best_val_loss = avg_val_loss
            bad_epochs = 0

            payload_best = {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "class_names": allowed_species,
                "config": dict(config),
                "best_val_loss": best_val_loss,
            }
            torch.save(payload_best, best_ckpt_path)
            print(f"[early_stop] ✅ New best val_loss={best_val_loss:.6f}. Saved -> {best_ckpt_path}")
            wandb.log({"best/val_loss": best_val_loss}, step=epoch + 1)
        else:
            bad_epochs += 1
            print(f"[early_stop] No improvement in val_loss. bad_epochs={bad_epochs}/{config.early_stop_patience}")
            if bad_epochs >= int(config.early_stop_patience):
                print(f"[early_stop] 🛑 Stopping early at epoch {epoch+1} (best val_loss={best_val_loss:.6f})")
                break


# ------------------------------------------------------------------------------
# Heatmaps
# ------------------------------------------------------------------------------
fig_tr = plot_confmat(
    last_train_trues, last_train_preds, allowed_species,
    normalize="true", title="Train Confusion Matrix (row-normalized)"
)
fig_va = plot_confmat(
    last_val_trues, last_val_preds, allowed_species,
    normalize="true", title="Validation Confusion Matrix (row-normalized)"
)
fig_ho = plot_confmat(
    last_hold_trues, last_hold_preds, allowed_species,
    normalize="true", title="Holdout Confusion Matrix (row-normalized)"
)

wandb.log({
    "last_train/confusion_matrix_heatmap": wandb.Image(fig_tr),
    "last_val/confusion_matrix_heatmap": wandb.Image(fig_va),
    "last_holdout/confusion_matrix_heatmap": wandb.Image(fig_ho),
})

plt.close(fig_tr)
plt.close(fig_va)
plt.close(fig_ho)


# ------------------------------------------------------------------------------
# Save prediction JSONs
# ------------------------------------------------------------------------------
saved_files = []
if bool(config.save_probs_json):
    train_json = os.path.join(run_dir, "last_epoch_predictions_train.json")
    val_json = os.path.join(run_dir, "last_epoch_predictions_valid.json")
    hold_json = os.path.join(run_dir, "last_epoch_predictions_holdout.json")

    save_predictions_json(train_json, "train", allowed_species, train_prob_rows)
    save_predictions_json(val_json, "valid", allowed_species, val_prob_rows)
    save_predictions_json(hold_json, "holdout", allowed_species, hold_prob_rows)

    saved_files = [train_json, val_json, hold_json]
    print("Saved last-epoch prediction JSONs:")
    for p in saved_files:
        print("  -", p)

    if bool(config.upload_probs_artifact):
        art = wandb.Artifact("last_epoch_predictions", type="predictions")
        for p in saved_files:
            art.add_file(p)
        wandb.log_artifact(art)


# ------------------------------------------------------------------------------
# Save final checkpoint
# ------------------------------------------------------------------------------
ckpt_dir = Path(run_dir)
ckpt_dir.mkdir(parents=True, exist_ok=True)

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
last_ckpt = ckpt_dir / f"last_model_state_resnet18_{stamp}.pkl"

payload_last = {
    "epoch": ran_epochs,
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "class_names": allowed_species,
    "config": dict(config),
}
torch.save(payload_last, last_ckpt)
print(f"[save] Wrote last model state to: {last_ckpt}")

wandb.finish()
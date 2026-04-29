#!/usr/bin/env python3
"""
training.py (ResNet18-only)

Train a ResNet18-based image classifier for wildlife species classification and log results to Weights & Biases.

Key behavior:
- If cached filtered CSV exists at `filtered_all_path`:
    -> load it and skip MegaDetector
- Else:
    -> build df from `image_dir` (filename -> species + site parsing)
    -> run MegaDetector to keep only images with >= 1 detection
    -> save the resulting df as `full_df_filtered.csv`
    -> continue training

Class selection:
- config.num_classes can be an int (top-N in TRAIN) or "all" (all unique species in TRAIN).
- config.include_species are appended if not already included.

Assumptions:
- build_df_from_folder(image_dir) returns columns: filename, species, site
- SpeciesImageDataset expects df with filename/species/site
- collate_keep_good may yield empty batches; we skip those safely.
"""

from __future__ import annotations

import os
import sys
import gc
import json
from pathlib import Path
import importlib.util
from datetime import datetime

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from torch.utils.data import DataLoader
import wandb

from dataloader import SpeciesImageDataset, collate_keep_good
from utilities import compute_allowed_species, plot_confmat, save_predictions_json, is_good, filter_bad_files
from detector import filter_df_with_megadetector

# ------------------------------------------------------------------------------
# Performance toggles
# ------------------------------------------------------------------------------
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")  # PyTorch 2.x
except Exception:
    pass


# ------------------------------------------------------------------------------
# Project root & sys.path configuration
# ------------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]  # .../Desktop/kaitlyn_catalyst

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ------------------------------------------------------------------------------
# Imports that depend on sys.path
# ------------------------------------------------------------------------------
from splitting import (
    build_df_from_folder,
    split_train_val_holdout,
)

# Import ResNet model from ct_classifier (package if possible; fallback to file import).
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

# Optional "raw" CSV (if present, used instead of scanning image_dir)
csv_path = BASE / "Desktop" / "kaitlyn_catalyst" / "notebooks" / "full_df_filtered.csv"

# Directory for writing cached outputs (filtered CSV + predictions + checkpoints)
run_dir = BASE / "Desktop" / "kaitlyn_catalyst" / "resnet_training"
run_dir.mkdir(parents=True, exist_ok=True)

# Images live here
image_dir = BASE / "images" / "all_species_images"

# Cached filtered dataframe (created if missing)
filtered_all_path = run_dir / "full_df_filtered.csv"

# Only image_dir is mandatory
if not image_dir.exists():
    raise FileNotFoundError(f"image_dir not found: {image_dir}")

# csv_path is optional; do NOT raise
if not csv_path.exists():
    print(f"[warn] csv_path not found (optional): {csv_path}")
    print(f"       Will either use cached filtered csv ({filtered_all_path}) or build+filter from image_dir.")

# Convert to str where needed later
csv_path = str(csv_path)
run_dir = str(run_dir)
image_dir = str(image_dir)
filtered_all_path = str(filtered_all_path)


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
        # - int: top-N most frequent species in TRAIN
        # - "all": all unique species in TRAIN
        "num_classes": "all",
        "include_species": [
            "nyala", "bushbuck",
            "reedbuck", "oribi",
            "civet", "genet",
            "mongoose_marsh", "mongoose_white_tailed", "mongoose_slender",
            "mongoose_banded", "mongoose_bushy_tailed", "mongoose_large_grey", "mongoose_dwarf",
            "bushpig", "warthog",
            "duiker_common", "duiker_red", "duiker",
            "buffalo", "wildebeest", "hippopotamus",
            "hartebeest", "eland", "pangolin"
        ],

        # Splitting
        "holdout_sites": ["d05", "d03", "g02", "e02", "e06", "f05", "i10", "i04", "i08", "d07", "b05", "g08"],
        "split_mode": "instance",  # "instance" or "sitewise"
        "min_in_each": 1,
        "test_size": 0.30,

        # Caching & filtering
        "cache_filtered_all": True,
        "use_megadetector_if_missing": True,
        "megadetector_model": "MDV5A",  # or a local path
        "megadetector_conf": 0.2,
        "megadetector_device": "cuda",  # "cuda" or "cpu"

        # Saving last-epoch predictions
        "save_probs_json": True,
        "upload_probs_artifact": True
    },
)

wandb.define_metric("epoch")
wandb.define_metric("*", step_metric="epoch")
config = wandb.config
wandb.run.name = f"resnet18_{config.split_mode}_N{config.num_classes}_ls{config.label_smoothing}"


# ------------------------------------------------------------------------------
# Load/build + MegaDetector-filter to create filtered_all_df
# ------------------------------------------------------------------------------
if os.path.exists(filtered_all_path):
    print(f"[info] Using cached filtered CSV: {filtered_all_path}")
    filtered_all_df = pd.read_csv(filtered_all_path)

else:
    print(f"[info] Filtered CSV not found: {filtered_all_path}")

    # Step A: build initial dataframe
    if os.path.exists(csv_path):
        print(f"[info] Using raw CSV: {csv_path}")
        full_df = pd.read_csv(csv_path)
    else:
        print("[info] Building dataframe by scanning image_dir...")
        full_df = build_df_from_folder(image_dir)

    if full_df is None or len(full_df) == 0:
        raise RuntimeError("build_df_from_folder() returned an empty dataframe. Check filename pattern / image_dir.")

    # Normalize + file sanity filtering
    for col in ("species", "site"):
        if col not in full_df.columns:
            raise ValueError(f"Expected column '{col}' in full_df but it is missing. Columns: {list(full_df.columns)}")
        full_df[col] = full_df[col].astype(str).str.strip().str.lower()

    if "filename" not in full_df.columns:
        raise ValueError(f"Expected column 'filename' in full_df. Columns: {list(full_df.columns)}")

    full_df = filter_bad_files(full_df, image_dir)

    # Step B: run MegaDetector filtering (only if enabled)
    if bool(config.use_megadetector_if_missing):
        filtered_all_df = filter_df_with_megadetector(
            full_df,
            image_dir=image_dir,
            conf_thresh=float(config.megadetector_conf),
            model_name_or_path=str(config.megadetector_model),
            device=str(config.megadetector_device),
        )
    else:
        print("[warn] MegaDetector disabled; using full_df after file sanity filtering.")
        filtered_all_df = full_df.copy()

    # Step C: save filtered CSV for future runs
    if bool(config.cache_filtered_all):
        print(f"[info] Saving filtered dataframe -> {filtered_all_path}")
        filtered_all_df.to_csv(filtered_all_path, index=False)

# Final normalization (defensive)
for col in ("species", "site"):
    filtered_all_df[col] = filtered_all_df[col].astype(str).str.strip().str.lower()


# ------------------------------------------------------------------------------
# Split (instance or sitewise)
# ------------------------------------------------------------------------------
train_df, val_df, holdout_df = split_train_val_holdout(
    filtered_all_df,
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
# Class list & datasets (ResNet only)
# ------------------------------------------------------------------------------
allowed_species = compute_allowed_species(
    train_df,
    config.num_classes,  # <-- DO NOT cast; supports "all"
    include_list=list(config.include_species) if hasattr(config, "include_species") else None,
)
num_classes_eff = len(allowed_species)
print("Classes:", allowed_species)
print("Effective num_classes:", num_classes_eff)

model = CustomResNet18(num_classes=num_classes_eff).to(
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
)


train_dataset = SpeciesImageDataset(train_df, image_dir, classifier=None, backbone="resnet18",
                                    top_n_species=None, include_species=allowed_species)
val_dataset = SpeciesImageDataset(val_df, image_dir, classifier=None, backbone="resnet18",
                                  top_n_species=None, include_species=allowed_species)
hold_dataset = SpeciesImageDataset(holdout_df, image_dir, classifier=None, backbone="resnet18",
                                   top_n_species=None, include_species=allowed_species)


# ------------------------------------------------------------------------------
# Loss/optim/loaders
# ------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")

criterion = nn.CrossEntropyLoss(label_smoothing=float(config.label_smoothing)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay))

common_loader_kw = dict(
    batch_size=int(config.batch_size),
    pin_memory=True,
    num_workers=8,
    persistent_workers=True,
    prefetch_factor=4
)

train_loader = DataLoader(train_dataset, shuffle=True, collate_fn=collate_keep_good, **common_loader_kw)
val_loader = DataLoader(val_dataset, shuffle=False, collate_fn=collate_keep_good, **common_loader_kw)
hold_loader = DataLoader(hold_dataset, shuffle=False, collate_fn=collate_keep_good, **common_loader_kw)

with torch.no_grad():
    dummy = torch.randn(2, 3, 224, 224).to(device)
    out = model(dummy)
    assert out.shape[1] == len(allowed_species), f"Model head {out.shape[1]} != classes {len(allowed_species)}"


# ------------------------------------------------------------------------------
# Train (record LAST epoch) + save probabilities, skipping empty batches
# ------------------------------------------------------------------------------
last_train_preds, last_train_trues = [], []
last_val_preds, last_val_trues = [], []
last_hold_preds, last_hold_trues = [], []

train_prob_rows, val_prob_rows, hold_prob_rows = [], [], []
num_epochs = int(config.epochs)

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    is_last = (epoch == num_epochs - 1)

    # ==================== TRAIN ====================
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

        if is_last and bool(config.save_probs_json):
            probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
            y_np = y_batch.detach().cpu().numpy()
            p_np = preds.detach().cpu().numpy()
            for i in range(len(y_np)):
                train_prob_rows.append({
                    "filename": str(names[i]),
                    "true_idx": int(y_np[i]),
                    "pred_idx": int(p_np[i]),
                    "true_label": allowed_species[int(y_np[i])],
                    "pred_label": allowed_species[int(p_np[i])],
                    "probs": probs[i].tolist()
                })

        train_preds.extend(preds.detach().cpu().numpy())
        train_trues.extend(y_batch.detach().cpu().numpy())

        del x_batch, y_batch, outputs, preds

    avg_train_loss = train_loss / max(1, len(train_loader))
    train_acc = accuracy_score(train_trues, train_preds)

    # ==================== VALIDATE ====================
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

            if is_last and bool(config.save_probs_json):
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                y_np = y_batch.cpu().numpy()
                p_np = preds.cpu().numpy()
                for i in range(len(y_np)):
                    val_prob_rows.append({
                        "filename": str(names[i]),
                        "true_idx": int(y_np[i]),
                        "pred_idx": int(p_np[i]),
                        "true_label": allowed_species[int(y_np[i])],
                        "pred_label": allowed_species[int(p_np[i])],
                        "probs": probs[i].tolist()
                    })

            val_preds.extend(preds.detach().cpu().numpy())
            val_trues.extend(y_batch.detach().cpu().numpy())

            del x_batch, y_batch, outputs, preds

    avg_val_loss = val_loss / max(1, len(val_loader))
    val_acc = accuracy_score(val_trues, val_preds)

    # ==================== HOLDOUT ====================
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

            if is_last and bool(config.save_probs_json):
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                y_np = y_batch.cpu().numpy()
                p_np = preds.cpu().numpy()
                for i in range(len(y_np)):
                    hold_prob_rows.append({
                        "filename": str(names[i]),
                        "true_idx": int(y_np[i]),
                        "pred_idx": int(p_np[i]),
                        "true_label": allowed_species[int(y_np[i])],
                        "pred_label": allowed_species[int(p_np[i])],
                        "probs": probs[i].tolist()
                    })

            hold_preds.extend(preds.detach().cpu().numpy())
            hold_trues.extend(y_batch.detach().cpu().numpy())

            del x_batch, y_batch, outputs, preds

    avg_hold_loss = hold_loss / max(1, len(hold_loader))
    hold_acc = accuracy_score(hold_trues, hold_preds)

    # Keep LAST epoch’s predictions
    last_train_preds, last_train_trues = train_preds, train_trues
    last_val_preds, last_val_trues = val_preds, val_trues
    last_hold_preds, last_hold_trues = hold_preds, hold_trues

    # Reports + logging
    train_report = classification_report(train_trues, train_preds, target_names=allowed_species,
                                         labels=list(range(len(allowed_species))), output_dict=True, zero_division=0)
    val_report = classification_report(val_trues, val_preds, target_names=allowed_species,
                                       labels=list(range(len(allowed_species))), output_dict=True, zero_division=0)
    hold_report = classification_report(hold_trues, hold_preds, target_names=allowed_species,
                                        labels=list(range(len(allowed_species))), output_dict=True, zero_division=0)

    wandb.log({
        "epoch": epoch + 1,
        "train/avg_loss": avg_train_loss, "train/accuracy": train_acc,
        "val/avg_loss": avg_val_loss, "val/accuracy": val_acc,
        "holdout/avg_loss": avg_hold_loss, "holdout/accuracy": hold_acc,
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


# ------------------------------------------------------------------------------
# Heatmaps (last epoch)
# ------------------------------------------------------------------------------
fig_tr = plot_confmat(last_train_trues, last_train_preds, allowed_species, normalize="true",
                      title="Train Confusion Matrix (row-normalized)")
fig_va = plot_confmat(last_val_trues, last_val_preds, allowed_species, normalize="true",
                      title="Validation Confusion Matrix (row-normalized)")
fig_ho = plot_confmat(last_hold_trues, last_hold_preds, allowed_species, normalize="true",
                      title="Holdout Confusion Matrix (row-normalized)")

wandb.log({
    "last_train/confusion_matrix_heatmap": wandb.Image(fig_tr),
    "last_val/confusion_matrix_heatmap": wandb.Image(fig_va),
    "last_holdout/confusion_matrix_heatmap": wandb.Image(fig_ho),
})
plt.close(fig_tr); plt.close(fig_va); plt.close(fig_ho)


# ------------------------------------------------------------------------------
# Save last-epoch probabilities to JSON + optional W&B artifact
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
# Save LAST model checkpoint (state_dict)
# ------------------------------------------------------------------------------
ckpt_dir = Path(run_dir)
ckpt_dir.mkdir(parents=True, exist_ok=True)

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
last_ckpt = ckpt_dir / f"last_model_state_resnet18_{stamp}.pkl"

payload = {
    "epoch": num_epochs,
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "class_names": allowed_species,
    "config": dict(config),
}
torch.save(payload, last_ckpt)
print(f"[save] Wrote last model state to: {last_ckpt}")

wandb.finish()

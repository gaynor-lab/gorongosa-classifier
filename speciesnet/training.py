#!/usr/bin/env python3
# training.py

from __future__ import annotations
import os
import sys
import gc
import json
from pathlib import Path
import importlib.util
from dataloader import SpeciesImageDataset, collate_keep_good
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

# training.py (right after imports)
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")  # PyTorch 2.x
except Exception:
    pass

# ────────────────────────────────────────────────────────────────────────────────
# Project root & sys.path
# training.py is in: .../Desktop/kaitlyn_catalyst/speciesnet/training.py
# So PROJECT_ROOT is one level up: .../Desktop/kaitlyn_catalyst
# ────────────────────────────────────────────────────────────────────────────────
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]  # .../Desktop/kaitlyn_catalyst

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ────────────────────────────────────────────────────────────────────────────────
# Imports that depend on sys.path
# ────────────────────────────────────────────────────────────────────────────────
from speciesnet.classifier import SpeciesNetClassifier
from model import AugmentedSpeciesNet
from splitting import (    # ensure your splitting.py is under speciesnet/
    build_df_from_folder,
    split_train_val_holdout,          # supports mode="instance" or "sitewise"
)
from dataloader import SpeciesImageDataset

# Try to import from ct_classifier as a proper package first.
# If ct_classifier lacks __init__.py, we fall back to importlib.
try:
    from ct_classifier.model import CustomResNet18
except Exception:
    CT_MODEL_PATH = PROJECT_ROOT / "ct_classifier" / "model.py"
    if not CT_MODEL_PATH.exists():
        raise FileNotFoundError(f"Could not find ct_classifier/model.py at {CT_MODEL_PATH}")
    spec = importlib.util.spec_from_file_location("ct_model", CT_MODEL_PATH)
    ct_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ct_model)
    CustomResNet18 = ct_model.CustomResNet18

# ────────────────────────────────────────────────────────────────────────────────
# Paths
# ────────────────────────────────────────────────────────────────────────────────
BASE = Path("/mnt/sharedstorage/sabdelazim")
csv_path        = BASE / "Desktop" / "kaitlyn_catalyst" / "notebooks" / "full_df_filtered.csv"
speciesnet_dir  = BASE / "Desktop" / "kaitlyn_catalyst" / "speciesnet"
image_dir       = BASE / "images" / "all_species_images"
target_species_txt  = speciesnet_dir / "target_species_selected.txt"
filtered_all_path   = speciesnet_dir / "filtered_all.csv"
id2species_path     = speciesnet_dir / "id_to_species_full.json"
speciesnet_dir.mkdir(parents=True, exist_ok=True)

# Sanity checks
for p, label in [(csv_path, "csv_path"), (image_dir, "image_dir")]:
    if not p.exists():
        raise FileNotFoundError(f"{label} not found: {p}")

# Cast to str where needed
csv_path = str(csv_path)
speciesnet_dir = str(speciesnet_dir)
image_dir = str(image_dir)
target_species_txt = str(target_species_txt)
filtered_all_path = str(filtered_all_path)
id2species_path = str(id2species_path)

# ────────────────────────────────────────────────────────────────────────────────
# W&B config
# ────────────────────────────────────────────────────────────────────────────────
wandb.init(
    project="Species-Classification",
    config={
        # Core
        "backbone": "resnet18",         # "speciesnet" or "resnet18"
        "epochs": 10,
        "lr": 1e-4,
        "batch_size": 256,
        "label_smoothing": 0.0,
        "weight_decay": 0.0,
        "dropout": 0.0,

        # Class selection
        "num_classes": 10,              # take top-N from TRAIN
        "include_species": [
            # keep individual mongoose variants
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
        "holdout_sites": ["d05","d03","g02","e02","e06","f05","i10","i04","i08","d07","b05","g08"],
        "split_mode": "instance",       # "instance" (NEW) or "sitewise" (OLD)
        "min_in_each": 1,
        "test_size": 0.30,

        # Misc
        "cache_filtered_all": False,

        # Saving last-epoch predictions
        "save_probs_json": True,
        "upload_probs_artifact": True
    },
)
wandb.define_metric("epoch")
wandb.define_metric("*", step_metric="epoch")
config = wandb.config
wandb.run.name = f"{config.backbone}_{config.split_mode}_N{config.num_classes}_ls{config.label_smoothing}"

# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────
def compute_allowed_species(train_df: pd.DataFrame, top_n, include_list=None):
    s = train_df["species"].astype(str).str.strip().str.lower()
    top = s.value_counts().nlargest(int(top_n)).index.tolist() if top_n is not None else []
    inc = [str(x).strip().lower() for x in (include_list or [])]
    return top + [x for x in inc if x not in top]

def plot_confmat(y_true, y_pred, class_names, normalize="true", title="Confusion Matrix"):
    labels = np.arange(len(class_names))
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    side = max(6, int(len(class_names) * 0.45))
    fig, ax = plt.subplots(figsize=(side, side))
    im = ax.imshow(cm, interpolation="nearest", aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=labels, yticks=labels,
           xticklabels=class_names, yticklabels=class_names,
           xlabel="Predicted", ylabel="True", title=title)
    for t in ax.get_xticklabels(): t.set_rotation(90)
    if len(class_names) <= 30:
        thresh = cm.max() / 2.0 if cm.size and np.isfinite(cm).any() else 0.5
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                txt = f"{val:.2f}" if normalize else f"{int(val)}"
                ax.text(j, i, txt, ha="center", va="center",
                        color="white" if val > thresh else "black", fontsize=7)
    fig.tight_layout()
    return fig

def save_predictions_json(path, split_name, class_names, rows):
    payload = {
        "split": split_name,
        "class_names": class_names,
        "num_classes": len(class_names),
        "num_samples": len(rows),
        "samples": rows
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

# ────────────────────────────────────────────────────────────────────────────────
# Load data (CSV preferred)
# ────────────────────────────────────────────────────────────────────────────────
if os.path.exists(csv_path):
    full_df = pd.read_csv(csv_path)
# else:
#     print("csv_path not found; scanning folder as fallback…")
    full_df = build_df_from_folder(image_dir)

# If you aren't actively using detector filtering, just normalize and go
if os.path.exists(filtered_all_path):
    print("Using existing filtered df.")
    filtered_all_df = pd.read_csv(filtered_all_path)
else:
    filtered_all_df = full_df.copy()
    if bool(config.cache_filtered_all):
        filtered_all_df.to_csv(filtered_all_path, index=False)

for col in ("species", "site"):
    filtered_all_df[col] = filtered_all_df[col].astype(str).str.strip().str.lower()


def is_good(path):
    try:
        return os.path.getsize(path) > 0
    except Exception:
        return False

def filter_bad_files(df, image_dir):
    paths = df["filename"].apply(lambda f: os.path.join(image_dir, f))
    mask = paths.apply(is_good)
    bad = df[~mask]
    good = df[mask].reset_index(drop=True)
    print(f"[precheck] kept {len(good)} | dropped {len(bad)} bad/empty files")
    return good

filtered_all_df = filter_bad_files(filtered_all_df, image_dir)
    
# ────────────────────────────────────────────────────────────────────────────────
# Split (instance or sitewise)
# ────────────────────────────────────────────────────────────────────────────────
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
    _df["site"]    = _df["site"].astype(str).str.strip().str.lower()

# ────────────────────────────────────────────────────────────────────────────────
# Class list & datasets
# ────────────────────────────────────────────────────────────────────────────────
allowed_species = compute_allowed_species(
    train_df,
    int(config.num_classes),
    include_list=list(config.include_species) if hasattr(config, "include_species") else None,
)
num_classes_eff = len(allowed_species)
print("Classes:", allowed_species)
print("Effective num_classes:", num_classes_eff)

backbone = str(config.backbone).lower()

if backbone == "speciesnet":
    with open(id2species_path, "r") as f:
        id_to_species = json.load(f)

    short_to_tl = {}
    for info in id_to_species.values():
        short = info["short_name"].replace(" ", "_").strip().lower()
        short_to_tl[short] = info["target_label"]

    selected_labels = []
    for s in allowed_species:
        if s in short_to_tl:
            selected_labels.append(short_to_tl[s])
        else:
            raise ValueError(f"Species '{s}' not in id_to_species_full.json mapping.")

    with open(target_species_txt, "w") as f:
        for label in selected_labels:
            f.write(label + "\n")

    classifier = SpeciesNetClassifier(
        model_name=os.path.expanduser("~/.cache/kagglehub/models/google/speciesnet/pyTorch/v4.0.1a/1"),
        target_species_txt=target_species_txt
    )
    original_outputs = len(classifier.labels)
    target_labels = len(classifier.target_labels)

    model = AugmentedSpeciesNet(
        classifier.model,
        original_outputs,
        target_labels,
        use_extra_head=True,
        dropout=float(config.dropout),
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.preprocess = classifier.preprocess

    train_dataset = SpeciesImageDataset(train_df, image_dir, classifier=model, backbone="speciesnet",
                                        top_n_species=None, include_species=allowed_species, return_meta=True)
    val_dataset   = SpeciesImageDataset(val_df, image_dir, classifier=model, backbone="speciesnet",
                                        top_n_species=None, include_species=allowed_species, return_meta=True)
    hold_dataset  = SpeciesImageDataset(holdout_df, image_dir, classifier=model, backbone="speciesnet",
                                        top_n_species=None, include_species=allowed_species, return_meta=True)

elif backbone == "resnet18":
    model = CustomResNet18(num_classes=num_classes_eff).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    train_dataset = SpeciesImageDataset(train_df, image_dir, classifier=None, backbone="resnet18",
                                        top_n_species=None, include_species=allowed_species)
    val_dataset   = SpeciesImageDataset(val_df, image_dir, classifier=None, backbone="resnet18",
                                        top_n_species=None, include_species=allowed_species)
    hold_dataset  = SpeciesImageDataset(holdout_df, image_dir, classifier=None, backbone="resnet18",
                                        top_n_species=None, include_species=allowed_species)
else:
    raise ValueError("config.backbone must be 'speciesnet' or 'resnet18'")

# ────────────────────────────────────────────────────────────────────────────────
# Loss/optim/loaders
# ────────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")
criterion = nn.CrossEntropyLoss(label_smoothing=float(config.label_smoothing)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay))

pin = torch.cuda.is_available()
# training.py (where you build loaders)

common_loader_kw = dict(
    batch_size=int(config.batch_size),
    pin_memory=True,                # pinned host -> faster H2D
    num_workers=8,                  # try 4–8; tune by CPU cores / disk
    persistent_workers=True,        # keep workers alive
    prefetch_factor=4               # each worker prefetches N batches
)

train_loader = DataLoader(train_dataset, shuffle=True,  collate_fn=collate_keep_good, **common_loader_kw)
val_loader   = DataLoader(val_dataset,   shuffle=False, collate_fn=collate_keep_good, **common_loader_kw)
hold_loader  = DataLoader(hold_dataset,  shuffle=False, collate_fn=collate_keep_good, **common_loader_kw)


with torch.no_grad():
    dummy = torch.randn(2, 3, 224, 224).to(device)
    out = model(dummy)
    assert out.shape[1] == len(allowed_species), f"Model head {out.shape[1]} != classes {len(allowed_species)}"

# ────────────────────────────────────────────────────────────────────────────────
# Train (record LAST epoch) + save probabilities, skipping empty batches
# ────────────────────────────────────────────────────────────────────────────────
last_train_preds, last_train_trues = [], []
last_val_preds,   last_val_trues   = [], []
last_hold_preds,  last_hold_trues  = [], []

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
        # Skip empty batches produced by collate_keep_good
        if x_batch.numel() == 0:
            continue

        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)

        # (Optional) collect probabilities on last epoch
        if is_last and bool(config.save_probs_json):
            probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
            y_np  = y_batch.detach().cpu().numpy()
            p_np  = preds.detach().cpu().numpy()
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
            # Skip empty batches
            if x_batch.numel() == 0:
                continue

            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            if is_last and bool(config.save_probs_json):
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                y_np  = y_batch.cpu().numpy()
                p_np  = preds.cpu().numpy()
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
            # Skip empty batches
            if x_batch.numel() == 0:
                continue

            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            hold_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            if is_last and bool(config.save_probs_json):
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                y_np  = y_batch.cpu().numpy()
                p_np  = preds.cpu().numpy()
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
    last_val_preds,   last_val_trues   = val_preds,   val_trues
    last_hold_preds,  last_hold_trues  = hold_preds,  hold_trues

    # Reports + logging
    train_report = classification_report(train_trues, train_preds, target_names=allowed_species,
                                         labels=list(range(len(allowed_species))), output_dict=True)
    val_report   = classification_report(val_trues,   val_preds,   target_names=allowed_species,
                                         labels=list(range(len(allowed_species))), output_dict=True)
    hold_report  = classification_report(hold_trues,  hold_preds,  target_names=allowed_species,
                                         labels=list(range(len(allowed_species))), output_dict=True)

    wandb.log({
        "epoch": epoch + 1,
        "train/avg_loss": avg_train_loss, "train/accuracy": train_acc,
        "val/avg_loss":   avg_val_loss,   "val/accuracy":   val_acc,
        "holdout/avg_loss": avg_hold_loss, "holdout/accuracy": hold_acc,
    })

    for split_name, report in [("train", train_report), ("val", val_report), ("holdout", hold_report)]:
        for cname in allowed_species:
            if cname in report:
                wandb.log({
                    f"{cname}/{split_name}_precision": report[cname]["precision"],
                    f"{cname}/{split_name}_recall":    report[cname]["recall"],
                    f"{cname}/{split_name}_f1":        report[cname]["f1-score"],
                }, step=epoch + 1)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[Epoch {epoch+1}] train_acc={train_acc:.4f} val_acc={val_acc:.4f} holdout_acc={hold_acc:.4f}")


# Heatmaps (last epoch)
fig_tr = plot_confmat(last_train_trues, last_train_preds, allowed_species, normalize="true",
                      title="Train Confusion Matrix (row-normalized)")
fig_va = plot_confmat(last_val_trues,   last_val_preds,   allowed_species, normalize="true",
                      title="Validation Confusion Matrix (row-normalized)")
fig_ho = plot_confmat(last_hold_trues,  last_hold_preds,  allowed_species, normalize="true",
                      title="Holdout Confusion Matrix (row-normalized)")

wandb.log({
    "last_train/confusion_matrix_heatmap":  wandb.Image(fig_tr),
    "last_val/confusion_matrix_heatmap":    wandb.Image(fig_va),
    "last_holdout/confusion_matrix_heatmap": wandb.Image(fig_ho),
})
plt.close(fig_tr); plt.close(fig_va); plt.close(fig_ho)

# Save last-epoch probabilities to JSON + optional W&B artifact
saved_files = []
if bool(config.save_probs_json):
    train_json = os.path.join(speciesnet_dir, "last_epoch_predictions_train.json")
    val_json   = os.path.join(speciesnet_dir, "last_epoch_predictions_valid.json")
    hold_json  = os.path.join(speciesnet_dir, "last_epoch_predictions_holdout.json")

    save_predictions_json(train_json, "train",   allowed_species, train_prob_rows)
    save_predictions_json(val_json,   "valid",   allowed_species, val_prob_rows)
    save_predictions_json(hold_json,  "holdout", allowed_species, hold_prob_rows)

    saved_files = [train_json, val_json, hold_json]
    print("Saved last-epoch prediction JSONs:")
    for p in saved_files: print("  -", p)

    if bool(config.upload_probs_artifact):
        art = wandb.Artifact("last_epoch_predictions", type="predictions")
        for p in saved_files: art.add_file(p)
        wandb.log_artifact(art)

        
# ────────────────────────────────────────────────────────────────────────────────
# Save LAST model (safest: state_dict pickle)
# ────────────────────────────────────────────────────────────────────────────────

ckpt_dir = Path(speciesnet_dir)
ckpt_dir.mkdir(parents=True, exist_ok=True)

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
last_ckpt = ckpt_dir / f"last_model_state_{config.backbone}_{stamp}.pkl"

payload = {
    "epoch": num_epochs,                      # last epoch index (1-based in your logs)
    "model_state": model.state_dict(),        # weights only (recommended)
    "optimizer_state": optimizer.state_dict(),# optional; useful if you resume
    "class_names": allowed_species,           # to map indices later
    "config": dict(config),                   # for reproducibility
}
torch.save(payload, last_ckpt)
print(f"[save] Wrote last model state to: {last_ckpt}")
        
        
wandb.finish()
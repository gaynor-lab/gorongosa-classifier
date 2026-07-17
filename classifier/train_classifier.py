#!/usr/bin/env python3
"""
train_classifier.py

Training-only script for crop-based species classification.

This script assumes crops already exist and a CSV is provided with:
- crop filename/path
- species label
- site

Expected inputs:
  --input_csv
  --crop_dir
  --output_dir

Example:
  python classifier/train_classifier.py \
    --input_csv /scratch/st-kgaynor-1/$USER/gorongosa_classifier/processing/resnet_training/combined_training_data.csv \
    --crop_dir /scratch/st-kgaynor-1/$USER/gorongosa_classifier/processing/resnet_training/all_training_crops \
    --output_dir /scratch/st-kgaynor-1/$USER/gorongosa_classifier/processing/resnet_training/model_outputs/test_run
"""

from __future__ import annotations

import argparse
import ast
import gc
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


# ------------------------------------------------------------------------------
# Project imports
# ------------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
CLASSIFIER_DIR = THIS_FILE.parent
PROJECT_ROOT = THIS_FILE.parents[1]

for p in [CLASSIFIER_DIR, PROJECT_ROOT]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def load_custom_resnet18():
    """
    Load CustomResNet18 from classifier/model.py.
    """
    try:
        from model import CustomResNet18

        return CustomResNet18
    except Exception:
        model_path = CLASSIFIER_DIR / "model.py"
        if not model_path.exists():
            raise FileNotFoundError(f"Could not find model.py at {model_path}")

        spec = importlib.util.spec_from_file_location("ct_model", model_path)
        ct_model = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(ct_model)
        return ct_model.CustomResNet18


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def parse_comma_list(value: str | None) -> list[str]:
    if value is None or str(value).strip() == "":
        return []
    return [x.strip().lower() for x in str(value).split(",") if x.strip()]


def parse_crop_list(value):
    """
    Accepts:
    - a plain filename string
    - a Python/JSON-like list string, e.g. "['a.jpg', 'b.jpg']"
    - a semicolon/comma-separated string
    """
    if pd.isna(value):
        return []

    if isinstance(value, list):
        return [str(x) for x in value]

    text = str(value).strip()
    if text == "":
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            pass

    if ";" in text:
        return [x.strip() for x in text.split(";") if x.strip()]

    return [text]


def detect_image_column(df: pd.DataFrame, preferred: str | None = None) -> str:
    if preferred:
        if preferred not in df.columns:
            raise ValueError(f"Requested image column '{preferred}' not found. Columns: {list(df.columns)}")
        return preferred

    candidates = [
        "crop_path",
        "filename_crop",
        "filename_crops",
        "image_path",
        "path",
        "file_path",
        "filename",
        "file_name",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        "Could not detect crop image column. "
        "Please provide --image_col. "
        f"Columns found: {list(df.columns)}"
    )


def build_training_dataframe(
    input_csv: Path,
    image_col: str | None,
    species_col: str,
    site_col: str,
    source_col: str | None,
) -> pd.DataFrame:
    df = pd.read_csv(input_csv)

    if species_col not in df.columns:
        raise ValueError(f"Missing species column '{species_col}'. Columns: {list(df.columns)}")

    if site_col not in df.columns:
        print(f"[warn] Missing site column '{site_col}'. Creating site='unknown'.")
        df[site_col] = "unknown"

    detected_image_col = detect_image_column(df, image_col)
    print(f"[info] Using image column: {detected_image_col}")

    rows = []

    for _, row in df.iterrows():
        species = str(row[species_col]).strip().lower()
        site = str(row[site_col]).strip().lower()

        if species in ["", "nan", "none"]:
            continue

        crop_values = parse_crop_list(row[detected_image_col])

        for crop_value in crop_values:
            crop_value = str(crop_value).strip()
            if crop_value == "":
                continue

            new_row = {
                "crop_path": crop_value,
                "species": species,
                "site": site,
            }

            if source_col and source_col in df.columns:
                new_row["source"] = str(row[source_col]).strip().lower()
            elif "source" in df.columns:
                new_row["source"] = str(row["source"]).strip().lower()
            else:
                new_row["source"] = "unknown"

            if "filename" in df.columns:
                new_row["original_filename"] = str(row["filename"])

            rows.append(new_row)

    out = pd.DataFrame(rows)

    if len(out) == 0:
        raise RuntimeError("No crop rows were created from the input CSV.")

    out["species"] = out["species"].astype(str).str.strip().str.lower()
    out["site"] = out["site"].astype(str).str.strip().str.lower()
    out["source"] = out["source"].astype(str).str.strip().str.lower()

    return out.reset_index(drop=True)


def resolve_crop_path(crop_value: str, crop_dir: Path | None) -> Path:
    p = Path(str(crop_value))

    if p.is_absolute():
        return p

    if crop_dir is not None:
        return crop_dir / p

    return p


def check_existing_files(df: pd.DataFrame, crop_dir: Path | None) -> pd.DataFrame:
    keep = []

    for crop_value in tqdm(df["crop_path"], desc="Checking crop files"):
        p = resolve_crop_path(crop_value, crop_dir)
        keep.append(p.exists())

    before = len(df)
    checked = df[keep].reset_index(drop=True)
    dropped = before - len(checked)

    print(f"[file check] kept {len(checked)} / {before} rows")
    print(f"[file check] dropped missing files: {dropped}")

    return checked


def choose_allowed_species(
    train_df: pd.DataFrame,
    num_classes: str,
    include_species: list[str],
) -> list[str]:
    if include_species:
        allowed_species = sorted(set(include_species))
    elif str(num_classes).lower() == "all":
        allowed_species = sorted(train_df["species"].dropna().unique().tolist())
    else:
        n = int(num_classes)
        allowed_species = train_df["species"].value_counts().head(n).index.tolist()
        allowed_species = sorted(allowed_species)

    if len(allowed_species) == 0:
        raise RuntimeError("No allowed species found.")

    return allowed_species


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def save_predictions_json(path: Path, split_name: str, class_names: list[str], rows: list[dict]):
    payload = {
        "split": split_name,
        "class_names": class_names,
        "predictions": rows,
    }
    save_json(path, payload)


def save_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    output_path: Path,
    title: str,
):
    if len(y_true) == 0:
        print(f"[warn] No rows for confusion matrix: {title}")
        return

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_classification_report(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    output_json: Path,
    output_csv: Path,
):
    if len(y_true) == 0:
        print(f"[warn] No rows for classification report: {output_json}")
        save_json(output_json, {})
        pd.DataFrame().to_csv(output_csv, index=False)
        return

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    save_json(output_json, report)
    pd.DataFrame(report).transpose().to_csv(output_csv)


def append_prediction_rows(
    row_list: list[dict],
    names,
    y_batch,
    probs,
    preds,
    class_names: list[str],
):
    y_np = y_batch.detach().cpu().numpy()
    probs_np = probs.detach().cpu().numpy()
    preds_np = preds.detach().cpu().numpy()

    for name, true_idx, pred_idx, prob_vec in zip(names, y_np, preds_np, probs_np):
        true_idx = int(true_idx)
        pred_idx = int(pred_idx)

        row_list.append(
            {
                "filename": str(name),
                "true_idx": true_idx,
                "pred_idx": pred_idx,
                "true_label": class_names[true_idx],
                "pred_label": class_names[pred_idx],
                "probs": prob_vec.tolist(),
            }
        )


# ------------------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------------------
class CropDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        crop_dir: Path | None,
        species_to_idx: dict[str, int],
        transform,
    ):
        self.df = df.reset_index(drop=True)
        self.crop_dir = crop_dir
        self.species_to_idx = species_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = resolve_crop_path(row["crop_path"], self.crop_dir)
        species = row["species"]

        with Image.open(img_path) as img:
            img = img.convert("RGB")

        x = self.transform(img)
        y = self.species_to_idx[species]

        return x, y, str(img_path)


# ------------------------------------------------------------------------------
# Train / eval loops
# ------------------------------------------------------------------------------
def run_eval(model, loader, criterion, device, class_names, desc):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_trues = []
    pred_rows = []

    with torch.no_grad():
        for x_batch, y_batch, names in tqdm(loader, desc=desc):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            total_loss += loss.item()
            all_preds.extend(preds.detach().cpu().numpy())
            all_trues.extend(y_batch.detach().cpu().numpy())

            append_prediction_rows(
                pred_rows,
                names,
                y_batch,
                probs,
                preds,
                class_names,
            )

    avg_loss = total_loss / max(1, len(loader))
    acc = accuracy_score(all_trues, all_preds) if len(all_trues) > 0 else 0.0

    return avg_loss, acc, all_trues, all_preds, pred_rows


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train species classifier from existing crop images and an input CSV."
    )

    # Inputs / outputs
    parser.add_argument("--input_csv", required=True, help="CSV with crop image column, species, and site.")
    parser.add_argument("--crop_dir", required=True, help="Folder containing crop images.")
    parser.add_argument("--output_dir", required=True, help="Folder for model outputs.")

    # CSV columns
    parser.add_argument("--image_col", default="", help="Crop image column. Auto-detected if not provided.")
    parser.add_argument("--species_col", default="species")
    parser.add_argument("--site_col", default="site")
    parser.add_argument("--source_col", default="source")

    # Split
    parser.add_argument(
        "--holdout_sites",
        default="d05,d03,g02,e02,e06,f05,i10,i04,i08,d07,b05,g08",
        help="Comma-separated site names to reserve as holdout.",
    )
    parser.add_argument("--val_size", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)

    # Classes
    parser.add_argument("--num_classes", default="all", help='"all" or an integer.')
    parser.add_argument("--include_species", default="", help="Comma-separated species list. Overrides num_classes.")

    # Training params
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=8)

    # Early stopping
    parser.add_argument("--early_stop_patience", type=int, default=3)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    parser.add_argument("--no_early_stop", action="store_true")

    # Debug / outputs
    parser.add_argument("--check_files", action="store_true", help="Check crop files exist before training.")
    parser.add_argument("--save_train_predictions", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Prepare splits and exit before training.")

    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    crop_dir = Path(args.crop_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    predictions_dir = output_dir / "predictions"
    plots_dir = output_dir / "plots"
    reports_dir = output_dir / "reports"
    splits_dir = output_dir / "splits"

    for d in [checkpoints_dir, predictions_dir, plots_dir, reports_dir, splits_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        raise FileNotFoundError(f"input_csv not found: {input_csv}")

    if not crop_dir.exists():
        raise FileNotFoundError(f"crop_dir not found: {crop_dir}")

    # Performance toggles
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    save_json(output_dir / "run_config.json", vars(args))

    # --------------------------------------------------------------------------
    # Load and prepare data
    # --------------------------------------------------------------------------
    image_col = args.image_col.strip() if args.image_col.strip() else None

    df = build_training_dataframe(
        input_csv=input_csv,
        image_col=image_col,
        species_col=args.species_col,
        site_col=args.site_col,
        source_col=args.source_col,
    )

    print("[info] Loaded crop rows:", len(df))
    print("[info] Species count:", df["species"].nunique())
    print("[info] Site count:", df["site"].nunique())

    if args.check_files:
        df = check_existing_files(df, crop_dir)

    df.to_csv(output_dir / "normalized_input_rows.csv", index=False)

    # --------------------------------------------------------------------------
    # Split train / val / holdout
    # --------------------------------------------------------------------------
    holdout_sites = set(parse_comma_list(args.holdout_sites))

    if holdout_sites:
        holdout_df = df[df["site"].isin(holdout_sites)].copy()
        train_val_df = df[~df["site"].isin(holdout_sites)].copy()
    else:
        holdout_df = pd.DataFrame(columns=df.columns)
        train_val_df = df.copy()

    if len(train_val_df) == 0:
        raise RuntimeError("No train/validation rows left after removing holdout sites.")

    species_counts = train_val_df["species"].value_counts()
    can_stratify = species_counts.min() >= 2 and len(species_counts) > 1
    stratify = train_val_df["species"] if can_stratify else None

    if not can_stratify:
        print("[warn] Stratified split not possible. Using regular random split.")

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=stratify,
    )

    include_species = parse_comma_list(args.include_species)
    allowed_species = choose_allowed_species(
        train_df=train_df,
        num_classes=args.num_classes,
        include_species=include_species,
    )

    def filter_species(split_df: pd.DataFrame) -> pd.DataFrame:
        return split_df[split_df["species"].isin(allowed_species)].reset_index(drop=True)

    train_df = filter_species(train_df)
    val_df = filter_species(val_df)
    holdout_df = filter_species(holdout_df)

    print("\n[data split]")
    print(f"Total rows: {len(df)}")
    print(f"Train rows: {len(train_df)}")
    print(f"Validation rows: {len(val_df)}")
    print(f"Holdout rows: {len(holdout_df)}")
    print(f"Classes ({len(allowed_species)}): {allowed_species}")

    train_df.to_csv(splits_dir / "train_split.csv", index=False)
    val_df.to_csv(splits_dir / "val_split.csv", index=False)
    holdout_df.to_csv(splits_dir / "holdout_split.csv", index=False)

    split_summary = (
        pd.concat(
            [
                train_df.assign(split="train"),
                val_df.assign(split="val"),
                holdout_df.assign(split="holdout"),
            ],
            ignore_index=True,
        )
        .groupby(["split", "species", "site"])
        .size()
        .reset_index(name="count")
    )
    split_summary.to_csv(output_dir / "split_species_site_summary.csv", index=False)

    species_to_idx = {species: i for i, species in enumerate(allowed_species)}
    idx_to_species = {str(i): species for species, i in species_to_idx.items()}

    save_json(
        output_dir / "class_mapping.json",
        {
            "allowed_species": allowed_species,
            "species_to_idx": species_to_idx,
            "idx_to_species": idx_to_species,
        },
    )

    if args.dry_run:
        print("[dry run] Prepared input rows and splits. Exiting before training.")
        return

    # --------------------------------------------------------------------------
    # Dataset / loaders
    # --------------------------------------------------------------------------
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    train_dataset = CropDataset(train_df, crop_dir, species_to_idx, train_transform)
    val_dataset = CropDataset(val_df, crop_dir, species_to_idx, eval_transform)
    holdout_dataset = CropDataset(holdout_df, crop_dir, species_to_idx, eval_transform)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    holdout_loader = DataLoader(holdout_dataset, shuffle=False, **loader_kwargs)

    print("\n[dataset stats]")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Holdout dataset size: {len(holdout_dataset)}")

    # --------------------------------------------------------------------------
    # Model / loss / optimizer
    # --------------------------------------------------------------------------
    CustomResNet18 = load_custom_resnet18()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[device] {device}")

    model = CustomResNet18(num_classes=len(allowed_species)).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # sanity check
    with torch.no_grad():
        dummy = torch.randn(2, 3, 224, 224).to(device)
        out = model(dummy)
        assert out.shape[1] == len(allowed_species), (
            f"Model output classes {out.shape[1]} != expected classes {len(allowed_species)}"
        )

    # --------------------------------------------------------------------------
    # Training
    # --------------------------------------------------------------------------
    best_val_loss = float("inf")
    bad_epochs = 0
    early_stop = not args.no_early_stop

    best_ckpt_path = checkpoints_dir / "best_model_state_resnet18.pkl"

    metrics = []

    last_train_trues = []
    last_train_preds = []
    last_val_trues = []
    last_val_preds = []
    last_holdout_trues = []
    last_holdout_preds = []

    last_train_rows = []
    last_val_rows = []
    last_holdout_rows = []

    ran_epochs = 0

    for epoch in range(args.epochs):
        ran_epochs = epoch + 1
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        model.train()
        train_loss = 0.0
        train_preds = []
        train_trues = []
        train_rows = []

        for x_batch, y_batch, names in tqdm(train_loader, desc="Training"):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            train_loss += loss.item()
            train_preds.extend(preds.detach().cpu().numpy())
            train_trues.extend(y_batch.detach().cpu().numpy())

            if args.save_train_predictions:
                append_prediction_rows(
                    train_rows,
                    names,
                    y_batch,
                    probs,
                    preds,
                    allowed_species,
                )

            del x_batch, y_batch, outputs, probs, preds

        avg_train_loss = train_loss / max(1, len(train_loader))
        train_acc = accuracy_score(train_trues, train_preds) if len(train_trues) > 0 else 0.0

        avg_val_loss, val_acc, val_trues, val_preds, val_rows = run_eval(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            class_names=allowed_species,
            desc="Validating",
        )

        avg_holdout_loss, holdout_acc, holdout_trues, holdout_preds, holdout_rows = run_eval(
            model=model,
            loader=holdout_loader,
            criterion=criterion,
            device=device,
            class_names=allowed_species,
            desc="Holdout",
        )

        row = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_accuracy": train_acc,
            "val_loss": avg_val_loss,
            "val_accuracy": val_acc,
            "holdout_loss": avg_holdout_loss,
            "holdout_accuracy": holdout_acc,
        }

        metrics.append(row)
        pd.DataFrame(metrics).to_csv(output_dir / "metrics_by_epoch.csv", index=False)

        print(
            f"[Epoch {epoch + 1}] "
            f"train_acc={train_acc:.4f} "
            f"val_acc={val_acc:.4f} "
            f"holdout_acc={holdout_acc:.4f}"
        )

        last_train_trues = train_trues
        last_train_preds = train_preds
        last_val_trues = val_trues
        last_val_preds = val_preds
        last_holdout_trues = holdout_trues
        last_holdout_preds = holdout_preds

        last_train_rows = train_rows
        last_val_rows = val_rows
        last_holdout_rows = holdout_rows

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        improved = (best_val_loss - avg_val_loss) > args.early_stop_min_delta

        if improved:
            best_val_loss = avg_val_loss
            bad_epochs = 0

            payload_best = {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "class_names": allowed_species,
                "species_to_idx": species_to_idx,
                "args": vars(args),
                "best_val_loss": best_val_loss,
            }

            torch.save(payload_best, best_ckpt_path)
            print(f"[save] New best checkpoint: {best_ckpt_path}")
        else:
            bad_epochs += 1
            print(f"[early_stop] No improvement. bad_epochs={bad_epochs}/{args.early_stop_patience}")

            if early_stop and bad_epochs >= args.early_stop_patience:
                print(f"[early_stop] Stopping at epoch {epoch + 1}")
                break

    # --------------------------------------------------------------------------
    # Final reports / plots / predictions
    # --------------------------------------------------------------------------
    save_predictions_json(
        predictions_dir / "last_epoch_predictions_valid.json",
        "valid",
        allowed_species,
        last_val_rows,
    )

    save_predictions_json(
        predictions_dir / "last_epoch_predictions_holdout.json",
        "holdout",
        allowed_species,
        last_holdout_rows,
    )

    if args.save_train_predictions:
        save_predictions_json(
            predictions_dir / "last_epoch_predictions_train.json",
            "train",
            allowed_species,
            last_train_rows,
        )

    save_classification_report(
        last_train_trues,
        last_train_preds,
        allowed_species,
        reports_dir / "classification_report_train.json",
        reports_dir / "classification_report_train.csv",
    )

    save_classification_report(
        last_val_trues,
        last_val_preds,
        allowed_species,
        reports_dir / "classification_report_valid.json",
        reports_dir / "classification_report_valid.csv",
    )

    save_classification_report(
        last_holdout_trues,
        last_holdout_preds,
        allowed_species,
        reports_dir / "classification_report_holdout.json",
        reports_dir / "classification_report_holdout.csv",
    )

    save_confusion_matrix(
        last_train_trues,
        last_train_preds,
        allowed_species,
        plots_dir / "confusion_matrix_train.png",
        "Train Confusion Matrix",
    )

    save_confusion_matrix(
        last_val_trues,
        last_val_preds,
        allowed_species,
        plots_dir / "confusion_matrix_valid.png",
        "Validation Confusion Matrix",
    )

    save_confusion_matrix(
        last_holdout_trues,
        last_holdout_preds,
        allowed_species,
        plots_dir / "confusion_matrix_holdout.png",
        "Holdout Confusion Matrix",
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    last_ckpt_path = checkpoints_dir / f"last_model_state_resnet18_{stamp}.pkl"

    payload_last = {
        "epoch": ran_epochs,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "class_names": allowed_species,
        "species_to_idx": species_to_idx,
        "args": vars(args),
    }

    torch.save(payload_last, last_ckpt_path)
    print(f"[save] Last checkpoint: {last_ckpt_path}")

    print("\n[done] Training complete.")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
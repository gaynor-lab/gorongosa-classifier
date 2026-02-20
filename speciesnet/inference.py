#!/usr/bin/env python3
"""
inference.py

Run inference using the latest saved ResNet18 checkpoint.

------------------------------------------------------------
USAGE EXAMPLES
------------------------------------------------------------

Single image:
    python inference.py --image /path/to/image.jpg

Folder (print predictions only):
    python inference.py --folder /path/to/images

Folder + save CSV:
    python inference.py --folder /path/to/images --output predictions.csv

------------------------------------------------------------

Features:
- Automatically finds most recent checkpoint
- Loads class names from checkpoint
- Runs inference on:
    • Single image
    • Entire folder
- Outputs top-k predictions
- Optionally saves CSV
"""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from torchvision import transforms

# -------------------------------------------------------------------------
# Project root
# -------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# -------------------------------------------------------------------------
# Import model
# -------------------------------------------------------------------------
try:
    from ct_classifier.model import CustomResNet18
except Exception:
    CT_MODEL_PATH = PROJECT_ROOT / "ct_classifier" / "model.py"
    spec = __import__("importlib.util").util.spec_from_file_location("ct_model", CT_MODEL_PATH)
    ct_model = __import__("importlib.util").util.module_from_spec(spec)
    spec.loader.exec_module(ct_model)
    CustomResNet18 = ct_model.CustomResNet18

# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------
CHECKPOINT_DIR = Path(
    "/mnt/sharedstorage/sabdelazim/Desktop/kaitlyn_catalyst/resnet_training"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

TOP_K = 5

# ImageNet normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
def get_latest_checkpoint(ckpt_dir: Path) -> Path:
    ckpts = list(ckpt_dir.glob("last_model_state_resnet18_*.pkl"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    return max(ckpts, key=os.path.getctime)


def load_model(ckpt_path: Path):
    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    class_names = ckpt["class_names"]
    num_classes = len(class_names)

    model = CustomResNet18(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"[INFO] Loaded model with {num_classes} classes")
    return model, class_names


def predict_image(model, class_names, image_path: Path):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    top_idx = probs.argsort()[::-1][:TOP_K]

    return [
        {"label": class_names[i], "probability": float(probs[i])}
        for i in top_idx
    ]


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Path to single image")
    parser.add_argument("--folder", type=str, help="Path to image folder")
    parser.add_argument("--output", type=str, help="Optional CSV output path")
    args = parser.parse_args()

    if not args.image and not args.folder:
        raise ValueError("Provide either --image or --folder")

    ckpt = get_latest_checkpoint(CHECKPOINT_DIR)
    model, class_names = load_model(ckpt)

    rows = []

    # ---------------- Single image ----------------
    if args.image:
        img_path = Path(args.image)
        results = predict_image(model, class_names, img_path)

        print(f"\nPredictions for {img_path.name}")
        for r in results:
            print(f"{r['label']:<25} {r['probability']:.4f}")

    # ---------------- Folder ----------------
    if args.folder:
        folder = Path(args.folder)
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

        for p in folder.iterdir():
            if p.suffix.lower() not in exts:
                continue

            results = predict_image(model, class_names, p)
            top1 = results[0]

            rows.append({

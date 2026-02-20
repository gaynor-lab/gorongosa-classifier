#!/usr/bin/env python3
# detector.py

from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from PIL import Image


def _clip(v: int, lo: int, hi: int) -> int:
    return max(lo, min(v, hi))


def _crop_with_padding(img: Image.Image, bbox, pad_frac: float = 0.10) -> Image.Image:
    """
    MegaDetector bbox is normalized [x, y, w, h] in [0,1].
    Convert to pixel coords, apply padding, clip, crop.
    """
    W, H = img.size
    x, y, w, h = bbox

    x1 = int(round(x * W))
    y1 = int(round(y * H))
    x2 = int(round((x + w) * W))
    y2 = int(round((y + h) * H))

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad_x = int(round(bw * pad_frac))
    pad_y = int(round(bh * pad_frac))

    x1 = _clip(x1 - pad_x, 0, W)
    y1 = _clip(y1 - pad_y, 0, H)
    x2 = _clip(x2 + pad_x, 0, W)
    y2 = _clip(y2 + pad_y, 0, H)

    if x2 <= x1 or y2 <= y1:
        return img
    return img.crop((x1, y1, x2, y2))


def filter_df_with_megadetector_and_crop(
    df: pd.DataFrame,
    image_dir: str,
    out_dir: str,
    conf_thresh: float,
    model_name_or_path: str,
    device: str = "cuda",
    animals_only: bool = True,
    pad_frac: float = 0.10,
    save_format: str = "jpg",
    jpeg_quality: int = 95,
) -> pd.DataFrame:
    """
    Run MegaDetector, keep images with >=1 detection above conf_thresh,
    save best bbox + crop image, and return df with crop filenames.

    Output columns added:
      - filename_crop
      - det_conf
      - det_category
      - bbox  (normalized [x,y,w,h])
    """
    try:
        from megadetector.detection.run_detector_batch import load_and_run_detector_batch
    except Exception as e:
        raise RuntimeError(
            "MegaDetector import failed. Expected MegaDetector v10.x.\n"
            "Try: pip install megadetector\n"
            f"Original error: {e}"
        )

    if "filename" not in df.columns:
        raise ValueError("df must contain a 'filename' column")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    image_paths = [os.path.join(image_dir, str(f)) for f in df["filename"].tolist()]

    print(f"[megadetector] running on {len(image_paths)} images, thr={conf_thresh}, animals_only={animals_only}")
    print(f"[megadetector] model: {model_name_or_path}")

    # NOTE: device selection is handled internally by MD/yolov5;
    # you already see it picking cuda:0 in logs.
    results = load_and_run_detector_batch(
        model_file=model_name_or_path,
        image_file_names=image_paths,
        confidence_threshold=float(conf_thresh),
        quiet=True,
        batch_size=1,
    )

    # map absolute path -> original row dict
    row_by_path = {os.path.join(image_dir, str(r["filename"])): r for _, r in df.iterrows()}

    kept = []
    for r in tqdm(results, desc="Cropping + saving"):
        fpath = r.get("file")
        if not fpath:
            continue

        dets = r.get("detections", []) or []
        dets = [d for d in dets if float(d.get("conf", 0.0)) >= conf_thresh]

        if animals_only:
            dets = [d for d in dets if str(d.get("category", "")) == "1"]

        if not dets:
            continue

        best = max(dets, key=lambda d: float(d.get("conf", 0.0)))
        bbox = best.get("bbox")
        if bbox is None:
            continue

        base = row_by_path.get(fpath)
        if base is None:
            # fallback: match by filename
            fname = os.path.basename(fpath)
            m = df[df["filename"].astype(str) == fname]
            if len(m) == 0:
                continue
            base = m.iloc[0].to_dict()
        else:
            base = dict(base)

        # load & crop
        try:
            img = Image.open(fpath).convert("RGB")
        except Exception:
            continue

        crop = _crop_with_padding(img, bbox, pad_frac=pad_frac)

        # save cropped
        orig_name = os.path.basename(fpath)
        stem = os.path.splitext(orig_name)[0]
        crop_name = f"{stem}_crop.{save_format.lower()}"
        crop_fpath = out_path / crop_name

        try:
            if save_format.lower() in ("jpg", "jpeg"):
                crop.save(crop_fpath, quality=jpeg_quality)
            else:
                crop.save(crop_fpath)
        except Exception:
            continue

        base["filename_crop"] = crop_name
        base["det_conf"] = float(best.get("conf", 0.0))
        base["det_category"] = best.get("category", None)
        base["bbox"] = bbox
        kept.append(base)

    out_df = pd.DataFrame(kept).reset_index(drop=True)
    print(f"[megadetector] kept+crops saved: {len(out_df)} / {len(df)}")
    return out_df
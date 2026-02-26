from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Tuple, Optional, List

import pandas as pd
from tqdm import tqdm
from PIL import Image


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------
def _clip(v: int, lo: int, hi: int) -> int:
    return max(lo, min(v, hi))


def _crop_with_padding(img: Image.Image, bbox, pad_frac: float = 0.10) -> Image.Image:
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


# -----------------------------------------------------------------------------
# MegaDetector (multi-detection version)
# -----------------------------------------------------------------------------
def filter_df_with_megadetector_and_crop(
    df: pd.DataFrame,
    image_dir: str,
    out_dir: str,
    conf_thresh: float,
    model_name_or_path: str,
    device: str = "cuda",              # kept for API compatibility
    animals_only: bool = True,
    pad_frac: float = 0.10,
    save_format: str = "jpg",
    jpeg_quality: int = 95,
    max_crops_per_image: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Multi-animal version.

    For each image:
      - Keep ALL detections >= conf_thresh (optionally animals-only)
      - Save one crop per bbox
      - Store bbox/conf/category/crop filenames as JSON lists

    Returns:
      kept_df, dropped_df
    """

    try:
        from megadetector.detection.run_detector_batch import load_and_run_detector_batch
    except Exception as e:
        raise RuntimeError(
            "MegaDetector import failed. Expected MegaDetector v10.x.\n"
            f"Original error: {e}"
        )

    if "filename" not in df.columns:
        raise ValueError("df must contain a 'filename' column")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    image_paths = [os.path.join(image_dir, str(f)) for f in df["filename"].tolist()]

    print(f"[megadetector] running on {len(image_paths)} images")
    print(f"[megadetector] threshold={conf_thresh}, animals_only={animals_only}, device={device}")
    print(f"[megadetector] model: {model_name_or_path}")

    results = load_and_run_detector_batch(
        model_file=model_name_or_path,
        image_file_names=image_paths,
        confidence_threshold=float(conf_thresh),
        quiet=True,
        batch_size=1,
    )

    row_by_path = {
        os.path.join(image_dir, str(r["filename"])): r.to_dict()
        for _, r in df.iterrows()
    }

    kept, dropped = [], []

    for r in tqdm(results, desc="Cropping + saving"):
        fpath = r.get("file")
        if not fpath:
            dropped.append({"filename": None, "drop_reason": "missing_result_file"})
            continue

        base = row_by_path.get(fpath)
        if base is None:
            fname = os.path.basename(fpath)
            m = df[df["filename"].astype(str) == fname]
            if len(m) == 0:
                dropped.append({"filename": fname, "drop_reason": "no_matching_row_in_df"})
                continue
            base = m.iloc[0].to_dict()

        dets_all = r.get("detections", []) or []
        if not dets_all:
            dropped.append({**base, "drop_reason": "no_detections"})
            continue

        # threshold filter
        dets_thr = [d for d in dets_all if float(d.get("conf", 0.0)) >= conf_thresh]
        if not dets_thr:
            dropped.append({
                **base,
                "drop_reason": "all_below_threshold",
                "max_conf": max((float(d.get("conf", 0.0)) for d in dets_all), default=0.0),
            })
            continue

        # animals-only filter
        dets_use = dets_thr
        if animals_only:
            dets_use = [d for d in dets_thr if str(d.get("category", "")) == "1"]
            if not dets_use:
                dropped.append({
                    **base,
                    "drop_reason": "no_animal_detections",
                    "max_conf": max((float(d.get("conf", 0.0)) for d in dets_thr), default=0.0),
                })
                continue

        # sort by confidence
        dets_use = sorted(dets_use, key=lambda d: float(d.get("conf", 0.0)), reverse=True)
        if max_crops_per_image is not None:
            dets_use = dets_use[:int(max_crops_per_image)]

        try:
            img = Image.open(fpath).convert("RGB")
        except Exception:
            dropped.append({**base, "drop_reason": "image_open_failed"})
            continue

        orig_name = os.path.basename(fpath)
        stem = os.path.splitext(orig_name)[0]

        crop_files: List[str] = []
        bboxes: List[list] = []
        det_confs: List[float] = []
        det_cats: List[str] = []

        for j, det in enumerate(dets_use):
            bbox = det.get("bbox")
            if bbox is None:
                continue

            crop = _crop_with_padding(img, bbox, pad_frac=pad_frac)
            crop_name = f"{stem}_crop{j}.{save_format.lower()}"
            crop_path = out_path / crop_name

            try:
                if save_format.lower() in ("jpg", "jpeg"):
                    crop.save(crop_path, quality=jpeg_quality)
                else:
                    crop.save(crop_path)
            except Exception:
                continue

            crop_files.append(crop_name)
            bboxes.append(list(bbox))
            det_confs.append(float(det.get("conf", 0.0)))
            det_cats.append(str(det.get("category", "")))

        if not crop_files:
            dropped.append({**base, "drop_reason": "no_crops_saved"})
            continue

        base["n_animals"] = len(crop_files)
        base["filename_crops"] = json.dumps(crop_files)
        base["bboxes"] = json.dumps(bboxes)
        base["det_confs"] = json.dumps(det_confs)
        base["det_categories"] = json.dumps(det_cats)

        # backward compatibility (use first crop as primary)
        base["filename_crop"] = crop_files[0]
        base["bbox"] = bboxes[0]
        base["det_conf"] = det_confs[0]
        base["det_category"] = det_cats[0]

        kept.append(base)

    kept_df = pd.DataFrame(kept).reset_index(drop=True)
    dropped_df = pd.DataFrame(dropped).reset_index(drop=True)

    print(f"[megadetector] kept: {len(kept_df)} / {len(df)}")
    print(f"[megadetector] dropped: {len(dropped_df)} / {len(df)}")

    return kept_df, dropped_df
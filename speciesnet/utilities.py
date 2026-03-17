import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import os
import json 
import matplotlib.pyplot as plt
# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def compute_allowed_species(train_df: pd.DataFrame, top_n, include_list=None) -> list[str]:
    """
    Pick which species/classes to train on.

    top_n:
      - int N  -> take the N most frequent species in TRAIN
      - "all"  -> take all unique species in TRAIN (sorted)
      - None   -> same as "all"
    include_list:
      - optional list of extra species to force-include (if present in data)
        Note: If include_list contains species not present in TRAIN, they will still be included
              in the returned list, but you may end up with empty classes. Use with care.
    """
    s = train_df["species"].astype(str).str.strip().str.lower()

    if top_n is None or (isinstance(top_n, str) and top_n.strip().lower() == "all"):
        base = sorted(s.unique().tolist())
    else:
        base = s.value_counts().nlargest(int(top_n)).index.tolist()

    inc = [str(x).strip().lower() for x in (include_list or [])]
    allowed = base + [x for x in inc if x not in base]
    return allowed


def plot_confmat(y_true, y_pred, class_names, normalize="true", title="Confusion Matrix"):
    """Plot a confusion matrix heatmap and return the matplotlib Figure."""
    labels = np.arange(len(class_names))
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    side = max(6, int(len(class_names) * 0.45))
    fig, ax = plt.subplots(figsize=(side, side))
    im = ax.imshow(cm, interpolation="nearest", aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=labels, yticks=labels,
           xticklabels=class_names, yticklabels=class_names,
           xlabel="Predicted", ylabel="True", title=title)
    for t in ax.get_xticklabels():
        t.set_rotation(90)

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
    """Save last-epoch per-sample predictions/probabilities to a JSON file."""
    payload = {
        "split": split_name,
        "class_names": class_names,
        "num_classes": len(class_names),
        "num_samples": len(rows),
        "samples": rows
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def is_good(path):
    """Return True if file exists and has non-zero size."""
    try:
        return os.path.getsize(path) > 0
    except Exception:
        return False


def filter_bad_files(df, image_dir):
    """Drop rows whose image files are missing or zero-byte."""
    paths = df["filename"].apply(lambda f: os.path.join(image_dir, f))
    mask = paths.apply(is_good)
    bad = df[~mask]
    good = df[mask].reset_index(drop=True)
    print(f"[precheck] kept {len(good)} | dropped {len(bad)} bad/empty files")
    return good


def expand_to_crop_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand one-row-per-original-image dataframe into one-row-per-crop dataframe.

    Expected input columns:
      - filename_crops: JSON string list or Python list of crop filenames
      - optionally bboxes: JSON string list or Python list of bounding boxes

    Output:
      - one row per crop
      - creates filename_crop
      - if bboxes exists, creates bbox aligned with filename_crop
      - drops filename_crops / bboxes list columns after expansion
    """
    df_expanded = df.copy()

    if "filename_crops" not in df_expanded.columns and "filename_crop" not in df_expanded.columns:
        raise ValueError("DataFrame must contain either 'filename_crops' or 'filename_crop'")

    if "filename_crops" in df_expanded.columns:
        df_expanded["filename_crops"] = df_expanded["filename_crops"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )

    if "bboxes" in df_expanded.columns:
        df_expanded["bboxes"] = df_expanded["bboxes"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )

    cols_to_explode = []
    if "filename_crops" in df_expanded.columns:
        cols_to_explode.append("filename_crops")
    if "bboxes" in df_expanded.columns:
        cols_to_explode.append("bboxes")

    if cols_to_explode:
        df_expanded = df_expanded.explode(cols_to_explode).reset_index(drop=True)

    if "filename_crops" in df_expanded.columns:
        df_expanded["filename_crop"] = df_expanded["filename_crops"]

    if "bboxes" in df_expanded.columns:
        df_expanded["bbox"] = df_expanded["bboxes"]

    df_expanded = df_expanded[df_expanded["filename_crop"].notna()].reset_index(drop=True)

    for col in ["filename_crops", "bboxes"]:
        if col in df_expanded.columns:
            df_expanded = df_expanded.drop(columns=col)

    return df_expanded
#!/usr/bin/env python3
"""
analyze_last_epoch.py

Load last-epoch prediction JSONs from a training run directory (default: resnet_training),
compute threshold-based metrics, and save Altair charts as HTML.

Expected files in RUN_DIR:
  - last_epoch_predictions_train.json
  - last_epoch_predictions_valid.json
  - last_epoch_predictions_holdout.json

Outputs (HTML) go to OUT_DIR (defaults to RUN_DIR).

----------------------------------------------------------------------
USAGE
----------------------------------------------------------------------
Default (run_dir is your resnet_training folder):
  python analyze_last_epoch.py

Specify run dir / output dir:
  python analyze_last_epoch.py --run-dir /path/to/resnet_training --out-dir /path/to/notebooks

Change thresholds:
  python analyze_last_epoch.py --thresholds 0.5 0.6 0.7 0.8 0.9 0.95

Make density overlay for a split at a threshold:
  python analyze_last_epoch.py --density-split valid --density-threshold 0.9
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import altair as alt


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------
DEFAULT_RUN_DIR = Path("/mnt/sharedstorage/sabdelazim/Desktop/kaitlyn_catalyst/resnet_training")
DEFAULT_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
SPLITS = ["train", "valid", "holdout"]


# ---------------------------------------------------------------------
# JSON loading
# ---------------------------------------------------------------------
def load_pred_file(p: Path, split: str) -> Tuple[pd.DataFrame, Optional[List[str]]]:
    """Load one predictions JSON and return (df, class_names or None)."""
    if not p.exists():
        return pd.DataFrame(), None

    d = json.loads(p.read_text())
    cls = d.get("class_names", []) or []
    df = pd.DataFrame(d.get("samples", []))
    if df.empty:
        return df, cls

    df["split"] = split

    # Ensure helpful columns exist / consistent
    if "true_label" not in df.columns and "true_idx" in df.columns and cls:
        df["true_label"] = df["true_idx"].apply(
            lambda i: cls[int(i)] if isinstance(i, (int, np.integer)) and 0 <= int(i) < len(cls) else str(i)
        )
    if "pred_label" not in df.columns and "pred_idx" in df.columns and cls:
        df["pred_label"] = df["pred_idx"].apply(
            lambda i: cls[int(i)] if isinstance(i, (int, np.integer)) and 0 <= int(i) < len(cls) else str(i)
        )

    df["correct"] = df["true_idx"] == df["pred_idx"]

    def _safe_pick(row, key_idx):
        try:
            probs = row["probs"]
            idx = int(row[key_idx])
            return float(probs[idx]) if isinstance(probs, (list, tuple)) and 0 <= idx < len(probs) else np.nan
        except Exception:
            return np.nan

    df["pred_conf"] = df.apply(lambda r: _safe_pick(r, "pred_idx"), axis=1)
    df["true_conf"] = df.apply(lambda r: _safe_pick(r, "true_idx"), axis=1)
    df["max_prob"] = df["probs"].apply(
        lambda v: float(np.max(v)) if isinstance(v, (list, tuple, np.ndarray)) and len(v) else np.nan
    )

    # Try to infer site if not present (useful for density overlay dropdown)
    if "site" not in df.columns and "filename" in df.columns:
        df["site"] = df["filename"].astype(str).str.extract(r"_([a-z]\d{2})_", expand=False)

    return df, cls


def load_all_preds(run_dir: Path) -> Tuple[pd.DataFrame, Optional[List[str]]]:
    paths = {
        "train": run_dir / "last_epoch_predictions_train.json",
        "valid": run_dir / "last_epoch_predictions_valid.json",
        "holdout": run_dir / "last_epoch_predictions_holdout.json",
    }

    dfs = []
    classes = None
    for split, p in paths.items():
        df, cls = load_pred_file(p, split)
        if not df.empty:
            dfs.append(df)
            if classes is None and cls:
                classes = cls

    all_pred = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    return all_pred, classes


# ---------------------------------------------------------------------
# Review-table chart (solid=review frac, dotted=accuracy)
# ---------------------------------------------------------------------
def compute_review_table(all_pred: pd.DataFrame, split: str, thresholds: List[float]) -> pd.DataFrame:
    cols = [
        "true_label", "threshold", "split",
        "n_total", "n_correct_total", "n_review", "review_frac",
        "n_kept_correct", "n_kept_wrong", "acc_at_t"
    ]

    if all_pred is None or all_pred.empty:
        return pd.DataFrame(columns=cols)

    df = all_pred.copy()
    df["split"] = df["split"].astype(str).str.lower()
    df = df[df["split"] == split.lower()].copy()
    if df.empty:
        return pd.DataFrame(columns=cols)

    df["true_label"] = df["true_label"].astype(str)
    df["pred_label"] = df["pred_label"].astype(str)
    df["correct"] = df["correct"].astype(bool)

    if "max_prob" not in df.columns:
        df["max_prob"] = df["probs"].apply(
            lambda v: float(np.max(v)) if isinstance(v, (list, tuple, np.ndarray)) and len(v) else np.nan
        )
    df = df[df["max_prob"].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=cols)

    denom_total = df.groupby("true_label").size().rename("n_total")
    correct_df = df[df["correct"]].copy()
    denom_correct = correct_df.groupby("true_label").size().rename("n_correct_total")

    rows = []
    for t in thresholds:
        t = float(t)

        n_review = (
            correct_df[correct_df["max_prob"] < t]
            .groupby("true_label")
            .size()
            .rename("n_review")
        )

        kept = df[df["max_prob"] >= t]
        n_kept_correct = (
            kept[kept["correct"]]
            .groupby("true_label")
            .size()
            .rename("n_kept_correct")
        )
        n_kept_wrong = (
            kept[~kept["correct"]]
            .groupby("true_label")
            .size()
            .rename("n_kept_wrong")
        )

        out = pd.concat([denom_total, denom_correct, n_review, n_kept_correct, n_kept_wrong], axis=1).fillna(0)
        for c in ["n_total", "n_correct_total", "n_review", "n_kept_correct", "n_kept_wrong"]:
            out[c] = out[c].astype(int)

        out["review_frac"] = np.where(
            out["n_correct_total"] > 0,
            out["n_review"] / out["n_correct_total"],
            np.nan
        )

        denom_acc = out["n_kept_correct"] + out["n_kept_wrong"]
        out["acc_at_t"] = np.where(
            denom_acc > 0,
            out["n_kept_correct"] / denom_acc,
            np.nan
        )

        out["threshold"] = t
        out["split"] = split
        rows.append(out.reset_index())

    return pd.concat(rows, ignore_index=True).sort_values(["true_label", "threshold"])


def build_faceted_chart_same_y(
    tbl: pd.DataFrame,
    split: str,
    columns: int = 4,
    width: int = 180,
    height: int = 120
):
    if tbl is None or tbl.empty:
        return alt.Chart(pd.DataFrame({"msg": [f"No data for split={split}"]})).mark_text().encode(text="msg:N")

    class_order = (
        tbl.drop_duplicates("true_label")
           .sort_values("n_total", ascending=False)["true_label"]
           .tolist()
    )

    base = alt.Chart(tbl).properties(width=width, height=height)

    review_line = base.mark_line(point=True).encode(
        x=alt.X("threshold:Q", title="Threshold"),
        y=alt.Y("review_frac:Q", title="Rate", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("true_label:N", legend=None),
    )

    acc_line = base.mark_line(point=True, strokeDash=[2, 2]).encode(
        x="threshold:Q",
        y=alt.Y("acc_at_t:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("true_label:N", legend=None),
    )

    n_text = (
        base.transform_aggregate(n_total="max(n_total)", groupby=["true_label"])
        .transform_calculate(label="'N=' + format(datum.n_total, ',')")
        .mark_text(align="center", baseline="middle", fontSize=12, opacity=0.85)
        .encode(text="label:N", x=alt.value(width * 0.55), y=alt.value(height * 0.55))
    )

    faceted = (
        alt.layer(review_line, acc_line, n_text)
        .facet(facet=alt.Facet("true_label:N", sort=class_order, title=None), columns=columns)
        .properties(title=f"{split} — Solid: review fraction | Dotted: accuracy")
    )

    legend_data = pd.DataFrame({"metric": ["Review fraction", "Accuracy"], "style": ["solid", "dotted"], "y": [2, 1]})
    legend_base = alt.Chart(legend_data).properties(width=220, height=120)

    legend_lines = legend_base.mark_line(strokeWidth=3).encode(
        x=alt.value(0), x2=alt.value(40), y=alt.Y("y:Q", axis=None),
        strokeDash=alt.StrokeDash(
            "style:N",
            scale=alt.Scale(domain=["solid", "dotted"], range=[[1, 0], [4, 2]]),
            legend=None
        ),
    )

    legend_text = legend_base.mark_text(align="left", baseline="middle", dx=50, fontSize=13).encode(
        y=alt.Y("y:Q", axis=None), text="metric:N"
    )

    legend_title = alt.Chart(pd.DataFrame({"t": ["Legend"]})).mark_text(
        align="left", fontSize=14, fontWeight="bold"
    ).encode(text="t:N").properties(height=30)

    legend_panel = alt.vconcat(legend_title, alt.layer(legend_lines, legend_text), spacing=10)

    return alt.hconcat(faceted, legend_panel, spacing=20)


# ---------------------------------------------------------------------
# Density overlay chart (Correct vs Incorrect in the SAME panel)
# ---------------------------------------------------------------------
def _filter_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    if split.lower() == "all":
        return df.copy()
    return df[df["split"].str.lower() == split.lower()].copy()


def build_maxprob_density_overlay(
    all_pred: pd.DataFrame,
    split: str = "valid",
    threshold_val: float = 0.90,
    width: int = 260,
    height: int = 120,
    bandwidth: float = 0.03,
    steps: int = 200,
):
    if all_pred is None or all_pred.empty:
        raise ValueError("all_pred is empty.")

    df = _filter_split(all_pred, split)
    if df.empty:
        raise ValueError(f"No rows for split='{split}'.")

    df = df.copy()
    df["true_label"] = df["true_label"].astype(str)
    df["pred_label"] = df["pred_label"].astype(str)
    df["correct"] = df["correct"].astype(bool)
    df["max_prob"] = df["max_prob"].astype(float)
    df["Correctness"] = np.where(df["correct"], "Correct", "Incorrect")

    # ensure site exists
    if "site" not in df.columns:
        if "filename" in df.columns:
            df["site"] = df["filename"].astype(str).str.extract(r"_([a-z]\d{2})_", expand=False)
        else:
            df["site"] = np.nan

    class_order = df["true_label"].value_counts().index.tolist()

    threshold = alt.param(
        name="threshold", value=float(threshold_val),
        bind=alt.binding_range(min=0.0, max=1.0, step=0.01, name="Threshold")
    )

    sites = sorted([s for s in df["site"].dropna().unique().tolist()])
    site_param = alt.param(
        name="site", value="All",
        bind=alt.binding_select(options=["All"] + sites, name="Site ")
    )

    site_filter = (site_param == "All") | (alt.datum.site == site_param)

    density = (
        alt.Chart(df)
        .transform_filter(site_filter)
        .transform_density(
            "max_prob",
            as_=["max_prob", "density"],
            groupby=["true_label", "Correctness"],
            extent=[0, 1],
            bandwidth=bandwidth,
            steps=steps,
        )
        .mark_area(opacity=0.35)
        .encode(
            x=alt.X("max_prob:Q", scale=alt.Scale(domain=[0, 1]), title="max_prob"),
            y=alt.Y("density:Q", title="density"),
            color=alt.Color("Correctness:N", legend=alt.Legend(title=None)),
        )
        .properties(width=width, height=height)
    )

    rule = (
        alt.Chart(df)
        .transform_filter(site_filter)
        .transform_calculate(xpos="threshold")
        .mark_rule(color="red", strokeDash=[5, 5])
        .encode(x=alt.X("xpos:Q", scale=alt.Scale(domain=[0, 1])))
        .properties(width=width, height=height)
    )

    chart = (
        alt.layer(density, rule)
        .facet(row=alt.Row("true_label:N", sort=class_order, title=None))
        .add_params(threshold, site_param)
        .properties(title=f"max_prob KDE overlay — split={split}")
        .resolve_scale(y="independent")
    )
    return chart


# ---------------------------------------------------------------------
# Per-class threshold metrics (threshold-as-negative)
# ---------------------------------------------------------------------
def compute_per_class_threshold_metrics_threshold_as_negative(
    all_pred: pd.DataFrame,
    split: str,
    thresholds: List[float],
) -> pd.DataFrame:
    df = all_pred.copy()
    df["split"] = df["split"].astype(str).str.lower()
    df = df[df["split"] == split.lower()].copy()
    if df.empty:
        return pd.DataFrame(columns=[
            "split", "threshold", "label", "metric", "value",
            "tp", "fp", "fn", "tn", "n_total", "n_above", "n_below"
        ])

    df["true_label"] = df["true_label"].astype(str)
    df["pred_label"] = df["pred_label"].astype(str)

    if "max_prob" not in df.columns:
        df["max_prob"] = df["probs"].apply(
            lambda v: float(np.max(v)) if isinstance(v, (list, tuple, np.ndarray)) and len(v) else np.nan
        )
    df["max_prob"] = df["max_prob"].astype(float)
    df = df[df["max_prob"].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=[
            "split", "threshold", "label", "metric", "value",
            "tp", "fp", "fn", "tn", "n_total", "n_above", "n_below"
        ])

    labels = sorted(set(df["true_label"].unique()).union(set(df["pred_label"].unique())))

    true = df["true_label"].values
    pred = df["pred_label"].values
    probs = df["max_prob"].values
    n_total = len(df)

    rows = []
    for t in thresholds:
        t = float(t)
        above = probs >= t
        n_above = int(np.sum(above))
        n_below = int(n_total - n_above)

        for lab in labels:
            true_is = (true == lab)
            pred_is = (pred == lab) & above  # predicted positive only if confident enough

            tp = int(np.sum(pred_is & true_is))
            fp = int(np.sum(pred_is & (~true_is)))
            fn = int(np.sum((~pred_is) & true_is))
            tn = int(np.sum((~pred_is) & (~true_is)))

            precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
            recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            f1 = (2 * precision * recall / (precision + recall)) if (
                (precision == precision) and (recall == recall) and (precision + recall) > 0
            ) else np.nan

            rows.extend([
                {"split": split, "threshold": t, "label": lab, "metric": "precision", "value": precision,
                 "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n_total": n_total, "n_above": n_above, "n_below": n_below},
                {"split": split, "threshold": t, "label": lab, "metric": "recall", "value": recall,
                 "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n_total": n_total, "n_above": n_above, "n_below": n_below},
                {"split": split, "threshold": t, "label": lab, "metric": "f1", "value": f1,
                 "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n_total": n_total, "n_above": n_above, "n_below": n_below},
            ])

    return pd.DataFrame(rows)


def plot_per_class_metrics_no_accuracy(
    per_class_long: pd.DataFrame,
    split: str,
    columns: int = 4,
    width: int = 180,
    height: int = 120,
    y_min: float = 0.0,
):
    if per_class_long is None or per_class_long.empty:
        return alt.Chart(pd.DataFrame({"msg": [f"No data for split={split}"]})).mark_text().encode(text="msg:N")

    df = per_class_long.copy()
    df = df[df["metric"].isin(["precision", "recall", "f1"])].copy()

    order = (
        df[df["metric"] == "recall"]
        .groupby("label")["tp"].max()
        .sort_values(ascending=False)
        .index.tolist()
    )

    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("threshold:Q", title="Threshold"),
            y=alt.Y("value:Q", title=None, scale=alt.Scale(domain=[y_min, 1]), axis=alt.Axis(format="%", tickCount=4)),
            color=alt.Color("metric:N", title=None),
            tooltip=[
                "label:N", "metric:N", "threshold:Q",
                alt.Tooltip("value:Q", format=".2%"),
                alt.Tooltip("tp:Q"), alt.Tooltip("fp:Q"), alt.Tooltip("fn:Q"), alt.Tooltip("tn:Q"),
                alt.Tooltip("n_total:Q", title="total"),
                alt.Tooltip("n_above:Q", title=">=thr"),
                alt.Tooltip("n_below:Q", title="<thr"),
            ],
        )
        .properties(width=width, height=height)
        .facet(facet=alt.Facet("label:N", sort=order, title=None), columns=columns)
        .properties(title=f"{split}: per-class precision/recall/f1 vs threshold (below-thr treated as negative)")
    )
    return chart


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default=str(DEFAULT_RUN_DIR), help="resnet_training folder")
    parser.add_argument("--out-dir", type=str, default=None, help="where to save HTML outputs (default: run-dir)")
    parser.add_argument("--thresholds", type=float, nargs="*", default=DEFAULT_THRESHOLDS, help="threshold list")

    # density overlay options (saved for whichever split you choose)
    parser.add_argument("--density-split", type=str, default="valid", choices=SPLITS,
                        help="split to use for density overlay")
    parser.add_argument("--density-threshold", type=float, default=0.90, help="initial threshold for density")

    # optional sizing controls
    parser.add_argument("--facet-columns", type=int, default=4, help="facet columns for per-class plots")
    parser.add_argument("--facet-width", type=int, default=180, help="facet width")
    parser.add_argument("--facet-height", type=int, default=120, help="facet height")
    parser.add_argument("--density-width", type=int, default=260, help="density facet width")
    parser.add_argument("--density-height", type=int, default=120, help="density facet height")
    parser.add_argument("--density-bandwidth", type=float, default=0.03, help="KDE bandwidth")
    parser.add_argument("--density-steps", type=int, default=200, help="KDE steps")

    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    thresholds = [float(x) for x in (args.thresholds or DEFAULT_THRESHOLDS)]

    # Altair safety for bigger data
    alt.data_transformers.disable_max_rows()

    print(f"[info] run_dir: {run_dir}")
    print(f"[info] out_dir: {out_dir}")
    print(f"[info] thresholds: {thresholds}")

    all_pred, classes = load_all_preds(run_dir)
    if all_pred.empty:
        raise FileNotFoundError(
            f"No prediction JSONs loaded from {run_dir}. "
            "Expected last_epoch_predictions_{train,valid,holdout}.json"
        )

    # quick sanity counts
    print(
        all_pred.groupby(["split", "correct"])
        .size()
        .rename("n")
        .reset_index()
        .to_string(index=False)
    )

    # ---- For each split: review/accuracy chart + PRF chart ----
    for sp in SPLITS:
        tbl = compute_review_table(all_pred, sp, thresholds)
        review_chart = build_faceted_chart_same_y(
            tbl, sp,
            columns=int(args.facet_columns),
            width=int(args.facet_width),
            height=int(args.facet_height),
        )
        review_path = out_dir / f"review_and_accuracy_same_y_{sp}.html"
        review_chart.save(str(review_path))
        print("[save]", review_path)

        per_class = compute_per_class_threshold_metrics_threshold_as_negative(all_pred, sp, thresholds)
        prf_chart = plot_per_class_metrics_no_accuracy(
            per_class, sp,
            columns=int(args.facet_columns),
            width=int(args.facet_width),
            height=int(args.facet_height),
            y_min=0.0,
        )
        prf_path = out_dir / f"per_class_prf_vs_threshold_{sp}.html"
        prf_chart.save(str(prf_path))
        print("[save]", prf_path)

    # ---- Density overlay (one split; controllable via args) ----
    density_chart = build_maxprob_density_overlay(
        all_pred,
        split=str(args.density_split),
        threshold_val=float(args.density_threshold),
        bandwidth=float(args.density_bandwidth),
        steps=int(args.density_steps),
        width=int(args.density_width),
        height=int(args.density_height),
    )
    density_path = out_dir / f"maxprob_density_overlay_{args.density_split}.html"
    density_chart.save(str(density_path))
    print("[save]", density_path)

    # optional: write all_pred as a parquet/csv for quick reuse
    out_csv = out_dir / "all_pred_last_epoch.csv"
    all_pred.to_csv(out_csv, index=False)
    print("[save]", out_csv)


if __name__ == "__main__":
    main()
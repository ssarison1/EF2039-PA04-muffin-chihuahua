"""Collect runs/*/metrics.json into tables (mean ± std over seeds) and paper figures.

Outputs in results/:
  all_runs.csv            one row per run
  rq1_label_noise.csv     table for RQ1 (full data, varying noise)
  rq2_data_size.csv       table for RQ2 (clean labels, varying images per class)
  fig_rq1_noise.png/.pdf  test accuracy vs label-noise rate
  fig_rq2_size.png/.pdf   per-class test accuracy vs images per class
  confusion_clean.png     confusion matrix of the clean full-data model (seed 0)
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import CLASSES

# Categorical colours (fixed order) and text colours.
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#1f1f1e", "#6b6a65", "#e4e3de"


def load(runs_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(runs_dir.glob("*/metrics.json")):
        m = json.loads(f.read_text())
        row = {k: m[k] for k in ["run", "n_per_class", "noise", "seed", "n_train", "n_flipped", "best_val_step"]}
        for tag in ["test_final", "test_best_val"]:
            for k in ["accuracy", "macro_f1"] + [f"acc_{c}" for c in CLASSES]:
                row[f"{tag}.{k}"] = m[tag][k]
        rows.append(row)
    if not rows:
        raise SystemExit(f"no metrics.json found under {runs_dir}")
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, key: str) -> pd.DataFrame:
    cols = [c for c in df.columns if c.startswith("test_")]
    g = df.groupby(key)[cols]
    mean, std, n = g.mean(), g.std(ddof=1).fillna(0.0), g.size()
    out = pd.DataFrame(index=mean.index)
    out["n_seeds"] = n
    for c in cols:
        out[c + ".mean"] = (100 * mean[c]).round(2)
        out[c + ".std"] = (100 * std[c]).round(2)
    return out.reset_index()


def style(ax, xlabel, ylabel):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, color=INK, fontsize=9)
    ax.set_ylabel(ylabel, color=INK, fontsize=9)


def line(ax, x, t, key, color, marker, label):
    ax.errorbar(x, t[key + ".mean"], yerr=t[key + ".std"], color=color, marker=marker,
                markersize=5, linewidth=2, capsize=3, label=label)


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=300)
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load(Path(args.runs_dir))
    df.to_csv(out / "all_runs.csv", index=False)

    # RQ1: full training data, varying noise
    rq1 = summarize(df[df["n_per_class"] == 0], "noise")
    rq1.to_csv(out / "rq1_label_noise.csv", index=False)
    # RQ2: clean labels, varying size (0 = all images)
    rq2 = summarize(df[df["noise"] == 0], "n_per_class")
    rq2.to_csv(out / "rq2_data_size.csv", index=False)

    print("RQ1 label noise (test, %, mean ± std):")
    print(rq1[["noise", "n_seeds", "test_final.accuracy.mean", "test_final.accuracy.std",
               "test_best_val.accuracy.mean", "test_best_val.accuracy.std"]].to_string(index=False))
    print("\nRQ2 data size (test, %, mean ± std):")
    print(rq2[["n_per_class", "n_seeds", "test_final.accuracy.mean", "test_final.accuracy.std",
               "test_final.acc_chihuahua.mean", "test_final.acc_muffin.mean"]].to_string(index=False))

    # Figure RQ1
    if len(rq1) > 1:
        fig, ax = plt.subplots(figsize=(3.3, 2.4))
        x = 100 * rq1["noise"]
        line(ax, x, rq1, "test_final.accuracy", C1, "o", "Final model")
        line(ax, x, rq1, "test_best_val.accuracy", C2, "s", "Best-validation checkpoint")
        ax.set_xticks(x)
        style(ax, "Flipped training labels (%)", "Test accuracy (%)")
        ax.legend(fontsize=7, frameon=False)
        save(fig, out, "fig_rq1_noise")

    # Figure RQ2 (x axis: images per class; "all" placed at the real per-class mean count)
    if len(rq2) > 1:
        full = df[(df["n_per_class"] == 0) & (df["noise"] == 0)]["n_train"]
        n_all = full.iloc[0] / len(CLASSES) if len(full) else None
        x = rq2["n_per_class"].astype(float).copy()
        if n_all is not None:
            x[x == 0] = n_all
        order = np.argsort(x.values)
        r = rq2.iloc[order]
        xs = x.values[order]
        fig, ax = plt.subplots(figsize=(3.3, 2.4))
        line(ax, xs, r, "test_final.accuracy", C1, "o", "Total")
        line(ax, xs, r, "test_final.acc_chihuahua", C2, "s", "Chihuahua")
        line(ax, xs, r, "test_final.acc_muffin", C3, "^", "Muffin")
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([str(int(v)) if n != 0 else f"all\n(~{int(v)})"
                            for v, n in zip(xs, r["n_per_class"])])
        ax.minorticks_off()
        style(ax, "Training images per class (log scale)", "Test accuracy (%)")
        ax.legend(fontsize=7, frameon=False)
        save(fig, out, "fig_rq2_size")

    # Confusion matrix of the clean full-data model, seed 0
    f = Path(args.runs_dir) / "nall_noise00_seed0" / "metrics.json"
    if f.exists():
        cm = np.array(json.loads(f.read_text())["test_final"]["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(2.4, 2.2))
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=10,
                        color="white" if cm[i, j] > cm.max() / 2 else INK)
        ax.set_xticks([0, 1], [c.capitalize() for c in CLASSES], fontsize=8)
        ax.set_yticks([0, 1], [c.capitalize() for c in CLASSES], fontsize=8)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("True", fontsize=9)
        save(fig, out, "confusion_clean")


if __name__ == "__main__":
    main()

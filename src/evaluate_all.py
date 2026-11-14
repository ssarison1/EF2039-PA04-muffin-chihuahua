"""Re-evaluate every trained checkpoint on the fixed test set and compare with the reported results.

Run from the repository root after downloading the checkpoints (scripts/download_weights.sh)
or after training them yourself (scripts/run_experiments.sh).

For each runs/<run>/model_final.pt it
  1. computes test accuracy, per-class accuracy, and macro F1, and
  2. compares the accuracy with the value reported in results/runs/<run>/metrics.json.

Output: a table on screen and results/reevaluation.csv

Example
  python src/evaluate_all.py
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import CLASSES, ImageCSV, build_model, compute_metrics, get_transforms, predict_loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs", help="folder with <run>/model_final.pt")
    ap.add_argument("--reported_dir", default="results/runs", help="folder with <run>/metrics.json")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--out_csv", default="results/reevaluation.csv")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    ckpts = sorted(Path(args.runs_dir).glob("*/model_final.pt"))
    if not ckpts:
        raise SystemExit(f"no checkpoints found under {args.runs_dir}/ "
                         "(run scripts/download_weights.sh or scripts/run_experiments.sh first)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_df = pd.read_csv(Path(args.data_root) / "splits" / "test.csv")
    loader = DataLoader(ImageCSV(test_df, args.data_root, get_transforms(False)),
                        batch_size=128, shuffle=False, num_workers=args.workers)
    model = build_model(pretrained=False).to(device)

    rows = []
    for ckpt in ckpts:
        run = ckpt.parent.name
        model.load_state_dict(torch.load(ckpt, map_location=device))
        y, p = predict_loader(model, loader, device)
        m = compute_metrics(y, p)
        reported_file = Path(args.reported_dir) / run / "metrics.json"
        reported = None
        if reported_file.exists():
            reported = json.loads(reported_file.read_text())["test_final"]["accuracy"]
        row = {"run": run, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}
        for c in CLASSES:
            row[f"acc_{c}"] = m[f"acc_{c}"]
        row["reported_accuracy"] = reported
        row["abs_diff"] = None if reported is None else abs(m["accuracy"] - reported)
        rows.append(row)
        diff = "n/a" if reported is None else f"{100 * row['abs_diff']:.2f}"
        print(f"{run:<22} test acc {100 * m['accuracy']:6.2f}%  reported "
              f"{'n/a' if reported is None else f'{100 * reported:6.2f}%'}  |diff| {diff} pts", flush=True)

    df = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    if df["abs_diff"].notna().any():
        print(f"\n{len(df)} checkpoints evaluated; largest |difference| from the reported accuracy: "
              f"{100 * df['abs_diff'].max():.2f} percentage points")
    print(f"saved {args.out_csv}")


if __name__ == "__main__":
    main()

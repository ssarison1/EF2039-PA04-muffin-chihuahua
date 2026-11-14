"""Evaluate a saved checkpoint on the fixed test set (or any split CSV).

Example
  python src/evaluate.py --ckpt runs/nall_noise00_seed0/model_final.pt
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
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(Path(args.data_root) / "splits" / f"{args.split}.csv")
    loader = DataLoader(ImageCSV(df, args.data_root, get_transforms(False)),
                        batch_size=128, shuffle=False, num_workers=args.workers)

    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.to(device)

    y, p = predict_loader(model, loader, device)
    m = compute_metrics(y, p)
    print(f"{args.split} set: {len(df)} images")
    print(f"  accuracy : {m['accuracy']:.4f}")
    for c in CLASSES:
        print(f"  acc {c:<10}: {m['acc_' + c]:.4f}")
    print(f"  macro F1 : {m['macro_f1']:.4f}")
    print(f"  confusion matrix (rows=true {CLASSES}, cols=pred): {m['confusion_matrix']}")
    print(json.dumps(m))


if __name__ == "__main__":
    main()

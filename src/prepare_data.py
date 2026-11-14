"""Build fixed, reproducible data splits from the raw Kaggle folders.

Input : data/raw/{train,test}/{chihuahua,muffin}/*.jpg
Output: data/splits/{train_pool,val,test}.csv and data/splits/summary.json

Steps
1. Verify every image can be decoded (corrupt files are dropped and listed).
2. Remove exact duplicates (MD5): duplicates inside the training folder are kept once,
   and training images that also appear in the test folder are removed (no leakage).
3. Use the official Kaggle test folder as the fixed test set.
4. Split the official train folder into train_pool (90%) and a clean validation set
   (10%), stratified by class, with a fixed seed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from common import CLASSES


def scan(raw_dir: Path, split: str):
    rows, bad = [], []
    for label, name in enumerate(CLASSES):
        for p in sorted((raw_dir / split / name).iterdir()):
            if not p.is_file():
                continue
            try:
                with Image.open(p) as im:
                    im.convert("RGB").load()
            except Exception:
                bad.append(str(p))
                continue
            md5 = hashlib.md5(p.read_bytes()).hexdigest()
            rows.append({"path": str(p.relative_to(raw_dir.parent)), "label": label, "md5": md5})
    return pd.DataFrame(rows), bad


def counts(df):
    return {name: int((df["label"] == i).sum()) for i, name in enumerate(CLASSES)} | {"total": len(df)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data", help="folder that contains raw/")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.data_root)
    raw = root / "raw"
    out = root / "splits"
    out.mkdir(parents=True, exist_ok=True)

    train, bad_tr = scan(raw, "train")
    test, bad_te = scan(raw, "test")
    raw_counts = {"train": counts(train), "test": counts(test)}

    # Same file content stored under both class folders = conflicting label.
    both = pd.concat([train, test])
    conflict = int((both.groupby("md5")["label"].nunique() > 1).sum())

    # Duplicates
    n_test_before = len(test)
    test = test.drop_duplicates("md5")
    n_train_before = len(train)
    train = train.drop_duplicates("md5")
    n_dup_train = n_train_before - len(train)
    overlap = train["md5"].isin(set(test["md5"]))
    n_overlap = int(overlap.sum())
    train = train[~overlap]

    train_pool, val = train_test_split(
        train, test_size=args.val_frac, stratify=train["label"], random_state=args.seed
    )

    cols = ["path", "label"]
    train_pool.sort_values("path")[cols].to_csv(out / "train_pool.csv", index=False)
    val.sort_values("path")[cols].to_csv(out / "val.csv", index=False)
    test.sort_values("path")[cols].to_csv(out / "test.csv", index=False)

    summary = {
        "classes": {i: c for i, c in enumerate(CLASSES)},
        "counts_after_corrupt_check": raw_counts,
        "corrupt_files_removed": bad_tr + bad_te,
        "duplicates_removed_within_test": n_test_before - len(test),
        "duplicates_removed_within_train": n_dup_train,
        "train_images_also_in_test_removed": n_overlap,
        "images_with_conflicting_labels": conflict,
        "split_seed": args.seed,
        "val_fraction": args.val_frac,
        "final_counts": {"train_pool": counts(train_pool), "val": counts(val), "test": counts(test)},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

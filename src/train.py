"""Fine-tune an ImageNet-pretrained ResNet-18 on Muffin vs Chihuahua.

One call = one run. A run is defined by
  --n_per_class : training images per class drawn from train_pool (0 = use all)
  --noise       : fraction of training labels flipped to the other class (per class)
  --seed        : controls subset sampling, label flipping, head init and data order

Every run uses the same budget of optimizer steps, so runs with different data sizes
receive the same amount of computation. The validation and test sets are always clean.

Outputs in <out_dir>/<run_name>/:
  train_list.csv  images actually used, with the original and the (possibly noisy) label
  log.csv         validation accuracy during training
  metrics.json    test metrics of the final model and of the best-validation checkpoint
  model_final.pt  weights of the final model
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import CLASSES, ImageCSV, build_model, compute_metrics, get_transforms, predict_loader, set_seed


def make_train_list(pool: pd.DataFrame, n_per_class: int, noise: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for label in range(len(CLASSES)):
        cls = pool[pool["label"] == label]
        if n_per_class > 0:
            if n_per_class > len(cls):
                raise ValueError(f"class {CLASSES[label]} has only {len(cls)} images")
            cls = cls.iloc[rng.choice(len(cls), n_per_class, replace=False)]
        cls = cls.copy()
        cls["clean_label"] = cls["label"]
        n_flip = int(round(noise * len(cls)))
        flip_idx = rng.choice(len(cls), n_flip, replace=False)
        noisy = cls["label"].to_numpy().copy()
        noisy[flip_idx] = 1 - noisy[flip_idx]  # binary task: flip to the other class
        cls["label"] = noisy
        parts.append(cls)
    df = pd.concat(parts).reset_index(drop=True)
    df["flipped"] = df["label"] != df["clean_label"]
    return df[["path", "clean_label", "label", "flipped"]]


def infinite(loader):
    while True:
        for batch in loader:
            yield batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--out_dir", default="runs")
    ap.add_argument("--n_per_class", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--eval_every", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no_pretrained", action="store_true", help="for quick code tests only")
    ap.add_argument("--overwrite", action="store_true", help="rerun even if metrics.json exists")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size = "all" if args.n_per_class == 0 else str(args.n_per_class)
    run_name = f"n{size}_noise{int(round(args.noise * 100)):02d}_seed{args.seed}"
    out = Path(args.out_dir) / run_name
    if (out / "metrics.json").exists() and not args.overwrite:
        print(f"[{run_name}] already finished, skipping (use --overwrite to rerun)")
        return
    out.mkdir(parents=True, exist_ok=True)

    splits = Path(args.data_root) / "splits"
    pool = pd.read_csv(splits / "train_pool.csv")
    val_df = pd.read_csv(splits / "val.csv")
    test_df = pd.read_csv(splits / "test.csv")

    train_df = make_train_list(pool, args.n_per_class, args.noise, args.seed)
    train_df.to_csv(out / "train_list.csv", index=False)

    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        ImageCSV(train_df, args.data_root, get_transforms(True)),
        batch_size=min(args.batch_size, len(train_df)), shuffle=True, drop_last=True,
        num_workers=args.workers, pin_memory=True, generator=g, persistent_workers=args.workers > 0,
    )
    eval_kw = dict(batch_size=128, shuffle=False, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(ImageCSV(val_df, args.data_root, get_transforms(False)), **eval_kw)
    test_loader = DataLoader(ImageCSV(test_df, args.data_root, get_transforms(False)), **eval_kw)

    model = build_model(pretrained=not args.no_pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    loss_fn = nn.CrossEntropyLoss()

    log, best_acc, best_state, best_step = [], -1.0, None, 0
    it = infinite(train_loader)
    t0 = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        x, y = next(it)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            loss = loss_fn(model(x), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()

        if step % args.eval_every == 0 or step == args.steps:
            yv, pv = predict_loader(model, val_loader, device)
            val_acc = float((yv == pv).mean())
            log.append({"step": step, "train_loss": loss.item(), "val_acc": val_acc})
            print(f"[{run_name}] step {step}/{args.steps} loss {loss.item():.4f} val_acc {val_acc:.4f}", flush=True)
            if val_acc > best_acc:
                best_acc, best_step = val_acc, step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            model.train()

    pd.DataFrame(log).to_csv(out / "log.csv", index=False)
    torch.save(model.state_dict(), out / "model_final.pt")

    yt, pt = predict_loader(model, test_loader, device)
    final = compute_metrics(yt, pt)
    model.load_state_dict(best_state)
    yb, pb = predict_loader(model, test_loader, device)
    best = compute_metrics(yb, pb)

    result = {
        "run": run_name,
        "n_per_class": args.n_per_class,
        "noise": args.noise,
        "seed": args.seed,
        "n_train": len(train_df),
        "n_flipped": int(train_df["flipped"].sum()),
        "config": vars(args),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "train_time_sec": round(time.time() - t0, 1),
        "test_final": final,
        "best_val_step": best_step,
        "best_val_acc": best_acc,
        "test_best_val": best,
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    print(f"[{run_name}] TEST final acc {final['accuracy']:.4f} | best-val ckpt acc {best['accuracy']:.4f}")


if __name__ == "__main__":
    main()

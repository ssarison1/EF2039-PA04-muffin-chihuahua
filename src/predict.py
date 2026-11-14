"""Classify one or more images with a trained checkpoint.

Example
  python src/predict.py --ckpt weights/resnet18_clean_full.pt my_image.jpg
"""
import argparse

import torch
from PIL import Image

from common import CLASSES, build_model, get_transforms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("images", nargs="+")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.to(device).eval()
    tf = get_transforms(False)

    for path in args.images:
        try:
            x = tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        except (OSError, ValueError) as e:
            print(f"{path}: cannot read image ({e})")
            continue
        with torch.no_grad():
            prob = torch.softmax(model(x), 1)[0].cpu()
        k = int(prob.argmax())
        print(f"{path}: {CLASSES[k]} (p={prob[k]:.3f})")


if __name__ == "__main__":
    main()

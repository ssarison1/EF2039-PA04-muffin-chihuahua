# Muffin vs Chihuahua: Label Quality vs Data Quantity

EF2039 Introduction to AI Programming, PA04 — Seungheon Lee (20230445), KENTECH

This repository contains the code, data splits, trained weights, and results for a short study of how
**training-data quality** (label noise) and **training-data quantity** (images per class) affect a
fine-tuned image classifier. Everything needed to reproduce the paper's numbers is listed below.

- **Task:** binary image classification, `chihuahua` (label 0) vs `muffin` (label 1)
- **Model:** ResNet-18 pretrained on ImageNet-1k (torchvision `IMAGENET1K_V1`), all layers fine-tuned
- **RQ1:** How much does test accuracy drop when 10/20/40% of training labels are wrong?
- **RQ2:** How many clean training images per class are needed (10/20/50/200/all)?

## Repository layout

```
src/
  common.py          class names, dataset, transforms, model, metrics
  prepare_data.py    checks images, removes duplicates/leakage, builds fixed splits
  train.py           one training run (data size, noise rate, seed) + test evaluation
  evaluate.py        evaluate a saved checkpoint on the test (or val) split
  evaluate_all.py    re-evaluate all 24 checkpoints and compare with the reported results
  predict.py         classify your own image files
  aggregate.py       mean ± std tables and figures from all runs
scripts/
  run_experiments.sh runs all 24 experiments of the paper, then aggregate.py
  download_weights.sh downloads the checkpoints of all 24 runs from the GitHub release
data/splits/         train_pool.csv, val.csv, test.csv (file lists + labels), summary.json
weights/             resnet18_clean_full_seed0.pt (main model, 45 MB), SHA256SUMS.txt (release checksums)
results/             tables (.csv), figures (.png/.pdf, same as in the paper), run_log.txt
results/runs/<run>/  per-run metrics.json, log.csv, train_list.csv
environment.yml      conda environment (exact versions)
requirements.txt     pip alternative (exact versions)
```

## 1. Data

- **Source:** Kaggle, "Muffin vs Chihuahua" by Samuel Cortinhas
  <https://www.kaggle.com/datasets/samuelcortinhas/muffin-vs-chihuahua-image-classification>.
  The images are not redistributed here; download them with the command in Step 3. See the Kaggle page
  for the dataset license.
- **Labels:** taken from the folder names (`chihuahua/`, `muffin/`). The labels were not edited by hand.
- **Raw class distribution (official Kaggle split):**

  | Folder | Chihuahua | Muffin | Total |
  |---|---:|---:|---:|
  | train | 2,559 | 2,174 | 4,733 |
  | test | 640 | 544 | 1,184 |

- **Cleaning (`src/prepare_data.py`):** every file was decoded, and exact duplicates were searched by MD5
  hash within each split and across train/test. Result: 0 corrupt files, 0 duplicates, 0 train–test
  overlaps, and 0 images with conflicting labels.
- **Splits (fixed, seed 0):** the official test folder is the test set. The official train folder is split
  90/10 (stratified) into a training pool and a clean validation set.

  | Split | Chihuahua | Muffin | Total |
  |---|---:|---:|---:|
  | train_pool | 2,303 | 1,956 | 4,259 |
  | val | 256 | 218 | 474 |
  | test | 640 | 544 | 1,184 |

  The exact file lists are in `data/splits/*.csv`.
- **Preprocessing:** RGB conversion and ImageNet mean/std normalization.
  - Training: `RandomResizedCrop(224, scale=(0.7, 1.0))` and horizontal flip.
  - Evaluation: `Resize(256)` and `CenterCrop(224)`.
- **Experimental subsets (`src/train.py`):**
  - RQ2: each run samples `n` images per class from `train_pool` with its seed.
  - RQ1: each run flips the labels of a fraction `p` of the images in each class to the other class.
  - The validation and test sets are never corrupted.
  - The images and labels used in every run, including which labels were flipped, are saved in
    `results/runs/<run>/train_list.csv`.

## 2. Hardware and software

**Minimum requirements**

| Item | Requirement |
|---|---|
| GPU | NVIDIA GPU with at least 4 GB memory recommended (CPU works but is much slower) |
| RAM | 8 GB or more |
| Disk | about 7.5 GB for the conda environment, 0.5 GB for the dataset, 1.1 GB for all 24 checkpoints |
| OS | Linux or Windows with WSL2 (tested on Ubuntu 24.04 under WSL2) |
| Tools | conda, git, curl, unzip (`sudo apt install curl unzip` if missing) |

**Hardware and software used for the paper**

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU, 8 GB (driver 616.56) |
| CPU / RAM | AMD Ryzen AI 7 350, 16 GB |
| OS | Windows 11 + WSL2 Ubuntu 24.04 |
| Python / PyTorch | Python 3.11.18, PyTorch 2.11.0 (CUDA 12.8), torchvision 0.26.0 |
| Time | 1–10 min per run (full-data runs about 70 s; small-subset runs longer), about 64 min for all 24 runs |

The RTX 50-series (Blackwell) GPU needs a CUDA 12.8+ build of PyTorch, which is why the `cu128` wheels are
used. Other NVIDIA GPUs work with the same commands. The code also runs on CPU, but much more slowly.

## 3. Reproduce step by step

Run all commands from the repository root, in a Linux or WSL terminal.

```bash
# 3.1 Get the code and create the environment
git clone https://github.com/ssarison1/EF2039-PA04-muffin-chihuahua.git
cd EF2039-PA04-muffin-chihuahua
conda env create -f environment.yml
conda activate pa04
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 3.2 Download the data (needs a Kaggle API token in ~/.kaggle/, see https://www.kaggle.com/settings)
mkdir -p data
kaggle datasets download -d samuelcortinhas/muffin-vs-chihuahua-image-classification -p data
unzip -q data/muffin-vs-chihuahua-image-classification.zip -d data/raw

# 3.3 Rebuild the splits (should print the same counts as Section 1 and match data/splits/*.csv)
python src/prepare_data.py

# 3.4 Evaluate the main model stored in this repository (no training needed)
python src/evaluate.py --ckpt weights/resnet18_clean_full_seed0.pt
#   expected: accuracy 0.9992, chihuahua 1.0000, muffin 0.9982, confusion [[640, 0], [1, 543]]

# 3.5 Download the checkpoints of all 24 runs (1.1 GB, GitHub release v1.0) and re-evaluate them
bash scripts/download_weights.sh
python src/evaluate_all.py
#   prints the test accuracy of every run next to the reported value in results/runs/<run>/metrics.json
#   and saves results/reevaluation.csv; the differences should be 0.00 points

# 3.6 Classify your own images
python src/predict.py --ckpt weights/resnet18_clean_full_seed0.pt path/to/image1.jpg path/to/image2.jpg

# 3.7 Re-run every experiment of the paper (24 runs, about 1 hour) and rebuild tables and figures
bash scripts/run_experiments.sh 2>&1 | tee results/run_log.txt
```

`run_experiments.sh` skips runs that already have a `metrics.json`, so an interrupted run can be restarted
with the same command. One run can also be started by hand:

```bash
python src/train.py --n_per_class 50 --noise 0.0 --seed 1   # 50 images per class, clean labels
python src/train.py --n_per_class 0  --noise 0.2 --seed 0   # all images, 20% flipped labels
```

**Training settings (all runs):**

| Setting | Value |
|---|---|
| Optimizer | AdamW, lr 1e-4, weight decay 1e-4 |
| Schedule | cosine decay over 600 steps (same budget for every data size) |
| Batch size | 64 (or the training-set size if it is smaller) |
| Mixed precision | yes |
| Validation | every 60 steps |
| Seeds | 0, 1, 2 |

Two models are evaluated on the test set per run:
- the **final** model (step 600);
- the **best-validation** checkpoint, chosen on the clean validation set.

Seeds fix the subset, the flipped labels, the head initialization, and the data order. GPU kernels are not
fully deterministic, so re-runs can differ slightly, usually by less than 0.3 percentage points.

## 4. Expected results

Test set: 1,184 images. Values are accuracy in % (mean ± std over 3 seeds). They are produced by
`src/aggregate.py` in `results/`.

**RQ1, label noise (all 4,259 training images)**

| Flipped labels | Final model | Final macro F1 | Best-val checkpoint |
|---|---:|---:|---:|
| 0% | 99.92 ± 0.00 | 99.91 ± 0.00 | 99.83 ± 0.08 |
| 10% | 96.59 ± 0.10 | 96.57 ± 0.10 | 98.82 ± 0.45 |
| 20% | 89.47 ± 1.28 | 89.43 ± 1.28 | 96.82 ± 0.97 |
| 40% | 65.09 ± 0.49 | 65.01 ± 0.48 | 78.91 ± 2.37 |

**RQ2, data size (clean labels, final model)**

| Images per class | Chihuahua | Muffin | Total |
|---|---:|---:|---:|
| 10 | 98.75 ± 0.81 | 97.98 ± 0.18 | 98.40 ± 0.37 |
| 20 | 98.33 ± 2.75 | 97.79 ± 2.05 | 98.09 ± 0.68 |
| 50 | 99.01 ± 0.70 | 98.47 ± 0.77 | 98.76 ± 0.13 |
| 200 | 98.75 ± 0.31 | 99.08 ± 0.66 | 98.90 ± 0.44 |
| all (2,303 / 1,956) | 100.00 ± 0.00 | 99.82 ± 0.00 | 99.92 ± 0.00 |

Main finding: 20 clean images per class (98.1%) outperform 4,259 images with 20% wrong labels (89.5%).
For this task, label quality matters far more than the amount of data. Selecting a checkpoint on a small
clean validation set recovers much of the accuracy lost to noise (+7.3 points at 20% noise).

Figures: `results/fig_rq1_noise.png`, `results/fig_rq2_size.png`, `results/confusion_clean.png`.

## 5. Weights

All 24 trained checkpoints are available as assets of the GitHub release
[v1.0](https://github.com/ssarison1/EF2039-PA04-muffin-chihuahua/releases/tag/v1.0).
Step 3.5 downloads them, checks them against `weights/SHA256SUMS.txt`, and unpacks them to
`runs/<run>/model_final.pt`.

| Release file | Runs | Size |
|---|---|---|
| `checkpoints_clean_full.zip` | all images, clean labels, seeds 0–2 (shared by RQ1 and RQ2) | 129 MB |
| `checkpoints_rq1_noise.zip` | all images, 10/20/40% flipped labels, seeds 0–2 | 385 MB |
| `checkpoints_rq2_size.zip` | 10/20/50/200 images per class, clean labels, seeds 0–2 | 513 MB |

Each file is the PyTorch `state_dict` of the final model (step 600) of that run (ResNet-18 with a 2-class head).
The main model `weights/resnet18_clean_full_seed0.pt` is also stored in this repository. Load any checkpoint with:

```python
import torch
from src.common import build_model   # run from the repository root
model = build_model(pretrained=False)
model.load_state_dict(torch.load("weights/resnet18_clean_full_seed0.pt", map_location="cpu"))
```

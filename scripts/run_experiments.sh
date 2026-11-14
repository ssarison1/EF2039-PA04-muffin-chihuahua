#!/usr/bin/env bash
# Runs every experiment in the paper (3 seeds each). Run from the repository root.
#   RQ1 (label noise): all training images, 0/10/20/40% flipped labels
#   RQ2 (data size)  : 10/20/50/200 images per class, clean labels
# The clean full-data run (nall_noise00) is shared by both questions.
set -e
SEEDS="0 1 2"

for seed in $SEEDS; do
  for noise in 0.0 0.1 0.2 0.4; do
    python src/train.py --n_per_class 0 --noise $noise --seed $seed "$@"
  done
  for n in 10 20 50 200; do
    python src/train.py --n_per_class $n --noise 0.0 --seed $seed "$@"
  done
done

python src/aggregate.py

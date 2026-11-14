#!/usr/bin/env bash
# Downloads the trained checkpoints of all 24 runs from the GitHub release and
# unpacks them into runs/<run>/model_final.pt. Run from the repository root.
#   checkpoints_clean_full.zip    3 runs: all images, clean labels (shared by RQ1 and RQ2)
#   checkpoints_rq1_noise.zip     9 runs: all images, 10/20/40% flipped labels
#   checkpoints_rq2_size.zip     12 runs: 10/20/50/200 images per class, clean labels
set -e
BASE=https://github.com/ssarison1/EF2039-PA04-muffin-chihuahua/releases/download/v1.0
FILES="checkpoints_clean_full.zip checkpoints_rq1_noise.zip checkpoints_rq2_size.zip"

mkdir -p downloads
for f in $FILES; do
  if [ ! -f "downloads/$f" ]; then
    echo "downloading $f"
    curl -L --fail -o "downloads/$f" "$BASE/$f"
  fi
done

echo "checking SHA-256 checksums"
(cd downloads && sha256sum -c ../weights/SHA256SUMS.txt)

for f in $FILES; do
  unzip -o -q "downloads/$f"
done
echo "done: $(ls runs/*/model_final.pt | wc -l) checkpoints in runs/"

#!/usr/bin/env bash
# Verify that a copy of the dataset matches this project's pinned snapshot.
# Usage: bash data/download_and_verify.sh <path-to-extracted-Brain_tumor_MRI_Dataset>
# (The snapshot zip ships as a GitHub Release asset: brain-tumor-mri-dataset-v2-snapshot.zip)
set -euo pipefail
dir="${1:-Brain_tumor_MRI_Dataset}"
here="$(cd "$(dirname "$0")" && pwd)"
cd "$(dirname "$dir")"
target="$(basename "$dir")"
sed "s|Brain_tumor_MRI_Dataset/|$target/|" "$here/checksums.sha256" > /tmp/checksums.$$.sha256
if sha256sum -c /tmp/checksums.$$.sha256 --quiet; then
  echo "OK: all 7,200 files match the snapshot this project used."
else
  echo "MISMATCH: this copy differs from the pinned snapshot (see lines above)."
  exit 1
fi
rm -f /tmp/checksums.$$.sha256

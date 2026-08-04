#!/usr/bin/env bash
# =============================================================================
# build_index.sh — build the bwa-mem2 index for the fq2vcf (FASTQ-in) path.
# =============================================================================
#
# ONLY needed for the optional FASTQ-in pipeline. The default booth demo is
# BAM-in and does not use this at all.
#
# This is slow (tens of minutes) and large (roughly 70 GB of index files next
# to the reference). Run it days before the show, not on the morning.
#
# Uses the Open-Omics fq2bams container so no host toolchain is required.
# =============================================================================

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PYTHON:-python3}"
[[ -x .venv/bin/python ]] && PY=.venv/bin/python

REF="$($PY -c '
from pathlib import Path
from app.config import get_config
gz = get_config().dataset_path("reference")
print(str(gz)[:-3] if str(gz).endswith(".gz") else str(gz))
')"

IMAGE="$($PY -c 'from app.config import get_config; print(get_config().engine("open_omics_fq2bams").image)')"

[[ -f "$REF" ]] || {
  echo "FAIL: reference not staged at $REF" >&2
  echo "      Run ./scripts/fetch_data.sh first." >&2
  exit 1
}

# Skip if a complete index already exists — this is far too slow to redo.
complete=1
for suffix in .0123 .amb .ann .bwt.2bit.64 .pac; do
  [[ -f "${REF}${suffix}" ]] || complete=0
done
if (( complete == 1 )); then
  echo "bwa-mem2 index already complete for $(basename "$REF") — nothing to do."
  exit 0
fi

REF_DIR="$(dirname "$REF")"
AVAIL=$(df -PB1 "$REF_DIR" | awk 'NR==2 {print $4}')
NEED=$(( 75 * 1024 * 1024 * 1024 ))
if (( AVAIL < NEED )); then
  echo "FAIL: bwa-mem2 index needs ~$(numfmt --to=iec $NEED) free, only $(numfmt --to=iec "$AVAIL") available at $REF_DIR" >&2
  exit 1
fi

if command -v bwa-mem2 >/dev/null; then
  echo "==> building index with the host bwa-mem2 (this takes a while)"
  bwa-mem2 index "$REF"
else
  echo "==> building index inside $IMAGE (this takes a while)"
  docker image inspect "$IMAGE" >/dev/null 2>&1 || {
    echo "FAIL: $IMAGE is not available locally." >&2
    echo "      It has no prebuilt Docker Hub image and must be built from" >&2
    echo "      the IntelLabs Open-Omics framework — see README, 'fq2vcf path'." >&2
    exit 1
  }
  docker run --rm \
    -u "$(id -u):$(id -g)" \
    -v "$REF_DIR:/ref" \
    "$IMAGE" \
    bwa-mem2 index "/ref/$(basename "$REF")"
fi

echo "==> done"
ls -lh "${REF}".* 2>/dev/null || true

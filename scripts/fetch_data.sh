#!/usr/bin/env bash
# =============================================================================
# fetch_data.sh — stage the public genomics datasets this demo runs on.
# =============================================================================
#
# Reads config.yaml so the file layout always matches what the app expects.
#
# Tiers (see `tier:` in config.yaml):
#   core   — reference + smoke BAM + chr20 BAM   (~2 GB)   ALWAYS fetched
#   truth  — GIAB HG002 v4.2.1 truth VCF + BED   (~170 MB) --with-truth
#   wgs    — full 35x WGS BAM                    (~46 GB)  --with-wgs
#   fastq  — paired FASTQ for the fq2vcf path    (large)   --with-fastq
#
# INTEGRITY POLICY — deliberately paranoid, and never invents a hash:
#   * Google Cloud Storage publishes an authoritative MD5 in the `x-goog-hash`
#     response header. We capture it at download time and verify against it.
#   * NCBI GIAB publishes sibling checksum files where available; we look for
#     them and verify when present.
#   * A local sha256 is computed for EVERY file and written to
#     data/CHECKSUMS.sha256.
#   * If config.yaml already carries a sha256 for a file, that value is
#     AUTHORITATIVE. A mismatch is a hard, loud failure — we never silently
#     regenerate a recorded checksum.
#
# Usage:
#   ./scripts/fetch_data.sh                      # core tier only
#   ./scripts/fetch_data.sh --with-truth         # + accuracy panel data
#   ./scripts/fetch_data.sh --with-wgs           # + the 46 GB headline BAM
#
#   --allow-fallback-root  download to ./data even if the data volume looks
#                          unmounted. Refused by default so an unmounted disk
#                          cannot silently fill the OS drive.
#   ./scripts/fetch_data.sh --all                # everything except FASTQ
#   ./scripts/fetch_data.sh --verify-only        # re-verify, download nothing
# =============================================================================

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${GENOMICS_DEMO_CONFIG:-$REPO_ROOT/config.yaml}"

WITH_TRUTH=0
WITH_WGS=0
WITH_FASTQ=0
VERIFY_ONLY=0
BUILD_INDEX=0
ALLOW_FALLBACK_ROOT=0

RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[0;33m'; BLU='\033[0;34m'; BLD='\033[1m'; NC='\033[0m'
say()  { echo -e "${BLU}==>${NC} $*"; }
ok()   { echo -e "${GRN} ok ${NC} $*"; }
warn() { echo -e "${YLW}warn${NC} $*"; }
die()  { echo -e "${RED}${BLD}FAIL${NC} $*" >&2; exit 1; }

usage() { sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-truth) WITH_TRUTH=1 ;;
    --with-wgs)   WITH_WGS=1 ;;
    --with-fastq) WITH_FASTQ=1 ;;
    --build-index) BUILD_INDEX=1 ;;
    --all)        WITH_TRUTH=1; WITH_WGS=1 ;;
    --verify-only) VERIFY_ONLY=1 ;;
    --allow-fallback-root) ALLOW_FALLBACK_ROOT=1 ;;
    -h|--help)    usage ;;
    *) die "unknown option: $1 (try --help)" ;;
  esac
  shift
done

command -v curl >/dev/null || die "curl is required"
command -v python3 >/dev/null || die "python3 is required"
command -v sha256sum >/dev/null || die "sha256sum is required"

# -----------------------------------------------------------------------------
# Ask the config loader where things go and what to fetch. Keeping this in
# Python means the script and the app can never disagree about layout.
# -----------------------------------------------------------------------------
DATA_ROOT="$(cd "$REPO_ROOT" && GENOMICS_DEMO_CONFIG="$CONFIG" python3 -c '
from app.config import get_config
print(get_config().data_root)
')" || die "could not read config.yaml"

say "Config     : $CONFIG"
say "Data root  : $DATA_ROOT"

# Refuse to download onto the OS disk when the data volume is merely unmounted.
# Without this the fallback root silently absorbs tens of GB and fills /.
UNMOUNTED="$(cd "$REPO_ROOT" && GENOMICS_DEMO_CONFIG="$CONFIG" python3 -c '
from app.config import get_config
print("yes" if get_config().data_disk_looks_unmounted else "no")
')" || die "could not read config.yaml"

if [ "$UNMOUNTED" = "yes" ] && [ "$ALLOW_FALLBACK_ROOT" -eq 0 ]; then
  CONFIGURED="$(cd "$REPO_ROOT" && GENOMICS_DEMO_CONFIG="$CONFIG" python3 -c '
from app.config import get_config
print(get_config().raw["paths"]["data_root"])
')"
  MOUNTPOINT="$(dirname "$CONFIGURED")"
  die "$(cat <<EOF
The data volume looks UNMOUNTED, not missing.

  configured data root : $CONFIGURED
  mountpoint           : $MOUNTPOINT  (exists, but is empty)
  would fall back to   : $DATA_ROOT

Downloading now would write tens of GB onto the OS disk. Your datasets are
almost certainly still on the data volume -- mount it and re-run:

  sudo mount $MOUNTPOINT

If the mount is missing from /etc/fstab it will not survive a reboot. See the
"Persisting the data mount" section of the README.

To download to the fallback path anyway, pass --allow-fallback-root.
EOF
)"
fi

# Used only if a derived dataset needs slicing and the host has no samtools.
ENGINE_IMAGE="$(cd "$REPO_ROOT" && GENOMICS_DEMO_CONFIG="$CONFIG" python3 -c '
from app.config import get_config
print(get_config().default_engine.image)
')" || die "could not read the engine image from config.yaml"

read -r -a CONTAINER_CLI <<< "$(cd "$REPO_ROOT" && GENOMICS_DEMO_CONFIG="$CONFIG" python3 -c '
import shlex
from app.config import get_config
cli = get_config().compute.get("container_cli") or ["docker"]
print(" ".join(cli) if isinstance(cli, list) else cli)
')" || die "could not read container_cli from config.yaml"

mkdir -p "$DATA_ROOT" || die "cannot create $DATA_ROOT"
[[ -w "$DATA_ROOT" ]] || die "$DATA_ROOT is not writable"

CHECKSUM_FILE="$DATA_ROOT/CHECKSUMS.sha256"

# Emits one TSV row per file to fetch: tier, url, local, md5_b64, size_bytes, key
MANIFEST="$(cd "$REPO_ROOT" && GENOMICS_DEMO_CONFIG="$CONFIG" python3 -c '
from app.config import get_config
cfg = get_config()
for key, ds in cfg.datasets.items():
    if ds.is_derived:
        continue  # built locally after the downloads, see "Derived datasets"
    print("\t".join([ds.tier, ds.url, ds.local, ds.md5_b64 or "-", str(ds.size_bytes or 0), key]))
    for sc in ds.sidecars:
        # A sidecar inherits its parent tier; size is unknown so we skip the check.
        print("\t".join([ds.tier, sc.url, sc.local, sc.md5_b64 or "-", "0", key + ":sidecar"]))
')" || die "could not build manifest from config.yaml"

# Datasets produced locally by slicing a verified parent: key, parent, region, local
DERIVED="$(cd "$REPO_ROOT" && GENOMICS_DEMO_CONFIG="$CONFIG" python3 -c '
from app.config import get_config
cfg = get_config()
for key, ds in cfg.datasets.items():
    if ds.is_derived:
        parent = cfg.dataset(ds.derived_from)
        print("\t".join([key, parent.local, ds.derive_region or "", ds.local]))
')" || die "could not build derived-dataset list from config.yaml"

# -----------------------------------------------------------------------------
# Free-space guard. Downloading 46 GB onto a full disk at 08:00 on show day is
# exactly the failure this demo cannot afford.
# -----------------------------------------------------------------------------
required_bytes=0
while IFS=$'\t' read -r tier url local md5 size key; do
  [[ -z "${tier:-}" ]] && continue
  case "$tier" in
    core) want=1 ;;
    truth) want=$WITH_TRUTH ;;
    wgs)   want=$WITH_WGS ;;
    *)     want=0 ;;
  esac
  [[ "$want" == "1" ]] || continue
  [[ -f "$DATA_ROOT/$local" ]] && continue
  required_bytes=$(( required_bytes + size ))
done <<< "$MANIFEST"

avail_bytes=$(df -PB1 "$DATA_ROOT" | awk 'NR==2 {print $4}')
# The reference is shipped bgzipped and DeepVariant wants it uncompressed, so
# leave headroom for the decompressed FASTA (~3.1 GB) plus run outputs.
headroom=$(( 8 * 1024 * 1024 * 1024 ))
need=$(( required_bytes + headroom ))
say "Need ~$(numfmt --to=iec "$need"), available $(numfmt --to=iec "$avail_bytes")"
if (( VERIFY_ONLY == 0 && need > avail_bytes )); then
  die "not enough free space at $DATA_ROOT.
     Need ~$(numfmt --to=iec "$need") (incl. $(numfmt --to=iec "$headroom") headroom), have $(numfmt --to=iec "$avail_bytes").
     Mount the U.2 NVMe at the configured paths.data_root (see README) or drop --with-wgs."
fi

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

# Authoritative MD5 straight from GCS, base64 as GCS reports it.
remote_gcs_md5() {
  curl -sIL --max-time 60 "$1" 2>/dev/null \
    | tr -d '\r' \
    | awk 'tolower($0) ~ /^x-goog-hash: *md5=/ { sub(/^[^=]*=/, ""); print; exit }'
}

local_md5_b64() {
  md5sum "$1" | awk '{print $1}' | xxd -r -p | base64
}

# Look for a sibling checksum file published next to a GIAB/NCBI object.
remote_sibling_sha256() {
  local url="$1" candidate
  for suffix in .sha256 .SHA256 .sha256sum; do
    candidate="${url}${suffix}"
    if curl -sfIL --max-time 20 "$candidate" >/dev/null 2>&1; then
      curl -sfL --max-time 60 "$candidate" 2>/dev/null | awk '{print $1; exit}'
      return 0
    fi
  done
  return 1
}

config_sha256_for() {
  (cd "$REPO_ROOT" && GENOMICS_DEMO_CONFIG="$CONFIG" python3 -c "
import sys
from app.config import get_config
key = sys.argv[1].split(':')[0]
ds = get_config().datasets.get(key)
print((ds.sha256 or '').strip() if ds else '')
" "$1")
}

download() {
  local url="$1" dest="$2"
  mkdir -p "$(dirname "$dest")"
  # -C - resumes a partial file, which matters for the 46 GB BAM over a
  # conference network. --retry rides out transient FTP hiccups.
  curl -fL --progress-bar -C - --retry 5 --retry-delay 5 --retry-connrefused \
       -o "$dest" "$url"
}

verify_file() {
  local url="$1" dest="$2" md5_expect="$3" size_expect="$4" key="$5"
  local base; base="$(basename "$dest")"

  # --- size -----------------------------------------------------------
  if [[ "$size_expect" != "0" ]]; then
    local actual; actual=$(stat -c%s "$dest")
    if [[ "$actual" != "$size_expect" ]]; then
      die "$base: size mismatch — expected $size_expect bytes, got $actual.
     The file is truncated or the upstream object changed. Delete it and re-run."
    fi
    ok "$base size $(numfmt --to=iec "$actual")"
  fi

  # --- MD5 (authoritative, from GCS) ----------------------------------
  if [[ "$md5_expect" == "-" && "$url" == https://storage.googleapis.com/* ]]; then
    md5_expect="$(remote_gcs_md5 "$url" || true)"
    [[ -n "$md5_expect" ]] && say "$base: captured authoritative GCS MD5 $md5_expect"
  fi
  if [[ -n "$md5_expect" && "$md5_expect" != "-" ]]; then
    local got; got="$(local_md5_b64 "$dest")"
    [[ "$got" == "$md5_expect" ]] \
      || die "$base: MD5 MISMATCH — upstream says $md5_expect, local file is $got"
    ok "$base MD5 matches upstream"
  fi

  # --- sibling checksum published by NCBI GIAB ------------------------
  if [[ "$url" == https://ftp-trace.ncbi.nlm.nih.gov/* ]]; then
    local sibling; sibling="$(remote_sibling_sha256 "$url" || true)"
    if [[ -n "$sibling" ]]; then
      local got; got="$(sha256sum "$dest" | awk '{print $1}')"
      [[ "$got" == "$sibling" ]] \
        || die "$base: sha256 MISMATCH against published NCBI checksum"
      ok "$base matches NCBI published sha256"
    fi
  fi

  # --- config.yaml sha256 is authoritative once populated -------------
  local recorded; recorded="$(config_sha256_for "$key")"
  local got_sha; got_sha="$(sha256sum "$dest" | awk '{print $1}')"
  if [[ -n "$recorded" && "$key" != *:sidecar ]]; then
    if [[ "$recorded" != "$got_sha" ]]; then
      die "$base: sha256 MISMATCH against config.yaml.
     config.yaml datasets.$key.sha256 = $recorded
     local file                        = $got_sha
     Checksums in config.yaml are authoritative. Investigate before the show —
     do NOT overwrite the config value to make this pass."
    fi
    ok "$base matches config.yaml sha256"
  fi

  # Record the path relative to DATA_ROOT so the manifest stays valid if the
  # data directory is moved (e.g. staged on the OS drive, then migrated to the
  # U.2 NVMe). Absolute paths would silently verify the ORIGINAL files after a
  # copy, which defeats the point of the check.
  echo "$got_sha  ${dest#"$DATA_ROOT"/}" >> "$CHECKSUM_FILE.tmp"
}

# -----------------------------------------------------------------------------
# Fetch loop
# -----------------------------------------------------------------------------
: > "$CHECKSUM_FILE.tmp"
skipped_tiers=()
(( WITH_TRUTH == 0 )) && skipped_tiers+=("truth (--with-truth)")
(( WITH_WGS == 0 ))   && skipped_tiers+=("wgs (--with-wgs)")

while IFS=$'\t' read -r tier url local md5 size key; do
  [[ -z "${tier:-}" ]] && continue
  case "$tier" in
    core)  want=1 ;;
    truth) want=$WITH_TRUTH ;;
    wgs)   want=$WITH_WGS ;;
    *)     want=0 ;;
  esac
  (( want == 1 )) || continue

  dest="$DATA_ROOT/$local"
  echo
  say "${BLD}$(basename "$local")${NC}  [$tier]"

  if [[ -f "$dest" ]]; then
    ok "already present, verifying"
  elif (( VERIFY_ONLY == 1 )); then
    warn "missing (verify-only mode, not downloading): $local"
    continue
  else
    download "$url" "$dest" || die "download failed: $url"
  fi

  verify_file "$url" "$dest" "$md5" "$size" "$key"
done <<< "$MANIFEST"

# -----------------------------------------------------------------------------
# Optional FASTQ (fq2vcf path only). GIAB rotates these filenames, hence config.
# -----------------------------------------------------------------------------
if (( WITH_FASTQ == 1 )); then
  echo
  say "${BLD}Paired FASTQ (fq2vcf path)${NC}"
  read -r FQ_BASE FQ_R1 FQ_R2 FQ_DIR < <(cd "$REPO_ROOT" && GENOMICS_DEMO_CONFIG="$CONFIG" python3 -c '
from app.config import get_config
f = get_config().fastq
print(f.get("base_url",""), f.get("r1",""), f.get("r2",""), f.get("local_dir","fastq"))
')
  [[ -n "$FQ_R1" && -n "$FQ_R2" ]] || die "fastq.r1/fastq.r2 not set in config.yaml"
  for fq in "$FQ_R1" "$FQ_R2"; do
    dest="$DATA_ROOT/$FQ_DIR/$fq"
    if [[ -f "$dest" ]]; then ok "already present: $fq"; else
      download "${FQ_BASE%/}/$fq" "$dest" \
        || die "FASTQ download failed. GIAB rotates these filenames — check
     fastq.r1 / fastq.r2 in config.yaml against the current GIAB listing."
    fi
    echo "$(sha256sum "$dest" | awk '{print $1}')  ${dest#"$DATA_ROOT"/}" >> "$CHECKSUM_FILE.tmp"
  done
fi

# -----------------------------------------------------------------------------
# Reference preparation: DeepVariant wants a plain FASTA + .fai.
# -----------------------------------------------------------------------------
echo
say "${BLD}Preparing reference${NC}"
REF_GZ="$DATA_ROOT/$(cd "$REPO_ROOT" && GENOMICS_DEMO_CONFIG="$CONFIG" python3 -c '
from app.config import get_config
print(get_config().dataset("reference").local)')"
REF_FA="${REF_GZ%.gz}"

if [[ -f "$REF_FA" && -f "$REF_FA.fai" ]]; then
  ok "uncompressed reference and .fai already in place"
elif [[ -f "$REF_GZ" ]]; then
  if [[ ! -f "$REF_FA" ]]; then
    say "decompressing reference (~3.1 GB uncompressed, takes a minute)"
    # Keep the .gz: it carries the published .fai/.gzi we verified against.
    gzip -dc "$REF_GZ" > "$REF_FA.partial" && mv "$REF_FA.partial" "$REF_FA"
    ok "wrote $(basename "$REF_FA")"
  fi
  if [[ ! -f "$REF_FA.fai" ]]; then
    if command -v samtools >/dev/null; then
      samtools faidx "$REF_FA" && ok "built .fai with samtools"
    else
      # samtools is not installed on every booth box, and the .fai format is
      # simple enough to generate directly rather than block the demo.
      say "samtools not found — building .fai directly (no extra dependencies)"
      python3 "$REPO_ROOT/scripts/build_fai.py" "$REF_FA" && ok "built .fai"
    fi
  fi
else
  warn "reference not staged; skipping preparation"
fi

# -----------------------------------------------------------------------------
# Optional bwa-mem2 index (fq2vcf path only) — slow and large, hence opt-in.
# -----------------------------------------------------------------------------
if (( BUILD_INDEX == 1 )); then
  echo
  say "${BLD}bwa-mem2 index${NC}"
  "$REPO_ROOT/scripts/build_index.sh" || die "bwa-mem2 index build failed"
fi

# -----------------------------------------------------------------------------
# Derived datasets: built here by slicing a parent we have already verified.
#
# The pre-flight smoke BAM is derived rather than downloaded on purpose. The
# obvious candidate -- DeepVariant's quickstart NA12878 BAM -- is aligned to
# hg19, so against our GRCh38 reference every run dies with "0 bases found in
# common among our input files". Slicing our own verified GRCh38 BAM keeps the
# entire demo on one reference build.
# -----------------------------------------------------------------------------
if [[ -n "${DERIVED//[[:space:]]/}" ]]; then
  echo
  say "${BLD}Derived datasets${NC}"
fi
while IFS=$'\t' read -r key parent_local region local; do
  [[ -z "${key:-}" ]] && continue
  dest="$DATA_ROOT/$local"
  parent="$DATA_ROOT/$parent_local"

  if [[ -f "$dest" && -f "$dest.bai" ]]; then
    ok "$(basename "$dest") already built"
  elif [[ ! -f "$parent" ]]; then
    warn "$key: parent $parent_local not staged — skipping"
    continue
  else
    say "slicing $region out of $(basename "$parent")"
    mkdir -p "$(dirname "$dest")"
    if command -v samtools >/dev/null; then
      samtools view -b -o "$dest.partial" "$parent" "$region" \
        && mv "$dest.partial" "$dest" && samtools index "$dest" \
        || die "failed to slice $key with samtools"
    else
      # samtools is not installed on the booth box, but the pipeline image
      # ships one -- reuse it rather than adding a host dependency.
      say "samtools not on PATH — using the one inside $ENGINE_IMAGE"
      "${CONTAINER_CLI[@]}" run --rm -u "$(id -u):$(id -g)" \
        -v "$DATA_ROOT:/d" --entrypoint /bin/bash "$ENGINE_IMAGE" -c "
          set -e
          export PATH=/opt/conda/envs/bio/bin:\$PATH
          samtools view -b -o '/d/$local.partial' '/d/$parent_local' '$region'
          mv '/d/$local.partial' '/d/$local'
          samtools index '/d/$local'
        " || die "failed to slice $key using the pipeline image.
     Is Docker reachable? Try: ./scripts/preflight.sh"
    fi
    ok "built $(basename "$dest") from a sha256-verified parent"
  fi

  echo "$(sha256sum "$dest" | awk '{print $1}')  $local" >> "$CHECKSUM_FILE.tmp"
  [[ -f "$dest.bai" ]] && \
    echo "$(sha256sum "$dest.bai" | awk '{print $1}')  $local.bai" >> "$CHECKSUM_FILE.tmp"
done <<< "$DERIVED"

# -----------------------------------------------------------------------------
# Publish checksums
# -----------------------------------------------------------------------------
sort -k2 "$CHECKSUM_FILE.tmp" > "$CHECKSUM_FILE"
rm -f "$CHECKSUM_FILE.tmp"

echo
echo -e "${GRN}${BLD}=== staging complete ===${NC}"
say "Checksums written to $CHECKSUM_FILE"
if (( ${#skipped_tiers[@]} )); then
  warn "Skipped tiers: ${skipped_tiers[*]}"
fi

# Nudge the operator to lock in the checksums exactly once, honestly.
missing_recorded=0
while IFS= read -r line; do
  path="${line#*  }"; hash="${line%%  *}"
  base_key=""
  while IFS=$'\t' read -r t u l m s k; do
    [[ "$DATA_ROOT/$l" == "$path" && "$k" != *:sidecar ]] && base_key="$k"
  done <<< "$MANIFEST"
  [[ -z "$base_key" ]] && continue
  if [[ -z "$(config_sha256_for "$base_key")" ]]; then
    if (( missing_recorded == 0 )); then
      echo
      warn "These files have no sha256 recorded in config.yaml yet."
      warn "Paste the values below into config.yaml under datasets.<name>.sha256"
      warn "so later runs are verified against a pinned value:"
      missing_recorded=1
    fi
    printf '    %-12s sha256: "%s"\n' "$base_key" "$hash"
  fi
done < "$CHECKSUM_FILE"

echo
say "Next: ./scripts/preflight.sh"

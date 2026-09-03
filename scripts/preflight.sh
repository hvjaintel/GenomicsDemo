#!/usr/bin/env bash
# =============================================================================
# preflight.sh — run this before the show floor opens.
# =============================================================================
# Verifies CPU features, Docker, the pipeline image, staged datasets and free
# space, then optionally runs a real 60-second smoke test through the whole
# stack. Exits non-zero if a live run would be impossible.
#
# Usage:
#   ./scripts/preflight.sh              # checks only
#   ./scripts/preflight.sh --pull       # also pull the pipeline image
#   ./scripts/preflight.sh --smoke      # also run the end-to-end smoke test
#   ./scripts/preflight.sh --pull --smoke
# =============================================================================

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DO_PULL=0
DO_SMOKE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pull)  DO_PULL=1 ;;
    --smoke) DO_SMOKE=1 ;;
    -h|--help) sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

if (( DO_PULL == 1 )); then
  IMAGE="$(python3 -c 'from app.config import get_config; print(get_config().default_engine.image)')"
  echo "==> pulling $IMAGE"
  docker pull "$IMAGE" || {
    echo "FAIL: could not pull $IMAGE" >&2
    echo "      If this is a permissions error, run:" >&2
    echo "        sudo usermod -aG docker \$USER && newgrp docker" >&2
    exit 1
  }
fi

set +e
python3 -m app.preflight
STATUS=$?
set -e

if (( DO_SMOKE == 1 )); then
  if (( STATUS != 0 )); then
    echo "Skipping smoke test — blocking checks failed above." >&2
    exit $STATUS
  fi
  echo "==> running end-to-end smoke test (tiny chr20 BAM)"
  python3 -c '
import sys
from app.config import get_config
from app.preflight import smoke_test
ok, message = smoke_test(get_config())
print(("PASS: " if ok else "FAIL: ") + message)
sys.exit(0 if ok else 1)
' || exit 1
fi

exit $STATUS

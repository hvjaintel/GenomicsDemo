#!/usr/bin/env bash
# =============================================================================
# run_demo.sh — one command to start the booth demo.
# =============================================================================
# Creates an isolated virtualenv on first run, installs the pinned
# dependencies, runs pre-flight checks, then launches the UI.
#
#   ./run_demo.sh                # set up if needed, then launch
#   ./run_demo.sh --skip-checks  # launch immediately (mid-show restart)
#   ./run_demo.sh --setup-only   # prepare the environment and exit
#   ./run_demo.sh --offline      # install from ./wheelhouse, no network
# =============================================================================

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

SKIP_CHECKS=0
SETUP_ONLY=0
OFFLINE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-checks) SKIP_CHECKS=1 ;;
    --setup-only)  SETUP_ONLY=1 ;;
    --offline)     OFFLINE=1 ;;
    -h|--help) sed -n '2,14p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

VENV="$REPO_ROOT/.venv"
STAMP="$VENV/.requirements.sha256"

# ---------------------------------------------------------------------------
# Environment setup. Ubuntu ships python3 without the venv module in some
# images, so fall back to virtualenv, which installs cleanly with --user and
# needs no root.
# ---------------------------------------------------------------------------
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "==> creating virtualenv at $VENV"
  if python3 -m venv "$VENV" 2>/dev/null; then
    :
  else
    echo "    python3-venv unavailable; falling back to virtualenv"
    python3 -m virtualenv --version >/dev/null 2>&1 || pip3 install --user --quiet virtualenv
    python3 -m virtualenv -q "$VENV"
  fi
fi

PY="$VENV/bin/python"
WANT="$(sha256sum requirements.txt | awk '{print $1}')"
HAVE="$(cat "$STAMP" 2>/dev/null || true)"

if [[ "$WANT" != "$HAVE" ]]; then
  echo "==> installing pinned dependencies"
  if (( OFFLINE == 1 )); then
    [[ -d wheelhouse ]] || { echo "FAIL: --offline needs a ./wheelhouse directory" >&2; exit 1; }
    "$PY" -m pip install --quiet --no-index --find-links wheelhouse -r requirements.txt
  else
    "$PY" -m pip install --quiet --upgrade pip
    "$PY" -m pip install --quiet -r requirements.txt
  fi
  echo "$WANT" > "$STAMP"
else
  echo "==> dependencies already up to date"
fi

if (( SETUP_ONLY == 1 )); then
  echo "==> setup complete"
  exit 0
fi

# ---------------------------------------------------------------------------
# Pre-flight. Never blocks the launch: a booth screen showing red checks is far
# more useful than no booth screen at all.
# ---------------------------------------------------------------------------
if (( SKIP_CHECKS == 0 )); then
  set +e
  "$PY" -m app.preflight
  PF=$?
  set -e
  if (( PF != 0 )); then
    echo
    echo "!! Pre-flight reported blocking issues (see above)."
    echo "!! Launching anyway so the Status screen can show them on the booth monitor."
    echo
    sleep 2
  fi
fi

PORT="$("$PY" -c 'from app.config import get_config; print(get_config().demo.get("port", 7860))')"
echo "==> starting the booth UI on http://localhost:$PORT"
echo "    Open it full screen on the booth monitor. Ctrl-C to stop."
exec "$PY" -m app.main

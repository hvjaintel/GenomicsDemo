#!/usr/bin/env bash
# =============================================================================
# install_service.sh — install the booth demo as a systemd service.
# =============================================================================
# After this, the machine goes from cold boot to a live demo on the booth
# monitor with nobody logged in.
#
#   sudo ./scripts/install_service.sh            # install and enable
#   sudo ./scripts/install_service.sh --status   # what is it doing now
#   sudo ./scripts/install_service.sh --uninstall
#
# Run it from the checkout you want to serve: the unit is pinned to that path.
# =============================================================================

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMPLATE="$REPO_ROOT/deploy/genomics-demo.service"
UNIT_NAME="genomics-demo.service"
UNIT_PATH="/etc/systemd/system/$UNIT_NAME"

say()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
ok()   { printf '  \033[1;32mok\033[0m   %s\n' "$*"; }
warn() { printf '  \033[1;33mwarn\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31mFAIL\033[0m %s\n' "$*" >&2; exit 1; }

case "${1:-}" in
  --status)
    systemctl status "$UNIT_NAME" --no-pager || true
    echo
    say "last 30 log lines"
    journalctl -u "$UNIT_NAME" -n 30 --no-pager || true
    exit 0
    ;;
  --uninstall)
    [[ $EUID -eq 0 ]] || die "needs root: sudo $0 --uninstall"
    systemctl disable --now "$UNIT_NAME" 2>/dev/null || true
    rm -f "$UNIT_PATH"
    systemctl daemon-reload
    ok "removed $UNIT_PATH"
    exit 0
    ;;
  "") ;;
  *) die "unknown option: $1" ;;
esac

[[ $EUID -eq 0 ]] || die "needs root: sudo $0"
[[ -f "$TEMPLATE" ]] || die "missing $TEMPLATE"

# ---------------------------------------------------------------------------
# Who will this run as? Not root -- the demo talks to Docker through group
# membership, and a booth UI has no business running privileged.
# ---------------------------------------------------------------------------
RUN_USER="${SUDO_USER:-}"
[[ -n "$RUN_USER" && "$RUN_USER" != "root" ]] \
  || die "could not determine the target user; run via sudo from that user's shell"

id -nG "$RUN_USER" | tr ' ' '\n' | grep -qx docker \
  || die "$RUN_USER is not in the 'docker' group — the service could start but every run would fail.
       Fix with: sudo usermod -aG docker $RUN_USER   (then log out and back in)"

# ---------------------------------------------------------------------------
# Refuse to pin the unit to a throwaway directory.
#
# The demo was found running out of a Copilot agent worktree, which works
# until that session is archived and the booth machine reboots into nothing.
# A service is a long-lived promise; it should not point somewhere designed to
# be disposable.
# ---------------------------------------------------------------------------
if [[ "$REPO_ROOT" == *"/copilot-worktrees/"* || "$REPO_ROOT" == *"/.copilot/"* ]]; then
  warn "this checkout looks like a temporary agent worktree:"
  warn "    $REPO_ROOT"
  warn "Such directories can be archived or cleaned up, taking the booth demo with them."
  warn "Prefer a permanent clone, e.g. /opt/genomics-demo or ~/GenomicsDemo."
  if [[ -z "${ALLOW_TEMP_CHECKOUT:-}" ]]; then
    die "refusing to install. Re-run from a permanent clone, or set ALLOW_TEMP_CHECKOUT=1 to override."
  fi
  warn "ALLOW_TEMP_CHECKOUT set — continuing against a disposable path."
fi

[[ -x "$REPO_ROOT/run_demo.sh" ]] || die "$REPO_ROOT/run_demo.sh is missing or not executable"

# ---------------------------------------------------------------------------
# The data mount becomes a hard systemd dependency, so read it from config
# rather than hardcoding this box's NVMe path.
# ---------------------------------------------------------------------------
DATA_ROOT="$(
  "${REPO_ROOT}/.venv/bin/python" - <<'PY' 2>/dev/null || true
from app.config import get_config
print(get_config().raw["paths"]["data_root"])
PY
)"
if [[ -z "$DATA_ROOT" ]]; then
  DATA_ROOT="$(grep -oP '^\s*data_root:\s*"\K[^"]+' "$REPO_ROOT/config.yaml" | head -1 || true)"
fi
[[ -n "$DATA_ROOT" ]] || die "could not determine paths.data_root from config.yaml"

if ! findmnt --target "$DATA_ROOT" >/dev/null 2>&1; then
  warn "$DATA_ROOT is not currently a mount point."
  warn "RequiresMountsFor will then resolve to whatever filesystem contains it,"
  warn "which is harmless but gives no protection against a missing data disk."
fi

say "installing $UNIT_NAME"
ok "user       $RUN_USER"
ok "checkout   $REPO_ROOT"
ok "data root  $DATA_ROOT"

sed -e "s|__REPO__|$REPO_ROOT|g" \
    -e "s|__USER__|$RUN_USER|g" \
    -e "s|__DATA_ROOT__|$DATA_ROOT|g" \
    "$TEMPLATE" > "$UNIT_PATH"

# Check the exact placeholder tokens, not any double underscore: the template's
# own comments discuss the substitution, and a blanket grep for "__" matched
# those and aborted every install.
if grep -qE '__(REPO|USER|DATA_ROOT)__' "$UNIT_PATH"; then
  die "a placeholder was left unsubstituted in $UNIT_PATH"
fi

chmod 0644 "$UNIT_PATH"
systemctl daemon-reload
systemctl enable "$UNIT_NAME" >/dev/null
ok "enabled at boot"

say "starting now"
systemctl restart "$UNIT_NAME"

PORT="$(
  "${REPO_ROOT}/.venv/bin/python" -c \
    'from app.config import get_config; print(get_config().demo.get("port", 7860))' 2>/dev/null || echo 7860
)"

# The first start may build a virtualenv, so give it room before declaring failure.
say "waiting for the UI on port $PORT"
for _ in $(seq 1 60); do
  if curl -fsS -o /dev/null "http://127.0.0.1:$PORT/" 2>/dev/null; then
    ok "serving on http://127.0.0.1:$PORT"
    echo
    say "verify it survives a reboot:   sudo reboot   then   $0 --status"
    exit 0
  fi
  sleep 2
done

die "service did not answer on port $PORT within 120s. Inspect with:
       sudo $0 --status"

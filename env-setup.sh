#!/usr/bin/env bash

# Bootstrap the development environment for the rationale benchmark project.

set -euo pipefail

log() {
  printf '==> %s\n' "$1"
}

fail() {
  printf 'Error: %s\n' "$1" >&2
  exit 1
}

SCRIPT_DIR=$(
  cd "$(dirname "${BASH_SOURCE[0]}")" && pwd
)
cd "$SCRIPT_DIR"

if [ -n "${PYTHON_BIN:-}" ]; then
  command -v "$PYTHON_BIN" >/dev/null 2>&1 || fail "PYTHON_BIN '$PYTHON_BIN' is not executable."
else
  for candidate in python3.12 python3.11 python3.10 python3.9 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON_BIN=$candidate
      break
    fi
  done
fi

[ -n "${PYTHON_BIN:-}" ] || fail "Python 3.9+ is required but not found."

"$PYTHON_BIN" - <<'PY'
import sys

if sys.version_info < (3, 9):
  raise SystemExit("Python 3.9 or newer is required.")
PY

log "Ensuring pip is available"
if ! "$PYTHON_BIN" -m ensurepip --upgrade >/dev/null 2>&1; then
  log "ensurepip not available, continuing if pip is already present"
fi

"$PYTHON_BIN" -m pip --version >/dev/null 2>&1 || fail "pip is required but missing for $PYTHON_BIN."

log "Installing or updating uv"
if ! "$PYTHON_BIN" -m pip install --user --upgrade uv >/dev/null 2>&1; then
  log "pip installation failed, falling back to official uv installer"
  command -v curl >/dev/null 2>&1 || fail "curl is required to install uv."
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 || fail "uv installer script failed."
fi

export PATH="$HOME/.local/bin:$PATH"
hash -r

UV_BIN=$(command -v uv || true)
[ -n "$UV_BIN" ] || fail "uv installation failed."

log "Creating virtual environment (./.venv)"
"$UV_BIN" venv --python "$PYTHON_BIN"

log "Synchronizing dependencies (including dev tools)"
"$UV_BIN" sync --dev

if [ -f ".env.example" ] && [ ! -f ".env" ]; then
  log "Seeding .env from template"
  cp .env.example .env
fi

cat <<'EOF'

Environment setup complete.
- Use "uv run" to execute project commands (e.g., uv run pytest).
- Optional: activate the venv with "source .venv/bin/activate".
- Review .env and update provider credentials before running benchmarks.
EOF

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$REPO_ROOT/.venv"
VENV_PY="$VENV_DIR/bin/python"
CONSTRAINTS_FILE="$REPO_ROOT/requirements/ci-constraints.txt"
cd "$REPO_ROOT"

# Require uv - no fallback
if ! command -v uv >/dev/null 2>&1; then
  echo "[wt_bootstrap] ERROR: uv is required but not found in PATH" >&2
  echo "[wt_bootstrap] Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

if [[ ! -f "$CONSTRAINTS_FILE" ]]; then
  echo "[wt_bootstrap] Missing constraints file: $CONSTRAINTS_FILE" >&2
  exit 1
fi

echo "[wt_bootstrap] Using uv"

if [[ ! -x "$VENV_PY" ]]; then
  uv venv "$VENV_DIR"
fi

# Single-path install (no fallback): install project + dev extras from this branch,
# then install package without dependencies for parity with strict CI bootstrap checks.
uv pip install --python "$VENV_PY" -c "$CONSTRAINTS_FILE" '.[dev]'
uv pip install --python "$VENV_PY" -c "$CONSTRAINTS_FILE" . --no-deps

# CI tools lane (explicit): avoid relying on optional extras implicitly containing these.
uv pip install --python "$VENV_PY" -c "$CONSTRAINTS_FILE" \
  pre-commit ruff mypy pytest pytest-cov

# Install pre-commit hooks (skip in CI - only needed for local dev)
if [[ -z "${CI:-}" ]]; then
  echo "[wt_bootstrap] Installing pre-commit hooks..."
  "$SCRIPT_DIR/wt_run.sh" pre-commit install
  "$SCRIPT_DIR/wt_run.sh" pre-commit install --hook-type pre-push
fi

echo
echo "[wt_bootstrap] Done. Next steps:"
echo "  1) Activate (optional):"
echo "     source .venv/bin/activate"
echo "  2) Run commands through enforcement wrapper:"
echo "     ./scripts/wt_run.sh pytest tests -q"
echo "     ./scripts/wt_run.sh pre-commit run --all-files"

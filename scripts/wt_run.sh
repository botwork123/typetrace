#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$REPO_ROOT/.venv"
VENV_PY="$VENV_DIR/bin/python"

if [[ $# -eq 0 ]]; then
  echo "usage: $0 <command> [args...]" >&2
  exit 2
fi

if [[ ! -x "$VENV_PY" ]]; then
  echo "[wt_run] Missing worktree-local venv at $VENV_PY" >&2
  echo "[wt_run] Run: ./scripts/wt_bootstrap.sh" >&2
  exit 1
fi

export VIRTUAL_ENV="$VENV_DIR"
export PATH="$VENV_DIR/bin:$PATH"

ACTIVE_PY="$(command -v python)"
ACTIVE_PY_ABS="$(python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$ACTIVE_PY")"
VENV_PY_ABS="$(python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$VENV_PY")"

if [[ "$ACTIVE_PY_ABS" != "$VENV_PY_ABS" ]]; then
  echo "[wt_run] Active python is not this worktree venv python" >&2
  echo "[wt_run] active: $ACTIVE_PY_ABS" >&2
  echo "[wt_run] expect: $VENV_PY_ABS" >&2
  echo "[wt_run] Run: ./scripts/wt_bootstrap.sh" >&2
  exit 1
fi

if ! "$VENV_PY" - "$REPO_ROOT" "$VENV_PY_ABS" "$VENV_DIR" <<'PY'
import importlib.util
import pathlib
import sys

repo_root = pathlib.Path(sys.argv[1]).resolve()
expected_python = pathlib.Path(sys.argv[2])
expected_venv_dir = pathlib.Path(sys.argv[3]).resolve()

if pathlib.Path(sys.executable) != expected_python:
    print("[wt_run] Interpreter drift detected", file=sys.stderr)
    print(f"[wt_run] sys.executable: {pathlib.Path(sys.executable).resolve()}", file=sys.stderr)
    print(f"[wt_run] expected: {expected_python}", file=sys.stderr)
    print("[wt_run] Run: ./scripts/wt_bootstrap.sh", file=sys.stderr)
    raise SystemExit(1)

if pathlib.Path(sys.prefix).resolve() != expected_venv_dir:
    print("[wt_run] Virtualenv prefix drift detected", file=sys.stderr)
    print(f"[wt_run] sys.prefix: {pathlib.Path(sys.prefix).resolve()}", file=sys.stderr)
    print(f"[wt_run] expected venv: {expected_venv_dir}", file=sys.stderr)
    print("[wt_run] Run: ./scripts/wt_bootstrap.sh", file=sys.stderr)
    raise SystemExit(1)

spec = importlib.util.find_spec("typetrace")
if spec is None:
    print("[wt_run] Unable to resolve typetrace import", file=sys.stderr)
    raise SystemExit(1)

origin_path: pathlib.Path | None = None
if spec.origin and spec.origin not in {"built-in", "frozen"}:
    origin_path = pathlib.Path(spec.origin).resolve()
elif spec.submodule_search_locations:
    origin_path = pathlib.Path(next(iter(spec.submodule_search_locations))).resolve() / "__init__.py"

if origin_path is None:
    print("[wt_run] Could not determine typetrace origin path", file=sys.stderr)
    raise SystemExit(1)

if not str(origin_path).startswith(str(repo_root) + "/"):
    print("[wt_run] typetrace import is outside this worktree", file=sys.stderr)
    print(f"[wt_run] typetrace.__file__: {origin_path}", file=sys.stderr)
    print(f"[wt_run] worktree root: {repo_root}", file=sys.stderr)
    raise SystemExit(1)
PY
then
  exit 1
fi

exec "$@"

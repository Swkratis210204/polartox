# Sourced by the other scripts: repo root, interpreter, and the helpers they share.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"          # override with: PYTHON=/path/to/python scripts/dices.sh
export PYTHONUTF8=1 PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

step() { printf '\n==== [%s] %s ====\n' "$(date +%H:%M:%S)" "$*"; }

# run_nb <notebook> : execute it in place; stop the script if any cell fails.
run_nb() {
  step "notebook: $1"
  "$PYTHON" scripts/run_notebook.py "$1"
}

# run_tests : the package test-suite.
run_tests() {
  step "tests"
  "$PYTHON" -m pytest -q
}

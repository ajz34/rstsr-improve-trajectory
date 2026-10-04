#!/usr/bin/env bash
#
# One-time setup for "grade NumPy against the Array API standard".
#
# What it produces:
#   array-api-tests/   the official conformance suite (data-apis/array-api-tests),
#                      pinned to a known commit, with the spec submodule checked out
#   .venv/             a venv that *overlays* the torch conda env: test-only packages
#                      (pytest, hypothesis, ndindex) live here, while numpy is
#                      imported from /home/a/miniconda3/envs/torch — no numpy build,
#                      no dev/ nightlies, nothing installed into the torch env itself.
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------- knobs ----
TORCH_PY="${TORCH_PY:-/home/a/miniconda3/envs/torch/bin/python}"
SUITE_REPO="${SUITE_REPO:-https://github.com/data-apis/array-api-tests.git}"
# Pinned so a later upstream change can't silently change the grade.
# (Same commit the notes' TESTING-ARRAY-API.md was written against.)
SUITE_COMMIT="${SUITE_COMMIT:-6c0b59f9ecd654f0c356614f2e421c9993f676d2}"
SUITE_DIR="$HERE/array-api-tests"
VENV="$HERE/.venv"

# ------------------------------------------------- 1. the numpy under test --
echo "==> numpy under test"
"$TORCH_PY" - <<'PY'
import numpy, sys
print(f"    python : {sys.executable}")
print(f"    numpy  : {numpy.__version__}")
print(f"    path   : {numpy.__file__}")
print(f"    claims : Array API {getattr(numpy, '__array_api_version__', '<no version declared>')}")
PY

# ------------------------------------------------- 2. the official suite ----
echo "==> conformance suite"
if [ ! -d "$SUITE_DIR/.git" ]; then
    git clone --quiet "$SUITE_REPO" "$SUITE_DIR"
fi
git -C "$SUITE_DIR" checkout --quiet "$SUITE_COMMIT"
# The spec repo (data-apis/array-api) is a submodule and is used as *executable
# test data*: stubs.py imports it, test_special_cases parses its docstrings.
# The suite asserts if it is missing, so this step is mandatory.
git -C "$SUITE_DIR" submodule update --init --quiet
echo "    suite : $(git -C "$SUITE_DIR" rev-parse --short HEAD)  ($(git -C "$SUITE_DIR" log -1 --format=%cs))"
echo "    spec  : $(git -C "$SUITE_DIR/array-api" rev-parse --short HEAD)"

# ------------------------------------------------- 3. venv overlay ----------
# --system-site-packages = the venv inherits the torch env's numpy, scipy, torch...
echo "==> venv overlay (test deps) -> $VENV"
if [ ! -x "$VENV/bin/python" ]; then
    uv venv --python "$TORCH_PY" --system-site-packages --quiet "$VENV"
fi
uv pip install --python "$VENV/bin/python" --quiet -r "$SUITE_DIR/requirements.txt"

# ------------------------------------------------- 4. prove it --------------
echo "==> check: venv sees exactly the torch env numpy"
"$VENV/bin/python" - <<'PY'
import numpy, sys
assert "miniconda3/envs/torch" in numpy.__file__, numpy.__file__
print(f"    ok: numpy {numpy.__version__} from {numpy.__file__}")
PY

echo
echo "Done.  Next:  ./run.sh            # full suite, 100 hypothesis examples/test"
echo "              ./run.sh --max-examples 20   # quick pass"

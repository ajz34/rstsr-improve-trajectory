#!/usr/bin/env bash
#
# One-time setup for "grade rstsr (via rstsr-faer-py) against the Array API
# standard" — S0 of DECISIONS.md.
#
# Differences from ../2026-10-04-arrayapi-compliance-notes/numpy-compliance/:
#   - the suite checkout lives in ~/Git-Others/array-api-tests (shared, not
#     cloned per task dir — user rule: repos go to ~/Git-Others)
#   - .venv overlay and reports/ live HERE (generated content, gitignored)
#
# What it produces:
#   ~/Git-Others/array-api-tests   suite @ pin 6c0b59f, spec submodule @ 5f847a3
#   conda env torch                test-only packages installed directly
#                                  (pytest, pytest-json-report, hypothesis,
#                                  ndindex) — no venv (user instruction);
#                                  numpy 2.5.1 already lives there, and the
#                                  rstsr_faer wheel is later installed the
#                                  same way
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------- knobs ----
TORCH_PY="${TORCH_PY:-/home/a/miniconda3/envs/torch/bin/python}"
SUITE_DIR="${SUITE_DIR:-$HOME/Git-Others/array-api-tests}"
SUITE_COMMIT="${SUITE_COMMIT:-6c0b59f9ecd654f0c356614f2e421c9993f676d2}"

# ------------------------------------------------- 1. the suite checkout ----
echo "==> conformance suite ($SUITE_DIR)"
if [ ! -d "$SUITE_DIR/.git" ]; then
    git clone --quiet https://github.com/data-apis/array-api-tests.git "$SUITE_DIR"
fi
git -C "$SUITE_DIR" checkout --quiet "$SUITE_COMMIT"
# Spec repo is executable test data (stubs.py imports it; test_special_cases
# parses its docstrings) — the suite asserts its presence at import time.
# Offline-first: the sibling clone ~/Git-Others/array-api contains the pinned
# commit; point the submodule at it (needs protocol.file.allow, git's CVE
# hardening blocks local-path submodule clones by default). The submodule's
# NAME differs from its path (name: array_api_tests/array-api) — resolve it
# from .gitmodules instead of assuming. SSH fallback if the local clone were
# missing the pin.
SPEC_LOCAL="${SPEC_LOCAL:-$HOME/Git-Others/array-api}"
SPEC_COMMIT="5f847a3858875c682ae901aa22b0413bf24be9da"
if ! git -C "$SUITE_DIR" submodule status 2>/dev/null | grep -q '^ '; then
    SUBMOD_NAME="$(git -C "$SUITE_DIR" config -f .gitmodules --get-regexp '\.path=' \
        | awk -F'= ' -v p="array-api" '$2==p {k=$1; sub(/\.path$/, "", k); sub(/^submodule\./, "", k); print k}')"
    if [ -d "$SPEC_LOCAL/.git" ] && git -C "$SPEC_LOCAL" cat-file -e "$SPEC_COMMIT" 2>/dev/null; then
        echo "    submodule: initializing offline from $SPEC_LOCAL"
        git -C "$SUITE_DIR" config "submodule.$SUBMOD_NAME.url" "$SPEC_LOCAL"
        git -C "$SUITE_DIR" -c protocol.file.allow=always submodule update --init --quiet
    elif ! git -C "$SUITE_DIR" submodule update --init --quiet 2>/dev/null; then
        echo "    submodule: local clone unusable; fetching via ssh"
        git -C "$SUITE_DIR" config url."git@github.com:".insteadOf "https://github.com/"
        git -C "$SUITE_DIR" submodule update --init --quiet
    fi
else
    git -C "$SUITE_DIR" submodule update --init --quiet
fi
echo "    suite : $(git -C "$SUITE_DIR" rev-parse --short HEAD)  ($(git -C "$SUITE_DIR" log -1 --format=%cs))"
echo "    spec  : $(git -C "$SUITE_DIR/array-api" rev-parse --short HEAD)"

# ------------------------------------------------- 2. test deps -------------
# Straight into the torch env (user instruction): no venv. uv does not touch
# numpy itself unless requirements demand it — the suite's requirements.txt
# lists only pytest, pytest-json-report, hypothesis, ndindex.
echo "==> test deps -> torch env"
uv pip install --python "$TORCH_PY" --quiet -r "$SUITE_DIR/requirements.txt"

# ------------------------------------------------- 3. prove it --------------
echo "==> check: interpreter + numpy"
"$TORCH_PY" - <<'PY'
import sys
import numpy, pytest, hypothesis, ndindex
print(f"    python    : {sys.executable}")
print(f"    numpy     : {numpy.__version__}")
print(f"    pytest    : {pytest.__version__}")
print(f"    hypothesis: {hypothesis.__version__}")
print(f"    ndindex   : {ndindex.__version__}")
PY

echo
echo "Done.  Next:"
echo "    MODULE=numpy ./run.sh                # S0 gate: reproduce the NumPy baseline"
echo "    MODULE=numpy ./run.sh array_api_tests/test_has_names.py   # surface check only"

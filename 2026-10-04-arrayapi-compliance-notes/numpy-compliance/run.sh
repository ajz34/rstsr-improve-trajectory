#!/usr/bin/env bash
#
# Grade a library against the official Array API conformance suite.
#
#   ./run.sh                          # whole suite, suite-default examples (100/test)
#   ./run.sh --max-examples 20        # quick pass
#   ./run.sh array_api_tests/test_has_names.py          # one module
#   MODULE=array_api_strict ./run.sh  # grade something else (control run)
#
# Anything after the script name is passed straight to pytest.
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUITE_DIR="$HERE/array-api-tests"
VENV="$HERE/.venv"
REPORTS="$HERE/reports"

MODULE="${MODULE:-numpy}"
mkdir -p "$REPORTS"

# Hypothesis keeps a persistent example database (.hypothesis/, gitignored).
# Each run replays and extends it, so a second run can surface a few failures
# the first missed.  FRESH=1 deletes it for a canonical, from-scratch number.
if [ -n "${FRESH:-}" ]; then
    rm -rf "$SUITE_DIR/.hypothesis"
fi

if [ ! -d "$SUITE_DIR" ] || [ ! -x "$VENV/bin/python" ]; then
    echo "run ./setup.sh first" >&2
    exit 1
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
REPORT="$REPORTS/${MODULE//./_}-$STAMP.json"

# Default target: the whole suite.  But if the caller named a real path
# (e.g. `./run.sh array_api_tests/test_has_names.py`), that replaces it, and
# any other flags after it are passed through to pytest.
PYTEST_ARGS=(array_api_tests/)
for arg in "$@"; do
    if [ "${arg#-}" = "$arg" ] && [ -e "$SUITE_DIR/$arg" -o -e "$arg" ]; then
        PYTEST_ARGS=()
        break
    fi
done
if [ -n "${MAX_EXAMPLES:-}" ]; then
    PYTEST_ARGS+=(--max-examples "$MAX_EXAMPLES")
fi
PYTEST_ARGS+=(
    --disable-deadline          # the suite's per-example 800ms deadline is for CI only
    --hypothesis-derandomize    # fixed seeds (counts still wobble +/-2; see README)
    --json-report --json-report-file "$REPORT"
    -q -p no:cacheprovider
)
PYTEST_ARGS+=("$@")

ENV_VARS=("ARRAY_API_TESTS_MODULE=$MODULE")
if [ -n "${API_VERSION:-}" ]; then
    # Default: whatever the library declares via __array_api_version__,
    # else the suite's own default. Set to e.g. 2023.12 to grade an older spec.
    ENV_VARS+=("ARRAY_API_TESTS_VERSION=$API_VERSION")
fi

# A small sidecar so the report is self-describing (pytest-json-report's own
# metadata hook comes out empty here): what library/version/API version, which
# suite commit, how many examples.
MAX_EX="${MAX_EXAMPLES:-}"
expect_value=0
for arg in "$@"; do
    if [ "$expect_value" = 1 ]; then MAX_EX="$arg"; expect_value=0; continue; fi
    case "$arg" in
        --max-examples|--hypothesis-max-examples) expect_value=1 ;;
        --max-examples=*|--hypothesis-max-examples=*) MAX_EX="${arg#*=}" ;;
    esac
done

M="$MODULE" C="$(git -C "$SUITE_DIR" rev-parse --short HEAD)" N="$MAX_EX" \
"$VENV/bin/python" -c '
import importlib, json, os, sys
mod = importlib.import_module(os.environ["M"])
json.dump({
    "module": os.environ["M"],
    "version": getattr(mod, "__version__", "?"),
    "api_version": getattr(mod, "__array_api_version__", "(suite default)"),
    "suite_commit": os.environ["C"],
    "max_examples": os.environ["N"] or "(suite default)",
}, open(sys.argv[1], "w"), indent=2)
' "${REPORT}.meta.json"

echo "==> $MODULE against array-api-tests @ $(git -C "$SUITE_DIR" rev-parse --short HEAD)"
echo "    report: $REPORT"
echo

cd "$SUITE_DIR"
START=$SECONDS
env "${ENV_VARS[@]}" "$VENV/bin/python" -m pytest "${PYTEST_ARGS[@]}" || true
ELAPSED=$((SECONDS - START))

echo
echo "==> finished in $((ELAPSED / 60))m $((ELAPSED % 60))s"
"$VENV/bin/python" "$HERE/summarize_report.py" "$REPORT" || true

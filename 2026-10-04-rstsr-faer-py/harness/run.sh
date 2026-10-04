#!/usr/bin/env bash
#
# Grade a library against the official Array API conformance suite.
#
#   MODULE=numpy ./run.sh                 # whole suite, suite-default examples
#   MODULE=numpy ./run.sh --max-examples 20
#   MODULE=rstsr_faer.api ./run.sh        # the actual subject (S1+)
#   ./run.sh array_api_tests/test_has_names.py   # one module (MODULE still applies)
#
# Anything after the script name is passed straight to pytest.
# Raw reports land in reports/ (gitignored); summaries are committed by hand.
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUITE_DIR="${SUITE_DIR:-$HOME/Git-Others/array-api-tests}"
TORCH_PY="${TORCH_PY:-/home/a/miniconda3/envs/torch/bin/python}"
REPORTS="$HERE/reports"

MODULE="${MODULE:-numpy}"
mkdir -p "$REPORTS"

# Hypothesis keeps a persistent example database in $SUITE_DIR/.hypothesis.
# Each run replays and extends it; FRESH=1 deletes it for a canonical number.
if [ -n "${FRESH:-}" ]; then
    rm -rf "$SUITE_DIR/.hypothesis"
fi

if [ ! -d "$SUITE_DIR/array_api_tests" ]; then
    echo "run ./setup.sh first" >&2
    exit 1
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
REPORT="$REPORTS/${MODULE//./_}-$STAMP.json"

# Default target: the whole suite. A named existing path replaces it.
PYTEST_ARGS=(array_api_tests/)
for arg in "$@"; do
    if [ "${arg#-}" = "$arg" ] && { [ -e "$SUITE_DIR/$arg" ] || [ -e "$arg" ]; }; then
        PYTEST_ARGS=()
        break
    fi
done
if [ -n "${MAX_EXAMPLES:-}" ]; then
    PYTEST_ARGS+=(--max-examples "$MAX_EXAMPLES")
fi
PYTEST_ARGS+=(
    --disable-deadline          # the suite's 800ms/example deadline is for CI only
    --hypothesis-derandomize    # fixed seeds (counts still wobble +/-2)
    --json-report --json-report-file "$REPORT"
    -q -p no:cacheprovider
)
PYTEST_ARGS+=("$@")

ENV_VARS=("ARRAY_API_TESTS_MODULE=$MODULE")
if [ -n "${API_VERSION:-}" ]; then
    ENV_VARS+=("ARRAY_API_TESTS_VERSION=$API_VERSION")
fi
# rstsr-faer-py gap machinery (S1+): skip/xfail files live HERE, not upstream.
if [ -n "${SKIPS_FILE:-}" ]; then
    PYTEST_ARGS+=("--skips-file" "$SKIPS_FILE")
fi
if [ -n "${XFAILS_FILE:-}" ]; then
    PYTEST_ARGS+=("--xfails-file" "$XFAILS_FILE")
fi

# Self-describing sidecar (pytest-json-report's own metadata hook is empty here).
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
"$TORCH_PY" -c '
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
env "${ENV_VARS[@]}" "$TORCH_PY" -m pytest "${PYTEST_ARGS[@]}" || true
ELAPSED=$((SECONDS - START))

echo
echo "==> finished in $((ELAPSED / 60))m $((ELAPSED % 60))s"
"$TORCH_PY" "$HERE/summarize_report.py" "$REPORT" || true

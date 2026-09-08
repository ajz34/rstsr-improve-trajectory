#!/usr/bin/env bash
# T6 `argmax-argmin` — reproduction script.
#
# PHASE 1 (current): baseline on clean rstsr 386948be + correctness gate +
# perf. Stages: correctness | portable | native | perf | numpy | all | baseline
#
# PHASE 2 (after G1): reruns the identical suite against the patched ../rstsr
# tree with `--save-baseline candidate` (stage placeholder `candidate`, marked
# TODO-PHASE2 below) and then produces proposed.patch.
#
# Machine: AMD Ryzen 9 9950X3D (Zen 5, AVX-512), 16 cores. rustc 1.97.1
# nightly. cargo release profile = stock defaults (see Cargo.toml).
#
# Run from this directory:  ./reproduce.sh            (everything)
#                     or    ./reproduce.sh <stage>
set -euo pipefail

cd "$(dirname "$0")"
RESULTS="$(pwd)/results"
mkdir -p "$RESULTS"
export RAYON_NUM_THREADS=16          # campaign convention: 16 physical cores
export CARGO_TERM_COLOR=never

STAGE="${1:-all}"

# ---------------------------------------------------------------------------
# Config matrix (plan D7):
#   portable: no RUSTFLAGS (x86-64 SSE2 baseline)
#   native:   RUSTFLAGS="-C target-cpu=native"
# ---------------------------------------------------------------------------

run_benches() {
    # $1: tag (portable|native|candidate)  $2: RUSTFLAGS ("" allowed)
    # benches saved as criterion baseline "$tag"; raw report tee'd to results/
    local tag="$1" flags="$2"
    local tagdir="$RESULTS/$tag"
    mkdir -p "$tagdir"
    if [ -n "$flags" ]; then
        export RUSTFLAGS="$flags"
    else
        unset RUSTFLAGS
    fi
    local bench
    for bench in arg anchors_ndarray; do
        echo "=== [$tag] bench: $bench"
        # --save-baseline stores full criterion data under target/criterion;
        # tee keeps the human-readable report in results/.
        cargo bench --bench "$bench" -- --save-baseline "$tag" 2>&1 \
            | tee "$tagdir/bench_${bench}.txt" \
            | grep -E "bench:|time:|change:|Performance|regress|improve|No change" || true
    done
}

stage_correctness() {
    echo "=== correctness gate (portable config) ==="
    unset RUSTFLAGS
    cargo run --release --example correctness 2>&1 | tee "$RESULTS/correctness_portable.txt" | grep -E "FAIL|PASSED"
    echo "=== correctness gate (native config) ==="
    RUSTFLAGS="-C target-cpu=native" cargo run --release --example correctness 2>&1 \
        | tee "$RESULTS/correctness_native.txt" | grep -E "FAIL|PASSED"
}

stage_portable() {
    run_benches portable ""
}

stage_native() {
    run_benches native "-C target-cpu=native"
}

# PHASE 2 stage: with the ../rstsr working tree PATCHED (proposed.patch
# applied), rerun the identical gate + suite as `candidate` criterion
# baselines and re-run perf. Verify with: git -C ../../rstsr status --short
# (4 modified files expected; feature_rayon/auto_impl/reduction.rs is the
# symlink target of device_faer/rayon_auto_impl/reduction.rs).
stage_candidate() {
    echo "=== [candidate] gate (portable) ==="
    unset RUSTFLAGS
    cargo run --release --example correctness 2>&1 | tee "$RESULTS/correctness_portable.txt" | grep -E "FAIL|PASSED"
    echo "=== [candidate] gate (native) ==="
    RUSTFLAGS="-C target-cpu=native" cargo run --release --example correctness 2>&1 \
        | tee "$RESULTS/correctness_native.txt" | grep -E "FAIL|PASSED"
    run_benches candidate ""
    run_benches candidate "-C target-cpu=native"
    mkdir -p "$RESULTS/perf"
    export RUSTFLAGS="-C target-cpu=native"
    cargo build --release --example profile_ops
    local bin="./target/release/examples/profile_ops"
    for op in argmax argmin argmax_1e6; do
        perf stat -d -o "$RESULTS/perf/perf_${op}_candidate.txt" -- "$bin" "$op"
    done
    python3 "$RESULTS/make_tables.py"
}

stage_perf() {
    mkdir -p "$RESULTS/perf"
    export RUSTFLAGS="-C target-cpu=native"
    cargo build --release --example profile_ops
    local bin="./target/release/examples/profile_ops"
    for op in argmax argmin argmax_1e6; do
        echo "=== perf stat -d : $op ==="
        perf stat -d -o "$RESULTS/perf/perf_${op}.txt" -- "$bin" "$op"
    done
    python3 "$RESULTS/make_tables.py"
}

stage_numpy() {
    # optional context anchor (numpy's argmax is L3-assisted here — context only)
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate torch
    python numpy_ref/anchors.py | tee numpy_ref/results.txt
}

case "$STAGE" in
    all)
        stage_correctness
        stage_portable
        stage_native
        stage_perf
        ;;
    correctness) stage_correctness ;;
    portable)    stage_portable ;;
    native)      stage_native ;;
    perf)        stage_perf ;;
    numpy)       stage_numpy ;;
    candidate)   stage_candidate ;;
    baseline)    stage_correctness && stage_portable && stage_native && stage_perf ;;
    *) echo "unknown stage: $STAGE (use correctness|portable|native|perf|numpy|candidate|baseline|all)"; exit 2 ;;
esac

echo "reproduce.sh: done (stage=$STAGE)"

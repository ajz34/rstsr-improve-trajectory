#!/usr/bin/env bash
# T0 `bench-harness-baseline` — full reproduction script.
#
# rstsr base commit: 386948be819baa334b8da02232f3a1944e5447d5 (path dep,
# READ-ONLY; T0 is a no-patch task).
#
# Machine: AMD Ryzen 9 9950X3D (Zen 5, AVX-512), 16 cores. rustc 1.97.1
# nightly (2026-07-14). cargo release profile = stock defaults.
#
# Run from this directory:  ./reproduce.sh            (everything)
#                     or    ./reproduce.sh <stage>    (one stage)
# Stages: correctness portable native d8 perf numpy
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
# D8 secondary column: native + --features dispatch_dim_layout_iter,
# large|odd cases only.
# ---------------------------------------------------------------------------

run_benches() {
    # $1: tag (portable|native|native_d8)  $2: features ("" allowed)  $3: filter ("" allowed)
    # benches saved as criterion baseline "$tag"
    local tag="$1" feats="$2" filter="$3"
    local tagdir="$RESULTS/$tag"
    mkdir -p "$tagdir"
    local bench
    for bench in triad transpose reduce vecdot elementwise fill_argmax anchors_ndarray; do
        echo "=== [$tag] bench: $bench  feats='$feats' filter='$filter'"
        # --save-baseline stores full criterion data under target/criterion;
        # tee keeps the human-readable report in results/.
        local cmd=(cargo bench --bench "$bench")
        [ -n "$feats" ]  && cmd+=(--features "$feats")
        [ -n "$filter" ] && cmd+=(-- "$filter" --save-baseline "$tag") || cmd+=(-- --save-baseline "$tag")
        "${cmd[@]}" 2>&1 | tee "$tagdir/bench_${bench}.txt" | grep -E "bench:|time:|change:|Performance|regress|improve|No change" || true
    done
}

stage_correctness() {
    echo "=== correctness gate (portable config) ==="
    cargo run --release --example correctness 2>&1 | tee "$RESULTS/correctness_portable.txt"
    echo "=== correctness gate (native config) ==="
    RUSTFLAGS="-C target-cpu=native" cargo run --release --example correctness 2>&1 | tee "$RESULTS/correctness_native.txt"
}

stage_portable() {
    unset RUSTFLAGS
    cargo bench --no-run
    run_benches portable "" ""
}

stage_native() {
    export RUSTFLAGS="-C target-cpu=native"
    cargo bench --no-run
    run_benches native "" ""
}

stage_d8() {
    export RUSTFLAGS="-C target-cpu=native"
    cargo bench --no-run --features dispatch_dim_layout_iter
    run_benches native_d8 dispatch_dim_layout_iter "large|odd"
}

stage_perf() {
    mkdir -p "$RESULTS/perf"
    export RUSTFLAGS="-C target-cpu=native"
    cargo build --release --example profile_ops
    local bin="./target/release/examples/profile_ops"
    for op in triad transpose sum_axis0 sum_axis1 vecdot add_contig argmax zeros; do
        echo "=== perf stat -d : $op ==="
        perf stat -e task-clock,cycles,instructions,branches,branch-misses,L1-dcache-loads,L1-dcache-load-misses,cache-references,cache-misses,page-faults -o "$RESULTS/perf/perf_${op}.txt" -- "$bin" "$op"
    done
}

stage_numpy() {
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate torch
    python numpy_ref/anchors.py | tee numpy_ref/results.txt
}

case "$STAGE" in
    all)
        stage_correctness
        stage_portable
        stage_native
        stage_d8
        stage_perf
        stage_numpy
        ;;
    correctness) stage_correctness ;;
    portable)    stage_portable ;;
    native)      stage_native ;;
    d8)          stage_d8 ;;
    perf)        stage_perf ;;
    numpy)       stage_numpy ;;
    *) echo "unknown stage: $STAGE (use correctness|portable|native|d8|perf|numpy)"; exit 2 ;;
esac

echo "reproduce.sh: done (stage=$STAGE)"

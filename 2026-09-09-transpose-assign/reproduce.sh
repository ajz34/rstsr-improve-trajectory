#!/usr/bin/env bash
# T1' `transpose-assign` — baseline reproduction script (PHASE 1).
#
# rstsr base commit: 386948be819baa334b8da02232f3a1944e5447d5 (path dep,
# READ-ONLY in phase 1; verified clean + HEAD 386948b before baselining).
#
# Machine: AMD Ryzen 9 9950X3D (Zen 5, AVX-512), 16 cores. rustc 1.97.1
# nightly (2026-07-14). cargo release profile = stock defaults.
#
# Run from this directory:  ./reproduce.sh            (everything)
#                     or    ./reproduce.sh <stage>    (one stage)
# Stages: correctness portable native portable_c native_c perf candidate
#
# Variant C is an env re-run of the A-filtered suite under the T7 glibc
# tunables (MALLOC_MMAP_THRESHOLD_=67108864 MALLOC_TRIM_THRESHOLD_=134217728).
#
# `candidate` is the PHASE-2 placeholder: after the planned rstsr edit
# (routing 2-D order-change through the blocked kernel — see PLAN.md),
# re-run `correctness portable native` and save baselines as `candidate`.
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
    # $1: tag  $2: extra env (array-ish string, "" allowed)  $3: criterion filter ("" allowed)
    local tag="$1" extra_env="$2" filter="$3"
    local tagdir="$RESULTS/$tag"
    mkdir -p "$tagdir"
    local bench
    for bench in transpose anchors_ndarray kernels_probe; do
        echo "=== [$tag] bench: $bench  filter='$filter'"
        local cmd=(cargo bench --bench "$bench")
        [ -n "$filter" ] && cmd+=(-- "$filter" --save-baseline "$tag") || cmd+=(-- --save-baseline "$tag")
        if [ -n "$extra_env" ]; then
            env $extra_env "${cmd[@]}" 2>&1 | tee "$tagdir/bench_${bench}.txt" | grep -E "bench:|time:|change:|Performance|regress|improve|No change" || true
        else
            "${cmd[@]}" 2>&1 | tee "$tagdir/bench_${bench}.txt" | grep -E "bench:|time:|change:|Performance|regress|improve|No change" || true
        fi
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

stage_portable_c() {
    unset RUSTFLAGS
    cargo bench --no-run
    # Variant C: allocating idiom under the T7 glibc tunables (filter "A")
    run_benches portable_c "MALLOC_MMAP_THRESHOLD_=67108864 MALLOC_TRIM_THRESHOLD_=134217728" "A"
}

stage_native_c() {
    export RUSTFLAGS="-C target-cpu=native"
    cargo bench --no-run
    run_benches native_c "MALLOC_MMAP_THRESHOLD_=67108864 MALLOC_TRIM_THRESHOLD_=134217728" "A"
}

stage_perf() {
    mkdir -p "$RESULTS/perf"
    export RUSTFLAGS="-C target-cpu=native"
    cargo build --release --example profile_ops
    local bin="./target/release/examples/profile_ops"
    local suffix="${2:-}"
    for op in tb_serial tb_faer16 ta_serial; do
        echo "=== perf stat -d : $op ==="
        perf stat -d -o "$RESULTS/perf/perf_${op}${suffix}.txt" -- "$bin" "$op"
    done
}

# PHASE-2 candidate stage: requires the planned rstsr edit APPLIED in the
# working tree (path dep picks it up). Runs the full gate + suite + perf,
# archiving everything under results/candidate_*.
stage_candidate() {
    echo "=== candidate correctness (both configs; aborts on failure) ==="
    unset RUSTFLAGS
    cargo run --release --example correctness 2>&1 | tee "$RESULTS/candidate_correctness_portable.txt"
    RUSTFLAGS="-C target-cpu=native" cargo run --release --example correctness 2>&1 | tee "$RESULTS/candidate_correctness_native.txt"
    echo "=== candidate benches ==="
    cargo bench --no-run
    run_benches candidate_portable "" ""
    export RUSTFLAGS="-C target-cpu=native"
    cargo bench --no-run
    run_benches candidate_native "" ""
    run_benches candidate_portable_c "MALLOC_MMAP_THRESHOLD_=67108864 MALLOC_TRIM_THRESHOLD_=134217728" "A"
    cargo bench --no-run
    run_benches candidate_native_c "MALLOC_MMAP_THRESHOLD_=67108864 MALLOC_TRIM_THRESHOLD_=134217728" "A"
    echo "=== candidate perf (after) ==="
    stage_perf "" "_after"
}

case "$STAGE" in
    all)
        stage_correctness
        stage_portable
        stage_native
        stage_portable_c
        stage_native_c
        stage_perf
        ;;
    correctness) stage_correctness ;;
    portable)    stage_portable ;;
    native)      stage_native ;;
    portable_c)  stage_portable_c ;;
    native_c)    stage_native_c ;;
    perf)        stage_perf ;;
    candidate)   stage_candidate ;;
    *) echo "unknown stage: $STAGE (use correctness|portable|native|portable_c|native_c|perf|candidate)"; exit 2 ;;
esac

echo "reproduce.sh: done (stage=$STAGE)"

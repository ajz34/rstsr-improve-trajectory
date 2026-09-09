#!/usr/bin/env bash
# T5 `fill-misc-structural` (PHASE 1) — reproduction script.
#
# rstsr base commit: 386948be819baa334b8da02232f3a1944e5447d5 (path dep,
# READ-ONLY in phase 1; verified clean at start and end).
#
# Machine: AMD Ryzen 9 9950X3D (Zen 5, AVX-512), 16 cores.
# rustc: nightly via rstsr's rust-toolchain.toml (record the exact version in
# README.md — the toolchain drifted from earlier campaign tasks, see README).
# Cargo release profile = stock defaults.
#
# Run from this directory:  ./reproduce.sh            (everything)
#                     or    ./reproduce.sh <stage>    (one stage)
# Stages: correctness portable native perf tunables
set -euo pipefail

cd "$(dirname "$0")"
RESULTS="$(pwd)/results"
mkdir -p "$RESULTS"
export RAYON_NUM_THREADS=16
export CARGO_TERM_COLOR=never

STAGE="${1:-all}"

stage_correctness() {
    echo "=== correctness gate (portable) ==="
    cargo run --release --example correctness 2>&1 | tee "$RESULTS/correctness_portable.txt"
    echo "=== correctness gate (native) ==="
    RUSTFLAGS="-C target-cpu=native" cargo run --release --example correctness 2>&1 | tee "$RESULTS/correctness_native.txt"
}

run_benches() {
    # $1: tag (portable|native)
    local tag="$1"
    local tagdir="$RESULTS/$tag"
    mkdir -p "$tagdir"
    echo "=== [$tag] bench: fill"
    if [ "$tag" = "native" ]; then
        RUSTFLAGS="-C target-cpu=native" cargo bench --bench fill -- --save-baseline "$tag" 2>&1 | tee "$tagdir/bench_fill.txt" | grep -E "time:|change:|regress|improve|No change" || true
    else
        cargo bench --bench fill -- --save-baseline "$tag" 2>&1 | tee "$tagdir/bench_fill.txt" | grep -E "time:|change:|regress|improve|No change" || true
    fi
}

stage_perf() {
    # perf stat on the kernel-vs-rider split (native build), serial device
    local tagdir="$RESULTS/perf"
    mkdir -p "$tagdir"
    RUSTFLAGS="-C target-cpu=native" cargo build --release --examples
    for mode in full_a ones_a zeros_c fill_b fill_b_faer fill_bound fill_bt; do
        echo "=== perf stat: $mode (native) ==="
        if [ "$mode" = "fill_b_faer" ]; then
            RAYON_NUM_THREADS=16 perf stat -d -o "$tagdir/perf_${mode}.txt" -- "./target/release/examples/profile_ops" "$mode" 200
        else
            perf stat -d -o "$tagdir/perf_${mode}.txt" -- "./target/release/examples/profile_ops" "$mode" 200
        fi
        cat "$tagdir/perf_${mode}.txt"
    done
    # size-regime runs for the zeros-laziness nuance (500 iters each)
    for mode in zeros_med zeros_odd full_med full_odd; do
        echo "=== perf stat: $mode (native, 500 iters) ==="
        perf stat -x ';' -e task-clock,page-faults,context-switches,cpu-cycles,instructions \
            -- "./target/release/examples/profile_ops" "$mode" 500 2>&1 | tee "$tagdir/perf_${mode}.txt" | grep -v WARNING || true
    done
}

stage_tunables() {
    # rt::full under the T7 glibc env tunables — rider removal for fill
    local tagdir="$RESULTS/tunables"
    mkdir -p "$tagdir"
    RUSTFLAGS="-C target-cpu=native" cargo build --release --examples
    echo "=== full_a default env (native) ==="
    perf stat -o "$tagdir/full_a_default.txt" -- "./target/release/examples/profile_ops" full_a 200
    cat "$tagdir/full_a_default.txt"
    echo "=== full_a MALLOC_MMAP_THRESHOLD_=67108864 MALLOC_TRIM_THRESHOLD_=134217728 ==="
    MALLOC_MMAP_THRESHOLD_=67108864 MALLOC_TRIM_THRESHOLD_=134217728 \
        perf stat -o "$tagdir/full_a_tunables.txt" -- "./target/release/examples/profile_ops" full_a 200
    cat "$tagdir/full_a_tunables.txt"
}

case "$STAGE" in
    correctness) stage_correctness ;;
    portable)    run_benches portable ;;
    native)      run_benches native ;;
    perf)        stage_perf ;;
    tunables)    stage_tunables ;;
    all)
        stage_correctness
        run_benches portable
        run_benches native
        stage_perf
        stage_tunables
        ;;
    *) echo "unknown stage $STAGE"; exit 2 ;;
esac

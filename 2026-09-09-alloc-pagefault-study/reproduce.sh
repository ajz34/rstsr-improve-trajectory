#!/usr/bin/env bash
# T7 `alloc-pagefault-study` — full reproduction script.
#
# rstsr base commit: 386948be819baa334b8da02232f3a1944e5447d5 (path dep,
# READ-ONLY; measurement-only task — no changes to the rstsr tree).
#
# Machine: AMD Ryzen 9 9950X3D (Zen 5, AVX-512), 16 cores. rustc 1.97.1
# nightly. cargo release profile = stock defaults. THP = madvise/madvise.
#
# Run from this directory:  ./reproduce.sh            (everything)
#                     or    ./reproduce.sh <stage>    (one stage)
# Stages: correctness portable native tunables perf thp
set -euo pipefail

cd "$(dirname "$0")"
RESULTS="$(pwd)/results"
mkdir -p "$RESULTS"
export RAYON_NUM_THREADS=16          # campaign convention: 16 physical cores
export CARGO_TERM_COLOR=never

STAGE="${1:-all}"

# Config matrix (campaign D7): portable = no RUSTFLAGS; native = -C target-cpu=native.
# This study benches BOTH (native primary for analysis, portable secondary).

stage_correctness() {
    echo "=== correctness gate (portable config) ==="
    unset RUSTFLAGS
    cargo run --release --example correctness 2>&1 | tee "$RESULTS/correctness_portable.txt"
    echo "=== correctness gate (native config) ==="
    RUSTFLAGS="-C target-cpu=native" cargo run --release --example correctness 2>&1 | tee "$RESULTS/correctness_native.txt"
}

run_benches() {
    # $1: tag (portable|native)
    local tag="$1"
    local tagdir="$RESULTS/$tag"
    mkdir -p "$tagdir"
    echo "=== [$tag] criterion suite: benches/reuse.rs ==="
    if [ "$tag" = native ]; then
        env RUSTFLAGS="-C target-cpu=native" cargo bench --bench reuse 2>&1 \
            | tee "$tagdir/bench_reuse.txt" | grep -E "time:|change:|Performance|regress|improve|No change" || true
    else
        env -u RUSTFLAGS cargo bench --bench reuse 2>&1 \
            | tee "$tagdir/bench_reuse.txt" | grep -E "time:|change:|Performance|regress|improve|No change" || true
    fi
    # keep criterion raw data under results/ too
    if [ -d target/criterion ]; then
        rm -rf "$tagdir/criterion"
        cp -r target/criterion "$tagdir/criterion"
    fi
}

stage_portable() {
    unset RUSTFLAGS
    cargo bench --no-run
    run_benches portable
}

stage_native() {
    export RUSTFLAGS="-C target-cpu=native"
    cargo bench --no-run
    run_benches native
}

# env-tunable stage: add_a_serial (and one faer case) under glibc malloc
# tunables; recovery % is computed against the default-env run of the same
# binary. NOTE: glibc reads MALLOC_* at process start — each run is a fresh
# process, so this is valid.
stage_tunables() {
    export RUSTFLAGS="-C target-cpu=native"
    cargo build --release --example profile_a_vs_b
    local bin="./target/release/examples/profile_a_vs_b"
    local out="$RESULTS/tunables"
    mkdir -p "$out"
    run_tunable() { # $1 label, remaining: env assignments
        local label="$1"; shift
        echo "=== tunable: $label ==="
        env "$@" "$bin" add_a_serial 2>&1 | tee "$out/add_a_${label}.txt" | grep RESULT || true
        env "$@" "$bin" add_a_faer 2>&1 | tee "$out/add_a_faer_${label}.txt" | grep RESULT || true
    }
    {
        echo "=== tunable: default ==="
        "$bin" add_a_serial  2>&1 | tee "$out/add_a_default.txt" | grep RESULT
        "$bin" add_a_faer    2>&1 | tee "$out/add_a_faer_default.txt" | grep RESULT
        "$bin" tr_a_serial   2>&1 | tee "$out/tr_a_default.txt" | grep RESULT
        "$bin" tr_a_faer     2>&1 | tee "$out/tr_a_faer_default.txt" | grep RESULT
        "$bin" add_mutc_serial 2>&1 | tee "$out/add_mutc_default.txt" | grep RESULT
        "$bin" add_2pass_serial 2>&1 | tee "$out/add_2pass_default.txt" | grep RESULT
        "$bin" add_bound_serial 2>&1 | tee "$out/add_bound_default.txt" | grep RESULT
        "$bin" tr_assign_serial 2>&1 | tee "$out/tr_assign_default.txt" | grep RESULT
        "$bin" tr_bound_serial  2>&1 | tee "$out/tr_bound_default.txt" | grep RESULT
        "$bin" sum_a_serial     2>&1 | tee "$out/sum_a_default.txt" | grep RESULT
        "$bin" full_a_serial    2>&1 | tee "$out/full_a_default.txt" | grep RESULT
        "$bin" full_b_serial    2>&1 | tee "$out/full_b_default.txt" | grep RESULT
        "$bin" zeros_a_serial   2>&1 | tee "$out/zeros_a_default.txt" | grep RESULT
    }
    run_tunable  mmapthr32M  MALLOC_MMAP_THRESHOLD_=33554432
    run_tunable  mmapthr64M  MALLOC_MMAP_THRESHOLD_=67108864
    run_tunable  mmapthr64M_trim  MALLOC_MMAP_THRESHOLD_=67108864 MALLOC_TRIM_THRESHOLD_=134217728
    run_tunable  trim128M    MALLOC_TRIM_THRESHOLD_=134217728
    run_tunable  toppad32M   MALLOC_TOP_PAD_=33554432
    run_tunable  arena1      MALLOC_ARENA_MAX=1
    run_tunable  mmapthr64M_arena1 MALLOC_MMAP_THRESHOLD_=67108864 MALLOC_ARENA_MAX=1
}

# perf stage: A-vs-B pairs with full -d counters (the brief's required pairs:
# add contig 2048^2 serial, transpose 2048^2 serial, one faer16 case).
stage_perf() {
    export RUSTFLAGS="-C target-cpu=native"
    cargo build --release --example profile_a_vs_b
    local bin="$(pwd)/target/release/examples/profile_a_vs_b"
    local out="$RESULTS/perf"
    mkdir -p "$out"
    for case_ in add_a_serial add_mutc_serial add_2pass_serial add_bound_serial \
                 tr_a_serial tr_assign_serial tr_bound_serial sum_a_serial \
                 full_a_serial full_b_serial zeros_a_serial \
                 add_a_faer add_mutc_faer add_bound_faer tr_a_faer tr_assign_faer; do
        echo "=== perf stat -d : $case_ ==="
        perf stat -d -e task-clock,cycles,instructions,branches,branch-misses,\
L1-dcache-loads,L1-dcache-load-misses,cache-references,cache-misses,page-faults,\
context-switches,cpu-migrations \
            -o "$out/perf_${case_}.txt" -- "$bin" "$case_"
    done
}

# THP stage: madvise(MADV_HUGEPAGE) quantification (system THP = madvise).
stage_thp() {
    export RUSTFLAGS="-C target-cpu=native"
    cargo build --release --example madvise_thp
    local bin="$(pwd)/target/release/examples/madvise_thp"
    local out="$RESULTS/thp"
    mkdir -p "$out"
    echo "THP enabled: $(cat /sys/kernel/mm/transparent_hugepage/enabled)" | tee "$out/thp_state.txt"
    echo "THP defrag:  $(cat /sys/kernel/mm/transparent_hugepage/defrag)"  | tee -a "$out/thp_state.txt"
    for mode in plain madvise reuse madvise_reuse; do
        echo "=== thp: $mode ==="
        perf stat -e page-faults,minor-faults,major-faults -o "$out/perf_${mode}.txt" -- "$bin" "$mode" 2>/dev/null || true
        "$bin" "$mode" 2>&1 | tee "$out/${mode}.txt" | grep RESULT || true
    done
}

case "$STAGE" in
    all)
        stage_correctness
        stage_portable
        stage_native
        stage_tunables
        stage_perf
        stage_thp
        ;;
    correctness) stage_correctness ;;
    portable)    stage_portable ;;
    native)      stage_native ;;
    tunables)    stage_tunables ;;
    perf)        stage_perf ;;
    thp)         stage_thp ;;
    *) echo "unknown stage: $STAGE (use correctness|portable|native|tunables|perf|thp)"; exit 2 ;;
esac

echo "reproduce.sh: done (stage=$STAGE)"

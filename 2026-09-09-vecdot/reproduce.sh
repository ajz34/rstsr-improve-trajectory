#!/usr/bin/env bash
# T3' `vecdot` — reproduction script (PHASE 1 = baseline; no rstsr changes).
#
# rstsr base commit: 386948be819baa334b8da02232f3a1944e5447d5 (path dep,
# READ-ONLY in phase 1; verify `git -C ../../rstsr status --short` is empty
# and HEAD is 386948b before baselining).
#
# Machine: AMD Ryzen 9 9950X3D (Zen 5, AVX-512), 16 cores. rustc 1.97.1
# nightly (2026-07-14). cargo release profile = stock defaults.
#
# Run from this directory:  ./reproduce.sh            (everything)
#                     or    ./reproduce.sh <stage>
# Stages: correctness portable native rerun probes perf numpy candidate candidate-restore
#
# Phase 2: the `candidate` stage applies proposed.patch to ../rstsr, runs the
# gates + benches vs the saved clean-tree baselines, and re-profiles;
# `candidate-restore` resets ../rstsr afterwards.
set -euo pipefail

cd "$(dirname "$0")"
RESULTS="$(pwd)/results"
mkdir -p "$RESULTS"
export RAYON_NUM_THREADS=16          # campaign convention: 16 physical cores
export CARGO_TERM_COLOR=never

STAGE="${1:-all}"

run_benches() {
    # $1: tag  $2: RUSTFLAGS suffix ("-" = none)  $3: bench filter ("" = all)
    local tag="$1" flags="$2" filter="$3"
    local tagdir="$RESULTS/$tag"
    mkdir -p "$tagdir"
    if [ "$flags" = "-" ]; then unset RUSTFLAGS; else export RUSTFLAGS="$flags"; fi
    cargo bench --no-run
    for bench in vecdot innerdot; do
        echo "=== [$tag] bench: $bench filter='$filter'"
        if [ -n "$filter" ]; then
            cargo bench --bench "$bench" -- "$filter" --save-baseline "$tag" 2>&1 \
                | tee "$tagdir/bench_${bench}.txt" | grep -E "bench:|time:|change:|Performance|regress|improve|No change" || true
        else
            cargo bench --bench "$bench" -- --save-baseline "$tag" 2>&1 \
                | tee "$tagdir/bench_${bench}.txt" | grep -E "bench:|time:|change:|Performance|regress|improve|No change" || true
        fi
    done
}

stage_correctness() {
    unset RUSTFLAGS
    echo "=== correctness gate (portable config) ==="
    cargo run --release --example correctness 2>&1 | tee "$RESULTS/correctness_portable.txt" | tail -3
    echo "=== correctness gate (native config) ==="
    RUSTFLAGS="-C target-cpu=native" cargo run --release --example correctness 2>&1 \
        | tee "$RESULTS/correctness_native.txt" | tail -3
}

stage_portable()  { run_benches portable "-" ""; }
stage_native()    { run_benches native "-C target-cpu=native" ""; }
stage_rerun()     { run_benches portable_rerun "-" "batched"; }

stage_probes() {
    unset RUSTFLAGS
    cargo build --release --example probe_kernels --example probe_branch
    ./target/release/examples/probe_branch | tee "$RESULTS/branch_probe.txt"
    ./target/release/examples/probe_kernels all | tee "$RESULTS/probe_portable.txt"
    RUSTFLAGS="-C target-cpu=native" cargo build --release --example probe_kernels
    ./target/release/examples/probe_kernels all | tee "$RESULTS/probe_native.txt"
}

stage_perf() {
    mkdir -p "$RESULTS/perf"
    export RUSTFLAGS="-C target-cpu=native"
    cargo build --release --example profile_ops
    local bin="./target/release/examples/profile_ops"
    for op in batched_am1 batched_axis0 batched_strided dot1d innerdot_serial batched_faer innerdot_faer; do
        echo "=== perf stat -d : $op ==="
        perf stat -d -o "$RESULTS/perf/perf_${op}.txt" -- "$bin" "$op"
    done
}

stage_numpy() {
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate torch
    OMP_NUM_THREADS=1 python numpy_ref/anchors.py | tee numpy_ref/results.txt
    cp numpy_ref/results.txt "$RESULTS/numpy_anchors_st.txt"
    python numpy_ref/anchors.py blis | grep -E "^np\.dot" | tee "$RESULTS/numpy_anchors_blas.txt" || true
}

stage_candidate() {
    # Phase-2 candidate measurement: requires the proposed.patch APPLIED to
    # ../../rstsr (the phase-2 tree). Compares against the clean-tree
    # criterion baselines "portable"/"native" saved by the portable/native
    # stages. faer16-suspect cells get extra trials (±8% band, see
    # results/candidate_faer_trials.md).
    echo "Applying proposed.patch to ../../rstsr (use 'candidate-restore' to undo)"
    git -C ../rstsr apply "$(pwd)/proposed.patch"
    RAYON_NUM_THREADS=16 cargo run --release --example correctness 2>&1 | tee "$RESULTS/correctness_candidate_portable.txt" | tail -2
    RAYON_NUM_THREADS=16 RUSTFLAGS="-C target-cpu=native" cargo run --release --example correctness 2>&1 | tee "$RESULTS/correctness_candidate_native.txt" | tail -2
    unset RUSTFLAGS
    cargo bench --bench vecdot   -- --baseline portable 2>&1 | tee "$RESULTS/bench_vecdot_candidate_portable.txt"   | grep -E "time:|change:" || true
    cargo bench --bench innerdot -- --baseline portable 2>&1 | tee "$RESULTS/bench_innerdot_candidate_portable.txt" | grep -E "time:|change:" || true
    RUSTFLAGS="-C target-cpu=native"
    cargo bench --bench vecdot   -- --baseline native 2>&1 | tee "$RESULTS/bench_vecdot_candidate_native.txt"   | grep -E "time:|change:" || true
    cargo bench --bench innerdot -- --baseline native 2>&1 | tee "$RESULTS/bench_innerdot_candidate_native.txt" | grep -E "time:|change:" || true
    # repeated trials for the volatile faer16/c64/small cells
    for i in 1 2; do
        cargo bench --bench vecdot -- "c64|axis0.*faer|dot1d_small|strided.*faer" --baseline native 2>&1 \
            | tee "$RESULTS/bench_vecdot_candidate_native_rerun_p$i.txt" | grep -E "^vecdot/|time:" || true
        cargo bench --bench vecdot -- "c64|axis0|dot1d_small|strided|batched_am1_4096x512" --baseline portable 2>&1 \
            | tee "$RESULTS/bench_vecdot_candidate_portable_rerun_p$i.txt" | grep -E "^vecdot/|time:" || true
    done
    # perf stat on the patched tree
    mkdir -p "$RESULTS/perf_candidate"
    cargo build --release --example profile_ops
    for op in batched_am1 batched_axis0 dot1d innerdot_serial batched_faer innerdot_faer; do
        perf stat -d -o "$RESULTS/perf_candidate/perf_${op}.txt" -- ./target/release/examples/profile_ops "$op"
    done
}

stage_candidate_restore() {
    git -C ../rstsr checkout -- .
    echo "rstsr tree restored: $(git -C ../rstsr status --short | wc -l) modified files; HEAD $(git -C ../rstsr rev-parse --short HEAD)"
}

case "$STAGE" in
    all)        stage_correctness; stage_portable; stage_native; stage_rerun; stage_probes; stage_perf; stage_numpy ;;
    correctness) stage_correctness ;;
    portable)    stage_portable ;;
    native)      stage_native ;;
    rerun)       stage_rerun ;;
    probes)      stage_probes ;;
    perf)        stage_perf ;;
    numpy)       stage_numpy ;;
    candidate)   stage_candidate ;;
    candidate-restore) stage_candidate_restore ;;
    *) echo "unknown stage: $STAGE (use correctness|portable|native|rerun|probes|perf|numpy|candidate|candidate-restore)"; exit 2 ;;
esac

echo "reproduce.sh: done (stage=$STAGE)"

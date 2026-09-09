#!/usr/bin/env bash
# T2' reductions — reproduce script (phase 1 baseline; phase 2 reuses it).
#
# Contract (plan §3): release mode, both RUSTFLAGS configs, both devices
# (RAYON_NUM_THREADS=16 asserted by the harness), correctness gate before
# perf, criterion 2 s + 0.7 s warm-up. Criterion outputs are kept per config
# under results/criterion_{portable,native}/ via CRITERION_HOME (they would
# otherwise clobber each other in target/criterion).
#
# Usage:
#   ./reproduce.sh              # everything (~25-30 min)
#   ./reproduce.sh correctness  # gate only, both configs (~2 min)
#   ./reproduce.sh portable     # criterion suite, portable (~6 min)
#   ./reproduce.sh native       # criterion suite, native (~6 min)
#   ./reproduce.sh anchors      # ndarray anchors, both configs (~2 min)
#   ./reproduce.sh perf         # perf stat -d, both configs (~2 min)
#   ./reproduce.sh probe        # manual-loop probes, both configs (<1 min)
#   ./reproduce.sh layout       # layout-branch probe (<1 min)
#   ./reproduce.sh numpy        # numpy anchors via conda torch (~1 min)
set -euo pipefail
cd "$(dirname "$0")"

export RAYON_NUM_THREADS=16

stage_correctness() {
    echo "== correctness (portable)"; cargo build --release --example correctness
    RAYON_NUM_THREADS=16 ./target/release/examples/correctness | tee results/correctness_portable.txt
    echo "== correctness (native)"
    RUSTFLAGS="-C target-cpu=native" cargo build --release --example correctness
    RAYON_NUM_THREADS=16 ./target/release/examples/correctness | tee results/correctness_native.txt
    # restore portable default build of the example
    cargo build --release --example correctness > /dev/null
}

stage_portable() {
    echo "== bench reduce (portable)"
    mkdir -p results/criterion_portable
    CRITERION_HOME="$PWD/results/criterion_portable" cargo bench --bench reduce -- --save-baseline portable \
        | tee results/bench_reduce_portable.log
}

stage_native() {
    echo "== bench reduce (native)"
    mkdir -p results/criterion_native
    CRITERION_HOME="$PWD/results/criterion_native" RUSTFLAGS="-C target-cpu=native" \
        cargo bench --bench reduce -- --save-baseline native | tee results/bench_reduce_native.log
}

stage_anchors() {
    echo "== anchors ndarray (portable)"
    mkdir -p results/criterion_portable
    CRITERION_HOME="$PWD/results/criterion_portable" cargo bench --bench anchors_ndarray -- --save-baseline portable \
        | tee results/bench_anchors_portable.log
    echo "== anchors ndarray (native)"
    CRITERION_HOME="$PWD/results/criterion_native" RUSTFLAGS="-C target-cpu=native" \
        cargo bench --bench anchors_ndarray -- --save-baseline native | tee results/bench_anchors_native.log
}

stage_perf() {
    echo "== perf stat -d (serial device), both configs"
    mkdir -p results/perf
    CARGO_TARGET_DIR=target-prof-portable cargo build --release --example profile_ops
    CARGO_TARGET_DIR=target-prof-native RUSTFLAGS="-C target-cpu=native" cargo build --release --example profile_ops
    for cfg in portable native; do
        if [ "$cfg" = native ]; then BIN=target-prof-native/release/examples/profile_ops; else BIN=target-prof-portable/release/examples/profile_ops; fi
        for op in sum_axis0 sum_axis1 sum_all min_axis0 min_all_1e7; do
            echo "== $cfg $op ==" | tee -a results/perf/raw_${cfg}_candidate.txt
            RAYON_NUM_THREADS=16 perf stat -d "$BIN" "$op" 2>&1 | tee -a results/perf/raw_${cfg}_candidate.txt
        done
    done
}

stage_probe() {
    CARGO_TARGET_DIR=target-prof-portable cargo build --release --example probe_manual
    CARGO_TARGET_DIR=target-prof-native RUSTFLAGS="-C target-cpu=native" cargo build --release --example probe_manual
    echo "--- NATIVE ---" | tee results/probe_manual_native.txt
    ./target-prof-native/release/examples/probe_manual | tee -a results/probe_manual_native.txt
    echo "--- PORTABLE ---" | tee results/probe_manual_portable.txt
    ./target-prof-portable/release/examples/probe_manual | tee -a results/probe_manual_portable.txt
}

stage_layout() {
    cargo build --release --example layout_probe
    ./target/release/examples/layout_probe > results/layout_probe.txt
}

stage_numpy() {
    # conda `torch` env carries numpy 2.5.1 on this machine
    source /home/a/miniconda3/etc/profile.d/conda.sh
    conda run -n torch python numpy_ref/anchors.py | tee numpy_ref/results.txt
}

stage_candidate() {
    # PHASE 2: after the rstsr edits are applied (see PLAN.md), run the
    # candidate against the saved phase-1 baselines, twice per config
    # (build-to-build layout lottery is up to +-5% on ~0.5 ms cells).
    echo "== correctness gates (patched tree, both configs)"
    cargo build --release --example correctness
    RAYON_NUM_THREADS=16 ./target/release/examples/correctness | tee results/correctness_portable.txt
    RUSTFLAGS="-C target-cpu=native" cargo build --release --example correctness
    RAYON_NUM_THREADS=16 ./target/release/examples/correctness | tee results/correctness_native.txt
    cargo build --release --example correctness > /dev/null
    echo "== candidate benches (2x per config)"
    for i in 1 2; do
        CRITERION_HOME="$PWD/results/criterion_portable" cargo bench --bench reduce -- --baseline portable \
            | tee results/bench_reduce_portable_candidate_r$i.log
        CRITERION_HOME="$PWD/results/criterion_native" RUSTFLAGS="-C target-cpu=native" \
            cargo bench --bench reduce -- --baseline native | tee results/bench_reduce_native_candidate_r$i.log
    done
    python3 results/make_compare.py
    echo "== perf stat after (serial, both configs + min_all 1e7)"
    stage_perf
}

stage="${1:-all}"
case "$stage" in
    correctness) stage_correctness ;;
    portable)    stage_portable ;;
    native)      stage_native ;;
    anchors)     stage_anchors ;;
    perf)        stage_perf ;;
    probe)       stage_probe ;;
    layout)      stage_layout ;;
    numpy)       stage_numpy ;;
    candidate)   stage_candidate ;;
    all)
        stage_correctness
        stage_portable
        stage_native
        stage_anchors
        stage_perf
        stage_probe
        stage_layout
        stage_numpy
        python3 results/make_tables.py
        ;;
    *) echo "unknown stage: $stage"; exit 2 ;;
esac
echo "stage '$stage' done."

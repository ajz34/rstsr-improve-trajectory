#!/usr/bin/env bash
# T4' elementwise — phase 1 (baseline on CLEAN rstsr @ 386948be).
#
# Stages:
#   correctness   — gate on both RUSTFLAGS configs (~2 min)
#   portable      — criterion suite, portable build (~4 min)
#   native        — criterion suite, native build (~4 min)
#   portable_c    — variant-A MALLOC-tunable re-run, portable (~3 min)
#   native_c      — variant-A MALLOC-tunable re-run, native (~3 min)
#   perf          — perf stat -d on add contig/strided reuse-B (+ add_a),
#                   native, serial (~2 min)
#   all (default) — everything above
#
# Variant C protocol (T7 carry-forward): re-runs the SAME A-id bench filter
# under MALLOC_MMAP_THRESHOLD_=67108864 MALLOC_TRIM_THRESHOLD_=134217728 to
# decouple the glibc page-fault rider from kernel time (no B variants; B is
# allocation-free by construction).
#
# Requires: nightly rustc (rstsr's rust-toolchain.toml applies via the path
# dep), RAYON_NUM_THREADS=16 (set below), perf for the perf stage.

set -euo pipefail
cd "$(dirname "$0")"

export RAYON_NUM_THREADS=16
STAGE="${1:-all}"

run_suite () {
    # $1 = results subdir name; RUSTFLAGS must be exported by the caller.
    local name="$1"
    mkdir -p "results/${name}"
    cargo bench --bench elementwise 2>&1 | tee "results/${name}/elementwise.log"
    cargo bench --bench anchors_ndarray 2>&1 | tee "results/${name}/anchors_ndarray.log"
    cargo bench --bench kernels_probe 2>&1 | tee "results/${name}/kernels_probe.log"
    rm -rf "results/${name}/criterion"
    cp -r target/criterion "results/${name}/criterion"
}

case "$STAGE" in
  correctness)
    echo "== correctness (portable) =="
    cargo run --example correctness 2>&1 | tee results/correctness_portable.txt
    echo "== correctness (native) =="
    export RUSTFLAGS="-C target-cpu=native"
    cargo run --example correctness 2>&1 | tee results/correctness_native.txt
    unset RUSTFLAGS
    ;;

  portable)
    unset RUSTFLAGS
    run_suite portable
    ;;

  native)
    export RUSTFLAGS="-C target-cpu=native"
    run_suite native
    unset RUSTFLAGS
    ;;

  portable_c)
    unset RUSTFLAGS
    mkdir -p results/portable_c
    env MALLOC_MMAP_THRESHOLD_=67108864 MALLOC_TRIM_THRESHOLD_=134217728 \
        cargo bench --bench elementwise -- "A" 2>&1 | tee results/portable_c/elementwise_A.log
    ;;

  native_c)
    export RUSTFLAGS="-C target-cpu=native"
    mkdir -p results/native_c
    env MALLOC_MMAP_THRESHOLD_=67108864 MALLOC_TRIM_THRESHOLD_=134217728 \
        cargo bench --bench elementwise -- "A" 2>&1 | tee results/native_c/elementwise_A.log
    unset RUSTFLAGS
    ;;

  perf)
    mkdir -p results/perf
    echo "== build (native) =="
    export RUSTFLAGS="-C target-cpu=native"
    cargo build --release --example profile_ops
    unset RUSTFLAGS
    echo "== perf: add contig reuse-B, serial =="
    perf stat -d -o "results/perf/perf_add_b_serial${PERF_SUFFIX}.txt" \
        -- target/release/examples/profile_ops add_b
    echo "== perf: add strided reuse-B, serial =="
    perf stat -d -o "results/perf/perf_add_strided_b_serial${PERF_SUFFIX}.txt" \
        -- target/release/examples/profile_ops add_strided_b
    echo "== perf: add contig alloc-A, serial =="
    perf stat -d -o "results/perf/perf_add_a_serial${PERF_SUFFIX}.txt" \
        -- target/release/examples/profile_ops add_a
    ;;

  pre2)
    # Extended-baseline pass on the CLEAN tree (adds stridedfirst / addasgn /
    # strided small+odd rows and the flip/i32/complex correctness fixtures).
    "$0" correctness
    unset RUSTFLAGS
    mkdir -p results/pre2_portable
    cargo bench --bench elementwise 2>&1 | tee results/pre2_portable/elementwise.log
    cargo bench --bench anchors_ndarray 2>&1 | tee results/pre2_portable/anchors_ndarray.log
    cargo bench --bench kernels_probe 2>&1 | tee results/pre2_portable/kernels_probe.log
    rm -rf results/pre2_portable/criterion; cp -r target/criterion results/pre2_portable/criterion
    export RUSTFLAGS="-C target-cpu=native"
    mkdir -p results/pre2_native
    cargo bench --bench elementwise 2>&1 | tee results/pre2_native/elementwise.log
    cargo bench --bench anchors_ndarray 2>&1 | tee results/pre2_native/anchors_ndarray.log
    cargo bench --bench kernels_probe 2>&1 | tee results/pre2_native/kernels_probe.log
    rm -rf results/pre2_native/criterion; cp -r target/criterion results/pre2_native/criterion
    unset RUSTFLAGS
    ;;

  candidate)
    # Phase-2 candidate pass: requires the blocked-kernel patch APPLIED to
    # ../rstsr (see PLAN.md §7). Same protocol as pre2 plus the C stages and
    # the after-patch perf pass (PERF_SUFFIX=_after is set by the caller for
    # the perf stage; here we run it directly).
    "$0" correctness
    unset RUSTFLAGS
    mkdir -p results/candidate_portable
    cargo bench --bench elementwise 2>&1 | tee results/candidate_portable/elementwise.log
    cargo bench --bench anchors_ndarray 2>&1 | tee results/candidate_portable/anchors_ndarray.log
    cargo bench --bench kernels_probe 2>&1 | tee results/candidate_portable/kernels_probe.log
    rm -rf results/candidate_portable/criterion; cp -r target/criterion results/candidate_portable/criterion
    PERF_SUFFIX=_after "$0" perf
    export RUSTFLAGS="-C target-cpu=native"
    mkdir -p results/candidate_native
    cargo bench --bench elementwise 2>&1 | tee results/candidate_native/elementwise.log
    cargo bench --bench anchors_ndarray 2>&1 | tee results/candidate_native/anchors_ndarray.log
    cargo bench --bench kernels_probe 2>&1 | tee results/candidate_native/kernels_probe.log
    rm -rf results/candidate_native/criterion; cp -r target/criterion results/candidate_native/criterion
    unset RUSTFLAGS
    env MALLOC_MMAP_THRESHOLD_=67108864 MALLOC_TRIM_THRESHOLD_=134217728 \
        cargo bench --bench elementwise -- "A" 2>&1 | tee results/candidate_native/elementwise_A_malloc.log
    ;;

  all)
    "$0" correctness
    "$0" portable
    "$0" native
    "$0" portable_c
    "$0" native_c
    "$0" perf
    ;;

  *) echo "unknown stage $STAGE"; exit 1;;
esac

echo "done: stage $STAGE"

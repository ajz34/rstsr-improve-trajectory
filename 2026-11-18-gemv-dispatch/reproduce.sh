#!/usr/bin/env bash
# P0 measurement driver for the GEMV-dispatch campaign (2026-11-18-gemv-dispatch).
#
# Threads: RAYON_NUM_THREADS=16 (physical cores) and the BLAS runtime pinned
# through the mechanism the installed library honours (the harness calls
# omp_set_num_threads for OpenMP builds; OMP_NUM_THREADS is the env fallback).
# RSTSR_DEV=1 makes rstsr-openblas's build script link this machine's
# libopenblas and libgomp (see its build.rs).
set -euo pipefail
cd "$(dirname "$0")"

export RAYON_NUM_THREADS=16
export OMP_NUM_THREADS=16
export RSTSR_DEV=1

mkdir -p results

echo "== environment" | tee results/environment.txt
{ echo "date: $(date -Is)"
  echo "host: $(uname -srm)"
  echo "cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')"
  echo "rustc: $(rustc --version)"
  echo "cargo: $(cargo --version)"
  echo "rstsr: $(git -C ../../rstsr rev-parse --short HEAD) ($(git -C ../../rstsr branch --show-current))"
  echo "rayon threads: ${RAYON_NUM_THREADS}"
  echo "blas config: $(nm -D /usr/lib/x86_64-linux-gnu/libopenblas.so.0 >/dev/null 2>&1 && echo system-libopenblas || echo unknown)"
} | tee -a results/environment.txt

echo "== correctness gate" | tee -a results/environment.txt
cargo run --release --example correctness 2>&1 | tee results/correctness.txt

echo "== criterion (3 passes)" | tee -a results/environment.txt
for i in 1 2 3; do
  cargo bench --bench gemv -- --save-baseline "run$i" 2>&1 | tee "results/bench-run$i.txt"
done

echo "== numpy anchors" | tee -a results/environment.txt
python3 numpy_ref/anchors.py 2>&1 | tee results/numpy.txt

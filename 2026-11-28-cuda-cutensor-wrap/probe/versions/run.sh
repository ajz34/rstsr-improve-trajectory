#!/bin/bash
# Run ./vs against every installed libcutensor.so.2; LD_LIBRARY_PATH (lib<V>/ symlink farms) selects the version.
# Writes run-<V>.log (full stdout+stderr). Usage: ./run.sh [V ...]   (default: all five versions)
# If the all-in-one process exits non-zero (crash), every case is re-run in its own process and appended to the log.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"
declare -A SRC=(
  [2.0.1]=/scratch/miniconda3-bak/envs/pyscf/lib/python3.10/site-packages/cutensor/lib/libcutensor.so.2
  [2.4.1]=/scratch/miniconda3-bak/pkgs/cutensor-2.4.1.4-h15eaa2f_0/lib/libcutensor.so.2
  [2.6.0]=/home/a/.cache/uv/archive-v0/hQ6viqq5PutHCWkI/cutensor/lib/libcutensor.so.2
  [2.7.0]=/home/a/miniconda3/envs/torch-gpu4pyscf/lib/python3.11/site-packages/cutensor/lib/libcutensor.so.2
  [2.8.0]=/home/a/miniconda3/envs/ruaa-dev/lib/python3.13/site-packages/cutensor/lib/libcutensor.so.2
)
# 2.0.1 links libcublasLt.so.12 (CUDA 13.2 does not ship it); only that one file is
# added to the lib2.0.1 farm: putting the whole conda lib dir into LD_LIBRARY_PATH breaks coreutils (libsystemd/tinfo).
declare -A EXTRA=( [2.0.1]=/scratch/miniconda3-bak/pkgs/libcublas-12.9.1.4-0/lib/libcublasLt.so.12 )
VERSIONS=${*:-"2.0.1 2.4.1 2.6.0 2.7.0 2.8.0"}
for V in $VERSIONS; do
  mkdir -p "lib$V"
  ln -sfn "${SRC[$V]}" "lib$V/libcutensor.so"
  ln -sfn "${SRC[$V]}" "lib$V/libcutensor.so.2"
  if [ -n "${EXTRA[$V]:-}" ]; then ln -sfn "${EXTRA[$V]}" "lib$V/$(basename "${EXTRA[$V]}")"; fi
  # per-invocation only (never export): the loader path must not leak into coreutils
  VSPATH="$HERE/lib$V:/usr/local/cuda/lib64"
  LOG="run-$V.log"
  { echo "# version $V"
    echo "# date $(date -Is)"
    echo "# LD_LIBRARY_PATH (for ./vs only) =$VSPATH"
    echo "# resolved: $(LD_LIBRARY_PATH="$VSPATH" ldd ./vs | grep -E 'cutensor|not found' | sed 's/^\s*//' | tr '\n' ';')"
  } > "$LOG"
  LD_LIBRARY_PATH="$VSPATH" timeout 300 ./vs all >> "$LOG" 2>&1; rc=$?
  echo "# exit(all in one process) = $rc" >> "$LOG"
  if [ $rc -ne 0 ] && [ $rc -ne 127 ]; then
    echo "# non-zero exit -> each case re-run in its own process" >> "$LOG"
    for C in S1 S2 S3 S4 S5 S6 S7; do LD_LIBRARY_PATH="$VSPATH" timeout 300 ./vs "$C" >> "$LOG" 2>&1; echo "# exit($C) = $?" >> "$LOG"; done
  fi
  echo "== $V (exit $rc) -> $LOG"
  grep -E "^S[1-7] " "$LOG" | cut -c1-150
done

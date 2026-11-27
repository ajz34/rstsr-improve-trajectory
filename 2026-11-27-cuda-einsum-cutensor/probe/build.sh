#!/bin/bash
# Build the cuTENSOR probe. Usage: ./build.sh <cutensor include dir> <dir containing libcutensor.so> <output binary>
# e.g. ./build.sh /home/a/miniconda3/envs/torch-gpu4pyscf/lib/python3.11/site-packages/cutensor/include ../lib ../bin/probe
set -euo pipefail
INC=${1:?include dir}
LIBDIR=${2:?lib dir}
OUT=${3:?output binary}
HERE=$(cd "$(dirname "$0")" && pwd)
CUDA=${CUDA_HOME:-/usr/local/cuda}
mkdir -p "$(dirname "$OUT")"
nvcc -O2 -std=c++17 -arch=sm_120 -Xcompiler -fopenmp \
    -I"$INC" \
    "$HERE/main.cu" "$HERE/check_strides.cu" "$HERE/check_conj.cu" "$HERE/check_dtype.cu" "$HERE/check_perf.cu" \
    -L"$LIBDIR" -L"$CUDA/lib64" -lcutensor -lcublas \
    -o "$OUT"
echo "built $OUT"

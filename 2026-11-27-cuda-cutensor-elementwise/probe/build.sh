#!/bin/bash
# Build the cuTENSOR elementwise probes. Usage: ./build.sh [facts|bench|all]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
INC=${CUTENSOR_INC:-/home/a/miniconda3/envs/torch-gpu4pyscf/lib/python3.11/site-packages/cutensor/include}
CUDA=${CUDA_HOME:-/usr/local/cuda}
what=${1:-all}
mkdir -p "$HERE/bin"
build() {
    local src=$1 out=$2
    "$CUDA/bin/nvcc" -O2 -std=c++17 -arch=sm_120 -I"$INC" -I"$HERE" "$HERE/$src" \
        -L"$HERE/lib" -lcutensor -o "$HERE/bin/$out"
    echo "built $HERE/bin/$out"
}
if [[ $what == dbg ]]; then build dbg.cpp dbg; fi
if [[ $what == facts || $what == all ]]; then build facts.cpp facts; fi
if [[ $what == bench || $what == all ]]; then build bench.cpp bench; fi


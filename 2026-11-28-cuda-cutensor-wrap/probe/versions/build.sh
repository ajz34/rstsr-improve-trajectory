#!/bin/bash
# Build the cross-version cuTENSOR probe (vs.cpp). Usage: ./build.sh
# Compiled once against the 2.7.0 header and linked against lib2.7.0/; the 2.x ABI is stable, so the same binary is run
# against every libcutensor.so.2 on the machine via LD_LIBRARY_PATH (see run.sh).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
INC=/home/a/miniconda3/envs/torch-gpu4pyscf/lib/python3.11/site-packages/cutensor/include
# -ccbin: the shell may have a conda gcc first in PATH whose cc1plus is missing; pin the system g++
nvcc -std=c++17 -O2 -arch=sm_120 -ccbin /usr/bin/g++ -I"$INC" "$HERE/vs.cpp" -L"$HERE/lib2.7.0" -lcutensor -o "$HERE/vs"
echo "built $HERE/vs"

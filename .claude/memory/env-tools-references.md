---
name: env-tools-references
description: How to run numpy/torch reference numbers (conda env "torch"), where reference C sources live (~/Git-Others), which profilers are installed.
metadata:
  type: reference
---

Environment facts verified 2026-09-08 (machine ajz34's 9950X3D box):

- Python references (numpy einsum, `.sum(axis)`, torch) require
  `conda activate torch` → numpy 2.5.1, torch 2.14.0+rocm7.2.
  Plain `python3` has NO numpy.
- `~/Git-Others` holds reference sources: numpy (C loops, einsumsimd),
  scipy, OpenBLAS, blis (blocked kernels), plus candle/pyscf/gpu4pyscf/libxc.
- Profilers: `perf` 7.0.14 at /usr/bin/perf works; valgrind and
  cargo-flamegraph are NOT installed.

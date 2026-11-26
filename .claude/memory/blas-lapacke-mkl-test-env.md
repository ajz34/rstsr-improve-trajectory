---
name: blas-lapacke-mkl-test-env
description: "How to run BLAS device tests under the lapacke feature and against MKL on this machine: conda OpenBLAS 0.3.34 symlink dir on LD_LIBRARY_PATH, MKL 2026.1 env vars, RSTSR_DEV=1 -- --test-threads=1; BLIS/AOCL/KML are check-only"
metadata:
  type: reference
---

Always `RSTSR_DEV=1` and `-- --test-threads=1`; `RUST_BACKTRACE=0` keeps
failure output short.

- `lapack` (default), system OpenBLAS 0.3.32:
  `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu`,
  `cargo test -p rstsr-openblas --features "rstsr/backtrace openmp"`.
- `lapacke`: the system OpenBLAS has no `LAPACKE_*`. Use conda's OpenBLAS
  0.3.34: in a temp dir, symlink `libopenblas.so` and `libopenblas.so.0`
  to `~/miniconda3/pkgs/libopenblas-0.3.34-openmp_h156ec52_2/lib/libopenblas.so.0`,
  prepend that dir to `LD_LIBRARY_PATH`, add `lapacke` to the features.
- MKL 2026.1: `MKLROOT=$HOME/intel/oneapi/mkl/2026.1`,
  `MKL_THREADING_LAYER=GNU`,
  `LD_LIBRARY_PATH=$HOME/intel/oneapi/mkl/2026.1/lib:$HOME/intel/oneapi/compiler/2026.1/lib`,
  `-p rstsr-mkl --features "rstsr/backtrace"` (optionally `+ lapacke`).
- BLIS, AOCL, KML: `cargo check --all-targets --features "rstsr/backtrace"`
  only; their libraries are not on the linker path.
- Known pre-existing failure: MKL `lapacke`
  `driver_impl::f64::gelsd::test_dgelsd_relations`, whose
  `DGESDD(compute_uv = false)` oracle reports "illegal argument in argument
  11" (passes under MKL `lapack`).
- `/tmp` is fine for small logs; see `tmp-tmpfs-usrquota` before pointing
  cargo at it.

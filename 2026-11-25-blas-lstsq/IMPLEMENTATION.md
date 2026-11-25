# lstsq for the BLAS devices — implementation report

Branch `261125/blas-lstsq` off `ed1f62d` in `/home/a/rstsr_pack/rstsr`
(working tree only; the main repo is never auto-committed). Implements
DECISIONS D1–D16 / PLAN B0–B6 of this directory.

## What landed

### `rstsr-blas-traits` (driver layer)

- `src/lapack_lstsq/{mod.rs, gelsd.rs}`: `GELSDDriverAPI<T>` and the `GELSD_`
  builder. `internal_run` returns `(x, s, rank, Option<residuals>)`; residuals
  are `None` when `M <= N` or `rank < min(M, N)` (the entry layer renders the
  SciPy NaN), and are read from the destroyed `b` workspace rows `n..m` with raw
  strides (no extra GEMM).
- Thunks `driver_impl/lapack/lstsq/{mod.rs, gelsd.rs}` (shared into all five
  device crates by the existing symlink sets): real/complex `duplicate_item`
  blocks, `LWORK`/`LIWORK`/`LRWORK` queries, `-1010/-1011` mapping, NaN/Inf
  info codes `-4/-5/-7/-10`.
- Thunks `driver_impl/lapacke/lstsq/{mod.rs, gelsd.rs}` (openblas/mkl only):
  one-liner forwarding to `LAPACKE_{s,d,c,z}gelsd`.
- `src/lib.rs`, `src/prelude.rs`, `src/trait_def.rs`: module export and
  `LapackDriverAPI<T>: GELSDDriverAPI<T>`.

### `rstsr-linalg-traits` (entry layer)

- `src/stack.rs`: `LstsqPlan` + `lstsq_plan` (rectangular `(M, N)`, vector or
  matrix `b`, broadcast batch; both matrix-axis orders). `solve_plan` keeps its
  square-only check.
- `src/blas_impl/lstsq.rs` (compiled per device crate through
  `src/linalg_auto_impl/lstsq.rs` symlinks): `blas_impl_lstsq_f` walks the
  stack, consumes `a` in place when it is owned/mutable and already in the
  device order, broadcasts `a`/`b` to the plan, and batches `x`, `s`, per-slice
  `rank` and `residuals`; eight `LstsqAPI` impls (`&A`, `TensorView`,
  `TensorMut`, `Tensor` × without/with `rcond`) all return dynamic-dim
  `LstsqResult`.
- `src/traits_def.rs` / `src/prelude.rs`: `LstsqAPI`, `lstsq`, `lstsq_f`,
  `LstsqResult` + `From` both ways.

### Tests and validators

- Driver: `blas_tests_impl/driver_impl/f64/gelsd.rs` — 4 fingerprints
  (tall 1024×512 K=1/K=3, wide 512×1024 K=1, rank-deficient with `rank = 511`)
  plus 16 layout combinations (call order × a/b order × padded `lda/ldb`)
  against the compact column-major reference.
- Function: `linalg_func/func_f64/lstsq.rs` (4 cases + explicit `rcond` + n-d
  batch consistency and owned-input consumption), `func_c64/lstsq.rs`
  (3 cases; `c64` is `Complex<f64>` here); both f64 tests cover the
  rank-deficient NaN residuals (vector and matrix `b`).
- OpenBLAS-only zero-size (`tests/zero_size/lapack.rs`): empty rows, empty
  cols (synthesized residuals), `K == 0` (padded to one zero column), empty
  batch.
- Python validators: GELSD sections in `driver_validation_f64.py`,
  `func_validation_f64.py`, `func_validation_c64.py` (same fingerprints).

### Docs

`blas_impl_lstsq_f` docstring; `docs/cheatsheet_linalg.md` prose bullet (no
table row — the cheatsheet rows are faer-backed) and `LstsqResult` in the
multi-output list; `docs/array_api_standard_extension.md` Notes bullet.

## Deviations from the plan

- No `ref_impl_lstsq_f` wrapper: the driver is already a single `GELSD` call,
  so the stack walk calls the builder directly (like `svd`/`det`, which also do
  not wrap `with_blas_num_threads`).
- `lapacke` is **compile-verified only** here: the system OpenBLAS
  (`/usr/lib/x86_64-linux-gnu/libopenblas.so.0`) is built without LAPACKE
  (`nm -D | grep LAPACKE_` is empty) and MKL is not on the linker path
  (`-lmkl_rt` missing), so `cargo test --features lapacke` cannot link on this
  machine. `cargo check -p rstsr-openblas --features lapacke` and
  `-p rstsr-mkl --features lapacke` pass.
- Residuals for `rank < min(M, N)`: the *driver* returns `None`, the *entry*
  fills NaN (SciPy semantics). The driver validator prints NaN for that case
  even though LAPACK leaves numbers in `B(n+1:m)`.

## Driver bug found by the tests

For `M < N`, `DGELSD` first zeroes `B(M+1:N, 1:nrhs)` (`DLASET('F', ...)`) and
writes the length-`N` solution into `B(1:N, :)`. A RowMajor bridge that copies
back only `M` rows returns a truncated (wrong) solution — same `s`, different
`x`. The thunk now copies `max(M, N)` rows for both the in and the back
transpose. This is the gap notice for the `lapacke-translation` skill (recorded
here, skill not edited): row-major bridges must size `b` by `max(M, N)` for
GELSD/GELSS/GELSY, and `nrhs = 0` must be avoided (`info = -4` from `DLALSD`).

## Verification (this machine)

- `cargo fmt --all --check`, `cargo check --workspace --all-targets`,
  `cargo clippy -p rstsr-blas-traits -p rstsr-linalg-traits -p rstsr-openblas
  --all-targets` clean.
- OpenBLAS: `tests_driver_impl` 85 passed, `tests_linalg_func` 97 passed,
  `tests_zero_size` 30 passed (`RSTSR_DEV=1`, `--test-threads=1`).
- Other devices: compile-checked only (their BLAS/LAPACK shared libraries are
  not on this machine's linker path); BLIS/AOCL/KML/MKL `cargo test` links fail
  before any test runs.
- Python validators run clean with the torch env interpreter
  (`~/miniconda3/envs/torch/bin/python`, NumPy 2.5.4 / SciPy 1.18.1).

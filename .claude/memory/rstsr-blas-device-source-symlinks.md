---
name: rstsr-blas-device-source-symlinks
description: "The five BLAS device crates consume the shared driver/thunk, entry and test sources through tracked symlinks — edit the shared tree once, then add the per-device symlink + per-device mod.rs line"
metadata:
  type: reference
---

All five `crates-device/rstsr-{openblas,mkl,blis,aocl,kml}` crates are thin
wrappers; the code is shared through symlinks (tracked in git as mode 120000):

- LAPACK/CBLAS thunks: `crates-device/*/src/driver_impl/cblas` and
  `.../src/driver_impl/lapack` (and `.../src/driver_impl/lapacke` for openblas
  and mkl only) are **directory symlinks** into the shared
  `rstsr-blas-traits/driver_impl/...` tree. The shared tree is *not* a module
  of the `rstsr-blas-traits` lib (it is compiled inside each device crate,
  where `crate::DeviceBLAS` resolves); `cargo check -p rstsr-blas-traits` does
  not compile it — check a device crate instead.
- Linalg entries: `crates-device/*/src/linalg_auto_impl/<fn>.rs` are
  **file symlinks** to `rstsr-linalg-traits/src/blas_impl/<fn>.rs`
  (e.g. `eig.rs` → `../../../../rstsr-linalg-traits/src/blas_impl/eig.rs`).
- Tests: `crates-device/*/tests/driver_impl/**` and
  `.../tests/linalg_func/**` are file symlinks to
  `rstsr-blas-traits/src/blas_tests_impl/**`.
- Still real per-device files: `src/driver_impl/mod.rs`,
  `src/linalg_auto_impl/mod.rs` (the `pub mod` lists differ per crate),
  the test entry binaries (`tests/tests_*.rs`), and the openblas-only
  `tests/zero_size/` harness.

Adding a new BLAS linalg function therefore means: one new file in
`rstsr-linalg-traits/src/blas_impl/`, one symlink per device
(`../../../../rstsr-linalg-traits/src/blas_impl/<fn>.rs`), and five
`pub mod <fn>;` lines. The relative link depth for test files is
`../../../../../rstsr-blas-traits/src/blas_tests_impl/...`.

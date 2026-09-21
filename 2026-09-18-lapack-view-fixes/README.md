# 2026-09-18 — LAPACK/BLAS3 builder view-layout fixes

- **rstsr commit**: `dee8807` (master, = merged #105), branch `260918/lapack-fix`
- **Scope**: functional-correctness audit of the `rstsr-blas-traits` builder
  wrappers (the wrapper layer above `driver_impl`), triggered by the
  2026-09-14 soundness check's note that BLAS3 wrappers pass allocation-base
  pointers. This task covers wrong-results bugs, not memory unsafety.
- **Date**: 2026-09-17/18 (PR + CI 2026-09-21)

## Findings and fixes

Split into multiple commits on `260918/lapack-fix`:

1. **`issue_blas3_view_layout`** (commit `acfe875`) — GEMM/SYHEMM/TRSM passed
   `raw().as_ptr()` (allocation base, layout offset ignored) and hard-coded
   `ldc`/`ldb = m`. Any f-prefer view with non-zero offset or padded ld got
   silently wrong results, and output writes landed in parent elements
   outside the view. Fixed with offset-aware `as_ptr`/`as_mut_ptr` and the
   output's real `ld_col()`. SYHEMM fixed in the same way (no driver impl
   exists for it in any backend, so untested).
2. **getrf/getri** (commit `404ee21`) — getrf allocated `ipiv` of length `n` but
   LAPACK defines `min(m, n)` entries, so wide matrices returned
   nondeterministic garbage in the tail (`ipiv -= 1` reads it); getri never
   validated `ipiv.len() >= n` (OOB read from the public builder).
3. **gesvd/gesdd** (commit `404ee21`) — wrapper `superb` sized `minmn - 1`
   underflowed for empty matrices (bogus allocation failure); gesvd's order
   mapping was inverted relative to gesdd (forced transpose-copies); the
   gesvd driver's RowMajor path queried LAPACK with the row-major `lda`
   (XERBLA `-6` for tall matrices — deviation from `lapacke_dgesvd_work.c`,
   which queries with `lda_t` etc.) and used the `r2c` transpose helper for
   the c2r output direction (unwrap panic; gesdd is the in-repo correct
   reference). Drivers are one physical tree in
   `rstsr-blas-traits/driver_impl/` shared by all five device crates via
   symlinks, so each driver fix applies to every backend at once.

Deferred (flagged, not fixed): syhemm computes `n = a.ncol()` (wrong for
`side=Left` with rectangular `b`) and its side=Right operand slots do not
match the CBLAS convention — dead API (no driver), untestable. (The gesvd
driver's `min_mn - 1` superb backup loop, also flagged, got the same
`saturating_sub` treatment in `404ee21`.)

## Tests

Moved into the issues harness (`crates-device/rstsr-openblas/tests/issues/`)
with word names instead of issue numbers, each carrying run notes that record
the exact wrong values produced by the unfixed code:

- `issue_blas3_view_layout.rs` — gemm operand offset (`[[0,3],[1,4]]` instead of
  `[[1,4],[2,5]]`), gemm output offset+padded-ld with parent-integrity check
  (stale `112.0` in `c[0,1]`, parent column 0 corrupted), trsm padded
  read-modify-write view.
- `issue_getrf_getri.rs` (commit `404ee21`) — wide-matrix LU exact values + pivot
  contract; short-ipiv rejection.
- `issue_gesvd_empty.rs` (commit `404ee21`) — empty-matrix SVD on both drivers.

## Environment and validation

- OpenBLAS 0.3.34 (local build `/home/a/Software/OpenBLAS-0.3.34`), linked
  via the crate's `RSTSR_DEV=1` build script; fixtures generated CI-style
  (`gen_rand_vec.py` + driver/func validation scripts) in a throwaway venv
  with numpy 2.5.3 / scipy.
- Full `cargo test -p rstsr-openblas --release --features="openmp linalg"`:
  **264 tests, 0 failures** (issues 4, driver_impl manifest 10, linalg_func
  manifest 28, core 208, workable/misc rest), plus
  `cargo check --workspace --all-targets` clean and linalg-traits faer 24/24.
  Each split commit was additionally validated in isolation via
  `git stash push --keep-index` before committing.
- Note: the manifest-based suites were previously believed "env-broken"
  locally; they run fine once fixtures are generated.

## PR and CI (2026-09-21)

- Branch pushed to the `ajz34` fork; **PR RESTGroup/rstsr#106** opened
  (title "rstsr (fix): LAPACK GESVD/GESDD empty-matrix handling and
  BLAS3/LAPACK view-layout contracts"), body in the #105 changelog style.
  Follow-up commits: `39a1df9` (rustfmt-only) and `74b7df7` (CI-only clippy
  fix). Awaiting review, not merged.
- **CI-only clippy fix (`74b7df7`)**: CI builds rstsr-openblas against an
  ILP64 OpenBLAS (`blas_int = i64`), where the test's `v as i64` pivot cast
  tripped `clippy::unnecessary_cast` under `-D warnings`; the local 32-bit
  build needed the cast, so it compiled clean locally. Tests must compare
  `blas_int` values in their native dtype (`ipiv.raw().to_vec()`), never via
  fixed-width casts.
- Final CI: **12/12 green** (clippy, rustfmt, doctests, integration-tests,
  unittests ×3, col-major, faer-linalg, pthread, no-std ×2).
- Local linking of the test target currently fails (`undefined symbol:
  dgetri_` on default features; `omp_set_num_threads` with
  `--features="openmp linalg"`), unlike 2026-09-18 when the full suite ran
  green — local link-env drift, not chased; CI is the gate that matters.

# Task split — zero-size stride policy and BLAS/LAPACK empty operands

Plan settled in `GRILL-R1-QUESTIONS.md` / `GRILL-R2-QUESTIONS.md` (2026-10-19).
Execution: **one task per turn**, stop for manual review; no PR, no push; the
task stays uncommitted until the reviewer's explicit instruction.

## Accepted decisions

- **P1** documentation only: docstrings + one ADR (no `order_semantics.md`
  section — that page is about row/column-major).
- **P2** `stride_f_contig` / `stride_c_contig` return all zeros when any shape
  axis is 0, for both `Ix<N>` and `IxD`.
- **P3a** `ld_col`: `shape[1] <= 1` → `max(1, shape[0])`; otherwise
  `stride[1] < max(1, shape[0])` → `None`, else the stride. `ld_row`
  symmetric. `None` now means "no usable leading dimension". **No
  `.unwrap()` at BLAS/LAPACK call sites**: the wrapper converts `None` to an
  `InvalidLayout` shape error through `require_ld` (revised 2026-10-19).
- **P3b** every BLAS/LAPACK wrapper early-returns the correct empty result,
  after the existing validation and before conversion/ld computation.
- **P4** zero-size BLAS tests live in a new OpenBLAS-only directory;
  `openblas_set_xerbla` harness is a future task.

## Order and dependencies

Guard tasks first: they make the `ld_*`/stride changes safe (an empty layout
can yield `ld = None` or zero, which unguarded call sites would trip on).

## T1 — BLAS-3 empty-operand guards

**Scope**

- `rstsr-blas-traits/src/blas3/trsm.rs` — guard in `run` **before**
  `flip_trans` (validate A/B shapes first; return the empty B); keep a guard in
  `internal_run` for direct callers.
- `rstsr-blas-traits/src/blas3/gemm.rs` — `m == 0 || n == 0` → return C;
  `k == 0` (with `m,n > 0`) → call the driver with shape-derived
  `lda = max(1, a.nrow())`, `ldb = 1`, `ldc` from the non-empty C, so BLAS
  performs `C := beta*C`. A/B are not dereferenced.
- `rstsr-blas-traits/src/blas3/syhemm.rs` — `m == 0 || n == 0` → return C.
- `rstsr-blas-traits/src/util/util_layout.rs` — add `require_ld`, the
  `Option<usize>` → `InvalidLayout` converter used by every BLAS/LAPACK call
  site (no `.unwrap()`).
- New tests: `crates-device/rstsr-openblas/tests/zero_size/` +
  `tests_zero_size.rs` (OpenBLAS-only; not in the shared
  `blas_tests_impl` symlinks).

**Tests**

- Issue #161: `solve_triangular`/`TRSM` with `[18, 0]` and `[0, 18]`, owned /
  mut / view, static and dynamic layouts — no `xerbla`.
- `gemm`: `m == 0`, `n == 0`, and `k == 0` with `beta != 1`
  (including a transposed empty operand).
- `syhemm`: `m == 0 || n == 0`.

**Exit** — targeted OpenBLAS test green; existing BLAS-3 tests unchanged;
no rstsr commit.

## T2 — LAPACK empty-operand guards

**Scope**

- `lapack_solve/gesv.rs`, `sysv.rs` — `n == 0` or `nrhs == 0` returns empty
  `ipiv` and the converted inputs without calling the driver (revised after
  CI: an empty B at the driver boundary was the failure).
- `lapack_solve/getrf.rs` (`m == 0 || n == 0`), `getri.rs` (`n == 0`),
  `potrf.rs` (`n == 0`).
- `lapack_eigh/{syev,syevd,sygv,sygvd}.rs` — `n == 0` returns empty
  eigenvalues/eigenvectors.
- `lapack_svd/{gesvd,gesdd}.rs` — move the existing `minmn == 0` guard above
  `lda` derivation (it currently computes `lda` first).
- Tests extend `tests_zero_size.rs`.

**Tests**

- `gesv`/`sysv`: `n == 0`; `nrhs == 0` with non-singular and singular A
  (both are unchanged-input no-ops).
- `getrf`/`getri` with a zero axis; `potrf` `0×0`; `syev(d)`/`sygv(d)` `0×0`.
- SVD empty remains covered by `tests/issues/issue_gesvd_empty.rs`.

**Exit** — targeted tests green; `map_lapack_info` untouched.

## T3 — `ld_row` / `ld_col` contract

**Scope**

- `rstsr-core/src/tensor/tensor2_impl.rs` — implement the Q2 rule, fix the
  swapped "row-vector"/"col-vector" comments, document the contract (the
  `None` meaning, the empty rule).
- Touched BLAS/LAPACK call sites: `ld(...).unwrap()` → `rstsr_unwrap()`
  (backtrace convention).
- Unit tests in the module: `[0,0]`, `[m,0]`, `[0,n]`, `[m,1]`, `[1,n]`,
  padded-stride `[m,1]`, and a deliberately too-small non-empty stride via
  `Layout::new_unchecked` (`None`).

**Exit** — `cargo test -p rstsr-core` relevant tests green; no behavior change
for non-empty valid layouts.

## T4 — P2 zero strides for zero-size shapes

**Scope**

- `rstsr-common/src/layout/shape.rs` — `stride_f_contig` / `stride_c_contig`
  return all zeros iff `self.iter().any(|d| d == 0)`, both `Ix<N>` and `IxD`;
  `stride_contig` inherits.
- `rstsr-common/src/layout/layoutbase.rs` — update the `new_c_contig` /
  `new_f_contig` safety comments; add the short type-level note on `Layout`.
- Docstrings on the stride methods.
- Unit tests in `shape.rs`: `[0,18]`, `[18,0]`, `[0,0]`, 3-D mixed, static and
  dynamic, both orders.

**Exit** — `cargo test -p rstsr-common` green; `rstsr-core` tests green;
DLPack/`Debug` outputs for empty tensors now show zero strides (expected).

## T5 — ADR-0011 (rstsr-book, separate repo)

- New `dev/adr/adr-0011-zero-size-layout-policy.mdx`, same frontmatter shape as
  ADR-0010: semantic rule (size 0 ⇒ strides unspecified, both contiguous),
  zero-stride constructor convention, `ld_*` contract, BLAS/LAPACK empty
  handling, NumPy references, rejected "as if zero were 1" alternative.
- Branch `261019/zero-size-layout-adr`; no push/PR; uncommitted for review.

## Deferred

- `openblas_set_xerbla` test harness (future task).
- Any change to `overwritable_convert_with_order` (documented caveat only).

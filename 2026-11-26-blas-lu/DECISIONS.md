# DECISIONS — `lu`, `lu_factor`, `lu_solve` on the BLAS devices

Recorded 2026-11-26 (Asia/Shanghai). rstsr `68f4891` (`main`), rstsr-ffi
`83efe50`, clean trees. Grill: `GRILL-R1-QUESTIONS.md` (Q1–Q15, facts +
recommendations), `GRILL-R2-QUESTIONS.md` (Q16–Q19, owner-delegated
evaluation), `GRILL-R3-QUESTIONS.md` (Q20–Q24). R2 and R3 answered "follow
your recommendation"; the answers of R1 are transcribed at the top of R2.

## Scope

- **D1** Three entries on the five BLAS devices, through the shared trees:
  `lu`, `lu_factor`, `lu_solve`. No faer implementation (precedent `lstsq`;
  faer 0.24.4's `partial_piv_lu` is a follow-up). Only OpenBLAS (OpenMP) is
  run locally, under `lapack` and `lapacke`; MKL/AOCL run the LU subsets;
  BLIS/KML are `cargo check`ed. No FFI change (`[sdcz]getrf_` and
  `LAPACKE_[sdcz]getrf` are bound everywhere).
- **D2** Out of scope: a permutation-matrix form (`p_indices=False`), full
  pivoting, `check_finite` / `overwrite_*` flags (ownership is the signal,
  no NaN pre-check), `cho_factor` / `cho_solve`, and any `?GETRS` / `?LASWP`
  translation.

## Shared types (`rstsr-linalg-traits/src/traits_def.rs`, `#region lu`)

- **D3** `pub struct LuResult<P, L, U> { pub p: P, pub l: L, pub u: U }`.
  Conversions: type-preserving `(P, L, U)` both ways (always); from
  `LuResult<Option<P>, L, U>`: `(P, L, U)` exactly when `p` is `Some`
  (panics otherwise) and `(L, U)` exactly when `p` is `None` (panics
  otherwise — the `EigResult` rule; a 2-tuple from an unpermuted result
  would silently drop the permutation). All coexist (coherence probed).
  Every BLAS `lu` form returns
  `LuResult<Option<Tensor<usize, B, D::SmallerOne>>, Tensor<T, B, D>,
  Tensor<T, B, D>>`.
- **D4** `pub struct LuFactorResult<LU, Piv> { pub lu: LU, pub piv: Piv }`
  with `(LU, Piv)` conversions both ways.
- **D5** `LuAPI`, `LuFactorAPI`, `LuSolveAPI` and `lu[_f]`, `lu_factor[_f]`,
  `lu_solve[_f]` in the three `duplicate_item` tables (after `LstsqAPI`),
  the traits in `rstsr_traits`, the functions in `rstsr_funcs`, the two
  structs in `rstsr_structs`.

## `lu`

- **D6** Semantics = SciPy `lu(a, permute_l, p_indices=True)`: `p` is the
  row-index array `(..., M)` of dtype `usize` with `A = L[P, :] @ U` (so
  `A = P L U` with `P[i, P[i]] = 1`; it is LAPACK's `ipiv` swap sequence
  applied in reverse order to `0..M-1`, rows `K..M` identity); `l` is
  `(..., M, K)` unit lower trapezoidal (rows `K..M` hold the multipliers for
  a tall slice); `u` is `(..., K, N)` upper trapezoidal; `K = min(M, N)`.
  Slices may be rectangular.
- **D7** Forms: the bare `(Tr)` is `permute_l = false`; `(Tr, bool)` takes
  the flag; four input kinds each (`&TensorAny`, `TensorView`, `Tensor`,
  `TensorMut`). `permute_l = true` gives `p: None` and `l = PL = L[P, :]`.
- **D8** Input policy: one `overwritable_convert_with_order` for the stack;
  an in-order owned / `TensorMut` input is consumed and destroyed by the
  factorization; an immutable input is copied; an off-order mutable input is
  factored from a work buffer and left untouched.
- **D9** A singular matrix (`info > 0`, factorization complete with an
  exactly-zero pivot) is a result: the factors are returned, no error.
- **D10** Zero-size: `K == 0` makes no driver call; `p = 0..M-1`, `l` and
  `u` zero, so `A = L[P] @ U` holds for every empty slice. An empty batch
  gives empty outputs. No `(1, 1)` special case. Divergences from SciPy's
  empty path: `p` is `(..., M)` (SciPy: `(..., 0)`) and the outputs are
  zero-initialized (SciPy: uninitialized).
- **D11** Errors: `ndim < 2` → `InvalidLayout`; `info < 0` → the
  internal-bug message; `-1010` / `-1011` → `MemoryError`.

## Driver tier (`rstsr-blas-traits`) — no new routine, no trait change

- **D12** `driver_impl/lapack/solve/getrf.rs` (both dtype blocks): the
  back-transpose is skipped only for `info < 0` (was `!= 0`). LAPACKE
  writes back for every `info`; `?GETRF` `info > 0` is a completed
  factorization whose factors `lu` hands out.
- **D13** `GETRF_::run()` is unchanged (`det` / `slogdet` / `inv` keep
  erroring on a singular matrix — a pre-existing NumPy divergence, a
  separate fix). New `GETRF_::run_with_info() -> Result<(TensorMutable2,
  Tensor<blas_int, Ix1>, blas_int)>` returns the raw `info` beside the
  factors and the 0-based pivots; `run()` is `run_with_info()` plus
  `map_lapack_info`. `lu` and `lu_factor` call `run_with_info()`.
- **D14** Skill `lapacke-translation` §3 gains one sentence: a routine whose
  `info > 0` is a valid result (`getrf`: completed factorization) writes
  back like LAPACKE (edit on contradiction).

## `lu_factor`

- **D15** SciPy's packed factorization:
  `LuFactorResult<Tensor<T, B, D>, Tensor<usize, B, D::SmallerOne>>` — `lu`
  is `(..., M, N)` packed (`U` in the upper triangle including the diagonal,
  the multipliers of `L` below it), `piv` is LAPACK's swap sequence made
  0-based, `(..., K)`, dtype `usize`. Rectangular slices and stacks as for
  `lu`. An owned / `TensorMut` input is factored in place and returned
  (`Out = LuFactorResult<Tr, ..>`, the `cholesky` rule, including the
  off-order work buffer with write-back) because the packed buffer is the
  result. A singular matrix returns its factors silently (SciPy prints a
  `LinAlgWarning`). Zero-size: `lu` is the (empty) input, `piv` is
  `(..., 0)`.

## `lu_solve`

- **D16** Two factor forms: `((lu, piv), b[, FlagTrans])` — SciPy's
  `lu_solve((lu, piv), b, trans)` with `piv` a `Tensor<usize>` — and
  `(&LuResult, b[, FlagTrans])`. `FlagTrans::{N, T, C}` are SciPy's `trans`
  0 / 1 / 2; `N` is the default. `b` is `(..., N)` or `(..., N, K)`; the
  batch dims of the factor and of `b` broadcast (`SolvePlan`).
- **D17** One composition, no `?GETRS`: per slice a permutation pass and two
  `TRSM` calls.
  - `A x = b`: forward swaps of the rows of `b` driven by `ipiv` / the
    scatter `y[P[i]] = b[i]`; then `TRSM(L, Lower, N, Unit)`,
    `TRSM(U, Upper, N)`.
  - `Aᵀ x = b` / `Aᴴ x = b`: `TRSM(U, Upper, T|C)`,
    `TRSM(L, Lower, T|C, Unit)`; then reverse swaps / the gather
    `x[i] = w[P[i]]`.
  The packed buffer is used as it is (`uplo` selects the triangle, `diag =
  Unit` leaves the diagonal — which holds `U`'s — unread). Same arithmetic
  as `?GETRS`, same `N²` traffic. The factor is only read.
- **D18** Argument tables, the `solve_general` kind classes (read-only =
  `&TensorAny` / `TensorView`, consumed = `TensorMut` / `Tensor`): A (all
  read-only → fresh `Tensor<T, B, IxD>`), B (`lu` / `piv` read-only, `b`
  consumed → `TrB`, solved in place), C (all consumed → `TrB`), each with
  and without `FlagTrans` (48 rows); the `&LuResult` form with `b` read-only
  (fresh) or consumed (`TrB`), with and without `FlagTrans` (8 rows). The
  static dims of `lu` (`D`) and `piv` (`D::SmallerOne`) are checked; the
  fresh result is dynamic-rank (the broadcast may change the rank).
- **D19** Edge rules: `p: None` → `InvalidValue` (a permuted factorization
  cannot be solved); a non-square factor or a `b` that breaks the shape /
  batch rules → `InvalidLayout`; `N == 0` or `K == 0` → an empty solution
  without a driver call; a singular factor yields `inf` / `nan`, no error.

## Tests, docs, delivery

- **D20** Driver `blas_tests_impl/driver_impl/f64/getrf.rs` (new; `GETRF`
  had one fingerprint test only): layout matrix for `a` (f-contig, f-padded,
  c-contig, c-padded, fully non-contiguous) with helpers that keep the
  layout, tall / square / wide, 0-based `ipiv` parity with the f-contig run,
  in-place buffer identity, destroyed-`a` parity of the row-major bridge
  against a direct column-major call, singular (valid factors in both
  orders), zero-size synthesis.
- **D21** Linalg `linalg_func/func_{f64,c64}/{lu,lu_factor,lu_solve}.rs` and
  validator sections `# ### lu`, `# ### lu_factor`, `# ### lu_solve`:
  fingerprints f64 4 / 2 / 3 and c64 2 / 1 / 2 large 2-D cases; at small
  sizes self-validation (`A ≈ L[P] U`, `A ≈ PL U`, unit diagonal,
  triangularity, `P` a permutation; `A x ≈ b`, `Aᵀ x ≈ b`, `Aᴴ x ≈ b`;
  agreement with `solve_general`; both `lu_solve` factor forms agree),
  singular inputs, batch parity against per-slice 2-D calls, layouts and
  both device orders, in-place and destroyed inputs, zero-size, the overload
  matrix, `ndim < 2` and the `lu_solve` errors.
- **D22** Docs: rustdoc on `LuResult` and `LuFactorResult`; `//!` docs on
  `blas_impl/{lu,lu_factor,lu_solve}.rs` in the eig / lstsq wording; one
  "DeviceFaer backend notes" bullet in `cheatsheet_linalg.md` for the three
  entries (no table row: the page's test is faer-backed); a note in
  `array_api_standard_extension.md`. No new page.
- **D23** Delivery: rstsr branch `261126/blas-lu` off `68f4891`; one PR; no
  auto-commit; PR only on instruction. Close-out: `README.md` here and a
  memory file.

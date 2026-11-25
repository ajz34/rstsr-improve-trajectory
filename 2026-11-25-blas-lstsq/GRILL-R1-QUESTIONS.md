# Grill R1 — least squares (`lstsq`) on the BLAS devices

rstsr `ed1f62d` (`main`, #171 merged; QR #169, eig #170 in), rstsr-ffi
`83efe50`; clean trees; **no rstsr file edited**. Facts verified 2026-11-25
(Asia/Shanghai) against NumPy 2.5.4, SciPy 1.18.1, the NumPy/SciPy sources,
LAPACK 3.12.x sources, the local faer 0.24.4 checkout, and the rstsr/ffi trees.

Recommended answers are marked ➡️; "yes" accepts the recommendation.

---

## Factual baseline (agent-authored)

### 1. Nothing exists today

- `lstsq` / `gels` / `least_squares`: **zero hits** in rstsr source or docs.
  No `LstsqAPI`, no `gelsd/gelss/gelsy` driver, no `GELSD_` builder.
- Closest entries: `pinv` (faer + BLAS; on BLAS it is `GESDD` + `GEMM` and
  returns `PinvResult { pinv, rank: usize }`, the rank being the **largest
  per-matrix rank over the stack**), `svd`/`svdvals`, `qr`, `solve_general`
  (square only — a rectangular `a` is explicitly rejected), `matrix_rank`
  (faer only, returns `Tensor<usize, _, IxD>`).
- The BLAS linalg entries are stacked (`StackLayout`, #168);
  `solve_general`/`solve_triangular`/`solve_symmetric` accept `b` as
  `(..., M, K)` or `(..., M)` and broadcast batch dims
  (`solve_plan` / `solve_stack_f`, `rstsr-linalg-traits/src/stack.rs:173,227`).
  **`solve_plan` asserts a square `a`** and sizes the solution as `M`, so a
  rectangular variant is needed.
- All five device crates consume the shared trees
  (`rstsr-blas-traits/src/lapack_*`, `driver_impl/{lapack,lapacke}`,
  `rstsr-linalg-traits/src/blas_impl/*` through `linalg_auto_impl/` symlinks).
  Only OpenBLAS (OpenMP) is run locally.
- Result-struct convention: plain structs with `From`/`Into` to the natural
  tuple (`QrResult`, `SVDResult`, `PinvResult`, `EigResult`). Overload
  convention: `duplicate_item` over `&TensorAny` / `TensorView` / `Tensor`
  (and `TensorMut` where the input is destroyed, as in `qr`/`det`; `svd` has
  no `TensorMut` form).

### 2. Reference semantics

**There is no standard to conform to.** `linalg.lstsq` was added to the
array-API draft in 2021-05 (PR #119) and **removed before the first published
revision** (PR #240, 2021-08, consortium vote); it is absent from the
2021.12–2025.12 stubs, the current 24-function linear-algebra extension, and
array-api-tests (`test_lstsq` never existed). The draft signature was
`lstsq(x1, x2, /, *, rtol=None)` returning a 4-tuple, with
`rtol = max(M,N)*eps` by default, `rank` shaped `shape(x1)[:-2]`, and a
`(K,)` residuals output with **no** emptiness rule — it never settled.

| | NumPy 2.5.4 | SciPy 1.18.1 |
|--|--|--|
| signature | `lstsq(a, b, rcond=None)` | `lstsq(a, b, cond=None, overwrite_a, overwrite_b, check_finite, lapack_driver=None)` |
| batching | **none** (`_assert_2d` rejects `a.ndim != 2`; `b` 1-D is promoted to `(M,1)`) | **batched** `(...,M,N)` × `(...,M)`/`(...,M,K)`, batch dims broadcast |
| default tolerance | `eps * max(M,N)` | `eps` (**no `max(M,N)` factor**) |
| driver | LAPACK **gelsd** (fixed) | gelsd (default) / gelss / gelsy |
| `x` | `(N,)` for 1-D `b`, else `(N,K)` | `(...,N)` / `(...,N,K)` |
| `residuals` | squared norms of GELSD workspace rows `n+1:m` when `rank==N && M>N`; else **empty `(0,)`**; 1-D `b` gives `(1,)` | same values but **`NaN`** where `rank<N` (`M>N`); `(0,)` when `M<=N`; 1-D `b` gives a **0-D/batch-shaped** result |
| `rank` | 0-D `np.int32` | 0-D `np.int64`; per-slice array when batched |
| `s` | `(min(M,N),)` descending | same; `None` for gelsy |
| negative tolerance | passed through (`-1` = machine eps) | **rejected** (`ValueError: Expected rcond >= 0.`) |
| f32 compute | promoted to f64 internally (`_commonType`) | stays f32 |

LAPACK's own RCOND rule (GELSD/GELSS): singular values `S(i) <= RCOND*S(1)`
are treated as zero; `RCOND < 0` selects machine epsilon, and `RCOND >= 1`
**also** silently falls to machine epsilon inside `DLALSD`
(`SRC/dlalsd.f:239-243`) — `rcond = 1.0` does not mean "keep everything".

Divergences that matter for a single rstsr API: tolerance default, residual
presentation, 1-D-`b` residual shape, `rank` container, batch support, and
gelsy's missing `s`.

### 3. LAPACK / FFI availability

- `[sdcz]gelsd_`, `[sdcz]gelss_`, `[sdcz]gelsy_` are bound in the shared
  `rstsr-lapack-ffi/src/lapack/ffi_extern.rs` (openblas/blis/aocl/kml) and in
  `rstsr-mkl-ffi/src/lapack/ffi_extern.rs`; `LAPACKE_[sdcz]{gelsd,gelss,gelsy}`
  and `_work` exist for the `lapacke` feature. `[sdcz]gelst_` is **MKL-only**
  (LAPACK 3.11, no LAPACKE wrapper); "GELSY2" does not exist in any tag.
- System OpenBLAS 0.3.32 (OpenMP) exports `dgelsd_`/`dgelss_`/`dgelsy_`
  (LAPACKE: none); the conda OpenBLAS 0.3.30 exports both.
- **GELSD workspace query**: one `LWORK = -1` call returns `WORK(1)` = optimal
  LWORK **and** `IWORK(1)` = minimum
  `LIWORK = max(1, 3*MINMN*NLVL + 11*MINMN)` (there is no LIWORK argument);
  complex `CGELSD`/`ZGELSD` additionally return `RWORK(1)` = minimum LRWORK in
  the same call (`SRC/dgelsd.f:355-356`, `cgelsd.f:379-381`). GELSS is a plain
  `lwork` query (complex `rwork` fixed `5*min(M,N)`); GELSY plain `lwork`
  (complex `rwork` fixed `2*N`).
- **GELSD destroys `A`; `B` is the solution workspace.** `LDB >= max(1,max(M,N))`,
  the solution is in the first `N` rows, and **the residual sum-of-squares for
  column i is the sum of squares of `B(n+1:m, i)` when `m >= n && rank == n`**
  (`dgelsd.f:98-105`). NumPy and SciPy both read residuals from there — no
  extra GEMM. GELSS documents the same; GELSY documents no residual property.
- **Zero size**: `M == 0 || N == 0` quick-returns rank 0 and empty `s`; the
  driver must not be called. **`NRHS == 0` is different**: DGELSD does not
  quick-return, and calling it with `nrhs = 0` makes the internal `DLALSD`
  report `info = -4` (SciPy 1.18.1 fails exactly this way: "Internal LAPACK
  errors: slice 0 emits lapack info=-4"). NumPy works around it by allocating
  one extra zero column and trimming; for `k == 0` the driver must be called
  with a padded column to obtain the real `rank`/`s`.
- **`info = -4` is special for GELSD**: "A contains a NaN or an infinity"
  (complex: or an entry whose absolute value overflows), and XERBLA is not
  called (`dgelsd.f:176-186`). Generic `map_lapack_info` would report it as
  "illegal argument 4 (internal bug)". Under the `lapacke` feature LAPACKE's
  own NaN checks return `-5` (A), `-7` (B), `-10` (rcond) instead — the two
  configs need one unified mapping.
- SciPy 1.18's batched zero-size path is **not batch-correct** (`x` drops the
  batch axes, `n == 0` residuals reduce over the batch axis, `rank` comes back
  as a scalar), so those rules have to be defined by us, not copied.

### 4. Machinery and policy that applies

- Helpers: `overwritable_convert[_with_order]`, `get_output_order`,
  `map_lapack_info`, `lwork_from_query`, `liwork_from_query`, `StackLayout`,
  `stack_slice_at[_mut]`, `broadcast_shapes_f`.
- `lapacke-translation` gap notice (as in QR/eig): §3a has **no row for
  gelsd** (query returns lwork + liwork in one call; complex adds lrwork via
  `rwork(1)`) and §3b has no `info = -4` NaN rule; §2 has no "solution and
  residuals live in the `B` workspace" pattern. Record the gaps; do not edit
  the skill without the owner's word.
- Test machinery: `blas_tests_impl/{driver_impl,linalg_func}` symlinked into
  the five crates, plus Python fingerprint validators with per-routine
  sections (`driver_validation_f64.py`, `func_validation_{f64,c64}.py`).
- The cheat-sheet linalg page/test is **faer-backed**
  (`rstsr/rstsr/tests/cheatsheet_linalg.rs`), so a BLAS-only entry cannot get a
  cheat-sheet row (the page/test/parity triad rule).

---

## Round 1 — the frontier

❓ **Q1 - Scope: the five BLAS devices only, no faer implementation.** The
faer device has no `lstsq` either, and faer's own `Svd::solve_lstsq` asserts
`m >= n` and divides by *all* singular values (no rank threshold, no
underdetermined support) — a faer implementation would be a custom
SVD-threshold-min-norm composition, not a thin call. Precedent:
`solve_symmetric` is BLAS-only. `LstsqAPI` is therefore implemented for
`DeviceBLAS` only; faer and the py shim are untouched, and the docs say so.
(Consequence: no cheat-sheet row, because its test is faer-backed.)

➡️ Yes. A faer impl is a self-contained follow-up; if it lands, the cheat-sheet
row can follow.

---

❓ **Q2 - Result surface: NumPy's four-output result as a rstsr struct, not a
solution-only function.** Add `LstsqAPI` to `traits_def.rs` and
`pub struct LstsqResult<X, Res, Rank, S> { x, residuals, rank, s }` with
`From`/`Into` to the natural tuple, free functions
`rt::linalg::lstsq`/`lstsq_f`, and the call forms `lstsq((&a, &b))` (default
tolerance) / `lstsq((&a, &b, rcond))`. The 2021 array-API removal argument
(the 4-tuple duplicates `pinv`/`svdvals`/`matrix_rank`) is real, but
composition would cost another full SVD per call, `matrix_rank` is faer-only,
and NumPy/SciPy's 4-tuple is the parity target; unused fields are free to
ignore.

➡️ Yes. `x` is the solution; `residuals`/`rank`/`s` are exactly what LAPACK
already returns or cheaply exposes.

---

❓ **Q3 - Stacking: n-dim input and output, SciPy-1.18-shaped, with the 2-D
case behaving like NumPy.** `a` is `(..., M, N)`; `b` is `(..., M)` or
`(..., M, K)` (a bare `(M,)` is allowed and broadcasts); batch dims of `a` and
`b` broadcast; `x` is `(..., N)` / `(..., N, K)`; `s` is `(..., min(M,N))`.
NumPy 2.5 is 2-D only; SciPy 1.18 is the only batch reference, and every BLAS
linalg entry in rstsr is stacked (#168). This needs a rectangular variant of
`solve_plan` (it currently asserts square `a`), and the batched zero-size
rules are ours to define (SciPy's are buggy).

➡️ Yes. 2-D calls still match NumPy/SciPy values exactly, so this is additive.

---

❓ **Q4 - `rcond`: scalar `T::Real`, default `eps * max(M,N)`, value passed
through to LAPACK.** NumPy and the deleted array-API draft default to
`max(M,N)*eps`; rstsr's `pinv` uses the same. SciPy 1.18 defaults to `eps`
(no factor) and rejects negatives. Recommendation: NumPy's default; the
explicit form is `lstsq((&a, &b, rcond))` with `rcond: T::Real` (for an f32
input write `1e-8_f32`, the documented `pinv` gotcha); no `Option` in the
tuple; negative values are passed through (`-1` → machine epsilon, NumPy
behaviour), documented; **no per-slice rcond array** in this campaign (unlike
`PinvRtol`), because NumPy/SciPy only accept scalars and a per-matrix
tolerance is its own API decision.

➡️ Yes.

---

❓ **Q5 - Driver: GELSD only.** GELSD is both references' default and the only
one that returns `rank`, `s`, and the residual workspace in one call. GELSS is
the same SVD family with a smaller/fixed workspace — a redundant second
translation; GELSY cannot supply `s` or residuals (a different return
contract), and GELST is MKL-only. No `driver` selector on the entry (SciPy's
selector changes the residual contract); the public `GELSD_` builder is the
extension point for a future driver campaign.

➡️ Yes. A composition over the existing `GESDD` (`pinv`-style) was considered
and rejected: it needs a second SVD for `s`/`rank` unless pinv's internals are
opened up, needs an extra GEMM for residuals, and does not match NumPy's
`gelsd` results bit-for-bit.

---

❓ **Q6 - Residuals presentation: SciPy-style `(..., K)` with `NaN` where
undefined, or NumPy-style empty `(0,)`?** Both read the same GELSD workspace
values; only presentation differs. NumPy: `(K,)` only when `rank == N && M > N`,
else an empty `(0,)`; 1-D `b` gives `(1,)`. SciPy: `(..., K)` with `NaN` on
slices where `rank < N`, `(..., 0)` when `M <= N`, 1-D `b` squeezes to
`(...,)`. Recommendation: **SciPy's shape-aligned convention with dtype
`T::Real`** (not SciPy's complex-residual quirk): it is the only one that
stays meaningful under stacking (a global empty cannot say "slice 1 is
deficient"), and it matches the LAPACK truth "undefined, not zero". This is a
deliberate, documented divergence from NumPy (precedent: `eig` always returns
complex).

➡️ Yes, SciPy shapes + real dtype.

---

❓ **Q7 - `rank` representation: per-slice `Tensor<usize, B, IxD>` (0-D when
unbatched) instead of a single `usize`.** `matrix_rank` returns
`Tensor<usize, _, IxD>` and `det` returns a 0-D tensor; `pinv` returns a scalar
`usize` = the largest per-matrix rank (a pre-existing lossy compromise). With
stacking, per-slice rank is the honest output (SciPy's batched `rank` is per
slice). Recommendation: tensor, shaped as the batch (empty batch → empty
tensor), with `LstsqResult` carrying it as the `Rank` generic; `pinv` is left
alone (out of scope, but noted as an inconsistency).

➡️ Yes. `LstsqResult<X, Res, Rank, S>` stays fully generic like `EigResult`.

---

❓ **Q8 - Input policy: destructive `a`/`b` under the QR/eig rule.** GELSD
destroys `A`; `B` is the solution workspace of shape `max(M,N) × K` in the call
order. Immutable inputs are copied; owned / `TensorMut` inputs are consumed as
scratch and **not written back**; all four input kinds are accepted (like
`qr`), and outputs are fresh in the call order. An owned `b` can serve
directly as the workspace only when `M >= N` and its layout already matches
`LDB`; otherwise it is copied. Recommendation: yes for `a`; accept the same
four kinds for `b` for symmetry, documenting that `b` is never written back
(even as `TensorMut`). No SciPy-style `overwrite_a`/`overwrite_b` flags —
ownership is the signal.

➡️ Yes.

---

❓ **Q9 - Zero-size rules (batch-correct, defined by us).**
(a) `min(M,N) == 0`: never call the driver; `x = zeros((..., N[, K]))`,
`rank = 0`, `s` empty `(..., 0)`, residuals = column sums of `|b|²` when
`N == 0 < M` (NumPy and SciPy agree), otherwise `(..., 0)`.
(b) `K == 0` with `M,N > 0`: **pad to one zero column and call GELSD** (NumPy's
workaround; direct `nrhs = 0` returns `info = -4`), then trim `x` and
residuals to zero columns — `rank` and `s` are real.
(c) empty batch: empty outputs with the batch shape, no driver calls.
(d) 1-D `b`: the same rules with `K = 1`.
The "never call the driver at zero size" rule is about a zero **matrix** axis;
`K == 0` is a zero **rhs** and still needs the SVD for `rank`/`s`.

➡️ Yes.

---

❓ **Q10 - Tests, docs and delivery.**
- Driver conformance `driver_impl/f64/gelsd.rs` (+ c64): layout matrix for both
  `a` and `b` (f-contig, f-prefer `ld > dim`, c-contig, c-prefer, fully
  non-contiguous), tall/square/wide/rank-deficient, residual sums checked
  against `||b - Ax||²` in the full-rank tall case, rcond rank sweep, the
  `nrhs = 0` pad path, zero-size, destroyed-input behaviour, `info = -4`
  (NaN in `A`).
- Linalg conformance `linalg_func/func_{f64,c64}/lstsq.rs`: NumPy fingerprints
  for 2-D `x`/`rank`/`s`/residuals; SciPy-1.18 fingerprints for batched
  `x`/`rank`/`s` (and our residual convention); 1-D and 2-D `b`; layouts;
  batch broadcast; zero-size; errors (`ndim < 2`, row mismatch, batch
  mismatch, NaN `a`); self-validation (`Aᴴ(Ax-b) ≈ 0`, `x ⊥ null(A)` for
  underdetermined, `s` descending, `rank` consistent with `s`/`rcond`,
  `‖x‖` minimal for underdetermined).
- Validator scripts get a `# ### lstsq` section; small sizes except one
  fingerprint-sized solve per dtype (test-size policy).
- Run both `lapack` and `lapacke` on OpenBLAS, MKL/AOCL subsets (QR/eig
  precedent); BLIS/KML compile-only (no local library).
- Docs: entry rustdoc + Overloads Table per `api-doc-conventions`, one `//!`
  module doc, a "BLAS-only, no faer" note on the cheat-sheet linalg page, and
  an `array_api_standard_extension.md` note that `lstsq` is **beyond** the
  array API (proposed and removed in 2021). No cheat-sheet row (the test is
  faer-backed).
- Gap notice: record the missing `lapacke-translation` §3a/§3b rows
  (gelsd/gelss/gelsy workspace queries; `info = -4`; B-workspace solution and
  residuals); do not edit the skill.
- Delivery: branch `261125/blas-lstsq` off `ed1f62d`, one PR, no auto-commit,
  PR only on instruction; `DECISIONS.md` + `PLAN.md` in this directory.

➡️ Yes.

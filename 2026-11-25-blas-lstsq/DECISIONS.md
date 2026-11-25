# DECISIONS — `lstsq` on the BLAS devices

Recorded 2026-11-25 (Asia/Shanghai). rstsr `ed1f62d`, rstsr-ffi `83efe50`,
clean trees. Grill R1: `GRILL-R1-QUESTIONS.md`; plan: `PLAN.md`.

## Answers as given (transcribed from chat, 2026-11-25)

- **Q1, Q2, Q4, Q6, Q7, Q8, Q9, Q10**: "Follow your recommendation."
- **Q3**: "existing code should be different on tensor default order … for
  column major, it may be a `(M, N, ...)`, b `(M, [K, ] ...)`, i.e. `...` at
  right side for col-major. Please make fact check on this behavior on
  existing BLAS solve plan." → fact-checked below (D3): confirmed.
- **Q5**: "Okay. … I just feel program efficiency take the lead, so I also
  want to consult if GELSD is better for efficiency." → benchmarked below
  (D5): GELSD is the efficient complete-contract choice.
- **Q10**: also "introduce python fingerprint tests; you need also create new
  fingerprint value tests in python, and test rust side compared to those
  fingerprints … only test few cases on fingerprints … usually be only of 2-5
  cases for one function … to test correctness on large 2-dim matrix."

---

### D1 — Scope: the five BLAS devices only, no faer implementation

`LstsqAPI` is implemented for `DeviceBLAS` only; `DeviceFaer` and the py shim
are untouched. faer has no lstsq either; its `Svd::solve_lstsq` asserts
`m >= n` and divides by all singular values, so a faer impl would be a custom
composition. Precedent: `solve_symmetric` is BLAS-only. Consequence: no
cheat-sheet row (its Rust test is faer-backed) — a note instead (D14).

### D2 — Result surface: the four NumPy outputs in a struct

`pub struct LstsqResult<X, Res, Rank, S> { pub x: X, pub residuals: Res, pub rank: Rank, pub s: S }`
in `rstsr-linalg-traits/src/traits_def.rs` (`#region lstsq`), with generic
`From`/`Into` for `(X, Res, Rank, S)`; exported from `rstsr_structs`.
`LstsqAPI`/`lstsq`/`lstsq_f` registered in the three `duplicate_item` tables;
call forms `lstsq((&a, &b))` (default tolerance) and `lstsq((&a, &b, rcond))`,
`_f` twins, plus the owned/view input-kind variants. Solution-only is rejected:
the extra outputs are exactly what LAPACK returns, and composition would cost
another SVD.

### D3 — Shape convention: device-default order, matrix axes last under RowMajor, first under ColMajor

Fact check (user's reading confirmed): the shared batch walk splits the
**per-matrix axes as the last ones under `RowMajor` and the first ones under
`ColMajor`** (`stack_split_shape` / `stack_shape`,
`rstsr-linalg-traits/src/stack.rs:19,29`), and the order used by the entries is
`a.device().default_order()` (`blas_impl/solve_general.rs:24`,
`blas_impl/svd.rs:35`), not the tensor's runtime layout. The walk is
stride-based, so any physical layout is sliced correctly as long as the shape
follows the device convention.

| order | `a` | `b` matrix | `b` vector | `x` | `s` / `residuals` |
|--|--|--|--|--|--|
| RowMajor | `(..., M, N)` | `(..., M, K)` | `(..., M)` | `(..., N[, K])` | `(..., min)` / `(..., K)` |
| ColMajor | `(M, N, ...)` | `(M, K, ...)` | `(M, ...)` | `(N[, K], ...)` | `(min, ...)` / `(K, ...)` |

`rank` is batch-shaped in both orders (no inner axis). `solve_plan` is
square-only and sizes the solution as `M`, so a rectangular `LstsqPlan` is
added (PLAN B3). Inputs are read, outputs are produced, in the device order;
the builder chooses its own call order per slice (as `GESDD_` does).

### D4 — `rcond`: scalar `T::Real`, default `eps * max(M, N)`, passed through

- Default (2-arg form): `eps * max(M, N)`, NumPy's and the deleted array-API
  draft's choice (rstsr `pinv` uses the same).
- Explicit value: passed to LAPACK unchanged. Documented LAPACK rule: a value
  `<= 0` **or `>= 1`** selects machine epsilon inside `DLALSD`
  (`dlalsd.f:239-243`) — `rcond = 1.0` does not mean "keep everything".
- `rcond = NaN` is rejected with `InvalidValue` (LAPACK would keep full rank;
  LAPACKE rejects it — rejecting keeps the two configs identical).
- No per-slice rcond array in this campaign (unlike `PinvRtol`); the tuple has
  no `Option` (the 2-arg form is the default).
- f32 callers must write the literal as f32 (`1e-8_f32`), the documented
  `pinv` gotcha.

### D5 — Driver: GELSD only; efficiency verdict from a local benchmark

Decision: `GELSD_` only, public in `rstsr-blas-traits`; GELSS/GELSY/GELST are
not exposed and there is no `driver` selector (SciPy's selector changes the
residual contract; `gelst` is MKL-only).

Efficiency (AMD Ryzen 9 9950X3D, 16 cores/32 threads; conda OpenBLAS 0.3.30,
SciPy 1.18.1 `lstsq(..., check_finite=False)`, best of 3; ms):

| problem (M×N×K) | gelsd | gelss | gelsy | SVD+GEMM |
|--|--|--|--|--|
| 8192×256×1, 1T | 51.1 | 628.5 | 45.2 | 83.2 |
| 8192×256×32, 1T | 57.0 | 710.9 | 47.9 | 84.7 |
| 4096×1024×1, 1T | 398.1 | 28999 | 237.2 | 560.2 |
| 2048×2048×1, 1T | 942.3 | 62636 | 338.3 | 1637.1 |
| 512×2048×1, 1T | 66.5 | 1317.9 | 83.3 | 134.4 |
| 4096×4096×1, 16T | 5734.3 | — | 1306.4 | 6543.8 |
| 2048×2048×1, 16T | 556.7 | — | 137.2 | 872.7 |
| 8192×256×1, 16T | 43.6 | — | 25.9 | 54.6 |

- `gelss` (QR + DBDSQR) is 10–60× slower — never a candidate.
- `gelsy` is 2.4–4.4× faster on square/large problems but **cannot supply `s`
  or residuals** (rank only), so it cannot implement the 4-tuple; it is also
  slower in the underdetermined 512×2048 case.
- The SVD+GEMM composition (GESDD + threshold + matmuls) is 1.1–1.9× slower
  than GELSD and needs U/V memory.
- GELSD is the fast path among drivers that satisfy the contract; the LAPACK
  call dominates and rstsr's own copies are O(MK+MN).
- Note for later: LAPACK's thread scaling is poor (2048²: 942→557 ms on 16
  cores) and the per-slice loop is serial, so batched efficiency is
  driver-bound. A solution-only `gelsy`/QR fast path is a **future** API, not
  this campaign.

### D6 — Residuals: SciPy shape convention, real dtype

- 2-D `b`, `M > N`: shape `batch + (K,)`; `NaN` on slices where `rank < N`.
- 2-D `b`, `M <= N`: shape `batch + (0,)` (residuals are exactly zero).
- 1-D `b`: same values squeezed to `batch` (0-D when unbatched); `M <= N` gives
  `batch + (0,)`.
- `K == 0`: `batch + (0,)`.
- dtype `T::Real` (NumPy's choice, not SciPy's complex-residual quirk).
- Rationale: the only convention that stays meaningful per slice under
  stacking; "undefined, not zero" matches LAPACK's contract. Deliberate,
  documented divergence from NumPy (precedent: `eig` always-complex).

### D7 — `rank`: per-slice `Tensor<usize, B, IxD>`

Shape = batch (0-D when unbatched), matching `matrix_rank`/`det`; `pinv`'s
scalar max-over-batch is a pre-existing inconsistency left alone.

### D8 — Input policy: destructive `a`, `b` always materialized into the workspace

GELSD destroys `A`. `a` is consumed when owned/`TensorMut` and its layout fits
the call order (`overwritable_convert_with_order`), otherwise copied from an
immutable input. `B` must be `max(M,N) × K` in the call order, so `b` is always
copied into a fresh workspace (rows `0..M`); nothing is written back, including
for `TensorMut` `b` (accepted only for overload symmetry). All four input kinds
for both arguments; no `overwrite_*` flags. Outputs are fresh, allocated in the
device order.

### D9 — Zero-size rules

- `min(M, N) == 0`: the driver is not called. `rank = 0`, `s` empty,
  `x = zeros((..., N[, K]))`; residuals `batch + (K,)` with column sums of
  `|b|²` when `N == 0 < M` (NumPy and SciPy agree), otherwise `(..., 0)`.
- `K == 0` with `M, N > 0`: **pad `nrhs` to 1** (one zero column) and call
  GELSD, then trim `x`/residuals to zero columns; `rank`/`s` are real. Direct
  `nrhs = 0` makes the internal `DLALSD` return `info = -4` (SciPy 1.18.1 fails
  this way); NumPy pads the same way.
- Empty batch: no driver calls; outputs have the batch shape with a 0 axis.
- 1-D `b`: same rules with `K = 1`.

### D10 — Errors

- `InvalidLayout`: `ndim(a) < 2`; `b` rank not `a` rank or one less; `b` row
  dimension mismatch; incompatible batch broadcast.
- `DeviceMismatch`: `a` and `b` on different devices.
- `InvalidValue`: `rcond` NaN; NaN/Inf in `a` (`info = -4` on the Fortran path,
  `-5`/`-7` under LAPACKE); other negative `info` (internal bug); `info > 0`
  (SVD did not converge).
- `MemoryError`: LAPACK workspace allocation sentinels `-1010`/`-1011`.
- Rank deficiency is not an error.

### D11 — Driver tier

- `rstsr-blas-traits/src/lapack_lstsq/{mod.rs, gelsd.rs}`:
  `pub trait GELSDDriverAPI<T> { unsafe fn driver_gelsd(order, m, n, nrhs,
  a: *mut T, lda, b: *mut T, ldb, s: *mut T::Real, rcond: T::Real,
  rank: *mut blas_int) -> blas_int; }`.
- `GELSD_` builder fields: `a: TensorReference<Ix2>`, `b:
  TensorReference<Ix2>`, `rcond: T::Real` (required); call order by
  `get_output_order` over both inputs; `b` workspace
  `zeros_f(([max(M,N), K], order, &device))` with rows `0..M` assigned from
  `b` (one pass; `K == 0` → one zero column); `x = b_work[0..N, :]` copy.
- Returns `(x: Tensor<T,B,Ix2>, s: Tensor<T::Real,B,Ix1>, rank: usize,
  residuals: Option<Tensor<T::Real,B,Ix1>>)`; `None` when `M <= N` or
  `rank < N`, `Some(len K)` otherwise (length 0 for `K == 0`). Residual sums
  are read from the destroyed `B` rows `N..M` (call-order strides, `|z|²`),
  no extra GEMM.
- Zero-size `min(M,N) == 0` synthesized inside the builder (D9); `nrhs`
  padding inside the builder.
- `-4` (Fortran) / `-5`, `-7` (LAPACKE) mapped by the builder to one
  `InvalidValue` "contains a NaN or infinity" message; `-10` (LAPACKE rcond
  NaN) cannot occur after the builder's own check.
- `lapack` thunk: query `LWORK=-1` once (work + `IWORK(1)`; complex also
  `RWORK(1)` with a 1-element `rwork` allocated before the query), allocate
  `lwork`/`liwork`(/`lrwork`), RowMajor bridge transposes `a` and `b`
  (`lda_t = max(1,M)`, `ldb_t = max(1,max(M,N))`), back-transposes both after
  the call (keeps the residual rows; LAPACKE does the same). `lapacke` thunk:
  pass-through `LAPACKE_[sdcz]gelsd` (it queries, allocates and transposes
  itself; user `lda >= N`, `ldb >= K`).
- Registration: `pub mod lapack_lstsq` in `lib.rs`, `pub use` in `prelude.rs`,
  `+ GELSDDriverAPI<T>` in `LapackDriverAPI`, family in each device's
  `driver_impl/{lapack,lapacke}/mod.rs`.

### D12 — Entry tier

- `rstsr-linalg-traits/src/ref_impl_blas.rs` `#region lstsq`:
  `ref_impl_lstsq_f(a: TensorView<Ix2>, b: TensorView<Ix2>, rcond) ->
  (x, s, rank, residuals)` with the `with_blas_num_threads` wrapper.
- `rstsr-linalg-traits/src/blas_impl/lstsq.rs`: `blas_impl_lstsq_f(a:
  TensorReference<IxD>, b: TensorReference<IxD>, rcond: Option<T::Real>)`:
  `order = a.device().default_order()`, `LstsqPlan`, per-slice
  `ref_impl_lstsq_f`, residual policy (D6), rank tensor (D7), shapes via
  `stack_shape`; empty batch returns empty outputs.
- All `LstsqAPI` impls return
  `LstsqResult<Tensor<T,B,IxD>, Tensor<T::Real,B,IxD>, Tensor<usize,B,IxD>,
  Tensor<T::Real,B,IxD>>` (dynamic shapes, like `solve_general`); 16 impl
  rows (4 input kinds × 4, `_f` twins) in `blas_impl/lstsq.rs`; owned/view
  inputs forward to `blas_impl_lstsq_f`.
- `LstsqPlan` (+ the rectangular split) in `stack.rs`; `solve_plan` keeps its
  square check on top of the shared shape-splitting helper.

### D13 — Tests and Python fingerprints

- Driver conformance (`driver_impl/f64/gelsd.rs`, f64 only — house style):
  layout matrix for `a` and `b`, tall/square/wide, rank-deficient, residual
  sums vs `‖b − Ax‖²`, rcond rank sweep, `K == 0` pad, zero-size,
  `info = -4` (NaN `a`).
- Entry conformance (`linalg_func/func_f64/lstsq.rs`,
  `func_c64/lstsq.rs`): 1-D/2-D `b`, layouts, batch broadcast, order
  (RowMajor + ColMajor), zero-size, errors, invariants (`Aᴴ(Ax−b) ≈ 0`,
  min-norm for wide, `s` descending, rank consistency).
- **Python fingerprints** (per user): 2–5 large 2-D cases per function, values
  generated and asserted in the validator scripts, hardcoded in Rust with the
  house `fingerprint` (`dot(cos(arange), x.ravel(C))`):
  - `driver_validation_f64.py`: 4 cases — (1024×512, K=1), (1024×512, K=3),
    (512×1024, K=1) min-norm, (1024×512 rank-deficient K=1); fingerprint `x`
    (full-rank only), `s`, residual sums; exact `rank`.
  - `func_validation_f64.py`: 4 cases through the entry — tall + 1-D `b`,
    tall + 3-column `b`, wide + 1-D `b`, rank-deficient (NaN residuals).
  - `func_validation_c64.py`: 3 cases — tall K=1, tall K=2, wide K=1.
  - Same manifest data (`a-{f64,c64}.npy`; RHS from `b-{f64,c64}.npy`); no new
    npy files.
- Small-size parameter/overload/batch/zero-size tests (no fingerprints).
- Zero-size LAPACK-boundary cases in
  `crates-device/rstsr-openblas/tests/zero_size/lapack.rs` (OpenBLAS-only
  harness).
- Runs: OpenBLAS `lapack` + `lapacke`; MKL and AOCL subsets; BLIS/KML
  compile-only.

### D14 — Documentation

- Source docstrings in the `blas_impl/svd.rs` house style on
  `blas_impl_lstsq_f` (shape/order convention, destructiveness, dtype rule,
  residual/rank policy, BLAS-only). The generated `rt::linalg::lstsq`
  carries no rustdoc, consistent with `svd`/`qr` (the API-doc skill governs
  `rstsr-core`; the generated linalg entries are undocorated by design).
- `rstsr/src/docs/cheatsheet_linalg.md` "DeviceFaer backend notes": **no
  `DeviceFaer` implementation** (BLAS-only), the four outputs, residuals
  `NaN`/empty convention, `rcond` default. No table row: the page's Rust test
  is faer-backed and a BLAS-only row cannot be executed there (same treatment
  as `solve_symmetric`).
- `rstsr/src/docs/array_api_standard_extension.md` Notes: `lstsq` is **beyond**
  the standard (its draft member was removed in 2021), BLAS-only.

### D15 — Gap notice for skill `lapacke-translation`

Recorded, skill not edited (QR/eig precedent): §3a has no `gelsd`/`gelss`/
`gelsy` workspace-query rows (GELSD returns LWORK + LIWORK in one
`LWORK=-1` call; complex adds LRWORK via `RWORK(1)`); §3b has no `info = -4`
"NaN/Inf in `A`" rule (and LAPACKE's `-5`/`-7`); §2 has no "solution and
residuals live in the `B` workspace, `LDB >= max(M,N)`" pattern.

Implementation findings to fold into a future skill revision (2026-11-25):
- The row-major bridge must copy **`max(M,N)` rows** of `B` in and back:
  for `M < N`, `DGELSD` calls `DLASET('F', N-M, NRHS, 0, 0, B(M+1,1), LDB)`
  and leaves the length-`N` solution in rows `1:N`. Copying `M` rows returns a
  truncated solution with an identical `S` (caught by the wide 512×1024
  fingerprint case).
- `NRHS = 0` must be avoided (`DLALSD` reports `info = -4`, SciPy 1.18.1
  raises "Internal LAPACK errors: slice 0 emits lapack info=-4"); pad to one
  zero column and trim the outputs.
- The `-4` NaN/Inf check is not reliable: LAPACK computes
  `DLANGE('Max')` over `A`, and on x86 the reduction may swallow NaN (MAXSS),
  so a NaN input can instead surface as a convergence failure, not `-4`. Keep
  the mapping, but tests should assert "an error", not the exact message.

### D16 — Delivery and out of scope

- Branch `261125/blas-lstsq` off `ed1f62d`; **no auto-commit** in rstsr, one
  PR only on instruction; task dir `README.md` + memory at close.
- Out of scope: faer/py-shim impl, GELSS/GELSY/GELST, driver selector,
  pivoting/`overwrite_*` flags, per-slice `rcond`, complex residual dtype,
  f32-in-f64 promotion, cheat-sheet row, batched per-slice parallelism,
  solution-only fast path.

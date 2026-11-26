# Grill R1 — LU decomposition (`lu`) on the BLAS devices

rstsr `68f4891` (`main`), rstsr-ffi `83efe50`; clean trees; **no rstsr file
edited**. Facts verified 2026-11-26 (Asia/Shanghai) against the SciPy 1.18.1
source and runtime, the LAPACK 3.12.1 / LAPACKE sources, the local faer 0.24.4
checkout and the rstsr tree.
Recommended answers are marked ➡️; "yes" accepts the recommendation.

---

## Factual baseline (agent-authored)

### 0. What the QR, eig and lstsq sessions settle for this campaign

Read in full: `2026-11-24-blas-qr/`, `2026-11-24-blas-eig/`,
`2026-11-25-blas-lstsq/` (init prompts, every round, answers, decisions, plans,
outcome READMEs). Carried over without re-asking:

- **Layers**: driver tier (`rstsr-blas-traits/src/lapack_*` + shared
  `driver_impl/{lapack,lapacke}`), composition `ref_impl_*` in
  `ref_impl_blas.rs`, entry `blas_impl/<fn>.rs` symlinked as
  `linalg_auto_impl/<fn>.rs` into the five device crates, tests in
  `blas_tests_impl/{driver_impl,linalg_func}` symlinked the same way.
- **Signature style**: SciPy-shaped; result struct with tuple `From`/`Into`;
  runtime options are tuple elements (`(a, QrMode)`); an option that changes
  the output arity is a type-level marker (`Homogeneous`, eig R3).
- **Stacking**: n-dim via `StackLayout`, matrix axes last under `RowMajor`
  and first under `ColMajor`; outputs keep the batch shape.
- **Input policy**: an in-order owned / `TensorMut` input is consumed and
  destroyed, an immutable or off-order one is only read; no `overwrite_*` /
  `check_finite` flags, no NaN pre-check.
- **Tests**: Python validators generate the fingerprints (2–5 large 2-D
  cases per function), overload / parameter / layout tests at size 3–5,
  layout tests must deliver real padded/strided layouts; OpenBLAS under
  `lapack` and `lapacke`, MKL/AOCL subsets, BLIS/KML `cargo check`.
- **Owner decisions on record**: thunks follow LAPACKE literally; plain
  `rows * ld` spans; a failed call skips the write-back; separate commits;
  no decision ids in rstsr code; one-sentence docstring titles.
- **Scope precedent**: `lstsq` is BLAS-only (no faer entry, no cheat-sheet
  row, a "DeviceFaer backend notes" bullet instead).

### 1. What exists today

- **No `lu` anywhere**: no `LuAPI`, no `faer_impl/lu.rs`, no `lu_factor` /
  `lu_solve`. NumPy and the array API have no `lu`; SciPy is the only
  reference.
- faer 0.24.4 offers `MatRef::partial_piv_lu()` → `PartialPivLu` with `L()`,
  `U()` (`(M, K)` / `(K, N)`, the same trapezoidal shapes as SciPy) and
  `P()` (`PermRef`, forward + inverse index arrays). Not wired in rstsr.
- **`GETRF` is already translated** (LAPACKE row exists, no workspace):
  - builder `GETRF_ { a }` (`lapack_solve/getrf.rs`) →
    `(TensorMutable2` holding the L/U factors in place`, ipiv:
    Tensor<blas_int, Ix1>)`, pivots made 0-based, `k == 0` synthesized
    without a driver call;
  - trait `GETRFDriverAPI<T>`, a member of `LapackDriverAPI<T>`;
  - thunks `driver_impl/lapack/solve/getrf.rs` (real / complex blocks,
    row-major bridge) and `driver_impl/lapacke/solve/getrf.rs` (pass-through);
  - consumers: `det`, `slogdet` (`ref_impl_slogdet_f`) and `inv`
    (`ref_impl_inv_f`).
- **Two properties of that path matter for `lu`**:
  1. the builder runs `map_lapack_info(info, "Lapack GETRF")?`, so
     `info > 0` — LAPACK: "`U(i,i)` is exactly zero; the factorization **has
     been completed**" — is an error. As a side fact, `det` / `slogdet` /
     `inv` of an exactly singular matrix therefore error out on the BLAS
     devices today (NumPy: `det` → `0`, `slogdet` → `(0, -inf)`, `inv` →
     `LinAlgError`).
  2. the row-major thunk returns before the back-transpose when
     `info != 0` (the accepted failed-call divergence), so on that path the
     valid factors of a singular matrix never reach the caller's buffer.
     `LAPACKE_dgetrf_work` transposes back for every `info` (even `< 0`).
- **Test assets**: `GETRF` has one fingerprint test only
  (`driver_impl/f64/getrf_getri.rs`: `lu` 5397.198541468395, pivots
  −14.694714160751573) — no layout matrix, no in-place or parity tests.
  The validators have a `# ### getrf, getri` driver section and no `lu`
  section.

### 2. Reference semantics (SciPy 1.18.1, `scipy/linalg/_decomp_lu.py` + `_linalg_lu_det.hh`)

| | SciPy 1.18.1 |
|--|--|
| signature | `lu(a, permute_l=False, overwrite_a=False, check_finite=True, p_indices=False)` |
| batching | `(..., M, N)`; one `?getrf` per slice on a column-major copy |
| `p_indices=True` | `P` `(..., M)` `int32`: row indices with **`A = L[P, :] @ U`** (⟺ `A = P L U`, `P[i, P[i]] = 1`) |
| `L` | `(..., M, K)` unit lower trapezoidal, `K = min(M, N)`; rows `K..M` hold the multipliers (non-zero) for a tall `A` |
| `U` | `(..., K, N)` upper trapezoidal |
| `permute_l=True` | `(PL, U)`, `PL = L[P, :]`, `A = PL @ U` |
| singular | `info > 0` is not surfaced: the factors are returned, no warning |
| `(1, 1)` | special-cased: `P = [0]` (int64 quirk), `L = 1`, `U = a` |
| empty (`min(shape) == 0`) | `P (..., 0)` in **every** empty case (also `(M, 0)` with `M > 0`), `L (..., M, K)`, `U (..., K, N)`; `permute_l`: `PL (..., M, K)`, `U (..., K, N)`; values uninitialized (`np.empty`) |
| `ndim < 2` | `ValueError('The input array must be at least two-dimensional.')` |

- The index array is LAPACK's `ipiv` swap sequence applied in **reverse**
  order to `0..M-1` (`ipiv_to_perm`: "Forward swaps give P, reverse gives
  P⁻¹" — the forward sequence is `lu_factor`'s `A[p] = L @ U` convention,
  the inverse permutation). Verified numerically against
  `scipy.linalg.lu(A, p_indices=True)`.
- Fingerprint portability (probe `/tmp/lu_probe/portability.py`: SciPy's
  conda OpenBLAS vs the system OpenBLAS 0.3.32 through ctypes): `P`
  identical; fingerprints of `L`, `U` and `PL` differ by ≤ `6e-11` for
  1024×1024, 1024×512 and 512×1024 — the 1e-8 house tolerance holds.

### 3. LAPACK / LAPACKE / FFI facts

- `[sdcz]getrf_` is bound in all five device trees, `LAPACKE_[sdcz]getrf` in
  openblas/mkl/aocl; the system OpenBLAS exports them. **No FFI change.**
- `?GETRF`: `A = P L U`, `L` lower trapezoidal for `m > n`, `U` upper
  trapezoidal for `m < n`; `IPIV` has `min(M,N)` entries; `INFO > 0`:
  factorization completed with `U(i,i)` exactly zero; `M == 0 || N == 0`
  quick-returns; no NaN check. No workspace (skill §3a row exists).
- `LAPACKE_dgetrf_work` (row-major): after the call, `if (info < 0) info =
  info - 1;` then the back-transpose — **unconditional**.

### 4. Policy that applies

- Skills `lapacke-translation` (§2 overwrite input, §3 thunk shape, §3b
  info mapping, §4 test matrix) and `test-device`.
- **Gap**: skill §3 says "the RowMajor output write-back runs only when the
  compute call returns `info == 0`", with the rationale "the builder maps any
  non-zero `info` to an `Err`". `?GETRF` `info > 0` is a *valid* result that
  `lu` hands to the user, so the rule as written loses the factors → Q11.

---

## Round 1 — the frontier

❓ **Q1 - Scope: the five BLAS devices only, SciPy semantics**: `rt::linalg::lu`
lands through the shared trees (new task directory `2026-11-26-blas-lu`); no
faer implementation in this campaign (precedent `lstsq`; faer's
`partial_piv_lu` makes a faer version a small, self-contained follow-up), and
no `lu_factor`, `lu_solve`, permutation-matrix form (`p_indices=False`),
full pivoting, `check_finite` or `overwrite_a` flags.

➡️ Yes. One entry, one reference, one PR.

---

❓ **Q2 - Returned factors = SciPy `p_indices=True`**: `lu` returns the row
indices `P` of shape `(..., M)` with `A = L[P, :] @ U` (so `A = P L U` with
`P[i, P[i]] = 1`), the unit lower trapezoidal `L` `(..., M, K)` and the upper
trapezoidal `U` `(..., K, N)`, `K = min(M, N)`; slices may be rectangular. `P`
is the `ipiv` swap sequence applied in reverse order to `0..M-1` (rows `K..M`
stay identity). This is SciPy's index convention, not `lu_factor`'s forward
one (`A[p] = L @ U`); the raw `ipiv` stays reachable through the public
`GETRF` builder.

➡️ Yes. Exactly the surface the prompt fixes.

---

❓ **Q3 - Index dtype `usize`**: `P` is `Tensor<usize, B, D::SmallerOne>`
(the `matrix_rank` / `lstsq` rank convention), not SciPy's `int32` or
`blas_int`.

➡️ Yes. Indexing tensors takes `usize`; no user wants `i32` here.

---

❓ **Q4 - `permute_l` as a type-level marker**: `LinalgAPI::Out` is fixed per
argument type and `permute_l` changes the arity (`(P, L, U)` → `(PL, U)`), so
a runtime `bool` cannot select it (a `(Tr, bool)` impl has one `Out`). Add a
unit-struct marker `pub struct PermuteL;` (the `Homogeneous` precedent) used
as a tuple element:

| form | `Out` |
|--|--|
| `lu(&a)` · `lu(a.view())` · `lu(a.view_mut())` · `lu(a)` | `LuResult<P, L, U> { p, l, u }` (↔ 3-tuple) |
| `lu((&a, PermuteL))` (+ the other three input kinds) | `LuPermutedResult<PL, U> { pl, u }` (↔ 2-tuple) |

No string form for the marker (as for `Homogeneous`). The static dims are
kept: for `a: Tensor<_, D>`, `P` has rank `D-1` and `L`/`U`/`PL` rank `D`
(the result rank never depends on the data, unlike `eig`).
Alternative rejected: one type `LuResult<Option<P>, L, U>` with `p: None`
for the permuted form (the `EigResult` Option-side pattern) — it would need
runtime-checked tuple conversions for a choice that is static here.

➡️ Yes. `PermuteL`, `LuResult`, `LuPermutedResult` are name proposals.

---

❓ **Q5 - Stacking and input policy**: n-dim stacks via `StackLayout`
(device order), one `overwritable_convert_with_order` for the whole stack: an
immutable input is factored from a fresh buffer; an owned / `TensorMut`
input is consumed and destroyed by `GETRF`; an off-order mutable input is
factored from a work buffer and left untouched (your lstsq rule). All four
input kinds in both forms.

➡️ Yes.

---

❓ **Q6 - A singular matrix is a result, not an error** (SciPy; LAPACK
`info > 0` = completed factorization with an exactly-zero pivot): `lu` returns
the factors. Two changes to the existing `GETRF` path, both LAPACKE-literal:

1. thunk `driver_impl/lapack/solve/getrf.rs` (both dtype blocks): skip the
   back-transpose only for `info < 0` (was `info != 0`).
2. builder: `GETRF_::run()` is unchanged for `det` / `slogdet` / `inv`; a new
   `GETRF_::run_with_info() -> Result<(TensorMutable2, Tensor<blas_int, Ix1>,
   blas_int)>` returns the raw `info` beside the factors, and `lu` calls it.

`det` / `slogdet` / `inv` keep erroring on a singular input (pre-existing
NumPy divergence, outside this campaign; a separate small fix if you want
it).

➡️ Yes.

---

❓ **Q7 - Zero-size rules (ours to define)**: `K == 0` → no driver call (the
builder already synthesizes): `P = 0..M-1` (identity, length `M`), `L` zeros
`(M, K)` (no unit diagonal when `K == 0`), `U` zeros `(K, N)` — so `A = L[P]
@ U` holds for every empty slice. An empty batch gives outputs with the batch
shape and no driver call. No `(1, 1)` special case (`GETRF` returns it
naturally). Divergences from SciPy's empty path: `P` is `(..., M)` instead of
`(..., 0)`, and every output is zero-initialized instead of uninitialized.

➡️ Yes. A permutation of zero rows for a non-empty `L` would break
`A = L[P] @ U` for nothing.

---

❓ **Q8 - Driver tier: no new routine, no trait change**: reuse
`GETRFDriverAPI<T>` (bound in every device); no `LapackDriverAPI<T>` edit,
hence no breaking change for external implementors; no `?LASWP`
translation (the permutation is O(M) plain Rust). Only the Q6 thunk/builder
changes touch `rstsr-blas-traits`.

➡️ Yes.

---

❓ **Q9 - Composition and buffers**: `ref_impl_lu_f(a: TensorReference<Ix2>,
permute_l: bool) -> Result<(Tensor<usize, Ix1>, Tensor<T, Ix2>, Tensor<T,
Ix2>)>` in `ref_impl_blas.rs`, under `with_blas_num_threads` (the `GETRF`
call is the O(M N K) part). `GETRF` in place on the converted input; `U` =
rows `0..K` from the diagonal rightwards (the QR `R` pattern), `L` =
below-diagonal plus unit diagonal, both fresh zero-filled tensors in the
device order; `P` by the reverse swap loop; with `permute_l` the rows of `L`
are gathered straight into `PL` (one pass, no extra copy) and `P` is still
returned (O(M), dropped by the entry). Outputs never alias the factor buffer.
Cost: O(M N) extra memory traffic against O(M N K) compute.

➡️ Yes.

---

❓ **Q10 - Entry tier**: `blas_impl/lu.rs` (symlinked ×5, `pub mod lu;`):
`blas_impl_lu_f(a: TensorReference<IxD>, permute_l: bool) -> Result<(Tensor<usize,
IxD>, Tensor<T, IxD>, Tensor<T, IxD>)>`; `LuAPI<DeviceBLAS>` impls for `(Tr)`
→ `LuResult<Tensor<usize, _, D::SmallerOne>, Tensor<T, _, D>, Tensor<T, _, D>>`
and `(Tr, PermuteL)` → `LuPermutedResult<Tensor<T, _, D>, Tensor<T, _, D>>`
over the four input kinds, `_f` twins generated by the trait; `LuAPI` /
`lu` / `lu_f` in the three `duplicate_item` tables and the preludes;
`ndim < 2` → `InvalidLayout`; `info < 0` → the internal-bug message;
`-1010` / `-1011` → `MemoryError`.

➡️ Yes.

---

❓ **Q11 - Skill gap notice**: `lapacke-translation` §3 gains one sentence
("a routine whose `info > 0` is a valid result, e.g. `getrf`'s completed
factorization, writes back like LAPACKE") — an edit on contradiction, as
you allowed in the QR grill and applied in the lstsq review.

➡️ Yes.

---

❓ **Q12 - Tests**:

- Driver `driver_impl/f64/getrf.rs` (new — `GETRF` had the fingerprint
  test only), small sizes: the layout matrix for `a` (f-contig, f-padded
  `ld > m`, c-contig, c-padded, fully non-contiguous) with `into_owned_keep_layout`
  helpers, tall / square / wide, 0-based `ipiv` parity with the f-contig
  run, in-place buffer identity, destroyed-`a` parity of the row-major
  bridge against a direct column-major call, **singular**: valid factors in
  both orders (`L[P] U ≈ A`, zero pivot at `info`), zero-size synthesis.
- Linalg `linalg_func/func_{f64,c64}/lu.rs`: fingerprints — f64 square
  1024 `a`, tall 1024×512, wide 512×1024, `permute_l` on the square; c64
  square + tall (4 + 2 cases) — plus self-validation (`A ≈ L[P] U`,
  `A ≈ PL U`, unit diagonal, triangularity, `P` is a permutation), a small
  singular case, batch parity against per-slot 2-D calls, layouts and both
  device orders, zero-size `(M, 0)`, `(0, N)`, `(0, 0)`, empty batch, the
  16 × 2 overload matrix on 4×4 blocks, `ndim < 2`.
- Validators: `# ### lu` sections in `func_validation_{f64,c64}.py`
  (`scipy.linalg.lu(a, p_indices=True)`); literals pasted into Rust at 1e-8.
- Runs: OpenBLAS `lapack` + `lapacke` (driver and linalg suites), MKL and
  AOCL `getrf` / `lu` subsets, BLIS/KML `cargo check`.

➡️ Yes.

---

❓ **Q13 - Docs**: rustdoc on `LuResult`, `LuPermutedResult`, `PermuteL`
(SciPy relation, shapes, index convention); a `//!` module doc on
`blas_impl/lu.rs` in the eig / lstsq wording; `cheatsheet_linalg.md` gets a
"DeviceFaer backend notes" bullet (BLAS-only, `P` as indices, `PermuteL`) and
no table row (the page's test is faer-backed);
`array_api_standard_extension.md` notes that `lu` is beyond the standard. No
new page.

➡️ Yes (if Q1 stands; a faer entry would turn the bullet into a table row).

---

❓ **Q14 - Delivery**: branch `261126/blas-lu` off `main` `68f4891`; one PR;
no auto-commit; PR only on your instruction. Close-out: `DECISIONS.md`,
`PLAN.md`, README + memory here.

➡️ Yes.

---

❓ **Q15 - Close the grill**: with Q1–Q14 accepted the frontier is empty.
Next: `DECISIONS.md`, then `PLAN.md`, then implementation.

➡️ Yes — confirm we have a shared understanding and I may start.

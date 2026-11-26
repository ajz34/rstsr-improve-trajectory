# Grill R3 — `lu`, `lu_factor`, `lu_solve` on the BLAS devices

## Answers to R2 (agent transcription; answers were given on screen 2026-11-26)

- **Additional request**: "I feel that maybe we introduce `lu_factor` as well
  is better. So now three functions: lu, lu_factor, lu_solve." → Q20.
- **Q16** — "Yes." The §A reading (`permute_l = true` → `{p: None, l: pl,
  u}`) and the one-type design stand.
- **Q17** — amended: "I think that input `lu: TensorViewAPI/Tensor/TensorMut`
  or `lu: &LuResult` should be both accepted. For `FlagTrans` part you are
  correct." → both factor forms: Q21–Q22.
- **Q18** — "Yes, and also `lu_factor`." → Q23.

## §C Fact check for Q21 (the packed factor through `TRSM`)

- BLAS `?TRSM` reads one triangle of its matrix, and with `diag = 'U'` the
  diagonal is not referenced. The packed `(N, N)` factor therefore serves
  both solves as is: `(Lower, Unit)` sees `L`, `(Upper, NonUnit)` sees `U`.
  No unpacking, no copy.
- Permutation from `ipiv` (0-based LAPACK swaps, as `lu_factor` returns):
  `A x = b` applies the swaps **forward** to the rows of `b`, then `L`-solve,
  `U`-solve; `Aᵀ/ᴴ x = b` does `U`-solve, `L`-solve, then applies the swaps
  in **reverse**. Verified against SciPy 1.18.1 `lu_factor` + `lu_solve`
  (`trans` 0/1/2, f64 / c128, vector and matrix right-hand sides; probe
  `/tmp/lu_probe/lu_solve_packed.py`, zero mismatches).
- This is exactly what `?GETRS` does (`?LASWP` forward, `?TRSM(L, 'N',
  'U')`, `?TRSM(U, 'N', 'N')`), so the numbers agree with it; the traffic is
  `N²` per slice for both.
- A singular factor: `lu_solve` returns `inf` / `nan` (SciPy does the same;
  `?GETRS` has no singularity check).
- `LAPACKE_dgetrs_work` (for the alternative): a two-matrix row-major
  bridge like `gesv` (`a` and `b` in, `b` back), `lda >= n`, `ldb >= nrhs`.

---

## Round 3 — the frontier opened by `lu_factor` and the second factor form

❓ **Q20 - `lu_factor` is SciPy's packed factorization**: forms
`lu_factor(&a)` / `lu_factor(a.view())` →
`LuFactorResult<Tensor<T, _, D>, Tensor<usize, _, D::SmallerOne>> { lu, piv }`
(↔ 2-tuple). `lu` is the `(M, N)` packed factor (`U` in the upper triangle
including the diagonal, the multipliers of `L` below it); `piv` is LAPACK's
swap sequence made 0-based, length `K = min(M, N)`, dtype `usize` (as the `P`
of `lu`). Rectangular slices and stacks as for `lu`.

An owned / `TensorMut` input is factored **in place and returned**
(`Out = LuFactorResult<Tr, ..>`, the `cholesky` rule; an off-order mutable
input is factored in a work buffer and written back), since the packed buffer
is the result. A singular matrix returns its factors silently (SciPy prints
a `LinAlgWarning`; the zero pivot is visible on the diagonal). Zero-size: `lu`
is the (empty) input, `piv` is `(..., 0)`.

➡️ Yes. `lu_factor` is the `GETRF` builder's result at the linalg tier; the
factor is the object `lu_solve` takes in the SciPy form.

---

❓ **Q21 - One solve composition for both factor forms; no `?GETRS`
translation**: `lu_solve` runs the §C composition per slice through the
existing `TRSM` builder, for

- the packed form `((lu, piv), b)`: `lu`'s two triangles are used directly
  (§C), `piv` drives the forward / reverse swaps;
- the `&LuResult` form: `l` and `u` as they are, `p` (the reverse-swap
  indices) drives the scatter / gather (§B of R2); `p: None` →
  `InvalidValue`.

No new trait, no thunk, one code path. Rejected alternative: translate
`?GETRS` for the packed form (the literal SciPy routine; a trait in
`LapackDriverAPI<T>`, two thunk trees, driver tests) while the `&LuResult`
form would still need the `TRSM` composition — two solve paths for the same
arithmetic.

➡️ Yes.

---

❓ **Q22 - `lu_solve` argument tables**: the `solve_general` kind classes —
read-only `{&TensorAny, TensorView}` and consumed `{TensorMut, Tensor}`:

| table | `lu` | `piv` | `b` | `Out` |
|--|--|--|--|--|
| A | read-only | read-only | read-only | fresh `Tensor<T, _, IxD>` |
| B | read-only | read-only | consumed | `TrB` (solved in place) |
| C | consumed | consumed | consumed | `TrB` (solved in place) |

each with and without the trailing `FlagTrans` (48 rows). The `&LuResult`
form: `b` read-only → fresh, `b` consumed → `TrB`, with and without
`FlagTrans` (8 rows). `piv` is `Tensor<usize>`; the static dims of `lu`
(`D`) and `piv` (`D::SmallerOne`) are checked against each other; the fresh
result is dynamic-rank (the batch broadcast may change the rank).

➡️ Yes. 56 mechanical `duplicate_item` rows (the eig tables had 16 per
entry).

---

❓ **Q23 - Tests and docs for the three functions**:

- Driver: as R1 Q12 (`driver_impl/f64/getrf.rs`: layout matrix, `ipiv`
  parity, in-place identity, bridge parity, singular, zero-size).
- Linalg `linalg_func/func_{f64,c64}/{lu,lu_factor,lu_solve}.rs` with
  validator sections:
  - `lu`: f64 4 cases (square 1024 `a`, tall 1024×512, wide 512×1024,
    `permute_l` on the square), c64 2 (square, tall);
  - `lu_factor`: f64 2 (square 1024: `lu` + `piv`; tall 1024×512), c64 1
    (square); plus a singular small case, in-place `lu_factor(a.view_mut())`
    returning the same buffer, the packed `lu` against `lu`'s `L` / `U`;
  - `lu_solve`: f64 3 (square 1024, `K = 1`, `K = 3`, one `T`), c64 2
    (`K = 1`, one `C`); both factor forms agree; `A x ≈ b`, `Aᵀ x ≈ b`,
    `Aᴴ x ≈ b`; agreement with `solve_general`; batch broadcast; in-place
    `b`; the permuted-factor and non-square errors; a singular factor gives
    non-finite values without an error; zero-size.
- Docs: rustdoc on `LuResult` and `LuFactorResult`; `//!` docs on the three
  `blas_impl` files; one "DeviceFaer backend notes" bullet covering the
  three entries; the `array_api_standard_extension.md` note.

➡️ Yes.

---

❓ **Q24 - Close the grill**: with Q20–Q23 accepted the frontier is empty.
Next: `DECISIONS.md`, then `PLAN.md`, then implementation on `261126/blas-lu`.

➡️ Yes — confirm we have a shared understanding and I may start.

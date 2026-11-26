# Grill R2 — LU decomposition (`lu`, `lu_solve`) on the BLAS devices (evaluation round)

## Answers to R1 (agent transcription; answers were given on screen 2026-11-26)

- **Q2, Q3, Q5–Q14** — "follow your recommendation."
- **Q1** — amended: "Also `lu_solve`. No faer, no `lu_factor`. Others follow
  yours." → `lu_solve` joins the campaign: Q17–Q18.
- **Q4** — amended: "We always have `LuResult` as `{p, l, u}`; if
  `permute_l = false`, we return `{p: None, l: pl, u}`. Also, `permute_l`
  should be plain `bool`, no new type needed if possible." → Q16 (reading in
  §A).
- **Q15** — "See if you are confident on Q1/Q4." → the owner delegated the
  closing evaluation; Q16–Q19 below are **agent-evaluated** and close the
  grill unless vetoed.

## §A Fact check for Q16 (reading of the amendment)

- The answer names `permute_l = false` for `{p: None, l: pl, u}`; `pl` is
  the permuted `L`, which is SciPy's `permute_l = True` product. Read as a
  slip: **`true` → `{p: None, l: pl, u}`; `false` (and the bare form) →
  `{p: Some(p), l, u}`.** Correct me if the reading is wrong.
- With a plain `bool` the tuple element cannot select `Out` (R1 Q4), so
  every form shares one `Out`: `LuResult<Option<P>, L, U>`.
- Coherence probed with rustc (`/tmp/lu_probe/coh`): the generic
  `(P, L, U)` conversions both ways and the two runtime-checked ones below
  coexist (an overlap would need `P = Option<P>`).
- Consequence: a permuted result (`p: None`) cannot feed `lu_solve`, since
  `PL` is not triangular. SciPy has the same property: the output of
  `lu(A, permute_l=True)` does not fit `lu_solve`.

## §B Fact check for Q17 (solving from `(P, L, U)`)

- Composition verified against SciPy 1.18.1 (`lu_factor` + `lu_solve`, all
  of `trans` 0/1/2, f64 and c128, vector and matrix right-hand sides; probe
  `/tmp/lu_probe/lu_solve.py`, no difference beyond 1e-12):
  - `A x = b`: scatter `y[P[i]] = b[i]`, then `L z = y` (unit lower), `U x = z`.
  - `Aᵀ x = b` / `Aᴴ x = b`: `Uᵀ/ᴴ y = b`, `Lᵀ/ᴴ w = y` (unit), gather
    `x[i] = w[P[i]]`.
- The `TRSM` builder (`blas3/trsm.rs`) already carries `uplo`, `transa` and
  `diag` (`Unit`); it reads one triangle of its matrix, so the two calls move
  `N²` elements per slice — the same as `?GETRS` on the packed factor. Both
  do the `O(N·K)` permutation pass.
- `?GETRS` needs the packed factor and `ipiv`; reached from a `LuResult` it
  would cost an `O(N²)` re-pack per solve (and a `P → ipiv` inversion), and
  as a translation it adds a trait to `LapackDriverAPI<T>`.
- LAPACK `?GETRS` runs no singularity check (neither does `TRSM`): a solve
  with an exactly-zero pivot yields `inf` / `nan`, not an error.

---

## Round 2 — evaluation (delegated by Q15; veto-able)

❓ **Q16 - One result type; `permute_l` is a plain `bool`**:
`pub struct LuResult<P, L, U> { pub p: P, pub l: L, pub u: U }` with

- the type-preserving `(P, L, U)` conversions both ways;
- from `LuResult<Option<P>, L, U>`: `(P, L, U)` when `p` is `Some`
  (panics otherwise), and `(L, U)` when `p` is `None` (panics otherwise —
  the `EigResult` rule; a 2-tuple from an unpermuted result would silently
  drop the permutation).

Every form returns `LuResult<Option<Tensor<usize, _, D::SmallerOne>>,
Tensor<T, _, D>, Tensor<T, _, D>>`: the bare `lu(&a)` / view / `view_mut()` /
owned forms mean `permute_l = false`; `lu((&a, bool))` and its three
siblings take the flag. No `PermuteL` type, no `LuPermutedResult`.

➡️ Accepted by delegation. Rejected alternative: a concrete `P` for the bare
form and `Option` only for the `bool` form — two `Out`s for one struct, and
`lu((&a, false))` would then differ in type from `lu(&a)`.

---

❓ **Q17 - `lu_solve` takes the `lu` result; two `TRSM` calls, no `GETRS`**:

- Forms: `lu_solve((&lu, &b))`, `lu_solve((&lu, &b, FlagTrans::T))`,
  `lu_solve((&lu, &b, FlagTrans::C))` (the existing `FlagTrans`; `N` is the
  default). The factor is the `&LuResult<..>` that `lu` returns and must be
  square; `b` is `(..., N)` or `(..., N, K)` in the four input kinds; the
  batch dims of the factor and of `b` broadcast (the `solve_general` plan).
- Output: an immutable `b` gets a fresh solution; a mutable or owned `b` is
  solved in place and returned (`Out = TrB`, the `solve_general` rule).
- Per slice: the §B composition — permutation pass, then `TRSM(L, Lower, N,
  Unit)` → `TRSM(U, Upper, N)`; for `T` / `C`: `TRSM(U, Upper, T|C)` →
  `TRSM(L, Lower, T|C, Unit)` and the gather. The factor is only read.
- Errors: `p: None` → `InvalidValue` (a permuted factorization cannot be
  solved); a non-square factor or a `b` that breaks the shape / batch rules
  → `InvalidLayout`. `N == 0` or `K == 0` → no driver call, empty solution.
  A singular factor gives `inf` / `nan` (LAPACK behaviour, documented).

Alternative: translate `?GETRS` and take SciPy's packed `(lu, piv)` object
(the public `GETRF` builder's output: the packed `TensorMutable2` plus
`blas_int` pivots). SciPy-literal, but it exposes `blas_int` and a builder
at the linalg tier, and a `LuResult` would need an `O(N²)` re-pack per solve.

➡️ Accepted by delegation.

---

❓ **Q18 - Tests and docs for `lu_solve`**: linalg
`linalg_func/func_{f64,c64}/lu_solve.rs` — fingerprints against SciPy
`lu_factor` + `lu_solve` (f64: square 1024 `a` with `K = 1` and `K = 3`, one
`T` case; c64: square 1024 `a` with `K = 1` and one `C` case), self-validation (`A x ≈ b`, `Aᵀ x ≈ b`, `Aᴴ x ≈ b`, agreement with
`solve_general`), batch broadcast, in-place `b`, the permuted-factor and
non-square errors, zero-size; a `# ### lu_solve` validator section in both
scripts; `LuSolveAPI` / `lu_solve` / `lu_solve_f` in the trait tables and
preludes; rustdoc on `lu_solve`'s module doc; the cheat-sheet note gains the
`solve` half.

➡️ Accepted by delegation.

---

❓ **Q19 - Confidence and close**: on Q4 / Q16 I am confident (with the §A
reading). `lu_solve` is the part you have not seen in detail; Q17 is the one
real fork (factor object `LuResult` + `TRSM` composition, versus a `GETRS`
translation on the packed factor). Give the word — or veto Q17 — and I write
`DECISIONS.md` and `PLAN.md`, then implement.

➡️ Yes — confirm the §A reading and Q17, and I may start.

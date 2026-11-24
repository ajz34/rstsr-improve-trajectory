# Grill R2 — eig / eigvals (complex) on the BLAS devices

## Answers to R1 (agent transcription; answers were given on screen 2026-11-24)

- **Q1** — acknowledged (standard + one generalized form; the three
  `itype` kinds are `eigh`'s).
- **Q2** — accepted, with "also refer to Q3/Q7, may change what you thought
  on this" → whether an `EigArgs` builder is needed is re-decided by Q17.
- **Q3 / Q7** — amended: "consider left/right and homogeneous_eigvals now;
  we do eig/eigvals once for all, not deferring any structural change edits
  to future." → carried to Q17–Q19.
- **Q13** — accepted with emphasis: "you may also introduce python
  fingerprint generation, and test against the fingerprint values" → Q21.
- **Q15** — amended: "first pull main branch, and start from main branch
  with `261125/blas-eig`" → Q22 (fact check §A).
- **Q4–Q6, Q8–Q12, Q14, Q16** — recommendation accepted.

## §A Fact check for Q15 (git state, fetched 2026-11-24 23:5x)

- `origin/main` = **`b01b6e0`** = "QR decomposition on the BLAS devices
  (#169)" — the QR PR is **merged**, and the squash includes the two
  docstring-fix commits (`svd.rs` on `origin/main` already carries the
  "mutable input whose layout matches the device order is destroyed"
  wording). Local `main` is stale at `c9fcc20` → a fast-forward pull brings
  it to `b01b6e0`; `261124/blas-qr` is fully merged.
- So `261125/blas-eig` branches off `main` @ `b01b6e0` with no dependency on
  unmerged work; the only overlap with #169 is the shared `mod.rs` /
  `LapackDriverAPI` / validator files, which the new branch edits on top.

## §B Fact check for Q17 (left vectors, homogeneous form)

- LAPACK normalizes **both** `VL` and `VR` of `?GEEV` (unit 2-norm, largest
  component real); SciPy's docstring sentence "the left eigenvector is not
  normalized" is stale for GEEV. For `?GGEV` SciPy renormalizes **both**
  `vl` and `vr` (`vl /= norm(vl)`, `vr /= norm(vr)`).
- Left-vector convention (LAPACK = SciPy): column `vl[:, j] = u(j)` with
  `u(j)ᴴ A = λ_j u(j)ᴴ`, i.e. `Aᴴ u = conj(λ_j) u`; generalized
  `u(j)ᴴ A = λ_j u(j)ᴴ B`.
- Homogeneous form: SciPy stacks `[alpha; beta]` on axis `-2` → `w` is
  `(..., 2, M)`; for the **standard** problem `beta = ones_like(alpha)`
  (`_make_eigvals`). `eigvals` accepts `homogeneous_eigvals` too.

---

## Round 2

❓ **Q17 - How `left` / `right` / `homogeneous_eigvals` reach the entry**:
`LinalgAPI::Out` is fixed per argument type, so the SciPy booleans become
**argument markers** (rstsr tuple style), not runtime flags:

- New shared types in `traits_def.rs`:
  - `#[non_exhaustive] enum EigVectors { None, Left, #[default] Right, Both }`
    (the SciPy `left`/`right` pair);
  - `pub struct Homogeneous;` — unit-struct marker used as a tuple element
    (like `Upper`/`Lower`), SciPy `homogeneous_eigvals=True`;
  - `pub struct EigBothResult<W, VL, VR> { eigenvalues, left, right }` with
    `From`/`Into (W, VL, VR)` (the three-output form). The existing
    `EigResult<W, V>` (↔ `(W, V)`) is reused for the one-sided forms.
- Tuple forms (positional order `a, b, vectors, homogeneous`, each element
  optional except `a`); `W1 = Tensor<c, B, D−1>` shaped `(..., M)`,
  `W2 = Tensor<c, B, D>` shaped `(..., 2, M)`, `c = Complex<T::Real>`:

| `eig` form | `Out` |
|--|--|
| `(Tr)` · `(Tr, EigVectors::Right)` | `EigResult<W1, VR>` (NumPy default) |
| `(Tr, EigVectors::Left)` | `EigResult<W1, VL>` (`eigenvectors` = left vectors, SciPy's `(w, vl)`) |
| `(Tr, EigVectors::Both)` | `EigBothResult<W1, VL, VR>` |
| `(Tr, EigVectors::None)` | `W1` (≡ `eigvals`) |
| `(Tr, Homogeneous)` | `EigResult<W2, VR>` |
| `(Tr, EigVectors, Homogeneous)` | as the four rows above with `W2` |
| `(Tr, &B, …)` | every row above with `b` inserted after `a` (generalized) |

| `eigvals` form | `Out` |
|--|--|
| `(Tr)` · `(Tr, &B)` | `W1` |
| `(Tr, Homogeneous)` · `(Tr, &B, Homogeneous)` | `W2` |

- Input kinds per matrix slot unchanged from R1 Q7 (`&TensorAny`,
  `TensorView`, `Tensor`, `TensorMut`; 4 rows standard, 16 generalized per
  form); the non-core forms delegate to the `(Tr[, &B], EigVectors,
  Homogeneous)` core in one line each, so each `duplicate_item` table stays
  one body (~35 lines); the two `blas_impl` files land around 300 lines each.
- Q2 stands: **no `EigArgs` builder** — the tuple forms are the complete
  SciPy surface with precise types (no `Option`, no dynamic rank).

Alternative (Route B): keep only the default tuple forms and add an
`EigArgs` builder (`a`, `b`, `left`, `right`, `homogeneous_eigvals`) →
`EigFullResult { eigenvalues: Tensor<c, IxD>, left: Option<..>, right:
Option<..> }`, SciPy runtime flags verbatim, ~40 rows + 1 builder, at the
price of `Option` unwrapping and a dynamic-rank spectrum.

➡️ Yes, Route T (markers). It is "once for all" at the type level and keeps
the house style (`QrMode`, `(a, Upper)`); the builder would make Q2 flip.

---

❓ **Q18 - Marker defaults and aliases**: `EigVectors::Right` is the
`#[default]`; the bare `(Tr)` form means `Right` (NumPy `eig`), and
`EigVectors::None` returns the bare spectrum, i.e. `eig((&a,
EigVectors::None))` ≡ `eigvals(&a)` (SciPy `left=False, right=False`).

➡️ Yes.

---

❓ **Q19 - Homogeneous standard problem**: with `Homogeneous` on a standard
`(Tr, …)` form the spectrum is `(..., 2, M)` with `beta ≡ 1` (SciPy
parity); the generalized form returns LAPACK's `α` and `β` as they come
(no division, so no inf/nan appears in the homogeneous form).

➡️ Yes.

---

❓ **Q20 - faer device**: unchanged in this campaign — no left vectors in
faer, and its `generalized_eigen` is not wired. The three new types live in
the shared `traits_def.rs`, so faer can adopt `(a, b)` / `Homogeneous`
later as plain impls (no structural change).

➡️ Yes.

---

❓ **Q21 - Tests for the new forms** (extends R1 Q13; Python generates every
fingerprint):

- Validator sections added to **both** `func_validation_{f64,c64}.py`
  (1024×1024 `a_raw`/`b_raw`, SciPy calls): `eig_left`, `eig_both`,
  `eig_generalized`, `eig_generalized_both`, `eigvals_generalized`,
  `eigvals_homogeneous` (standard, `[w; 1]`),
  `eigvals_generalized_homogeneous`; canonicalization = spectrum sorted
  lexicographically (`np.sort` on complex), vector columns reordered by
  the same order, `abs(...)` before fingerprinting; homogeneous arrays
  fingerprinted as the `(2, M)` stack after sorting columns by `α/β`.
- Rust `linalg_func/func_{f64,c64}/{eig,eigvals}.rs` assert those literals
  **and** self-validate: `A V ≈ V diag(w)`, `Aᴴ VL ≈ VL diag(conj w)`,
  `A V ≈ B V diag(w)`, `Aᴴ VL ≈ Bᴴ VL diag(conj w)`; unit column norms for
  GGEV (and GEEV); homogeneous shape `(..., 2, M)` with `β ≡ 1` (standard)
  and `α/β ≈ w` wherever `β ≠ 0`; `EigVectors::None` ≡ `eigvals`;
  `(Tr, EigVectors::Right)` ≡ `(Tr)`; 3-D stack with broadcast `b`;
  Fortran-ordered and fully non-contiguous inputs; zero-size; non-square
  `a`/`b`; `TensorMut` inputs destroyed; one exact small case (SciPy
  docstring: `[[0, -1], [1, 0]]` → `w = [i, −i]`,
  `vr = [[0.707, 0.707], [−0.707i, 0.707i]]`).
- Driver-level `driver_impl/f64/{geev,ggev}.rs` as in R1 Q13 plus the
  `left`/`right` combinations against the single-sided runs.

➡️ Yes.

---

❓ **Q22 - Delivery**: fast-forward local `main` to `origin/main`
(`b01b6e0`), branch `261125/blas-eig` from it; one PR, no auto-commit; the
merged `261124/blas-qr` local branch is left for you to delete.

➡️ Yes.

---

❓ **Q23 - Close the grill**: with Q17–Q22 answered the frontier is empty
(driver tier, entry, output rules, tests, docs, delivery all settled). Next:
`DECISIONS.md` here, then implement on `261125/blas-eig`.

➡️ Yes — confirm we have a shared understanding and I may start.

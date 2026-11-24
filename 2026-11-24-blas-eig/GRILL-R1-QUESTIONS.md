# Grill R1 — eig / eigvals (complex) on the BLAS devices

rstsr `9e8dac4` (`261124/blas-qr`), rstsr-ffi `83efe50`; no rstsr file edited.
Recommended answers are marked ➡️; "yes" accepts the recommendation.

---

## Factual baseline (agent-authored, verified 2026-11-24)

### 1. Fact check: how many eigenproblem kinds are there?

- **Non-symmetric (this task)**: LAPACK has exactly **two** drivers —
  `GEEV` (standard `A v = λ v`) and `GGEV` (generalized `A v = λ B v`). `GGEV`
  has **no `itype`**: the three "kinds" (`A x = λ B x`, `A B x = λ x`,
  `B A x = λ x`, `itype = 1/2/3`) belong to the **symmetric/Hermitian**
  generalized problem (`*SYGV`/`*HEGV`, `*SYGVD`/`*HEGVD`) and are already
  implemented on the BLAS devices for `rt::linalg::eigh` (`eig_type`).
  SciPy agrees: `scipy.linalg.eig(a, b)` accepts one `b` form only; the
  `type=` keyword exists only on `eigh`/`eigvalsh`.
- Other LAPACK drivers in this area, for completeness: `GEEVX`/`GGEVX`
  ("expert" drivers: balancing knobs + condition numbers, same eigenvalues)
  and the QZ/Schur family `GGES` (+`HGEQZ`/`TGEVC`, SciPy `qz`/`ordqz`).
- **No driver choice exists for `eig`**: the `driver = "ev"/"evd"` (QR vs
  divide-and-conquer) selection of `eigh` has no counterpart — `GEEV` is the
  one algorithm (Hessenberg + QR iteration with internal balancing).
- faer: `MatRef::eigen()` (→ `U`, `S`), `eigenvalues()`, and
  `generalized_eigen(B)` (→ `U`, `S_a = α`, `S_b = β`, QZ-based). No left
  vectors, no balancing/homogeneous knobs. rstsr's faer device uses only the
  two standard ones (`faer_impl/{eig,eigvals}.rs`).

### 2. What exists today

- Entry traits `EigAPI` / `EigvalsAPI` and `EigResult<W, V>` (↔ `(W, V)`)
  in `rstsr-linalg-traits/src/traits_def.rs`; **faer-only** impls
  (`eig(&a)`, `eigvals(&a)`, n-dim batched) whose output dtype is
  `Complex<T::Real>` for every input — f32→c64, f64→c128, c64→c64, c128→c128
  (the array-API rule). No docstrings on the traits. The BLAS devices have no
  `EigAPI`/`EigvalsAPI` impl at all.
- BLAS linalg entries follow the three-layer pattern (SVD/QR precedent):
  1. driver tier `rstsr-blas-traits/src/lapack_*/` (`XxxDriverAPI<T>` +
  `derive_builder` struct) and the shared vendor trees
  `rstsr-blas-traits/driver_impl/{lapack,lapacke}/` (symlinked into all five
  crates; `lapacke` = openblas+mkl, mutually exclusive feature);
  2. composition `ref_impl_*_simple_f` in `rstsr-linalg-traits/src/ref_impl_blas.rs`;
  3. entry `blas_impl/*.rs` (stack walk over `StackLayout`), reached from every
  device crate by a symlink `crates-device/*/src/linalg_auto_impl/*.rs`.
- `LapackDriverAPI<T>` lists every driver trait explicitly (new traits are a
  breaking change for external implementors — accepted precedent).
- Input policy: `overwritable_convert_with_order` consumes owned / `&mut`
  inputs whose layout matches the call order (QR R2-Q15 added `TensorMut`
  overloads on the owner's request); outputs are fresh tensors.

### 3. NumPy / SciPy signatures

- `scipy.linalg.eig(a, b=None, left=False, right=True, overwrite_a=False,
  overwrite_b=False, check_finite=True, homogeneous_eigvals=False)` → `w`
  (`(..., M)` complex; `(..., 2, M)` with `homogeneous_eigvals`), plus `vl`
  if `left`, plus `vr` if `right`. Batched over leading axes (SciPy ≥1.18 has
  a C++ `_batched_linalg._eig`).
  - GGEV eigenvectors are **renormalized** to unit 2-norm by SciPy (LAPACK
    returns max-`|re|+|im|` = 1 scaling); GEEV vectors are already unit 2-norm
    with the largest component real (no renormalization).
  - Generalized `w = α/β` with explicit rules: `β ≠ 0 → α/β`;
    `β = 0, α ≠ 0 → inf`; `α = β = 0 → nan`.
  - Backward-compat: real `a` with an all-real spectrum → `vr` is made real
    (`w` stays complex).
- `scipy.linalg.eigvals(a, b=None, overwrite_a, overwrite_b, check_finite,
  homogeneous_eigvals)` = `eig(a, b, left=0, right=0)`.
- `numpy.linalg.eig(a)` → `(w, vr)`; `eigvals(a)` → `w`; no `b`, no `left`;
  real-typed `w`/`vr` when the spectrum is real.
- array-API `linalg.eig(x)` → `(eigenvalues, eigenvectors)`, **always complex
  of the input precision**, order unspecified; `eigvals` likewise. rstsr's
  faer device follows this (never downcasts).

### 4. LAPACK facts that shape the driver tier

- Real `?GEEV` (`s`/`d`): outputs `WR`, `WI` (real, split) and real `VL`/`VR`
  with the **conjugate-pair packing rule**: real eigenvalue `j` → `v(j) =
  VR(:, j)`; pair `(j, j+1)` → `v(j) = VR(:, j) + i VR(:, j+1)`,
  `v(j+1) = VR(:, j) − i VR(:, j+1)`. Complex `?GEEV` (`c`/`z`): complex
  `W`, `VL`, `VR` directly. SciPy/numpy assemble the complex outputs from
  the real ones (scipy `_linalg_eig.hh::transform_eigvecs`).
- `?GGEV` real: `ALPHAR`, `ALPHAI`, `BETA` (real), `VL`/`VR` real with the
  same packing rule; complex: `ALPHA` (complex), `BETA` (real), complex
  `VL`/`VR`.
- Both drivers **destroy** `A` (and `B`) and compute the left/right vectors
  independently (`JOBVL`/`JOBVR`).
- Workspaces: real GEEV → `lwork` query (min `3N`, `4N` with vectors); complex
  GEEV → query **plus** `rwork = 2N` (no query). Real GGEV → query (min
  `8N`); complex GGEV → query (min `2N`) plus `rwork = 8N`. LAPACKE allocates
  the formula-sized `rwork` **before** the query call.
- `info > 0`: GEEV — QR iteration failed to compute all eigenvalues (no
  vectors computed); GGEV — QZ failure (`≤ N`), `N+1` (`HGEQZ`), `N+2`
  (`TGEVC`).
- Row-major bridge (LAPACKE): transpose `A` (and `B`) in; transpose `VL`/`VR`
  out; the back-transpose of the destroyed `A` is kept by LAPACKE.

### 5. FFI and vendor coverage (no FFI change needed)

`[sdcz]geev_`, `[sdcz]ggev_` (plus `geevx`, `ggevx`, `gges`) are bound in
**all five** crates; `LAPACKE_[sdcz]geev` / `LAPACKE_[sdcz]ggev` exist for
openblas/mkl/aocl. The system OpenBLAS exports `[sdcz]geev_`/`[sdcz]ggev_`
(`nm` checked); it exports no `LAPACKE_*` (lapacke config = conda OpenBLAS
only, as in the QR grill).

### 6. Test and doc assets

- `blas_tests_impl/linalg_func/func_validation_{f64,c64}.py` already carry
  `# ### eig` and `# ### eigvals` sections (from #146): 1024×1024 standard
  problem, fingerprint of the **sorted** spectrum and of `abs(vr[:, order])`
  (phase/sign-invariant). No Rust test consumes them yet; no generalized
  sections exist.
- No `linalg_func/func_*/eig.rs`; no `driver_impl/f64/geev.rs`.
- `cheatsheet_linalg.md` is faer-backed; every table row needs the Rust test
  + Python parity triad, so BLAS-only forms go to its "DeviceFaer backend
  notes" list (precedent: `solve_symmetric`, generalized `eigvalsh`).
  `array_api_standard_extension.md` already marks `eig`/`eigvals` `Y`.
- Skills in force: `lapacke-translation` (§3a has **no** `geev`/`ggev` row →
  gap notice; the QR grill's rule: record here, edit the skill only on
  contradiction), `test-device`, `api-cheatsheet-conventions`.

---

## Round 1 — the frontier

❓ **Q1 - Scope and the "3 kinds" fact check**: this campaign implements
`eig` and `eigvals` on the BLAS devices in exactly **two** forms — standard
(`GEEV`) and generalized `A v = λ B v` (`GGEV`) — on all five devices through
the shared trees. The three `itype` kinds are the Hermitian forms already
done by `eigh`; `GGEVX`/`GEEVX` (condition numbers) and the QZ/Schur family
are out of scope.

➡️ Yes. That is the complete non-symmetric surface NumPy/SciPy expose
through `eig`/`eigvals`; the expert and Schur drivers would be separate
entries (`qz`).

---

❓ **Q2 - No driver option, no `EigArgs` builder**: `eig`/`eigvals` get no
`driver` argument (GEEV/GGEV are the only algorithms) and no `EigArgs`
builder struct; every option rides on tuple overloads (Q7). The `EighArgs`/
`SVDArgs` precedent is not extended.

➡️ Yes. A builder whose only knobs are `b`/`left` would duplicate the tuple
forms and add a third public type for nothing.

---

❓ **Q3 - Left eigenvectors and homogeneous eigenvalues**: deferred at the
entry tier — `eig` returns right vectors only (`EigResult`), `eigvals` the
spectrum as `α/β`. The **driver-tier builders keep the full LAPACK surface**
(`left`/`right` flags returning `Option<vl>`/`Option<vr>`; the GGEV builder
returns `α` and `β` separately), so `left=True` and `homogeneous_eigvals`
can be added later without touching the driver tier.

➡️ Yes. `left` changes the output arity (needs a three-field result type or a
marker argument — a signature decision of its own), and nothing consumes
the homogeneous form today.

---

❓ **Q4 - Driver-tier shape**: a new family `rstsr-blas-traits/src/lapack_eig/`
with two builders — `GEEV_` (`GEEVDriverAPI<T>`) and `GGEV_`
(`GGEVDriverAPI<T>`) — vendor thunks in the shared trees
`driver_impl/{lapack,lapacke}/eig/{geev.rs,ggev.rs}` (Fortran-faithful
`?geev_work`/`?ggev_work` + thin trait impls, two `duplicate_item` blocks
each), both traits registered in `LapackDriverAPI<T>` and the preludes;
composition `ref_impl_eig_simple_f` / `ref_impl_eig_generalized_simple_f` in
`ref_impl_blas.rs`. No monolithic "eig driver".

➡️ Yes. Mirrors the LAPACK split and every existing family.

---

❓ **Q5 - One trait per routine with complex output slots** (the real/complex
divergence): the trait signature carries the **assembled complex** outputs —
`driver_geev(order, jobvl, jobvr, n, a: *mut T, lda, w: *mut Complex<T::Real>,
vl: *mut Complex<T::Real>, ldvl, vr: *mut Complex<T::Real>, ldvr)` and
`driver_ggev(..., a, lda, b, ldb, alpha: *mut Complex<T::Real>, beta:
*mut T::Real, vl, ldvl, vr, ldvr)`. For real `T` the vendor impl allocates
the real `WR`/`WI`/`VR` (and `ALPHAR`/`ALPHAI`) temporaries in the call
order, calls the Fortran-faithful thunk, and assembles the complex outputs
through **one shared helper** (`rstsr-blas-traits/src/util/util_eig.rs`:
pack `wr`/`wi → w`, unpack `vr` by the conjugate-pair rule, layout-aware
for both orders) so the assembly is written once and both vendor trees call
it. Complex `T` passes its pointers straight through. Alternative: two
dtype-split traits with the assembly in the builder's real block.

➡️ Yes, one trait. The builder layer stays dtype-agnostic (like `SYEV`/
`SYGV` through `T::Real` slots), the trait keeps its LAPACK name, and
output-convention post-processing in the driver tier has a precedent
(`superb` copy-out of GESVD).

---

❓ **Q6 - Builder options and outputs**:
- `GEEV_`: `a`, `left: bool` (default `false`), `right: bool` (default
  `true`) → `(w: Tensor<Complex<T::Real>, Ix1>, Option<vl>, Option<vr>)`.
- `GGEV_`: `a`, `b`, `left`, `right` → `(alpha: Tensor<Complex<T::Real>, Ix1>,
  beta: Tensor<T::Real, Ix1>, Option<vl>, Option<vr>)`.
- Aliases `SGEEV`/`DGEEV`/`CGEEV`/`ZGEEV` and `SGGEV`/`DGGEV`/`CGGEV`/`ZGGEV`.
- `a` (and `b`) go through `overwritable_convert_with_order` once (destroyed
  in place when owned/mutable in the call order, else copied); fresh outputs
  in the call order; `testify_ld` on every leading dimension.

➡️ Yes.

---

❓ **Q7 - Entry signature (tuple overloads, SciPy-shaped)**:
- `rt::linalg::eig(&a)` → `EigResult<Tensor<c, D−1>, Tensor<c, D>>`;
  `rt::linalg::eig((&a, &b))` → generalized, same `Out`.
- `rt::linalg::eigvals(&a)` → `Tensor<c, D−1>`; `eigvals((&a, &b))`.
- Each matrix slot accepts `&TensorAny`, `TensorView`, `Tensor`,
  `TensorMut` (owned/mutable inputs are consumed as scratch — GEEV/GGEV
  destroy `a`/`b` — and are never written back; outputs are always fresh).
  The standard form is a 4-row `duplicate_item` table; the generalized form
  4×4 = 16 rows (mechanical).
- `c = Complex<T::Real>`; `eigvals` is `GEEV`/`GGEV` with both vector flags
  off, not a wrapper around `eig`.

➡️ Yes. Same overload set as the QR R2 decision; the `.into()` tuple
conversion of `EigResult` already exists.

---

❓ **Q8 - Output dtype, order, normalization**: always complex
(`Complex<T::Real>`) — NumPy/SciPy's real-downcast of `vr` (and `w`) for an
all-real spectrum is **not** followed (the output type is fixed per input
type under `LinalgAPI::Out`, and faer + the array API never downcast; the
divergence goes in the entry docs). Eigenvalues in LAPACK order, unsorted
(NumPy/faer). GEEV vectors as LAPACK returns them (unit 2-norm, largest
component real); GGEV vectors **renormalized** to unit 2-norm (SciPy parity).

➡️ Yes. Same spectrum, same vectors, same order as NumPy/SciPy up to the
type-level rule.

---

❓ **Q9 - Generalized eigenvalues**: `w = α/β` computed elementwise with
SciPy's rules — `β ≠ 0 → α/β`; `β = 0, α ≠ 0 → inf`; `α = β = 0 → nan` —
on the tensors, after the driver call (no in-driver division, no overflow
pruning).

➡️ Yes. Matches `scipy.linalg.eig` exactly, including singular `B`.

---

❓ **Q10 - Stacked input**: n-dim stacks via `StackLayout` for both forms
(matrix axes last under `RowMajor`, first under `ColMajor`; batch dims of `a`
and `b` broadcast against each other, as `blas_impl_eigh_generalized_f`);
outputs carry the broadcast batch shape and the device order.

➡️ Yes.

---

❓ **Q11 - Zero size and errors**: `n == 0` or an empty batch → `w (…, 0)`,
`vr (…, 0, 0)` synthesized without calling the drivers; non-square `a` or
`b`, or mismatching inner shapes, or `ndim < 2` → `InvalidLayout`;
`info > 0` → the generic `map_lapack_info` message (`"Lapack GEEV (info =
i)"`), `-1010`/`-1011` → `MemoryError`; no finite pre-check (policy).

➡️ Yes.

---

❓ **Q12 - Workspace policy and bridge copies** (recorded here as the
`lapacke-translation` §3a gap notice; the skill table is not edited unless a
contradiction shows up — QR precedent):
`geev` real → query; `geev` complex → query + `rwork = MAX(1, 2N)` allocated
**before** the query; `ggev` real → query; `ggev` complex → query + `rwork =
MAX(1, 8N)` before the query. Query with the real leading dimensions, the
`lwork_from_query` hardening, no fudge factors. Bridge copies LAPACKE-
faithful: `a`/`b` in, `vl`/`vr` out, the back-transpose of the destroyed
`a`/`b` kept (O(n²) against an O(n³) solve).

➡️ Yes.

---

❓ **Q13 - Tests**:
- driver conformance `blas_tests_impl/driver_impl/f64/{geev,ggev}.rs`
  (layout matrix for every matrix input: f-contig, f-prefer `ld > n`,
  c-contig, c-prefer, fully non-contiguous; spectra with real eigenvalues
  only, conjugate pairs only, mixed; `left`/`right` on and off; in-place
  destruction of `a`/`b` by buffer identity; zero-size), symlinked into the
  five crates. Complex-path coverage stays at the linalg tier (c64 files
  below), as in the QR grill — the pass-through + `rwork` path is the simple
  one; the assembly is the risky one.
- linalg conformance `linalg_func/func_{f64,c64}/{eig,eigvals}.rs`:
  fingerprints from the validator scripts (existing `# ### eig`/`eigvals`
  standard sections; new generalized sections in both scripts), plus
  self-validating checks — `A V ≈ V diag(w)`, `A V ≈ B V diag(w)`, unit
  column norms, dtype/shape checks, 3-D stacks with broadcast `b`,
  Fortran-ordered and non-contiguous inputs, zero-size, non-square error.
- Run `tests_driver_impl` + `tests_linalg_func` on `rstsr-openblas` under
  `lapack` and `lapacke` (`--test-threads=1`, `RSTSR_DEV=1`), re-run both
  validator scripts.

➡️ Yes.

---

❓ **Q14 - Docs**: no new doc page. `cheatsheet_linalg.md` gets no new table
row (BLAS-only forms cannot run the faer-backed Rust/parity triad) but a
"DeviceFaer backend notes" bullet: generalized `eig((&a, &b))` /
`eigvals((&a, &b))` are BLAS-only. `array_api_standard_extension.md`'s last
bullet mentions the generalized eig extension. `//!` docs on the new
`blas_impl/{eig,eigvals}.rs` using the input-destruction wording of
`8ebd56a`. CHANGELOG by release-plz.

➡️ Yes.

---

❓ **Q15 - Delivery**: one PR on a new rstsr branch `261125/blas-eig`
**stacked on** `261124/blas-qr` (`9e8dac4`): both touch the same
`LapackDriverAPI`, `linalg_auto_impl/mod.rs`, test `mod.rs` and validator
files, and the two docstring-fix commits set the wording eig should follow;
rebased onto `main` once the QR PR merges. No auto-commit; PR only on your
instruction. Alternative: branch off `main` for an independent PR (trivial
conflicts expected).

➡️ Yes, stacked.

---

❓ **Q16 - Close-out**: at the end, write
`.claude/memory/rstsr-blas-eig.md` (+ index line) with the durable facts
(the "3 kinds" correction, the complex-slot driver shape, the GGEV
renormalization, the validator sections) and a `README.md` here with the
results table.

➡️ Yes.

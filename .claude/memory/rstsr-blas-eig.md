---
name: rstsr-blas-eig
description: "eig/eigvals (complex) on the BLAS devices (2026-11-24/25 grill, closed; impl awaits go): GEEV/GGEV, EigResult<W, VL, VR> with Option sides, EigSide runtime enum (no per-variant Out), Homogeneous marker, complex-slot driver traits + shared real→complex assembly helper, GGEV renormalization; PLAN in 2026-11-24-blas-eig/"
metadata:
  type: project
---

2026-11-24/25. Grill closed in 3 rounds; decisions in
`2026-11-24-blas-eig/DECISIONS.md` (D1–D23), implementation plan in
`PLAN.md`. Code not started; branch will be `261125/blas-eig` off `main`
`b01b6e0` (QR #169 merged). rstsr itself untouched.

- **Fact check that reshaped the prompt**: the non-symmetric problem has
  ONE generalized form (`GGEV`, no `itype`); the three `itype` kinds are
  `eigh`'s (already on BLAS). `eig` has no driver choice (`GEEV` is the
  only algorithm; `GEEVX`/`GGEVX` are expert drivers, `GGES` the Schur/QZ
  family). faer has `eigen`, `eigenvalues`, and an unwired QZ-based
  `generalized_eigen` (U, α, β) — no left vectors.
- **`Out` cannot depend on an enum variant** — `(Tr, EigSide)` is one
  argument type (my R2 table was wrong; owner's "uniformly describe three
  outputs" fixed it). Settled: `EigResult<W, VL, VR> { eigenvalues, left,
  right }`; every BLAS `eig` form returns `EigResult<W, Option<VL>,
  Option<VR>>`; conversions: generic `(W, VL, VR)`, `(W, T)` when exactly
  one side is `Some` (panics otherwise — keeps the old faer `(w, vr)`
  `.into()`), `(W, T, T)` when both are `Some`. Coexistence of the three
  impls probed with rustc (the unwrapped ones would need `T = Option<T>` to
  overlap). Owner allowed the breaking change on faer (output type only).
- `#[non_exhaustive] enum EigSide { None, Left, #[default] Right, Both }`
  with `TryFrom<&str>` (`"none"|"left"|"right"|"both"`) and
  `From<FlagSide>` (`L`→Left, `R`→Right; `Left`/`Right` consts already in
  `rstsr_structs`). `pub struct Homogeneous;` tuple marker → spectrum
  `(..., 2, M)` (`β ≡ 1` for the standard problem). Forms (positional
  `a, b, side, homogeneous`): 16 for `eig`, 4 for `eigvals`. No `EigArgs`.
- LAPACK facts: real `?GEEV`/`?GGEV` return split `WR`/`WI` (or
  `ALPHAR`/`ALPHAI`) + **real** `VR` packed by the conjugate-pair rule →
  assembly needed (SciPy `_linalg_eig.hh::transform_eigvecs`); GEEV
  vectors are unit-norm (SciPy's "left not normalized" docstring is stale);
  GGEV vectors (both sides) are renormalized by SciPy; `rwork` = `2N`
  (zgeev) / `8N` (zggev) allocated **before** the query (LAPACKE);
  `w = α/β` with inf/nan rules (`_make_eigvals`).
- Driver tier: ONE trait per routine with complex output slots
  (`w: *mut Complex<T::Real>`), real-dtype vendor impls assemble through one
  shared helper (`util/util_eig.rs`) — precedent: GESVD `superb` copy-out in
  the driver tier. `lapacke-translation` §3a has no `geev`/`ggev` row: gap
  recorded, skill not edited (owner rule from the QR grill).
- Test assets: validators already carry `# ### eig`/`eigvals` (sorted
  spectrum, `abs` of reordered columns); new sections for left/both/
  generalized/homogeneous. Shim blast radius of the `EigResult` refactor:
  `crates-interop/rstsr-faer-py/linalg/src/linalg.rs:124` (`.eigenvectors`),
  faer tests `test_eig`; `eigh` unaffected.

# Decisions — eig / eigvals (complex) on the BLAS devices (grill closed 2026-11-25)

Rounds: R1 (scope, driver tier, entry, output rules, tests, docs, delivery),
R2 (the `left`/`right`/`homogeneous_eigvals` signature), R3 (owner-delegated
evaluation: `Option`-sided `EigResult`, `EigSide` casts, faer output
refactor). Answers were given on screen and transcribed at the top of
`GRILL-R2-QUESTIONS.md` / `GRILL-R3-QUESTIONS.md`; R3's Q24–Q26 are
agent-evaluated under the owner's delegation (veto-able until the go).
Facts gathered at rstsr `origin/main` `b01b6e0` (QR #169 merged),
rstsr-ffi `83efe50`.

## Scope

- **D1** `eig`/`eigvals` on the BLAS devices in two forms: standard (`GEEV`)
  and generalized `A v = λ B v` (`GGEV`). The three `itype` kinds are
  `eigh`'s (already done); `GEEVX`/`GGEVX` and the QZ/Schur family are out.
- **D2** No driver option (GEEV/GGEV are the only algorithms) and no
  `EigArgs` builder — the tuple forms below are the complete SciPy surface.
- **D3** All five devices through the shared trees + symlinked
  `linalg_auto_impl/{eig,eigvals}.rs`; only OpenBLAS (openmp; `lapack` and
  `lapacke`) is run locally. No FFI change (all symbols bound).

## Shared types (`rstsr-linalg-traits/src/traits_def.rs`)

- **D4** `EigResult<W, VL, VR> { eigenvalues, left, right }` replaces the
  2-field struct (breaking). Conversions (owner amendment 2026-11-25):
  generic `Into (W, VL, VR)` (type-preserving, always);
  `Into (W, T)` from `EigResult<W, Option<T>, Option<T>>` when **exactly
  one** side is `Some`, panics otherwise — so the previous faer
  `.into() → (w, vr)` keeps working; `Into (W, T, T)` when **both** sides
  are `Some`, panics otherwise. All three coexist (coherence probed with
  rustc: the unwrapped impls would need `T = Option<T>` to overlap).
  `EigRightResult` is discarded.
- **D5** `#[non_exhaustive] enum EigSide { None, Left, #[default] Right,
  Both }` with `TryFrom<&str>`/`FromStr` (`"none"`/`"left"`/`"right"`/
  `"both"`) and `From<FlagSide>` (`L`→`Left`, `R`→`Right`).
- **D6** `pub struct Homogeneous;` — tuple-element marker for SciPy
  `homogeneous_eigvals=True`.

## Entry tier (`rstsr-linalg-traits/src/blas_impl/{eig,eigvals}.rs`)

- **D7** Every BLAS `eig` form returns `EigResult<W, Option<VL>,
  Option<VR>>`; `left`/`right` are `Some` exactly for the requested sides;
  the bare form means `EigSide::Right`; `EigSide::None` gives `None`/`None`.
  `W = W1 = Tensor<c, D−1>` (`(..., M)`) or `W2 = Tensor<c, D>` (`(..., 2,
  M)`) with `Homogeneous`; `c = Complex<T::Real>`.
- **D8** Overloads (positional order `a, b, side, homogeneous`; matrix slots
  accept `&TensorAny`/`TensorView`/`Tensor`/`TensorMut`):
  `eig`: `(Tr)`, `(Tr, EigSide)`, `(Tr, &str)`, `(Tr, FlagSide)`, each also
  with `&B` after `a`, with `Homogeneous` appended, or both (16 forms);
  `eigvals`: `(Tr)`, `(Tr, &B)` → `W1`; `(Tr, Homogeneous)`, `(Tr, &B,
  Homogeneous)` → `W2`. `&str`/`FlagSide` rows delegate to `EigSide` rows.
- **D9** Owned / `&mut` inputs are consumed as scratch (GEEV/GGEV destroy
  `a`/`b`; nothing written back); outputs are fresh, in the call order.
- **D10** Output rules: always complex (no NumPy real-downcast; documented);
  LAPACK order, unsorted; GEEV vectors as returned (unit 2-norm, largest
  component real); GGEV vectors on both sides renormalized to unit 2-norm
  (SciPy); generalized `w = α/β` with SciPy's inf/nan rules; homogeneous
  standard problem has `β ≡ 1`, generalized returns raw `α`/`β`.
- **D11** Stacked input via `StackLayout`; `a`/`b` batch dims broadcast.
- **D12** Zero size synthesized (never calls the drivers); non-square /
  mismatch / `ndim < 2` → `InvalidLayout`; `info > 0` → generic
  `map_lapack_info`; no finite pre-check.
- **D13** `eigvals` = GEEV/GGEV with both vector flags off (not a wrapper
  over `eig`).

## Driver tier (`rstsr-blas-traits`)

- **D14** New family `src/lapack_eig/{geev,ggev}.rs`: `GEEV_` (`a`, `left`,
  `right`) → `(w, Option<vl>, Option<vr>)`; `GGEV_` (`a`, `b`, `left`,
  `right`) → `(alpha, beta, Option<vl>, Option<vr>)`; traits
  `GEEVDriverAPI<T>`/`GGEVDriverAPI<T>` with **complex output slots**
  (`*mut Complex<T::Real>`, `beta: *mut T::Real`), registered in
  `LapackDriverAPI<T>` and the preludes; aliases `S/D/C/ZGEEV`, `S/D/C/ZGGEV`.
- **D15** Vendor thunks `driver_impl/{lapack,lapacke}/eig/{geev,ggev}.rs`:
  Fortran-faithful `?geev_work`/`?ggev_work` + trait impls (real/complex
  `duplicate_item` blocks). Real `T`: real `WR`/`WI`/`VR` (and
  `ALPHAR`/`ALPHAI`) temporaries in the call order, assembled into the
  complex slots by **one** shared helper `src/util/util_eig.rs` (pack
  `wr`/`wi`, unpack `vr` by the conjugate-pair rule, both orders). Complex
  `T`: pointers passed through.
- **D16** Workspace (recorded as the `lapacke-translation` §3a gap notice;
  skill table not edited): `geev` real → query; complex → query + `rwork =
  MAX(1, 2N)` before the query; `ggev` real → query; complex → query +
  `rwork = MAX(1, 8N)` before the query; `lwork_from_query` hardening; no
  fudge factors. Bridges LAPACKE-faithful (`a`/`b` in, `vl`/`vr` out,
  back-transpose of destroyed inputs kept).
- **D17** Composition `ref_impl_eig_simple_f` / `ref_impl_eig_generalized_simple_f`
  in `ref_impl_blas.rs` (renormalization, `α/β`, β-ones).

## faer device

- **D18** Output-type refactor only: `eig(&a)` → `EigResult<W1, Option<VL>,
  Option<VR>>` with `left: None`, `right: Some(vr)`; `eigvals` unchanged.
  The faer tests and the py shim keep their `(w, vr)` destructuring through
  the `(W, T)` conversion (D4); only the field access `.eigenvectors` is
  retired in `crates-interop/rstsr-faer-py/linalg/src/linalg.rs:135` and
  `rstsr-linalg-traits/tests/test_faer_func/func_{f64,c64}.rs`.

## Tests, docs, delivery

- **D19** Driver conformance `driver_impl/f64/{geev,ggev}.rs` (layout
  matrix per matrix input, real / conjugate-pair / mixed spectra, `left`/
  `right` combinations, buffer identity, zero-size), symlinked into the five
  crates.
- **D20** Linalg conformance `linalg_func/func_{f64,c64}/{eig,eigvals}.rs`:
  Python-generated fingerprints (validator sections `eig`, `eigvals`,
  `eig_left`, `eig_both`, `eig_generalized`, `eig_generalized_both`,
  `eigvals_generalized`, `eigvals_homogeneous`,
  `eigvals_generalized_homogeneous`; sorted spectrum, reordered `abs`
  columns, homogeneous `(2, M)` stacks sorted by `α/β`) plus
  self-validation (`A V ≈ V diag w`, `Aᴴ VL ≈ VL diag conj w`, `A V ≈ B V
  diag w`, `Aᴴ VL ≈ Bᴴ VL diag conj w`, unit norms, `β ≡ 1`, `α/β ≈ w`,
  `EigSide::None` ≡ spectrum, `Right` ≡ bare, stacks + broadcast `b`,
  layouts, zero-size, non-square, destroyed `TensorMut`, the SciPy 2×2
  docstring case). Run `tests_driver_impl` + `tests_linalg_func` on
  `rstsr-openblas` under `lapack` and `lapacke`.
- **D21** Docs: no new page; "DeviceFaer backend notes" bullet for the
  BLAS-only generalized/left/homogeneous forms (no table rows — the
  faer-backed triad cannot run them); `array_api_standard_extension.md`
  bullet extended; rustdoc on `EigResult`/`EigSide`/`Homogeneous`; `//!`
  docs on the new `blas_impl` files in the `8ebd56a` wording; CHANGELOG by
  release-plz.
- **D22** Delivery: fast-forward local `main` to `b01b6e0`, branch
  `261125/blas-eig` from it; one PR; no auto-commit; PR only on instruction.
- **D23** Close-out: `.claude/memory/rstsr-blas-eig.md` + index line;
  `README.md` here with the results table.

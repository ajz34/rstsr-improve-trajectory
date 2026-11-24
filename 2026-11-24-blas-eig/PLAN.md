# PLAN — eig / eigvals (complex) on the BLAS devices

Decisions: `DECISIONS.md` (D1–D23). Branch `261125/blas-eig` off
`main` @ `b01b6e0`. Batches in dependency order; each batch ends with its
own verification before the next starts. No commit in the rstsr repo (PR on
instruction only).

## B0 — branch

```bash
cd ~/rstsr_pack/rstsr
git checkout main && git pull --ff-only origin main      # -> b01b6e0 (QR #169 merged)
git checkout -b 261125/blas-eig
```

## B1 — shared types + faer output refactor (`rstsr-linalg-traits`, shim)

Files:
- `rstsr-linalg-traits/src/traits_def.rs` `#region eig`:
  - `pub struct EigResult<W, VL, VR> { pub eigenvalues: W, pub left: VL, pub
    right: VR }` (replaces the 2-field struct);
  - `impl<W, VL, VR> From<(W, VL, VR)> for EigResult<W, VL, VR>` and the
    reverse (generic, type-preserving);
  - `impl<W, T> From<EigResult<W, Option<T>, Option<T>>> for (W, T)` —
    exactly one side `Some`, else `panic!` (message names the two-sided and
    no-sided cases);
  - `impl<W, T> From<EigResult<W, Option<T>, Option<T>>> for (W, T, T)` —
    both `Some`, else `panic!`;
  - `#[non_exhaustive] #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub enum EigSide { None, Left, #[default] Right, Both }` +
    `TryFrom<&str>` (`"none"|"left"|"right"|"both"`, else
    `rstsr_error!(InvalidValue, ...)`), `FromStr`, `From<FlagSide>`
    (`L → Left`, `R → Right`), rustdoc (type floor: summary + conversions);
  - `pub struct Homogeneous;` with rustdoc.
- `rstsr-linalg-traits/src/prelude.rs`: `EigSide`, `Homogeneous` into
  `rstsr_structs` (`EigResult` is already exported).
- `faer_impl/eig.rs`: `type Out = EigResult<Tensor<c, DeviceFaer, D::SmallerOne>,
  Option<Tensor<c, DeviceFaer, D>>, Option<Tensor<c, DeviceFaer, D>>>`;
  `left: None, right: Some(v)`; module doc updated.
- `tests/test_faer_func/func_{f64,c64}.rs` `test_eig`: `(r.eigenvalues,
  r.eigenvectors)` → `let (w, v) = rt::linalg::eig(a.view()).into();`
  (the `(W, T)` conversion — same destructuring as before).
- `crates-interop/rstsr-faer-py/linalg/src/linalg.rs:124`: `r.eigenvectors`
  → `r.right.unwrap()` (or the 2-tuple `.into()`).

Verify:
```bash
cargo test -p rstsr-linalg-traits --features faer          # faer suite incl. test_eig
cargo check -p rstsr-faer-py-linalg                        # shim compiles
cargo test -p rstsr --test cheatsheet_linalg               # facade parity
```

## B2 — driver tier (`rstsr-blas-traits`)

Files (new unless noted):
- `src/util/util_eig.rs` (+ `pub mod util_eig;` in `util/mod.rs`,
  `prelude_dev.rs` re-export):
  - `pack_eigenvalues(wr: &[R], wi: &[R]) -> Vec<Complex<R>>` (or write
    into a `*mut Complex<R>` slice);
  - `unpack_real_eigvecs(order: FlagOrder, n, ldv, v_real: &[R], wi: &[R],
    out: &mut [Complex<R>])` — the conjugate-pair rule, column `j` real →
    `v[:, j]`; pair `(j, j+1)` → `v[:, j] ± i v[:, j+1]`; strides from
    `order` (`[1, ldv]` ColMajor, `[ldv, 1]` RowMajor).
- `src/lapack_eig/mod.rs`, `geev.rs`, `ggev.rs`:
  - `pub trait GEEVDriverAPI<T> { unsafe fn driver_geev(order, jobvl: char,
    jobvr: char, n, a: *mut T, lda, w: *mut Complex<T::Real>, vl: *mut
    Complex<T::Real>, ldvl, vr: *mut Complex<T::Real>, ldvr) -> blas_int; }`
  - `pub trait GGEVDriverAPI<T> { unsafe fn driver_ggev(order, jobvl, jobvr,
    n, a, lda, b, ldb, alpha: *mut Complex<T::Real>, beta: *mut T::Real, vl,
    ldvl, vr, ldvr) -> blas_int; }`
  - `GEEV_` builder: `a: TensorReference<Ix2>`, `left: bool` (false),
    `right: bool` (true); `internal_run` → `(Tensor<c, Ix1>,
    Option<Tensor<c, Ix2>>, Option<Tensor<c, Ix2>>)`; order by
    `get_output_order(&[], &[(a.c_prefer(), a.f_prefer())], default)`;
    `overwritable_convert_with_order(a, order)`; `testify_ld` on `a`, and
    on `vl`/`vr` when allocated (`empty_f(([n, n], order, &device))`);
    `n == 0` → empty `w`, empty `(0, 0)` sides, no driver call;
    `map_lapack_info(info, "Lapack GEEV")`.
  - `GGEV_` builder: `a`, `b`, `left`, `right` → `(alpha: Tensor<c, Ix1>,
    beta: Tensor<T::Real, Ix1>, Option<vl>, Option<vr>)`; order voted by
    `a` **and** `b`; `same_device` check; both inputs through
    `overwritable_convert_with_order`; zero-size as above.
  - Aliases `SGEEV/DGEEV/CGEEV/ZGEEV`, `SGGEV/DGGEV/CGGEV/ZGGEV`.
- `src/lib.rs` (`pub mod lapack_eig;`), `src/prelude.rs` (`pub use
  crate::lapack_eig::*;`), `src/trait_def.rs` (`LapackDriverAPI<T>` +=
  `GEEVDriverAPI<T>` + `GGEVDriverAPI<T>`).
- `driver_impl/lapack/eig/{mod.rs,geev.rs,ggev.rs}` and
  `driver_impl/lapacke/eig/{mod.rs,geev.rs,ggeev.rs}`;
  `driver_impl/{lapack,lapacke}/mod.rs` += `pub mod eig;`.
  - `lapack/eig/geev.rs`: `#[duplicate_item]` real `[f32]/[f64]` →
    `{s,d}geev_work` (Fortran argument order: `jobvl, jobvr, n, a, lda, wr,
    wi, vl, ldvl, vr, ldvr, work, lwork, info`; ColMajor direct; RowMajor
    `lda_t = MAX(1, n)`, ld checks with Fortran indices `-5`/`-8`/`-10`/
    `-12`... (per the dgeev.f argument numbering), query short-circuit,
    `a_t`/`vl_t`/`vr_t` temporaries, `orderchange_*_ix2_cpu_serial`
    bridges, `-1010`/`-1011` sentinels) + the trait impl: query →
    `lwork_from_query` → `work` → compute; then **assembly** into the
    complex slots (`pack_eigenvalues`, `unpack_real_eigvecs`); complex
    `[Complex<f32>]/[Complex<f64>]` → `{c,z}geev_work` (+`rwork`, `lrwork`)
    + trait impl with `rwork = MAX(1, 2N)` allocated **before** the query,
    pointers passed through.
  - `lapack/eig/ggev.rs`: same shape for `{s,d}ggev_work` /
    `{c,z}ggev_work`; real outputs `alphar`/`alphai`/`beta`/`vl`/`vr`;
    assembly of `alpha` and the vectors; complex `rwork = MAX(1, 8N)`.
  - `lapacke/eig/*.rs`: `LAPACKE_{s,d,c,z}geev` / `...ggev` pass-through
    with the same real-dtype assembly (LAPACKE returns row-major real
    buffers for RowMajor).
- Device crates: nothing beyond the symlinked trees (openblas/mkl get
  `lapack` + `lapacke`; aocl/blis/kml `lapack` only) — verify each crate's
  `driver_impl/mod.rs` picks the module up.

Verify:
```bash
cargo build -p rstsr-blas-traits
RSTSR_DEV=1 cargo test -p rstsr-openblas --test tests_driver_impl -- --test-threads=1   # after B4 driver files exist
```

## B3 — composition + entries (`rstsr-linalg-traits`)

- `src/ref_impl_blas.rs` `#region eig`:
  - `ref_impl_eig_simple_f(args: GEEV_) -> (Tensor<c, Ix1>,
    Option<Tensor<c, Ix2>>, Option<Tensor<c, Ix2>>)` —
    `device.with_blas_num_threads(nthreads, || GEEV::default().a(a)
    .left(left).right(right).build()?.run())`.
  - `ref_impl_eig_generalized_simple_f(args: GGEV_) -> (alpha, beta,
    Option<vl>, Option<vr>)`; renormalization is done at the entry tier on
    the stacked tensors (see B3 entry) — or here per slice; decide at
    implementation (per-slice keeps the helper local).
- `src/blas_impl/eig.rs`:
  - `blas_impl_eig_f(a: TensorReference<IxD>, b: Option<TensorReference<IxD>>,
    side: EigSide, homogeneous: bool) -> EigResult<Tensor<c, IxD>,
    Option<Tensor<c, IxD>>, Option<Tensor<c, IxD>>>`: square check,
    broadcast (generalized), `overwritable_convert_with_order` on `a`/`b`,
    per-slice driver call, GGEV renormalization (both sides, unit 2-norm),
    `w = α/β` (SciPy inf/nan rules) unless homogeneous; homogeneous → `w`
    stacked `(..., 2, M)` (`[α; β]`, `β ≡ 1` for GEEV) in the device
    order; zero-size short-circuit.
  - `blas_impl_eigvals_f(a, b, homogeneous) -> Tensor<c, IxD>` (`D−1` or
    `D`), both vector flags off.
  - `duplicate_item` tables per D8: `eig` — `(Tr)`, `(Tr, EigSide)`,
    `(Tr, &str)`, `(Tr, FlagSide)`, `(Tr, Homogeneous)`, `(Tr, EigSide,
    Homogeneous)`, `(Tr, &str, Homogeneous)`, `(Tr, FlagSide, Homogeneous)`
    and the eight `(Tr, &B, …)` twins; `eigvals.rs` — `(Tr)`, `(Tr, &B)`,
    `(Tr, Homogeneous)`, `(Tr, &B, Homogeneous)`. Matrix slots:
    `&TensorAny<R, ..>`, `TensorView`, `Tensor`, `TensorMut` (owned/mut
    consumed as scratch). `&str`/`FlagSide` rows delegate (parse/convert,
    call the `EigSide` row). `'a` in the generics column only for rows whose
    `Tr` carries a lifetime (QR lesson).
  - `//!` docs in the `8ebd56a` wording (input destruction condition, dtype
    rule, order note, `EigSide`/`Homogeneous` semantics, SciPy accordance
    incl. the no-real-downcast divergence).
- Device crates ×5: `src/linalg_auto_impl/eig.rs` and `eigvals.rs`
  symlinks → `../../../../rstsr-linalg-traits/src/blas_impl/{eig,eigvals}.rs`;
  `linalg_auto_impl/mod.rs` += `pub mod eig; pub mod eigvals;`.

Verify:
```bash
cargo test -p rstsr-linalg-traits
cargo clippy -p rstsr-linalg-traits -p rstsr-blas-traits -p rstsr-openblas --all-targets
cargo fmt --all -- --check
```

## B4 — tests

- Validators `rstsr-blas-traits/src/blas_tests_impl/linalg_func/func_validation_{f64,c64}.py`
  (SciPy 1.18.1 / NumPy 2.5.x, env `torch`): keep `# ### eig`, `# ###
  eigvals`; add `# ### eig_left` (`left=True, right=False`), `# ### eig_both`,
  `# ### eig_generalized` (`scipy.linalg.eig(a, b)`), `# ###
  eig_generalized_both`, `# ### eigvals_generalized`, `# ###
  eigvals_homogeneous` (`np.linalg.eigvals` → `[w; 1]`), `# ###
  eigvals_generalized_homogeneous` (`scipy.linalg.eigvals(a, b,
  homogeneous_eigvals=True)`). Canonical form: spectrum sorted
  lexicographically (`np.sort`), vector columns reordered by the same
  order, `abs(...)`; homogeneous `(2, M)` stacks with columns sorted by
  `α/β`. Also record the SciPy 2×2 docstring case values.
- Driver bodies `blas_tests_impl/driver_impl/f64/{geev,ggev}.rs` (+ `f64/mod.rs`):
  layout matrix per matrix input (f-contig, f-prefer `ld > n`, c-contig,
  c-prefer, fully non-contiguous), spectra (real only / conjugate pairs
  only / mixed), `left`/`right` on-off (results equal the single-sided
  runs), buffer identity of destroyed `a`/`b`, zero-size; self-validating
  (`A V ≈ V diag w`, `Aᴴ VL ≈ VL diag conj w`).
- Linalg bodies `blas_tests_impl/linalg_func/func_{f64,c64}/{eig,eigvals}.rs`
  (+ both `mod.rs`): the validator fingerprints; `EigSide::None` ≡
  `eigvals`, `Right` ≡ bare, `(Tr, "left")`/`(Tr, Left)` casts,
  `(W, V)`/`(W, VL, VR)` conversions (and the panic cases under
  `#[should_panic]`), stacks with broadcast `b`, layouts, zero-size,
  non-square, destroyed `TensorMut`, exact 2×2 case.
- Symlinks: `tests/driver_impl/f64/{geev,ggev}.rs` and
  `tests/linalg_func/func_{f64,c64}/{eig,eigvals}.rs` into all five crates
  (+ the `mod.rs` edits there); `tests/README.md` unchanged (no dedicated
  exceptions expected).
- Run:
  ```bash
  RSTSR_DEV=1 cargo test -p rstsr-openblas --test tests_driver_impl -- --test-threads=1
  RSTSR_DEV=1 cargo test -p rstsr-openblas --test tests_linalg_func  -- --test-threads=1
  # lapacke config against the conda OpenBLAS (system one exports no LAPACKE_*)
  conda run -n rest-user env RSTSR_DEV=1 cargo test -p rstsr-openblas --features lapacke --test tests_driver_impl --test tests_linalg_func -- --test-threads=1
  python rstsr-blas-traits/src/blas_tests_impl/linalg_func/func_validation_f64.py   # (torch env)
  python rstsr-blas-traits/src/blas_tests_impl/linalg_func/func_validation_c64.py
  ```
  Optional: MKL/AOCL eig subsets if the libraries link locally.

## B5 — docs, wrap-up

- `rstsr/src/docs/cheatsheet_linalg.md`: "DeviceFaer backend notes" bullet
  — generalized `(a, b)`, `EigSide` (`left`/`both`/`none`) and
  `Homogeneous` forms are BLAS-only; `EigResult` has three fields.
- `rstsr/src/docs/array_api_standard_extension.md`: extend the last bullet
  (generalized + side/homogeneous extensions are BLAS-only).
- CHANGELOG: release-plz (conventional commit titles).
- Task dir: `README.md` (results table, gotchas); memory
  `.claude/memory/rstsr-blas-eig.md` + index line; commit here.

## Risk register

- Shim crate build needs the python env (`cargo check -p rstsr-faer-py-linalg`);
  if it cannot build locally, keep the one-line edit mechanical and state it.
- 1024×1024 GGEV in the validators: seconds; fine.
- Coherence of the three `EigResult` conversions: probed OK (2026-11-25).
- `EigSide::None` reads next to `Option::None` in match arms — always
  qualified.
- KML/BLIS/AOCL `lapack` trees get the files by symlink; compile-only risk
  (no local libs) — keep to the shared-tree discipline.

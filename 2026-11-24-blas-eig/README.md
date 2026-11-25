# eig / eigvals (complex) on the BLAS devices — outcome (2026-11-25)

Decisions: `DECISIONS.md` (D1–D23). Plan: `PLAN.md` (B0–B5, all done).
Implementation: rstsr branch `261125/blas-eig` on top of `b01b6e0` (`main`),
committed by the owner as `23f726f`. The review fixes of 2026-11-25 are
**uncommitted** on top of it (owner rule: no auto-commit; PR on instruction).
Changelog and versions untouched.

## What was built

- `rstsr-linalg-traits/src/traits_def.rs`: `EigResult<W, VL, VR>` (three
  generic fields; tuple conversions `(W, VL, VR)`, `(W, T)` for exactly one
  requested side, `(W, T, T)`), `#[non_exhaustive] EigSide { None, Left,
  #[default] Right, Both }` (`TryFrom<&str>`, `From<FlagSide>`), `Homogeneous`
  marker. The faer `eig` was refactored onto the same output type
  (`faer_impl/eig.rs`), shim `rstsr-faer-py/linalg` adapted.
- `rstsr-blas-traits`: `GEEVDriverAPI<T>` / `GGEVDriverAPI<T>` +
  `GEEV_`/`GGEV_` builders (`src/lapack_eig/{geev,ggev}.rs`), shared real →
  complex assembly `util/util_eig.rs`, vendor thunks
  `driver_impl/{lapack,lapacke}/eig/{geev,ggev}.rs`.
- `rstsr-linalg-traits/src/ref_impl_blas.rs`: `ref_impl_eig_simple_f`,
  `ref_impl_eig_generalized_simple_f`, the n-dim walk `ref_impl_eig_nd_f`
  (shared by `eig` and `eigvals`); `blas_impl/{eig,eigvals}.rs` (16 + 4
  overload forms, symlinked as `linalg_auto_impl/` in the five device crates).
- Tests: `blas_tests_impl/driver_impl/f64/{geev,ggev}.rs` (34 tests),
  `linalg_func/func_{f64,c64}/{eig,eigvals}.rs` (38 tests), validators
  extended (`# ### eig (left / both)`, `eig (generalized …)`,
  `eigvals (generalized / homogeneous)`, driver `dgeev/zgeev/dggev/zggev`).
- Docs: `array_api_standard_extension.md` notes, `cheatsheet_linalg.md`
  "DeviceFaer backend notes" bullet. No cheat-sheet rows (plan B5).

## Results

| Config | tests_driver_impl | tests_linalg_func |
|---|---|---|
| OpenBLAS 0.3.32 (system), `lapack` | 80/80 | 74/74 (full suite) |
| OpenBLAS 0.3.30 (conda `rest-user`), `lapacke` | eig subset 32/32 | — |
| MKL 2026.1, `lapack` | 80/80 | eig subset 28/28 |
| MKL 2026.1, `lapacke` | eig subset 32/32 | — |
| AOCL 5.3.0, `lapack` | 80/80 | eig subset 28/28 |

Also green: `cargo test -p rstsr-linalg-traits --features faer` (70/70),
facade `cheatsheet_linalg` (11/11), clippy (default lints, tests included) on
`rstsr-linalg-traits`, `rstsr-blas-traits`, `rstsr-openblas`, `rstsr-mkl`,
`rstsr-aocl`, `rstsr`; `cargo check` on `rstsr-blis`/`rstsr-kml`;
`rstsr-faer-py-linalg` check; rustfmt. BLIS/KML not exercised (no local
library); both consume the same shared trees. MKL `lapacke`: the pre-existing
`gesdd` failure (argument 11) is unrelated to eig. The system OpenBLAS exports
no `LAPACKE_*` symbols: the `lapacke` config links against the conda OpenBLAS
only (`LIBRARY_PATH`/`LD_LIBRARY_PATH` to the env's `lib`). MKL tests need
`LD_LIBRARY_PATH` with `compiler/<ver>/lib` (libiomp5) or
`MKL_THREADING_LAYER=SEQUENTIAL` from Python.

## Deviations from the plan

1. **GGEV `beta` is complex for C/ZGGEV** (`beta: *mut T` end to end; type
   `T` in `GGEVDriverAPI`, the vendor thunks and the simple impl), not real
   as the plan assumed. Found as a heap overflow: the first draft allocated
   `n` reals where LAPACK writes `n` complex. The simple impl returns
   `(alpha, beta)` both as `Tensor<Complex<T::Real>>`; the linalg tier still
   hands out `w = α/β` with SciPy's inf/nan rules.
2. The n-dim walk lives in `ref_impl_blas.rs` (`ref_impl_eig_nd_f`, a
   closure-walk with a `kernel` per slice) and is shared by `eig` and
   `eigvals` (`EigSide::None`) rather than duplicated per entry.
3. In-place rule sharpened: **each** input is destroyed independently when
   its batch equals the output batch; a slice of a broadcast input is copied
   out of a view broadcast to the output batch. The first draft only worked
   in place when *both* operands were in place (found by the nd tests).

## Review (2026-11-25) and fixes

Two-axis review of `23f726f` in three sub-agents (values / SciPy compliance,
API shape and docstrings, driver conventions), every point re-derived in the
main session before a fix. Owner priorities: value correctness and SciPy
compliance first, then API shape and docs. All fixes are uncommitted.

| # | Sev | Finding (re-verified) | Fix |
|---|---|---|---|
| 1 | med | `w = α/β` used `num`'s naive complex division (`|β|²`): zeros for `‖B‖ ≳ 1.8e19` (f32) / `1.3e154` (f64), inf/NaN for `‖B‖ ≲ 1e-19` / `1e-154`; SciPy divides overflow-safe | `cdiv` (real `β` component-wise, Smith's algorithm otherwise) in `ref_impl_blas.rs`; regression tests f64/c64 |
| 2 | med | static-dim generalized overloads: `b` with more batch axes than `a` compiled and then panicked inside `*_f` (`into_dim` unwrap) | `InvalidLayout` from the static-dim cores only (`shape().const_ndim()`), documented, tested |
| 3 | med | layout tests delivered compact inputs (`make_case` ended in `into_owned()`) | `into_owned_keep_layout()`, `view()` + `view_mut()` runs, stride asserts, padding integrity; conflicting-order pencils and a shape-rejection test for GGEV |
| 4 | med | linalg tests: no residual identities, no conversion `should_panic`, mismatch test hit the wrong branch, no `ndim < 2` | per-slot `A v = λ v` / `A v = λ B v` / left forms; 5 conversion tests; square-size mismatch, non-square `b`, 1-D inputs |
| 5 | med | docs overstated faer coverage (cheat sheet, array-API note, `EigSide`/`Homogeneous`); cheat sheet named the two-field struct; `eig((a, Left))` is not exported by the prelude; cheat-sheet rows had no Rust/parity twins | corrected (`FlagSide::L`); rows `eig_standard` / `eigvals_standard` added to `cheatsheet_linalg.rs` and `parity.py` |
| 6 | low | `ndim < 2` → `AxisError` (D12 says `InvalidLayout`) | assert in `stack_split_layout` (all stacked entries) |
| 7 | low | complex GGEV thunk had `beta: *mut Real` | `*mut T` |
| 8 | low | real LAPACK-path impls dropped the compute call's return (`-1011` swallowed) | query and compute return values are used |
| 9 | low | `unpack_real_eigvecs` read past `n` for an unpaired complex eigenvalue in release builds | real column fallback, debug assert kept |
| 10 | low | builder docs said views are destroyed; NaN form per slice vs SciPy per batch undocumented; col-major batch alignment undocumented; `eig:` prefix in `eigvals` errors | docs; `entry` name threaded through `ref_impl_eig_nd_f` |

Left open (judgement / owner decisions): the 16-row slot table repeated per
impl and two ~15-line column normalizers (sibling-endorsed style); thunk glue
duplicated between the `lapack` and `lapacke` trees (each mirrors its vendor
API); the layout helpers of `svd_layout.rs` / `geqrf.rs` / `orgqr.rs` still
gather (pre-existing, follow-up); the `from_raw_parts_mut(a, n * lda)` nominal
overrun (ADR-0009 pattern); a pre-existing rustdoc `redundant_explicit_links`
in the `pinv` doc.

### Test sizes (owner rule)

Owner's rule after seeing the run time: large (256 / 1024) matrices are for
*fingerprint* tests only; overload and parameter tests use the smallest
workable matrix (3–5). Before: eight 1024-size tests re-solved the same
matrix for every form — about 34 full solves per dtype, 68 in all, 490 s on
OpenBLAS (89% of the linalg suite) and 434 s on AOCL. After:

- Fingerprint tests, one full-size solve each: `test_eig` (GEEV, 1024;
  `EigSide::Both` gives `w`, `vl`, `vr` in one call), `test_eig_generalized`
  (GGEV, moved to the 256 leading blocks: one full-size QZ solve costs
  27–35 s), `test_eigvals` (values-only GEEV at 1024, GGEV at 256). The 256
  constants were regenerated with SciPy and written into the validator
  blocks; the homogeneous fingerprints were dropped (the form is checked
  against the plain spectrum at small size).
- `test_eig_overloads`, `test_eigvals_overloads`, `test_eig_homogeneous`:
  4x4 blocks of the manifest data, all sixteen operand-kind combinations x
  every side / homogeneous form, compared against the small `Both` result.
- Driver tier: `test_dgeev_relations` 64 → 6.

| Config (final state) | linalg suite | driver suite | `eig` name filter |
|---|---|---|---|
| OpenBLAS 0.3.32, `lapack` | 94/94 in 34 s | 82/82 in 22 s | 48/48 in 22 s |
| OpenBLAS 0.3.30 (conda), `lapacke` | — | 82/82 | 48/48 in 22 s |
| MKL 2026.1, `lapack` | — | eig 34/34 | 48/48 in 21 s |
| AOCL 5.3.0, `lapack` | — | eig 34/34 | 48/48 in 24 s |

Also green at the final state: clippy `-D warnings` on all targets
(`rstsr-openblas` lapack + lapacke, `rstsr-mkl`, `rstsr-aocl`,
`rstsr-linalg-traits`, `rstsr-blas-traits`, `rstsr` all features), the
`rstsr-blas-traits` lib tests (debug and release), the `rstsr` tests with
`linalg` (cheat sheet 13/13), rustdoc of the umbrella crate, `cargo check`
of `rstsr-blis` / `rstsr-kml`, rustfmt, and the three Python validators
(6 s / 10 s, down from 63 s / 84 s).

## Notes for the implementer / reviewer

- **Fingerprint robustness (validator + tests, all learned the hard way)**:
  - The *sorted complex order* is not robust when distinct eigenvalues share
    a real part (conjugate pairs at roundoff): fingerprint the sorted real and
    imaginary parts independently.
  - Eigenvector blocks: per-column fingerprints of `|v|`, sorted
    (order-invariant). Reordering columns by a canonical spectrum order is
    fragile for the same reason.
  - The raw random `b` (and `b + n·I`) gives an ill-conditioned 1024 pencil:
    three LAPACK builds agree on the real parts (1e-13) but differ by 0.06 in
    the imaginary parts and in the eigenvectors (residuals all ~1e-18). The
    generalized fingerprints use an SPD `B = b·bᵀ + n·I` (f64); the raw
    complex `b` is fine (c64). The linalg tier fingerprints 256 blocks.
  - The **raw LAPACK scaling of GGEV vectors differs between builds** (MKL vs
    OpenBLAS: column norms 1.0–5.9, "largest component ±1", but per-column
    scaling differs): driver-tier fingerprints rescale every column to unit
    2-norm first. The linalg tier already does that.
  - Homogeneous fingerprints (`|α|, |β|` scaled by `1/(|α|+|β|)` per column,
    rows sorted independently) were dropped in the review: the form is now
    checked against the plain spectrum on a 4x4 matrix.
- **rstsr test-API facts**:
  - `view.raw()` spans the whole storage; iterate with `i(k).to_scalar()`.
  - `i((.., i, j))` does **not** expand `..` over all leading axes (a single
    `..` is one axis): spell the leading axes out.
  - `TensorReference` is `TensorAny<DataReference>`: an owned copy must be
    bound and passed as `t.view_mut().into()`; `broadcast_to` borrows a
    bound view (E0716 otherwise).
  - Column-major stacks: matrix axes first (`[n, n, batch…]`), batch
    broadcast aligns *leading* batch dims (a one-axis `b` pairs with the
    first batch axis of `a`), the batch iteration order is row-major over the
    batch dims.
  - `(&a, &b)` with `b: &Tensor` is a double reference; `eig((a, Homogeneous))`
    keeps the default right side.
- `duplicate_item` tables: the core tables may elide the lifetime of a sole
  `&TensorAny` operand (clippy `needless_lifetimes`); the delegation tables
  may not (E0637, the associated `Out` carries the lifetime). The generator
  (`/tmp/eiggen/gen_eig.py`, scratch) post-processes only the core tables.
- Cross-vendor driver-tier facts: DGEEV/ZGEEV and DGGEV/ZGGEV fingerprints of
  the spectrum agree across OpenBLAS, MKL and AOCL to 1e-8.

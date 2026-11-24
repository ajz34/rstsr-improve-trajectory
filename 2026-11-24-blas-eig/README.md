# eig / eigvals (complex) on the BLAS devices — outcome (2026-11-25)

Decisions: `DECISIONS.md` (D1–D23). Plan: `PLAN.md` (B0–B5, all done).
Implementation: rstsr branch `261125/blas-eig` on top of `b01b6e0` (`main`),
**uncommitted** (owner rule: no auto-commit; PR on instruction). Changelog and
versions untouched.

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
- Tests: `blas_tests_impl/driver_impl/f64/{geev,ggev}.rs` (32 tests),
  `linalg_func/func_{f64,c64}/{eig,eigvals}.rs` (28 tests), validators
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
    generalized 1024 fingerprints use an SPD `B = b·bᵀ + 1024·I` (f64); the
    raw complex `b` is fine (c64).
  - The **raw LAPACK scaling of GGEV vectors differs between builds** (MKL vs
    OpenBLAS: column norms 1.0–5.9, "largest component ±1", but per-column
    scaling differs): driver-tier fingerprints rescale every column to unit
    2-norm first. The linalg tier already does that.
  - Homogeneous fingerprints: `|α|, |β|` scaled by `1/(|α|+|β|)` per column,
    rows sorted independently.
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

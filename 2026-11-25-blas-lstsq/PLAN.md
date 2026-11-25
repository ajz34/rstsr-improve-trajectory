> Status: implemented on `261125/blas-lstsq` — see `IMPLEMENTATION.md` for the
> landed code, deviations and verification.

# PLAN — `lstsq` on the BLAS devices

Decisions: `DECISIONS.md` (D1–D16). Branch `261125/blas-lstsq` off `ed1f62d`
(`main`). No commit in the rstsr repo (PR on instruction only).

**Shared sources.** The device crates consume the `rstsr-blas-traits` /
`rstsr-linalg-traits` sources through symlinks (not copies):
`crates-device/*/src/driver_impl/{lapack,lapacke}` → the shared
`rstsr-blas-traits/driver_impl/...` trees;
`crates-device/*/src/linalg_auto_impl/<fn>.rs` → the shared
`rstsr-linalg-traits/src/blas_impl/<fn>.rs`;
`crates-device/*/tests/{driver_impl,linalg_func}/**` → the shared
`rstsr-blas-traits/src/blas_tests_impl/**`. So most edits happen once in the
shared trees; per-device work is limited to new tracked symlinks and the
per-device `linalg_auto_impl/mod.rs` lists (which are real per-device files).

## B0 — branch

```bash
cd ~/rstsr_pack/rstsr
git checkout main && git pull --ff-only origin main           # -> ed1f62d
git checkout -b 261125/blas-lstsq
git status --short            # expect the two pre-existing trajectory-only files untouched
```

## B1 — result type and trait surface (`rstsr-linalg-traits`, shared)

Files:
- `src/traits_def.rs` `#region lstsq`:
  - `#[derive(Debug, Clone)] pub struct LstsqResult<X, Res, Rank, S> { pub x: X,
    pub residuals: Res, pub rank: Rank, pub s: S }` — rustdoc: the four NumPy
    outputs, residuals convention (D6), rank per slice (D7);
  - `impl<X, Res, Rank, S> From<(X, Res, Rank, S)> for LstsqResult<...>` and the
    reverse (type-preserving, no panic paths);
  - the three `duplicate_item` tables (`traits_def.rs:8,41,61`): add
    `[LstsqAPI] [lstsq] [lstsq_f]` rows (alphabetical after `inv`).
- `src/prelude.rs`: `LstsqAPI` in `rstsr_traits`, `lstsq, lstsq_f` in
  `rstsr_funcs`, `LstsqResult` in `rstsr_structs`.
- No `LstsqArgs` builder type: the entry surface is the tuple (D2).

Verify: `cargo check -p rstsr-linalg-traits` (the faer feature is not involved:
no faer impl, D1).

## B2 — driver API (`rstsr-blas-traits/src`)

New `src/lapack_lstsq/{mod.rs, gelsd.rs}`; family registration in `src/lib.rs`,
`src/prelude.rs`, `src/trait_def.rs` (`LapackDriverAPI<T> +=
GELSDDriverAPI<T>`).

`gelsd.rs`:
- `pub trait GELSDDriverAPI<T> { unsafe fn driver_gelsd(order: FlagOrder,
  m: blas_int, n: blas_int, nrhs: blas_int, a: *mut T, lda: blas_int, b: *mut T,
  ldb: blas_int, s: *mut T::Real, rcond: T::Real, rank: *mut blas_int) ->
  blas_int; }` (documented `# Safety` payload pointer contract);
- `#[derive(Builder)] #[builder(pattern = "owned", no_std,
  build_fn(error = "Error"))] pub struct GELSD_<'a, T, B> { pub a:
  TensorReference<'a, T, B, Ix2>, pub b: TensorReference<'a, T, B, Ix2>,
  pub rcond: T::Real }` (`setter(into)` on `a`/`b`; `rcond` required);
- `impl GELSD_` → `internal_run(order, a_ptr, lda, b_ptr, ldb)`:
  - `order = get_output_order(&[], &[(a.c_prefer(), a.f_prefer()),
    (b.c_prefer(), b.f_prefer())], device.default_order())`;
  - `b_work = zeros_f(([max(m, n), k], order, &device))` (k = 0 → one zero
    column), rows `0..m` from `b` in one assign; `x = zeros_f(([n, k], order))`
    + assign `b_work[0..n]`;
  - call order `m == 0 || n == 0` → synth outputs without a driver call (D9);
  - residual sums from `b_work.raw()` (call-order strides; `norm_sqr()`) when
    `m > n && rank == n`, into a `Tensor<T::Real, B, Ix1>` of length `k`, else
    `None`;
  - error mapping D10/D11: `-4`/`-5`/`-7`/`-10` → `InvalidValue` "contains a NaN
    or infinity"; `-1010`/`-1011` → `MemoryError`; `info > 0` → convergence;
    other negatives → internal-bug path via `map_lapack_info`; NaN `rcond` →
    `InvalidValue` before the call.
- Return `(Tensor<T,B,Ix2>, Tensor<T::Real,B,Ix1>, usize, Option<Tensor<T::Real,B,Ix1>>)`.
- Aliases `SGELSD/DGELSD/CGELSD/ZGELSD`.

Bounds on `impl GELSD_`: `T: BlasFloat`, `B: BlasDriverBaseAPI<T> +
GELSDDriverAPI<T> + DeviceCreationAnyAPI<T> + DeviceCreationAnyAPI<T::Real> +
OpAssignArbitaryAPI<T, Ix2, Ix2> + OpAssignAPI<T, Ix2>` (plus what the `assign_f`
of the `b` slice needs — verify at implementation).

Verify: `cargo check -p rstsr-blas-traits`.

## B3 — thunks (`rstsr-blas-traits/driver_impl`, shared tree)

- `driver_impl/lapack/lstsq/mod.rs` + `driver_impl/lapack/lstsq/gelsd.rs`;
  `driver_impl/lapack/mod.rs` += `pub mod lstsq;`.
  - real `[f32]/[f64]` → `{s,d}gelsd_`; complex `[Complex<f32>]/[Complex<f64>]`
    → `{c,z}gelsd_` (`rwork`, `lrwork`);
  - query: `lwork = -1`, one-element `iwork` (and `rwork` for complex); read
    `work(1)`, `iwork(1)`, `rwork(1)`; allocate; real call;
  - ColMajor: pass through, `lda/ldb` from `testify_ld`; RowMajor bridge (same
    shape as `svd/gesdd.rs`): `lda_t = max(1, m)`, `ldb_t = max(1, max(m, n))`,
    check user `lda >= n` (`-5`) / `ldb >= nrhs` (`-8`), query short-circuit with
    the user buffers, transpose `a` and `b` in, call, transpose **both** back
    (LAPACKE parity: the destroyed `A` and the solution/residual `B` stay in the
    caller's layout);
  - allocation sentinels `-1010`/`-1011`; negative `info` returned unchanged
    (the builder maps them, D11).
- `driver_impl/lapacke/lstsq/mod.rs` + `gelsd.rs`; `lapacke/mod.rs` +=
  `pub mod lstsq;`: `LAPACKE_[sdcz]gelsd(LAPACK_ROW_MAJOR | LAPACK_COL_MAJOR, m,
  n, nrhs, a, lda, b, ldb, s, rcond, rank)` pass-through (LAPACKE performs the
  query, allocation and transposes; RowMajor user `lda >= n`, `ldb >= nrhs`).
- Unit test for the bridge/ld checks lives in the driver test file (B5); no
  device-crate `mod.rs` edits (the `driver_impl/mod.rs` device files already
  switch lapack/lapacke).

Verify: `RSTSR_DEV=1 cargo check -p rstsr-openblas` (lapack) and
`RSTSR_DEV=1 cargo check -p rstsr-openblas --features lapacke`; then
`cargo check` of `rstsr-mkl`/`rstsr-aocl` (lapack) and compile-only
`rstsr-blis`/`rstsr-kml` (no local libs, `cargo check` still builds the code).

## B4 — entry tier (`rstsr-linalg-traits`, shared) + device wiring

Files:
- `src/stack.rs`: extract the shape-splitting half of `solve_plan` into a
  helper and add
  `pub struct LstsqPlan { pub batch_out: Vec<usize>, pub m: usize, pub n: usize,
  pub k: usize, pub is_vec: bool }` + `pub fn lstsq_plan(a_shape, b_shape,
  order) -> Result<LstsqPlan>`:
  - `a` rank ≥ 2, inner `(m, n)`;
  - `b` rank = `a.rank()` (matrix, inner `(m, k)`) or `a.rank() - 1` (vector,
    inner `(m)`, `k = 1`);
  - batch broadcast via the existing `broadcast_shapes_f` path;
  - `out_shape(order)` (`n` [, `k`]) and `residual_shape(order)` (`k` / empty /
    batch only for the vector form); `solve_plan` keeps its square check.
- `src/ref_impl_blas.rs` `#region lstsq`:
  `pub fn ref_impl_lstsq_f<T, B>(a: TensorView<Ix2>, b: TensorView<Ix2>,
  rcond: T::Real) -> Result<(...)>` with the `get_current_pool` +
  `device.with_blas_num_threads` wrapper and `GELSD_` construction.
- `src/blas_impl/lstsq.rs` (new; shared entry source):
  - `pub fn blas_impl_lstsq_f<T>(a: TensorReference<IxD>, b:
    TensorReference<IxD>, rcond: Option<T::Real>) -> Result<LstsqResult<...>>`
    (docstring in the `blas_impl/svd.rs` style: order convention, destructive
    `a`, `b` copied, 1-D/2-D `b`, residual/rank policy, BLAS-only);
  - `order = device.default_order()`; `LstsqPlan`; `rcond.unwrap_or(eps *
    max(m, n))`; empty batch → empty outputs;
  - per-slice `ref_impl_lstsq_f`; `x`/`s` assigned into `stack_shape` outputs;
    `rank` written through `IterLayout::new(rank.layout(), TensorIterOrder::C)`
    (the `blas_impl/det.rs` pattern); residuals: `stack_slice_at_mut` with inner
    1 for the matrix form, `IterLayout` scalar/squeezed writes for the vector
    form, `NaN` where the per-slice residuals are `None` (D6); `K == 0`/`m == 0`
    paths come from the builder;
  - 16 `LstsqAPI` impl rows (4 input kinds × 4) + the two tuple forms
    (`(a, b)` and `(a, b, rcond)`), all returning dynamic-dim
    `LstsqResult<Tensor<T,B,IxD>, Tensor<T::Real,B,IxD>, Tensor<usize,B,IxD>,
    Tensor<T::Real,B,IxD>>`; `_f` twins generated by `duplicate_item`.
- Device wiring (per device, new tracked symlinks + 5 small edits):
  - `crates-device/rstsr-{aocl,blis,kml,mkl,openblas}/src/linalg_auto_impl/lstsq.rs`
    → `../../../../rstsr-linalg-traits/src/blas_impl/lstsq.rs`;
  - `pub mod lstsq;` in each `src/linalg_auto_impl/mod.rs` (after `inv`).

Verify:
```bash
RSTSR_DEV=1 cargo check -p rstsr-openblas
cargo check -p rstsr-mkl && cargo check -p rstsr-aocl && cargo check -p rstsr-blis && cargo check -p rstsr-kml
cargo test -p rstsr-openblas --test tests_zero_size -- --test-threads=1
```

## B5 — tests, validators and fingerprints

Shared test sources (new files + `mod` lines; then symlinks in the five device
crates):

- `src/blas_tests_impl/driver_impl/f64/gelsd.rs` (+ `f64/mod.rs` `mod gelsd;`):
  `test_dgelsd_relations` (tall/square/wide 4–6 sizes: `A x ≈ b`,
  `Aᴴ(Ax−b) ≈ 0`, `residuals = Σ|b−Ax|²`, `s` vs `DGESDD`), `test_dgelsd_rank`
  (`rcond` sweep on a graded matrix, `rank`/`s` agreement),
  `test_dgelsd_layouts` (RowMajor/ColMajor, c/f-strided, padded ld),
  `test_dgelsd_zero_rhs` (K=0 pad), `test_dgelsd_zero_size`, `test_dgelsd_nan`
  (`a` NaN → one `InvalidValue` message), `test_dgelsd_fingerprint` (4 cases).
  plus `crates-device/rstsr-openblas/tests/zero_size/lapack.rs`: driver-level
  `m == 0`/`n == 0`/`K == 0` boundary cases (OpenBLAS-only harness).
- `src/blas_tests_impl/linalg_func/func_f64/lstsq.rs` and
  `.../func_c64/lstsq.rs` (+ mod lines): `test_lstsq_basic` (1-D/2-D `b`, `x`
  vs SVD reference, residuals, rank), `test_lstsq_wide` (min-norm), 
  `test_lstsq_batch` (broadcast/full; per-slice rank, NaN residuals),
  `test_lstsq_order` (`ColMajor`), `test_lstsq_zero_size`, `test_lstsq_overloads`
  (all 16 input-kind combos, 2-arg/3-arg), `test_lstsq_rcond` (explicit,
  `<= 0`, `>= 1` documented behavior, NaN → error), `test_lstsq_errors`
  (ndim/rows/device), `test_lstsq_fingerprint` (4 f64 / 3 c64 cases, D13).
- Device symlinks (per device):
  `tests/driver_impl/f64/gelsd.rs` → shared;
  `tests/linalg_func/func_f64/lstsq.rs` and `.../func_c64/lstsq.rs` → shared.
- Python validators (shared tree, run manually):
  - `driver_impl/driver_validation_f64.py` `# ### gelsd` (4 cases: 1024×512
    K=1, 1024×512 K=3, 512×1024 K=1, 1024×512 rank-deficient `a[:,2] =
    a[:,0]+a[:,1]`; `x`/`s`/residual fingerprints, `rank`);
  - `linalg_func/func_validation_f64.py` `# ### lstsq` (4 cases, 1-D and 2-D
    `b`, wide, rank-deficient);
  - `linalg_func/func_validation_c64.py` `# ### lstsq` (3 cases).
  - Python default tolerance: pass `cond=eps*max(m,n)` to
    `scipy.linalg.lstsq(..., lapack_driver='gelsd', check_finite=False)`;
    SciPy's own default is `eps`. Fingerprint = `np.dot(np.cos(np.arange(size)),
    a.ravel())`, as in the existing files.
  - Run `python -c "import numpy, scipy"` first: the active env is
    `~/miniconda3/envs/torch` (NumPy 2.5.4 / SciPy 1.18.1).
- Fingerprint values generated by running the three validators, then hardcoded
  in the Rust tests with `abs(...) < 1e-8`.

Device runs (final matrix, documented in `README.md`):
- OpenBLAS: `tests_linalg_func`, `tests_driver_impl`, `tests_zero_size`
  (`lapack` and `lapacke`); `--test-threads=1`.
- MKL: driver + linalg subsets (§ names `gelsd`/`lstsq`); AOCL: same;
  BLIS/KML: compile-only.

## B6 — docs, verification, close-out

- `src/docs`/rustdoc: the `blas_impl_lstsq_f` docstring (B4) is the source-level
  documentation; no cheat-sheet row (D14); add the cheatsheet note in
  `rstsr/src/docs/cheatsheet_linalg.md` ("DeviceFaer backend notes") and the
  `array_api_standard_extension.md` Notes bullet (D14).
- Record the `lapacke-translation` gap notice in `DECISIONS.md` D15 only (no
  skill edit without the owner's word).
- CHANGELOG: release-plz derives it from the PR title; no manual entry.
- Full verification: `cargo fmt --check`, clippy `-D warnings` on all targets
  (openblas both features, mkl, aocl, linalg-traits, blas-traits, rstsr),
  `cargo test` of `rstsr-blas-traits` (debug/release), `rstsr` tests with
  `linalg` (cheat sheet 13/13), rustdoc of `rstsr`, the three validators.
- Task dir `README.md` (results table, runtimes, gotchas) + memory
  `.claude/memory/rstsr-blas-lstsq.md` + `MEMORY.md` index line; commit the
  trajectory repo (co-author trailers).

## Risk register

- **GELSD `nrhs = 0`**: LAPACK's internal `DLALSD` reports `info = -4`; the
  builder's one-zero-column pad is mandatory (verified with SciPy 1.18.1).
- **NaN/Inf info codes differ by config** (`-4` Fortran vs `-5`/`-7` LAPACKE):
  both map to one message; the builder validates `lda`/`ldb` first so those
  codes cannot be layout errors.
- **LAPACK's `rcond >= 1` clamp to machine epsilon**: documented in D4; test
  `test_lstsq_rcond` asserts the documented behavior, not NumPy's.
- **Residual sums for complex**: `norm_sqr` from the destroyed `B`; the
  real-valued dtype is a deliberate divergence (documented).
- **Batched rank assembly**: `IterLayout` writes the logical C order of the
  batch; verify against a 2×2 batch with distinct per-slice ranks in
  `test_lstsq_batch` (the `det` pattern is the precedent).
- **Fingerprint drift across BLAS builds**: GELSD's D&C SVD is deterministic
  per build; 1e-8 absolute on O(10–100) fingerprints is the house tolerance,
  but the rank-deficient `s` values near the threshold are excluded from
  fingerprints (only exact `rank` is asserted there).
- **Memory/host load**: the 1024×512 solves are ~0.4 s (1 thread each); the
  whole linalg suite stays in the tens of seconds.

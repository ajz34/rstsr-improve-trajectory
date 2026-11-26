# PLAN — `lu`, `lu_factor`, `lu_solve` on the BLAS devices

Decisions: `DECISIONS.md` (D1–D23). Branch `261126/blas-lu` off `main` @
`68f4891`. Batches in dependency order; each batch ends with its own
verification before the next starts. No commit in the rstsr repo (PR on
instruction only).

**Shared sources.** The five device crates consume `rstsr-blas-traits` /
`rstsr-linalg-traits` through tracked symlinks: `driver_impl/{lapack,lapacke}`
trees, `linalg_auto_impl/<fn>.rs` (→ `rstsr-linalg-traits/src/blas_impl/`),
and `tests/{driver_impl,linalg_func}/**` (→
`rstsr-blas-traits/src/blas_tests_impl/`). Most edits happen once in the
shared trees; per-device work is new symlinks plus the per-device `mod.rs`
lines.

## B0 — branch

```bash
cd ~/rstsr_pack/rstsr
git checkout main && git pull --ff-only origin main            # -> 68f4891
git checkout -b 261126/blas-lu
git status --short                                             # expect empty
```

## B1 — shared types and trait surface (`rstsr-linalg-traits`, shared)

Files:

- `src/traits_def.rs`, new `#region lu`:
  - `#[derive(Debug, Clone)] pub struct LuResult<P, L, U> { pub p: P, pub
    l: L, pub u: U }` — rustdoc: SciPy relation, shapes, index convention,
    `p: None` for the `permute_l` form;
  - `impl<P, L, U> From<(P, L, U)> for LuResult<P, L, U>` and the reverse;
  - `impl<P, L, U> From<LuResult<Option<P>, L, U>> for (P, L, U)` — `p` must
    be `Some`, else `panic!` naming the `(L, U)` conversion;
  - `impl<P, L, U> From<LuResult<Option<P>, L, U>> for (L, U)` — `p` must be
    `None`, else `panic!` naming the `(P, L, U)` conversion;
  - `#[derive(Debug, Clone)] pub struct LuFactorResult<LU, Piv> { pub lu: LU,
    pub piv: Piv }` + `(LU, Piv)` conversions both ways.
- `src/traits_def.rs`, the three `duplicate_item` tables: `[LuAPI] [lu]
  [lu_f]`, `[LuFactorAPI] [lu_factor] [lu_factor_f]`, `[LuSolveAPI]
  [lu_solve] [lu_solve_f]` after the `LstsqAPI` row.
- `src/prelude.rs`: the three traits in `rstsr_traits`, the six functions in
  `rstsr_funcs`, `LuResult` / `LuFactorResult` in `rstsr_structs`.

Verify:

```bash
cargo check -p rstsr-linalg-traits
cargo test  -p rstsr-linalg-traits --lib          # existing unit tests
```

## B2 — driver tier (`rstsr-blas-traits`)

Files:

- `driver_impl/lapack/solve/getrf.rs` (shared into the five crates): both
  dtype blocks, `if *info != 0 { return *info; }` → `if *info < 0 { return
  *info; }` (the back-transpose runs for a completed factorization, as in
  `LAPACKE_dgetrf_work`).
- `src/lapack_solve/getrf.rs`:
  - `pub fn run_with_info(self) -> Result<(TensorMutable2<'a, T, B>,
    Tensor<blas_int, B, Ix1>, blas_int)>` — the body of today's
    `internal_run` without `map_lapack_info`; rustdoc states that `info > 0`
    is a completed factorization with an exactly-zero pivot;
  - `run()` / `internal_run()` = `run_with_info()` + `map_lapack_info(info,
    "Lapack GETRF")` → `(a, ipiv)` (contract unchanged for `det` / `slogdet`
    / `inv`).
- `~/rstsr_pack/rstsr-agents/skills/lapacke-translation/SKILL.md` §3: one
  sentence — a routine whose `info > 0` is a valid result (`getrf`: completed
  factorization) writes back like LAPACKE. **Edit only, no commit**
  (rstsr-agents is under the no-auto-commit rule).

Verify (system OpenBLAS, `RSTSR_DEV=1`, `-- --test-threads=1`):

```bash
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu RSTSR_DEV=1 RUST_BACKTRACE=1 \
  cargo test -p rstsr-openblas --features "rstsr/backtrace openmp" --test tests_driver_impl -- getrf det
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu RSTSR_DEV=1 RUST_BACKTRACE=1 \
  cargo test -p rstsr-openblas --features "rstsr/backtrace openmp" --test tests_linalg_func -- det inv slogdet
```

## B3 — compositions and the stack walk (`rstsr-linalg-traits`, shared)

Files:

- `src/stack.rs`: new `pub fn solve_stack_enumerated_f(..)` — the body of
  today's `solve_stack_f` with the kernel typed `FnMut(usize, TensorView<Ix2>,
  TensorMut<Ix2>) -> Result<()>` (batch index first); `solve_stack_f` becomes
  a thin wrapper passing `|_, a, b| kernel(a, b)`. No call site changes.
- `src/ref_impl_blas.rs`, new `#region lu`:
  - `pub fn lu_ipiv_to_perm(ipiv: &[blas_int], m: usize) -> Vec<usize>` — the
    0-based `?GETRF` pivots applied in reverse order to `0..m`;
  - `pub fn lu_perm_to_ipiv(p: &[usize]) -> Vec<usize>` — its inverse (the
    forward transposition factorization of the permutation);
  - `pub fn ref_impl_lu_factor_f<'a, T, B>(a: TensorReference<'a, T, B, Ix2>)
    -> Result<(TensorMutable<'a, T, B, Ix2>, Vec<usize>, usize)>` — one
    `GETRF` under the device BLAS thread policy, pivots as `usize`, raw
    `info`;
  - `pub fn ref_impl_lu_into_f<T, B>(a, p: &mut [usize], l: TensorViewMut<Ix2>,
    u: TensorViewMut<Ix2>, permute_l: bool) -> Result<()>` — factor `a` in
    place, then write into the pre-zeroed outputs: `p` by
    `lu_ipiv_to_perm`; `U` rows `0..K` from the diagonal rightwards; `L`
    columns `0..K` below the diagonal plus the unit diagonal — or, with
    `permute_l`, `PL` rows gathered `PL[i, ..min(P[i], K)] = LU[P[i], ..]`
    and the unit entry at `P[i] < K`. No intermediate tensors;
  - `pub fn lu_swap_rows<T, B>(b: TensorViewMut<'_, T, B, Ix2>, ipiv:
    &[usize], forward: bool)` — `?LASWP` in place through the raw storage;
  - `pub fn ref_impl_lu_solve_f<T, B>(l: TensorView<'_, T, B, Ix2>, u:
    TensorView<'_, T, B, Ix2>, ipiv: &[usize], b: TensorReference<'_, T, B,
    Ix2>, trans: FlagTrans) -> Result<TensorMutable<'_, T, B, Ix2>>` —
    `N`: forward swaps, `TRSM(L, Lower, Unit, N)`, `TRSM(U, Upper, NonUnit,
    N)`; `T` / `C`: `TRSM(U, Upper, T|C)`, `TRSM(L, Lower, Unit, T|C)`,
    reverse swaps; `CN` → `rstsr_invalid!`; `N == 0 || K == 0` → nothing.
    Each `TRSM` result is `clone_to_mut`-ed before the next step so the data
    is always in the caller-visible buffer when the swaps run; the returned
    `TensorMutable` keeps the variant of the converted `b`.
- `src/stack.rs` unit tests: `lu_ipiv_to_perm` / `lu_perm_to_ipiv` round trip
  belongs beside the compositions (unit test module of `ref_impl_blas.rs`,
  device-free).

Verify:

```bash
cargo check -p rstsr-linalg-traits
cargo test  -p rstsr-linalg-traits --lib
```

## B4 — entries (`rstsr-linalg-traits/src/blas_impl/`, compiled per device)

New files, each with a `//!` module doc (title one sentence; details in named
paragraphs; eig / lstsq wording for input destruction, shapes, errors):

- `lu.rs`: `blas_impl_lu_f(a: TensorReference<IxD>, permute_l: bool) ->
  Result<(Tensor<usize, IxD>, Tensor<T, IxD>, Tensor<T, IxD>)>` — `StackLayout`
  walk, one `overwritable_convert_with_order` for the stack, outputs
  `zeros_f` in the device order (`p: batch ++ [M]`, `l: batch ++ [M, K]`,
  `u: batch ++ [K, N]`), per slice `ref_impl_lu_into_f`. Overloads:
  `(Tr, bool)` core (4 input kinds, `D: DimAPI + DimSmallerOneAPI`) →
  `LuResult<Option<Tensor<usize, _, D::SmallerOne>>, Tensor<T, _, D>,
  Tensor<T, _, D>>`; `(Tr)` delegates with `false` (4 rows).
- `lu_factor.rs`: `blas_impl_lu_factor_f(a: TensorReference<IxD>) ->
  Result<(TensorMutable<IxD>, Tensor<usize, IxD>)>` — the `inv` shape: whole
  stack converted once, loop under `with_blas_num_threads`,
  `ref_impl_lu_factor_f` + `clone_to_mut` (write-back), pivots copied into
  `zeros_f(batch ++ [K])`. Overloads: read-only kinds →
  `LuFactorResult<Tensor<T, _, D>, Tensor<usize, _, D::SmallerOne>>`;
  owned / `TensorMut` kinds → `LuFactorResult<Tr, Tensor<usize, _, D::SmallerOne>>`.
- `lu_solve.rs`:
  - `blas_impl_lu_solve_f(lu: TensorView<IxD>, piv: TensorView<usize, IxD>,
    b: TensorReference<'b, IxD>, trans) -> Result<TensorMutable<'b, IxD>>`;
  - `blas_impl_lu_solve_plu_f(lu: &LuResult<Tensor<usize, IxD>, Tensor<IxD>,
    Tensor<IxD>>, b, trans)` (the impl unwraps `p`, `None` → `InvalidValue`);
  - one private core: devices equal, `solve_plan(l.shape(), b.shape(), ..)`
  (square factor, `b` vector or matrix, broadcast batch), `u.shape() ==
    l.shape()`, `piv.shape() == batch_l ++ [N]`, then
    `solve_stack_enumerated_f` with `l` as `a`; the kernel fetches the `u`
    and `piv` slices of batch index `i` (broadcast to the plan) and calls
    `ref_impl_lu_solve_f`; the `LuResult` form converts `P → ipiv` per slice.
  - Overloads (`solve_general` kind classes; read-only = `&TensorAny` /
    `TensorView`, consumed = `TensorMut` / `Tensor`):
    - packed core `(TrLu, TrPiv, TrB, FlagTrans)`: table A (all read-only →
      `Tensor<T, _, IxD>`), B (`lu` / `piv` read-only, `b` consumed → `TrB`),
      C (all consumed → `TrB`), 8 rows each;
    - packed delegation `(TrLu, TrPiv, TrB)` → `NoTrans` (24 rows);
    - `(&LuResult<..>, TrB, FlagTrans)` core: `b` read-only → fresh, `b`
      consumed → `TrB` (4 rows); `(&LuResult<..>, TrB)` delegates (4 rows).
- Per device ×5 (`openblas`, `mkl`, `blis`, `aocl`, `kml`):
  `src/linalg_auto_impl/{lu,lu_factor,lu_solve}.rs` symlinks
  (`../../../../rstsr-linalg-traits/src/blas_impl/<fn>.rs`) and three
  `pub mod` lines in `src/linalg_auto_impl/mod.rs` (after `lstsq`).

Verify:

```bash
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu RSTSR_DEV=1 RUST_BACKTRACE=1 \
  cargo check -p rstsr-openblas --all-targets --features "rstsr/backtrace openmp"
cargo check -p rstsr-blis  --all-targets --features "rstsr/backtrace"
cargo check -p rstsr-aocl  --all-targets --features "rstsr/backtrace"
cargo check -p rstsr-kml   --all-targets --features "rstsr/backtrace"
cargo check -p rstsr-mkl   --all-targets --features "rstsr/backtrace"
cargo check -p rstsr-openblas --all-targets --features "rstsr/backtrace openmp ilp64"
```

## B5 — tests and validators

- Driver `blas_tests_impl/driver_impl/f64/getrf.rs` (new; symlink ×5, `mod
  getrf;` in each `driver_impl/f64/mod.rs`; the fingerprint test stays in
  `getrf_getri.rs`): a `make_case` helper that keeps padded / strided
  layouts (`into_owned_keep_layout`, strides asserted), sizes `m, n ∈ {1, 2,
  3, 5, 6}`; `test_dgetrf_layout` (f-contig, f-padded, c-contig, c-padded,
  fully non-contiguous × `view()` / `view_mut()`: `L[P] U ≈ A`, pivots equal
  the f-contig run, 0-based, in place exactly when the layout is in order —
  read from `c_prefer() || f_prefer()`), padding integrity,
  `test_dgetrf_bridge_parity` (destroyed `a` of a row-major in-place call is
  bit-identical to the packed factor of a direct column-major call),
  `test_dgetrf_singular` (`info = i`, valid factors, both orders, in-place
  write-back), `test_dgetrf_zero_size`.
- Linalg `blas_tests_impl/linalg_func/func_f64/{lu,lu_factor,lu_solve}.rs`
  and `func_c64/{lu,lu_factor,lu_solve}.rs` (symlink ×5 each, `mod` lines):
  - `lu.rs`: `test_lu` (fingerprints: f64 square 1024 `a`, tall 1024×512,
    wide 512×1024, `permute_l` on the square; c64 square, tall),
    `test_lu_relations` (small tall / square / wide: `A ≈ L[P] U`, `A ≈ PL
    U`, unit diagonal, triangularity, `P` a permutation, rows `K..M` of `L`
    hold multipliers), `test_lu_singular`, `test_lu_batch` (per-slot parity
    with 2-D calls, both device orders), `test_lu_layouts` (f / c / strided
    inputs), `test_lu_inplace` (owned / `TensorMut` consumed and destroyed;
    off-order mutable input untouched; padded), `test_lu_overloads` (8
    forms on a 4×4 block, tuple conversions incl. the two panicking ones
    with `should_panic`), `test_lu_zero_size`, `test_lu_ndim_error`;
  - `lu_factor.rs`: `test_lu_factor` (fingerprints f64 square + tall, c64
    square), `test_lu_factor_relations` (packed `lu` equals `lu`'s `L` / `U`,
    `piv` round-trips to `P`, singular, in-place returns the same buffer),
    `test_lu_factor_layouts`, `test_lu_factor_batch`, `test_lu_factor_zero_size`;
  - `lu_solve.rs`: `test_lu_solve` (fingerprints f64 1024 `K = 1`, `K = 3`,
    one `T`; c64 `K = 1`, one `C`), `test_lu_solve_relations` (`A x ≈ b`,
    `Aᵀ x ≈ b`, `Aᴴ x ≈ b`, equality with `solve_general`, both factor
    forms agree), `test_lu_solve_batch` (factor vs `b` broadcast, vector
    form), `test_lu_solve_inplace` (`b.view_mut()` returns the same buffer;
    layouts; both device orders), `test_lu_solve_overloads` (the argument
    tables on small blocks), `test_lu_solve_errors` (permuted factor,
    non-square, `piv` length, `b` rows, in-place batch growth),
    `test_lu_solve_singular`, `test_lu_solve_zero_size`.
- Validators `blas_tests_impl/linalg_func/func_validation_{f64,c64}.py`:
  `# ### lu`, `# ### lu_factor`, `# ### lu_solve` sections (SciPy 1.18.1,
  `check_finite=False`); literals pasted into the Rust tests at `1e-8`.
- `tests_zero_size` bodies (OpenBLAS-only harness): `lu`, `lu_factor`,
  `lu_solve` corner shapes if the harness has per-function files — check
  `crates-device/rstsr-openblas/tests/zero_size/` first and follow its
  pattern.

Run (env per `blas-lapacke-mkl-test-env` memory):

```bash
# OpenBLAS lapack (system 0.3.32)
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu RSTSR_DEV=1 RUST_BACKTRACE=1 \
  cargo test -p rstsr-openblas --features "rstsr/backtrace openmp" --test tests_driver_impl -- --test-threads=1
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu RSTSR_DEV=1 RUST_BACKTRACE=1 \
  cargo test -p rstsr-openblas --features "rstsr/backtrace openmp" --test tests_linalg_func -- --test-threads=1
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu RSTSR_DEV=1 RUST_BACKTRACE=1 \
  cargo test -p rstsr-openblas --features "rstsr/backtrace openmp" --test tests_zero_size -- --test-threads=1
# validators
conda run -n torch python rstsr-blas-traits/src/blas_tests_impl/linalg_func/func_validation_f64.py
conda run -n torch python rstsr-blas-traits/src/blas_tests_impl/linalg_func/func_validation_c64.py
# OpenBLAS lapacke (conda 0.3.34 symlink dir on LD_LIBRARY_PATH), MKL 2026.1
# (MKLROOT, MKL_THREADING_LAYER=GNU) and AOCL: LU subsets of both suites
```

Timing per the test-size policy (`--report-time` on the LU subset).

## B6 — docs, lint, close-out

- `rstsr/src/docs/cheatsheet_linalg.md`: one "DeviceFaer backend notes"
  bullet for the three entries (BLAS-only; `P` as indices; `permute_l` as a
  `bool`; `lu_factor` packed form and in-place rule; `lu_solve` factor forms
  and `FlagTrans`); `rstsr/src/docs/array_api_standard_extension.md`: note
  that `lu` is beyond the array API.
- Lint / docs:

```bash
cargo clippy -p rstsr-blas-traits   --all-targets -- -D warnings
cargo clippy -p rstsr-linalg-traits --all-targets -- -D warnings
cargo clippy -p rstsr-openblas --all-targets --features "rstsr/backtrace openmp" -- -D warnings
cargo clippy -p rstsr-mkl      --all-targets --features "rstsr/backtrace" -- -D warnings
cargo clippy -p rstsr          --all-targets --features "rstsr/backtrace row_major linalg" -- -D warnings
cargo fmt --check
cargo doc --no-deps -p rstsr-linalg-traits
cargo doc --no-deps -p rstsr --features linalg
cargo test -p rstsr --features linalg --test cheatsheet_linalg
```

- Task directory: `README.md` (what was built, results table, gotchas);
  `.claude/memory/rstsr-blas-lu.md` + index line; commit here.

## Risk register

- `TRSM` on the packed buffer relies on `diag = 'U'` leaving the diagonal
  unread — core BLAS semantics, verified on OpenBLAS first, then MKL/AOCL.
- `TensorMutable::ToBeCloned` holds the valid data in its *view*, not its
  buffer: no `clone_to_mut` before a builder has filled the buffer.
- `duplicate_item` tables: cells in brackets, locals never named like items,
  the `'a` lifetime only in rows whose `Tr` carries one.
- Fingerprints across LAPACK builds: `P` exact, `L` / `U` / `PL` within
  6e-11 on two OpenBLAS builds; 1e-8 house tolerance.
- Test time: one 1024² `GETRF` ≈ 70 ms single-thread; the LU linalg subset
  stays in seconds. Debug builds: extraction loops are per row / column.
- `/tmp` quota: build with `TMPDIR=$HOME/.cache/tmp-cargo`.

# linalg convergence checklist — rstsr_faer.api × array-api-tests (2026-10-10)

The last remaining block of the array-API convergence. With `fft` declared out
of scope for this version, the *only* thing standing between rstsr_faer.api and
the NumPy band is the **`linalg` extension** (the array-API `xp.linalg`
namespace, its 23 member functions, and the four top-level linear-algebra
functions). This directory is the work plan for it; the frozen status read-out
and the other (now-drained) buckets live in
`../2026-10-07-arrayapi-convergence/`; register ids (G-nnn) refer to
`../2026-10-04-rstsr-faer-py/GAP-REGISTER.md`.

**Current: `qr` (R6) + `eig`/`eigvals` (R8) — the last members — `1332 / 17 / 33`**
(suite `array-api-tests@6c0b59f`, API `2025.12`, module `rstsr_faer.api`):

- Main (`06aa7e5`, post-`#143` `cross`): **`1321 / 28 / 33`** of 1382.
- `261011/shim-trace-solve` (`#144`, trace shim + solve promotion):
  **`1325 / 24 / 33`** of 1382 — **+4 passed / −4 failed, 0 regressions**
  (flipped: `test_has_names[linalg-trace]`, `test_extension_func_signature[linalg.trace]`,
  `test_linalg.py::test_{trace,solve}`).
- `261012/matrix-rank` (R4): **`1327 / 22 / 33`** of 1382 — **+2 passed / −2
  failed, 0 regressions** (flipped:
  `test_has_names[linalg-matrix_rank]`, `test_extension_func_signature[linalg.matrix_rank]`;
  `test_matrix_rank` stays red, blocked by the shared faer SVD `NoConvergence`).
- `eig`/`eigvals` (`#146`), then `qr` (`#147`): **`1327 / 22 / 33` → `1332 / 17 /
  33`** — **+5 passed / −5 failed, 0 regressions** (flipped: `test_eig`,
  `test_eigvals`, `test_qr`, `test_has_names[linalg-qr]`,
  `test_extension_func_signature[linalg.qr]`).

The `linalg` extension is now down to **3 graded failures** on top of the 14
out-of-scope `fft` name nodes (§0): the value tests `matrix_rank`, `pinv`,
`svd`, all gated by the one faer SVD `NoConvergence` gap — **no `linalg` member
is absent** from the surface. The 33 skips are 28 `fft` + the 5
backend-independent `test_remainder`. Merged chain (all on main): shim init
`#133` → `1287 / 62 / 33`; R3 `tensordot` `#135` → `1294 / 55 / 33`; R2
mixed-dtype `#137` → `1300 / 49 / 33`; R4 `outer` `#138` → `1303 / 46 / 33`; R1
batched `#140` → `1309 / 40 / 33`; `matrix_power` `#141` → `1312 / 37 / 33`; R5
norms `#142` → `1318 / 31 / 33`; R4 `cross` `#143` → `1321 / 28 / 33`; trace shim
+ solve `#144` → `1325 / 24 / 33`; R4 `matrix_rank` → `1327 / 22 / 33`;
`eig`/`eigvals` `#146` → `1329 / 20 / 33`; R6 `qr` `#147` → `1332 / 17 / 33`.

## Progress (2026-10-12): `qr` (R6) + `eig`/`eigvals` (R8) — the last members

`eig`/`eigvals` merged as `#146`; `qr` on branch `261012/qr` (off main `6786abe`,
i.e. post-`#146`; PR RESTGroup/rstsr#147). All three are **faer device only**
(like the rest of the extension), with shim bindings.

- `qr`: new trait `QrAPI` in `rstsr-linalg-traits/traits_def.rs` (three
  `duplicate_item` tables + `prelude.rs`), kernel in the new `faer_impl/qr.rs`.
  The `mode` argument is a **`QrMode` enum** (`#[non_exhaustive]`,
  `#[default] Reduced`) with `TryFrom<&str>`/`FromStr` — the array-API `mode`
  string (`'reduced'`/`'complete'`) is parsed at the boundary; `QrAPI` is
  implemented for the bare tensor (= reduced), `(tensor, QrMode)`, and
  `(tensor, &str)`. `QrResult<Q, R>` converts to/from `(Q, R)`. faer's
  `MatRef::qr()` is infallible: reduced = `compute_thin_Q()` + `R()`, complete =
  `compute_Q()` + the thin `R` zero-padded to `(M, N)`; batched via
  `map_stack_slices2`; a zero-size axis short-circuits (complete-mode
  empty-column `Q` is the identity completion, matching NumPy).
- `eig`/`eigvals`: `EigAPI`/`EigvalsAPI` + `faer_impl/{eig,eigvals}.rs`; general
  (non-Hermitian) eigendecomposition, **complex** output for every input, via
  faer's `eigen()`/`eigenvalues()`. `eig`'s eigenvector columns are rescaled to
  unit 2-norm (faer leaves them unnormalized).
- Shim: `linalg.qr(x, /, *, mode='reduced')` → `(Q, R)` namedtuple; `linalg.eig`
  → `(eigenvalues, eigenvectors)`; `linalg.eigvals` → the spectrum. The "qr is
  absent" prose in `linalg.rs`/`api.py` was dropped — **no `linalg` member is
  absent now**.
- Fingerprint tests (`test_faer_func/func_{f64,c64}.rs`): `test_qr` sign/phase-
  canonicalizes (`abs(Q)` over the first `K` columns — complete-mode extras are
  an arbitrary orthonormal completion — plus `R` with a real-positive diagonal),
  and adds `test_qr_empty`; `test_eig`/`test_eigvals` sort the unordered spectrum.
- Conformance: **`1327 / 22 / 33` → `1332 / 17 / 33`** (+5 / −5, 0 regressions).
  Faer fingerprint suite 64 → 68.
- Review round on `qr` (`/code-review high`, applied): the zero-size guard gave
  complete-mode `Q` as **zeros** where NumPy expects the identity completion —
  array-API `test_qr` never checks `Q`'s orthonormality, so conformance had
  missed it; fixed by writing `eye` per batch slice. Also dropped the dead
  `From<()>`/`From<Option<()>> for QrMode` and added `test_qr_empty` + a
  wide+complete case. Re-verified: conformance unchanged **`1332 / 17 / 33`**,
  faer tests 68/68, fmt/clippy clean.

## Progress (2026-10-12): `matrix_rank` — R4, third member

Branch `261012/matrix-rank` (off main `06aa7e5`, i.e. post-`#143` `cross`).

- New trait `MatrixRankAPI` in `rstsr-linalg-traits/src/traits_def.rs` (added to
  the three `duplicate_item` tables + `prelude.rs`), kernel in the new
  `faer_impl/matrix_rank.rs`. `Out = Tensor<usize, DeviceFaer, IxD>` (the rstsr
  index dtype), batch-shaped `shape(x)[:-2]`.
- **`MatrixRankRtol<TR, B> { None, Float(TR), Array(Tensor<TR, B, IxD>) }`** is
  the entry's argument type (no wrapper struct — one field would add nothing).
  The enum exists because a **scalar has no device but an array does**: `Array`
  carries its device and the kernel `same_device`-checks it against the input
  before ranking, while `Float` needs no check. Overloads are `From` impls —
  `Tensor` → `Array`; concrete `f32`/`f64` → `Float`; `()` and `Option<()>` →
  `None`. The scalar impl is **per real type, not a generic `From<TR>`**: a
  generic one overlaps `From<()>` / `From<Option<()>>` at `TR = ()` (E0119 —
  the same coherence wall the `SortArgs`/`SearchSortedArgs` concrete impls
  dodge). The `(Tr, A)` impl takes `A: TryInto<MatrixRankRtol<TR, DeviceFaer>>`.
- Algorithm: `s = svdvals(x)`; `smax = max(s, k_ax, keepdims)`; `tol = smax *
  rtol` (`smax * max(M,N)*eps` when `None`); `rank = sum(s > tol)` over the
  singular-value axis, composed from core reductions (`greater_f`, the bool-sum
  `TensorSumBoolAPI`). **No Hermitian fast path** (NumPy has one; the array-API
  has no `hermitian` argument). Empty matrix axes / batches short-circuit to
  zero rank, mirroring NumPy.
- Shim: `rstsr-faer-py` binds `linalg.matrix_rank(x, /, *, rtol=None)` —
  `src/linalg.rs` (`linalg_matrix_rank`, rtol taken as `Option<&Bound<PyAny>>`:
  a Python float → `Float` in `x`'s real dtype, a `NativeArray` → `Array` via
  `T::Real`'s tensor ref + clone) + `python/rstsr_faer/api.py` (only an array is
  cast to `x`'s real dtype; a scalar stays a Python float — binding marshalling,
  *not* a promotion pre-cast, so not a `DTYPE-CASTS.md` entry). The rank is
  lifted to int64 by the shared `idx_lift`.
- New tests: `rstsr-linalg-traits/tests/test_faer_func/func_f64.rs`
  (`test_matrix_rank{,_numpy,_empty,_nd,_colmajor,_tolerance,_rtol_device_mismatch}`),
  the NumPy cases from `numpy/linalg/tests/test_linalg.py::TestMatrixRank` /
  `test_empty_matrix_rank` translated (1-D/scalar/Hermitian inputs are out of the
  array-API contract and skipped). The mismatch test pins the device check: a
  `ColMajor` rtol tensor against a `RowMajor` input errors.
- Conformance: **`1325 / 24 / 33` → `1327 / 22 / 33`** (+2 passed / −2 failed, 0
  regressions; flipped `test_has_names[linalg-matrix_rank]`,
  `test_extension_func_signature[linalg.matrix_rank]`).
- Review round (post-commit `b065a49`, on-screen code review, 6 of 7 findings
  applied): `rtol` is now **validated up front** — the device `same_device` and
  the batch `broadcast_shapes_f` run *before* the empty-input short-circuit and
  the SVD, so an invalid tolerance errors on an empty input instead of being
  silently ignored (matches the `vector_norm` precedent); a new
  `test_matrix_rank_empty_validates_rtol` pins it. The `Array` order behaviour
  is now **documented** rather than fixed (`traits_def.rs` + the kernel module
  doc): the matrix axes are the **leading** logical axes under a row-major device
  and the **trailing** axes under a column-major one, and the tolerance
  broadcast follows that same device order — so row-major matches NumPy's
  trailing alignment while column-major mirrors it. Shim docstring corrected
  (the kernel broadcasts; the Python layer only casts the dtype); the Python
  wrapper raises `TypeError` for a non-broadcast dtype (array `rtol`) instead of
  `KeyError`, and rejects a non-real-numeric `rtol` (str/complex) instead of
  coercing via `float()`. The `Default` impl is kept (hand-written:
  `#[derive(Default)]` would add spurious `TR: Default, B: Default` bounds), now
  exercised by a test. **Declined (intentional):** no
  `From<f64> for MatrixRankRtol<f32, B>` — a scalar must be the input's own real
  type, so an `f32` caller writes `1e-8_f32` (documented).
  Re-verified: conformance **`1327 / 22 / 33`**, non-passed node set
  byte-identical, linalg faer tests **59/59**, clippy/fmt/doc clean.
- **`test_matrix_rank` stays red**, but for a new reason: the first drawn
  example is a rank-1 `f32` 3×5 matrix of large equal values
  (`5.83e18`), and faer's SVD raises `NoConvergence` on it — the same gap that
  already blocks `test_svd` / `test_pinv` (`svdvals` fails identically; `f64`
  converges). Not a `matrix_rank` defect and not shim-fixable: a rust-side faer
  SVD issue (large-magnitude `f32`). It now gates three value tests.

## Progress (2026-10-11): norms (`matrix_norm`, `vector_norm`) — R5

Branch `261011/worktree` (off main `bbc6a19`, i.e. post-`#141`; PR
RESTGroup/rstsr#142). Two new faer entries plus their shim bindings; **faer
device only** (no BLAS column), as the plan predicted (the `nuc` order needs the
singular values).

- New traits `MatrixNormAPI` / `VectorNormAPI<Inp>` in
  `rstsr-linalg-traits/src/traits_def.rs` (added to the three `duplicate_item`
  tables + `prelude.rs`), kernel in the new `faer_impl/norm.rs`. Receivers are
  the usual `(Tr, A)` where `A: TryInto<MatrixNormArgs|VectorNormArgs>`, plus a
  bare-`Tr` impl for the no-argument form.
- **Shared `NormOrd { None, Fro, Nuc, P(f64), Inf, NegInf }`** — `None` is the
  default and resolves *per operation* (`'fro'` for a matrix, `2` for a vector),
  so the same value names different orders. Each op rejects the other's orders
  with `InvalidValue` (a matrix only takes `±1`/`±2` numerically; a vector
  rejects `'fro'`/`'nuc'`). Overloads come from `TryFrom` impls on the args
  structs (`()`, `Option<()>`, `bool`, `NormOrd`, `&str`, `isize`, `f64`, the
  2/3-tuples) — concrete impls, no blanket `TryFrom<O>` (coherence), per the
  `AxesIndex` orphan-rule note.
- **Composition, not new kernels:** `matrix_norm` = core reductions
  (`'fro'` → the `l2_norm` reduction; `1/-1` → abs → sum → max/min over rows;
  `inf/-inf` → abs → sum over columns → max/min; `2/-2/'nuc'` → the **batched
  `svdvals` entry** → max/min/sum). `vector_norm` reduces `|x|` (l2 for `p = 2`,
  a plain sum for `p = 1`, a nonzero count for `p = 0`, the general
  `(Σ|x|^p)^(1/p)` otherwise). The matrix axes follow the **device order**
  (`batch_and_matrix_shape`), like `matrix_power`. Output is dynamic `IxD` (rank
  depends on `keepdims`), with the real element type pinned to a generic `TR`
  (`T: ComplexField<Real = TR> + ExtNum<AbsOut = TR>`) so the faer SVD and the
  core reductions agree on one scalar.
- **Empty input yields `0` under every order** — a deliberate divergence from
  NumPy, which is inconsistent here (it raises for the matrix `-2` and `-inf`
  and returns `inf` for the vector `-1`). The cause is not a norm decision: the
  faer `max`/`min` reductions reject *any* zero-size input, so both entries
  early-return `zeros_f` whenever the input has any zero dim (this also covers
  the empty **stack** dims the suite's `finite_matrices` draws). Stated in the
  module doc and locked by the empty-input test.
- Shim: `linalg_matrix_norm` / `linalg_vector_norm` `#[pyfunction]`s +
  `_LinalgNamespace` entries + the `api.py` wrappers. `ord` is marshalled by
  `py_to_ord` (`None`/`'fro'`/`'nuc'`/int/float); `axis` reuses the Python
  layer's existing `_norm_axes` (`None` / int / tuple → `Option<Vec<isize>>`),
  so the Rust entry takes `Option<Vec<isize>>` rather than a `PyAny`.
- **Measured `1312 / 37 / 33` → `1318 / 31 / 33`** (+6: the two name nodes, the
  two signature nodes, `test_matrix_norm`, `test_vector_norm`), 0 regressions.
  Gates: faer tests **50/50** (NumPy's own `test_matrix_2x2` goldens for
  `[[1,3],[5,7]]` → `fro` √84, `nuc` 10, `1`/`inf` 10/12, `±2`
  9.123105625617661 / 0.8768943743823404, plus n-dim / col-major / empty cases);
  fmt / clippy clean.
- **Review round** (`/code-review high`, fixes in `2c2e8f9`): validate the
  vector-norm axes **before** the empty-input short-circuit (otherwise a bad
  axis was swallowed on an empty input); drop the redundant `abs` copy in the
  `'fro'` and vector `p = 2` paths (`l2_norm` already takes the magnitude); add
  `FromStr` on `NormOrd` and both args structs; route the shim axis through
  `_norm_axes`; extend the empty-input test to the min-based orders. Conformance
  re-run after the fixes: `1318 / 31 / 33` unchanged.

## Progress (2026-10-11): `matrix_power` — R4 deferred member landed

Merged as `#141` (`bbc6a19`), branch `261011/matrix-power` (off `2e93deb`, i.e.
post-`#140`). This is exactly where the deferral note below predicted it would
land: **not** rstsr-core (the `n < 0` case needs the matrix inverse, which lives
in `rstsr-linalg-traits`).

- New faer-only `MatrixPowerAPI<Inp>` (`(Tr, isize)` receiver) + the
  `faer_impl/matrix_power.rs` kernel, composing the **batched `inv_f`** (R1) with
  the tensor-tier matmul (`matmul_f`). NumPy-parity: `n = 0` → identity, `n < 0`
  → invert once then the positive power, and the general case by repeated
  squaring LSB→MSB.
- **Measured `1309 / 40 / 33` → `1312 / 37 / 33`** (3 nodes:
  `test_has_names[linalg-matrix_power]`,
  `test_extension_func_signature[linalg.matrix_power]`,
  `test_linalg.py::test_matrix_power`), 0 regressions. `matrix_power` is now
  exposed by the shim (all three of its nodes pass), so it is no longer an
  absent member.

## Progress (2026-10-11): batched (stacked) linalg — R1, the dominant lever

Branch `261011/batched-linalg` (off main `39f83d0`, i.e. post-`#138` faer 0.24;
**merged as `#140`**), 5 commits. Every faer linalg entry that asserted `ndim == 2` now accepts a
matrix **or a stack of matrices** (`(..., M, N)` / `(..., M, M)`) — the array-API
contract the value tests exercise immediately. **Measured: `1303 / 46 / 33` →
`1308 / 41 / 33`** (batch wave: `test_{cholesky,eigh,eigvalsh,inv,svdvals}`)
**→ `1309 / 40 / 33`** (det wave: `test_det`); node set test-for-test identical,
0 regressions. Provenance: built from `rstsr` `261011/batched-linalg` @ `95e089f`
(clean tree), `maturin build --release` (opt-level 3) →
`/tmp/wheels/rstsr_faer_py-0.10.0a1-cp310-abi3-manylinux_2_39_x86_64.whl`, sha256
`cc46566c…b34610`, built 2026-10-11 05:20; run `FRESH=1 NO_EXPLAIN=1 CHUNKED=1`,
19/19 chunks.

- **Members now n-dim:** `cholesky`, `inv`, `eigh` (standard), `eigvalsh`,
  `svd`, `svdvals`, `pinv`, `det`; the master (`#133`) `slogdet` was rerouted
  through the same walk. `solve_general` / `solve_triangular` are n-dim too, and
  generalized `eigh` gained batch **broadcast** (below). Only the
  **BLAS-device** solve stays 2-D (out of scope for this branch).
- **Shared helper — `rstsr-linalg-traits/src/faer_impl/batch.rs`** (`pub(crate)`):
  `map_stack_slices{,2,3}` / `map_stack_slices2x2` (1–3 outputs),
  `map_stack_scalars` / `map_stack_scalars2` (scalar-per-matrix), and
  `map_stack_slices_inplace` (the in-place solve walk), plus `stack_shape`,
  `batch_and_matrix_shape`, `with_parallel`. The layout splits into
  (batch, inner) by the device order (matrix axes **last** under `RowMajor`,
  **first** under `ColMajor`); each output is pre-allocated as a fresh
  contiguous `zeros_f`; then the input batch offsets and every output's batch
  offsets are walked **in lockstep** (`IterLayout::new(.., TensorIterOrder::C)`)
  and each entry's private 2-D `*_ix2` kernel is called on a per-slice view.
  Design mirrors the `cholesky` entry (`2586c37`) — *pre-allocated outputs,
  lockstep offset walk* — not the rejected recursive `linalg_util` collect.
- **In-place solve (no clone).** An owned / contiguous-mutable `b` is solved
  **in place**; an immutable view `b` allocates one output; a non-contiguous
  mutable `b` falls back to a contiguous scratch buffer with write-back. This is
  `overwritable_convert` (`Mut` / `Owned` / `ToBeCloned`) at the entry, with
  `map_stack_slices_inplace` walking a's and b's batch offsets in lockstep and
  widening a 1-D `(M,)` slice to `(M, 1)` **per slice** (order-agnostic) for the
  vector case. **RSS-verified no clone** (gitignored scratch `rstsr/tmp/memtest/`):
  a 4-GiB owned `b` (524288×32×32) solved in place → **RSS +0 MiB** for both
  `solve_general` and `solve_triangular`; an explicit `b.clone()` control →
  +4096 MiB; the allocating (view-`b`) path → +4096 MiB.
- **Broadcast (following numpy / `array-api-tests` `solve_args`)**:
  `solve_general`, `solve_triangular` and generalized `eigh` broadcast their
  **batch** dims. Key rule from the reference: **`b` is a vector iff `ndim ==
  1`**, otherwise it is a `(..., M, K)` matrix stack; only batch dims broadcast.
  `SolvePlan` / `solve_plan` compute the broadcast batch shape and the
  order-aware `a_shape` / `out_shape`; the target broadcasts through the fallible
  `to_broadcast_f`.
- **API change (approved):** the allocating `solve_general` /
  `solve_triangular` impls now return `Tensor<_, IxD>` (was the fixed `DB`),
  generalized `eigh` returns `EighResult<_, IxD, IxD>` (+ `Num`), and `det`
  returns `Tensor<T, DeviceFaer, IxD>` (0-dim for a 2-D input). The faer-py shim
  is unchanged except `linalg_det` (returns the tensor directly now) — the
  wrappers route through `rt::linalg::*_f`, so **no new shim code was needed for
  R1**.
- **`linalg_util` removed:** the square split moved into `faer_impl::batch`;
  `slogdet` dropped its recursive collect-into-Vecs walk and the `reverse_axes`
  col-major hack.
- Gates: faer tests **35/35**; fmt / clippy clean. **ColMajor gotcha** (for n-D
  linalg tests): `asarray((v, shape, &dev))` follows the device order, but
  `asarray((v, layout, &dev))` uses the layout verbatim — so `[B,M,K].c()` on an
  F device pins C-contiguous and lands the batch axis where an F device expects
  a matrix axis; use the bare-shape form (or pin `RowMajor`) in n-D tests.
- **Not shim-visible:** `solve` still fails on **G-009** (mixed-dtype on the
  solve path), `pinv` / `svd` still fail on faer SVD `NoConvergence` (hits 2-D
  too — not a batching defect), and generalized `eigh` is not an array-API
  member. So the wave's graded flips are exactly the 6 above; the rest of R1 is
  rust-side capability (the R1 unblock is real, but `solve`'s node is gated by
  G-009, not by batching).

## Progress (2026-10-10): shim-side init

Branch `261010/faer-py-linalg-init` (off rstsr `main` `9db98e8`; since merged
as `#133`, `d7056ba`). Purely shim-side — thin pass-throughs, no algorithm and
no batching in the wrapper:

- New `crates-interop/rstsr-faer-py/src/linalg.rs` (13 `#[pyfunction]`s) +
  registration in `src/lib.rs`; the `rstsr` dep gains the `linalg` feature.
- `python/rstsr_faer/api.py`: `_LinalgNamespace` exposing **13 of the 23**
  members (`cholesky, det, diagonal, eigh, eigvalsh, inv, matmul,
  matrix_transpose, pinv, solve, svd, svdvals, vecdot`), the top-level
  `matmul`/`matrix_transpose`/`vecdot`, and `Array.__matmul__`.
- Measured: **1249 / 51 / 82 → 1287 / 62 / 33**. Exposing the partial
  namespace flipped the 49 `linalg` skips → run (16 pass / 33 fail); **0
  regressions** outside the linalg/fft name family.
- **Dominant finding (new — the pre-init plan missed it):** rstsr's faer
  linalg entries are **2-D only** (`rstsr_assert_eq!(ndim, 2)`), so with a
  pass-through wrapper every *stacked* input (`ndim > 2`) raises. The
  array-API value tests are hypothesis-generated and hit stacks on the first
  example, so of the 13 exposed members only `diagonal` and `matrix_transpose`
  pass their value tests today. **Batched (stacked) linalg is a rust-side
  gap** — it is now the single biggest lever. *(Resolved 2026-10-11 by R1 —
  see the batched-linalg Progress section.)*
- Second finding: `matmul`/`vecdot` are same-dtype only (mixed pairs need the
  G-009 promotion table), so their value tests fail on the first mixed pair.

## Progress (2026-10-10): tensordot (R3)

Landed rust-side + shim-side (branch `261010/tensordot` off `d7056ba`, PR
RESTGroup/rstsr#135). `rt::tensordot` is a new tensor-tier entry
(`tensor/linalg/tensordot.rs`) with a device op on `DeviceCpuSerial` and the
rayon auto-impl, and a view-only GEMM fast path; the shim exposes it as a
top-level `tensordot` and a `_LinalgNamespace` member.

- **4 nodes flipped to pass:** `test_has_names[linear_algebra-tensordot]`,
  `test_has_names[linalg-tensordot]`, `test_func_signature[tensordot]`,
  `test_extension_func_signature[linalg.tensordot]`.
- `test_tensordot` and `test_linalg_tensordot` now **run** (were name-absent)
  and fail **only** on R2/G-009 — the mixed-dtype promotion gap shared with
  `matmul`/`vecdot`, not a tensordot defect.
- Measured: **`1287 / 62 / 33` → `1294 / 55 / 33`**, 0 regressions.

## Progress (2026-10-10): mixed-dtype linalg (R2)

Branch `261010/ext-matmul` (off rstsr `main` `2f5cf1d`, i.e. post-`#135`; PR
RESTGroup/rstsr#137, CI 13/13 green). A rust-side `ext_linalg` family plus the
shim rewiring that consumes it — this is the G-009 promotion gap, closed for
the whole product family.

- New `rt::ext_matmul` / `rt::ext_vecdot` / `rt::ext_tensordot`: the
  promotion-compatible siblings of the same-dtype entries; each operand pair is
  computed in its promoted common dtype (`DTypePromoteAPI`, the NumPy rule).
  Implemented on `DeviceCpuSerial` and `DeviceFaer`; `ext_vecdot` conjugates its
  first operand per the array-API contract.
- Promotion is **fused into the kernel inner loop** (`promote_pair`), not done
  by promoting into intermediate buffers, so the mixed-dtype path allocates no
  temporaries (the cost is a thin per-dtype copy of the kernel).
  `ext_tensordot` reuses the view-only GEMM fast path.
- Shim: `linalg_matmul` / `linalg_vecdot` / `linalg_tensordot` route both
  same- and mixed-dtype pairs through the `ext_` entries, so the G-009 fallback
  is gone for them. The binary dispatch gained the missing `i8 × i16` arm, and
  `conj` now preserves integer dtypes (the array-API rule) — the latter was
  **required**: `test_vecdot`'s reference computes `xp.conj(int)`.
- **Measured: `1294 / 55 / 33` → `1296 / 53 / 33` (matmul) → `1300 / 49 / 33`
  (vecdot + tensordot)**, 0 regressions. Flipped nodes:
  `test_matmul`, `test_linalg_matmul`, `test_vecdot`, `test_linalg_vecdot`,
  `test_tensordot`, `test_linalg_tensordot`.
- Note the branch carries **no batching** (that is R1, a separate PR); `matmul`
  and `vecdot` are already n-D capable, so their value tests pass without it.
  The two branches are not yet combined.

## Progress (2026-10-10): `outer` / `ext_outer` (R4, first member)

Branch `261010/linalg-outer` (off rstsr `main` `f03c3ca`, i.e. post-`#137`; not
yet pushed). The first R4 member, implemented as a real device-op family rather
than the plan's broadcast-mul shortcut (the owner's call).

- New same-dtype `rt::outer` + promotion `rt::ext_outer`, mirroring the
  `matmul`/`vecdot` pair: device traits `DeviceOuterAPI` / `DeviceExtOuterAPI`,
  a naive serial kernel + rayon twin (promotion fused via `promote_pair` in the
  ext twin), and per-device impls. `outer` returns a rank-2 `Tensor<_, _, Ix2>`
  (the shim narrows it to its `IxD` handles).
- **Device scope** mirrors the family: same-dtype `outer` is on *all* devices —
  `DeviceCpuSerial` explicitly, every `DeviceRayonAutoImpl` (faer + the five
  BLAS crates) through the shared `feature_rayon/auto_impl/outer.rs`, reached by
  one symlink per crate (the same mechanism as `vecdot`/`tensordot`).
  `ext_outer` is `DeviceCpuSerial` + `DeviceFaer` only, like the rest of `ext_`.
- **Contract: 1-D only**, per array-API `linalg.outer` (the suite draws only 1-D
  and comments "outer does not work on stacks"); a non-1-D operand is
  `InvalidValue` from `outer_f` (panic otherwise). No conjugation (unlike
  `vecdot`); the result is `(N, M)` in the operands' common dtype.
- **`outer` does grade binary promotion**: `test_outer` draws its operands from
  `two_mutual_arrays(dtypes=dh.real_dtypes, …)` and asserts
  `assert_dtype(..., expected=result_type(x1.dtype, x2.dtype))`, so the shim
  routes `linalg.outer` through `ext_outer`, never the same-dtype entry.
- Shim: `linalg_outer` (+ `_LinalgNamespace.outer`, the native import, and a
  positional-only `def outer(x1, x2, /)` wrapper); `outer` dropped from the
  module's "absent members" prose.
- **Measured: `1300 / 49 / 33` → `1303 / 46 / 33`**, 0 regressions. Flipped:
  `test_linalg.py::test_outer`, `test_has_names[linalg-outer]`,
  `test_extension_func_signature[linalg.outer]`.
- Gates: 272 doctests (+2), 179 lib (+10), 591 entry-row (+5); `cargo check
  --all-targets` clean on `rstsr-openblas` (the BLAS test tree symlinks
  `core_func`, so this covers it) and `--all-targets`-free checks on the other
  four BLAS crates; fmt/clippy/rustdoc clean; lib tests re-run clean under
  `col_major`.
- **Review round (`/code-review high`, fixed in `aa9e345`).** One real defect:
  the fresh result's arrangement came from `TensorIterOrder::default()`, which
  is `K`, so the match always took `_ => [n, m].c()` and every result was
  C-contiguous even on a ColMajor device — against the sibling entries and
  `docs/order_semantics.md`. `outer` now allocates in the input device's
  `default_order()` (the owner's rule: *the output follows the input's default
  order*), with a layout test pinning it against `tensordot(a, b, 0)`. Also:
  kernel-level parallel-branch tests for both new rayon twins (all earlier cases
  sat below `PARALLEL_SWITCH`, so those paths never ran); the parity test
  rewritten as a true transfer of NumPy's `linalg/tests/test_linalg.py::TestOuter`
  (the only value source — a **class-body assert**, so `sync_numpy.py` gained a
  class-body index and the header cites `path::Class` with an empty method); the
  `np.outer` (flattens) vs `np.linalg.outer` (1-D, raises) distinction recorded
  in `numpy_differences.md`; a stale `slogdet` dropped from the shim's
  absent-names doc. Conformance re-run after the fix: **`1303 / 46 / 33`
  unchanged**.
- **Follow-up: `rt::outer_from` / `outer_from_f`** (owner's request; **no**
  `ext_outer_from`). The output-providing variant of `outer`, mirroring
  `vecdot_from`: `c: impl TensorViewMutAPI<Dim = Ix2>` (the device op is pinned
  to `Layout<Ix2>`, so the rank is fixed rather than a generic `DC`), a broadcast
  gate (`cannot write into broadcasted tensor`), a shape check against
  `(a.size, b.size)`, and the same `transmute::<Raw<TC>, Raw<MaybeUninit<TC>>>`
  to hand the initialized buffer to the uninit-taking device op. Unlike
  `matmul_from` there is no `alpha`/`beta` — `outer` has no accumulate form, so
  `c` is fully overwritten. Method form lives in an `Ix2`-specialized
  `TensorAny` impl block (`c.outer_from(&a, &b)`); exported from the prelude;
  listed in `outer`'s "Variants of this function". Lib tests: overwrite
  semantics + method equivalence, wrong-shape error, broadcast-layout error.

## Progress (2026-10-10): `trace` (R4, second member)

Same branch. `matrix_power` is out of scope for rstsr-core (see the R4 item; deferred to `rstsr::linalg`); this is the other
derivable op.

- **Pure composition, no new device op** (`tensor/linalg/trace.rs`):
  `trace` = `diagonal` + `sum_axes`, exactly the plan's suggestion. The one
  wrinkle is that `diagonal`'s *default* axes are the first two while array-API
  `linalg.trace` uses the **last two**, so a small private helper supplies the
  default axes when the caller leaves them unset; `diagonal` always appends the
  diagonal as the *last* axis, so `sum_axes(-1)` is then correct for any axes
  the caller does pass.
- **The default axes follow the device default order** (owner's rule): the last
  two under `RowMajor` (the array-API contract the suite checks over stacks) and
  the first two under `ColMajor` (NumPy's). Explicit axes are order-independent.
  This is `diagonal`'s own convention generalized — rstsr already ties
  "leading vs trailing" to the default order elsewhere (flatten order,
  `l2_norm` axes). Recorded in `numpy_differences.md`.
- **Signature follows the reduction pair**: `trace(x, offset)` returns
  `Tensor<B::TOut, B, IxD>` (sibling of `sum_with_args`), and
  `trace_with_dtype::<TOut>(x, offset)` is the array-API `dtype=` keyword
  (sibling of `sum_with_dtype`, which already folds in `TOut` with no cast copy).
  The `offset` parameter is the existing local `DiagonalArgs`, so `()`, `None`,
  an integer, and `(offset, axis1, axis2)` all work (per the "argument groups
  travel as one tuple" convention) — no new `*Args` type, so no orphan-rule
  trouble.
- **Bounds stay on the device, not the element type**: the only new requirement
  is `B: OpSumAPI<T, D::SmallerOne>` (or `OpSumDtypeAPI<T, TOut, D::SmallerOne>`)
  plus `D: DimAPI + DimSmallerOneAPI`, i.e. the reduction bound re-indexed at
  the reduced rank. No new element-type bound.
- **Zero-size diagonals work**: an out-of-range offset yields an empty diagonal,
  and `reduce_axes_cpu_serial`'s `la.size() == 0` branch fills every output cell
  with the sum identity, so the trace is `0` rather than an error.
- **NumPy provenance is `_core/tests/test_numeric.py::TestNonarrayArgs::test_trace`**
  (L349, hash `eeea38bf1267`): a 3×2 input where NumPy's first-two and our
  row-major last-two defaults coincide. The stacked-input divergence is recorded
  in `numpy_differences.md`; `sync_numpy.py` gained the SURFACE row.
- Gates: 273 doctests (+1), 187 lib (+6), 596 entry-row (+5, incl. the
  `doc_draft` twin `doc_trace`); fmt/clippy/rustdoc clean; `col_major` lib run
  green; `cargo check --all-targets` clean on `rstsr-openblas` and
  `rstsr-faer-py`.
- **Not done here**: the shim binding (`linalg.trace` / `linalg.matrix_power`)
  — this wave is rstsr-core only.
- **Review round (`/code-review high`).** One real bug, and it is *not* in
  `trace`: `Layout::diagonal` had no duplicate-axis guard, so `axis1 == axis2`
  dropped only one axis while still appending the diagonal — `[2,2].c()
  .diagonal(None, Some(0), Some(0))` yields shape `[2,2]` stride `[1,4]` over a
  4-element buffer. Reproduced: `rt::sum_axes(&a.into_dyn().diagonal((0,0,0)),
  -1)` panics with `range end index 6 out of range for slice of length 4`; the
  `into_diagonal_f` SAFETY comment ("addresses a subset") is violated, so this
  is a soundness hole in **rstsr-common**, reachable from `rt::diagonal` before
  `trace` existed (trace just made the `(offset, axis1, axis2)` form more
  visible). Fixed with a `rstsr_assert!(axis1 != axis2, InvalidValue, ...)`
  (NumPy raises `ValueError` here) plus a layout test. Doc policy §4 also
  requires the (b) **Row/Column Major Notice** div for order-dependent
  functions, which `trace` now carries (div + one example per order +
  `order_semantics` link), and the `trace` row was missing from
  `array_api_standard.md`'s Linear Algebra table. Both shim docstrings said
  `trace` was a *rust-side gap*, now false.
- **CI round**: `unittests-col-major` failed on the new parity test — its 3-d
  input was `rt::arange(8).into_shape([2, 2, 2])`, and reshape fills in the
  device default order, so under `ColMajor` it holds different values than
  NumPy's row-major `np.arange(8).reshape(2,2,2)` (which is where the expected
  values came from). Rebuilt from `tensor_from_nested!` literals. **Generalizable:
  `arange().into_shape()` is not order-independent in content; a local
  `--features col_major` lib run is not a substitute for the CI job's full suite.**
  Branch pushed, **PR RESTGroup/rstsr#138** open (14/14 checks green, no
  auto-merge).
- **Not acted on**: the reviewer's `ext_outer`-driver duplication and the
  `outer_from` `Ix2` pin (deliberate — the device op takes `Layout<Ix2>`; a
  dim-generic form would need `into_dim` in the entry). Its CRLF claim about
  `numpy_coverage.csv` was **false** (`git show` diff is 1 insertion; 281 → 282
  CRLF lines, zero bare-LF).

## 0. The red map this checklist must clear

**Current failures (17) = `fft` 14 + `linalg` 3** — every remaining node, from
the `qr` run (see **Progress (2026-10-12): `qr` + `eig`/`eigvals`**). The `fft`
14 are declared **out of scope** for this version; the `linalg` 3 are all value
tests gated by one shared faer SVD gap.

`fft` 14 (out of scope) — `test_has_names[fft-*]`:
`fft, fftfreq, fftn, fftshift, hfft, ifft, ifftn, ifftshift, ihfft, irfft,
irfftn, rfft, rfftfreq, rfftn`.

`linalg` 3 — every remaining `linalg` node, with its blocker:

| node | status | blocker |
|---|---|---|
| `test_matrix_rank`, `test_pinv`, `test_svd` | exposed | faer SVD `NoConvergence` on large-magnitude `f32` (hits 2-D too — **not** a batching defect; `matrix_rank` inherits it via `svdvals`, `pinv` via `svd`) |

**No `linalg` member is absent** from `xp.linalg` — `qr` (the last one) landed
in R6. The residual `matrix_rank` / `pinv` / `svd` failures are all the one faer
SVD `NoConvergence` gap, not missing surface.

> **Resolved since this table was written**: `cross` (`#143`), the `trace` shim
> hookup and `solve` promotion (`#144`), `matrix_rank` (R4), `eig`/`eigvals`
> (`#146`), and `qr` (`#147`). `solve` closed with **G-009**; `matrix_rank`'s
> `has_names` + signature nodes closed, but its value test is gated by the
> shared SVD gap above.

**Skips (33)** — 28 `fft` (out of scope) + 5 `test_remainder` (upstream
`@pytest.mark.skip("flaky")`). No `linalg` skips remain.

> **Historical — the init-time red map (62 = `linalg` 48 + `fft` 14)**, kept for
> provenance. R1 has since batched every faer entry, R4 landed `outer` +
> `trace` (rust), the matrix_power and norm waves have landed, so the two
> "exposed, value-failing" / "absent" rows below are long superseded — the
> current map is the table above.

| group | nodes | what it checked |
|---|---|---|
| `test_has_names[linalg-*]` | 10 | the members then absent from `xp.linalg` |
| `test_has_names[linear_algebra-tensordot]` | 1 | top-level `tensordot` absent |
| `test_signatures.py::test_extension_func_signature[linalg.*]` | 10 | the same members |
| `test_signatures.py::test_func_signature[tensordot]` | 1 | top-level `tensordot` signature |
| `test_linalg.py` (name absent) | 12 | `test_{cross,eig,eigvals,matrix_norm,matrix_power,matrix_rank,outer,qr,slogdet,trace,vector_norm,tensordot}` |
| `test_linalg.py` (exposed, value-failing) | 13 | `test_{cholesky,det,eigh,eigvalsh,inv,linalg_matmul,linalg_vecdot,matmul,pinv,solve,svd,svdvals,vecdot}` |

At init, the 10 absent members were `cross, matrix_norm, matrix_power,
matrix_rank, outer, qr, slogdet, tensordot, trace, vector_norm`; all ten have
since landed.

## 1. Rust-side inventory — what exists today

Rust linalg lives in **`rstsr-linalg-traits`** (`LinalgAPI` family, one
`func`/`func_f` per op) with per-device impls under `faer_impl/` and
`blas_impl/`; the tensor-level linear algebra (`matmul`, `vecdot`,
`matrix_transpose`, `diagonal`) is in **`rstsr-core/src/tensor/`**. DeviceFaer
(the shim's device) implements the faer column.

`✓` present · `◐` partial · `✗` absent. "shim" = exposed by the init branch.

| array-API name | rstsr entry today | rust | shim |
|---|---|---|---|
| `matmul` | `rt::matmul` (tensor tier), `rt::ext_matmul` (promotion) | ✓ same-dtype; ✓ promote (R2) | ✓ |
| `matrix_transpose` | `rt::matrix_transpose` | ✓ | ✓ |
| `vecdot` | `rt::vecdot` (tensor tier), `rt::ext_vecdot` (promotion) | ✓ same-dtype; ✓ promote (R2) | ✓ |
| `diagonal` | `rt::diagonal` (indexing) | ✓ | ✓ |
| `cholesky` | `CholeskyAPI` (faer) | ✓ n-dim (R1) | ✓ |
| `det` | `DetAPI` (faer) | ✓ n-dim (R1) | ✓ |
| `eigh` | `EighAPI` (faer) | ✓ n-dim (R1) | ✓ |
| `eigvalsh` | `EigvalshAPI` (faer) | ✓ n-dim (R1) | ✓ |
| `inv` | `InvAPI` (faer) | ✓ n-dim (R1) | ✓ |
| `pinv` | `PinvAPI` (faer) | ✓ n-dim (R1) | ✓ |
| `solve` | `SolveGeneralAPI` (faer) | ✓ n-dim (R1) | ✓ |
| `svd` | `SVDAPI` (faer) | ✓ n-dim (R1) | ✓ |
| `svdvals` | `SVDvalsAPI` (faer) | ✓ n-dim (R1) | ✓ |
| `tensordot` | `rt::tensordot` (tensor tier), `rt::ext_tensordot` (promotion) | ✓ same-dtype (R3); ✓ promote (R2) | ✓ |
| `outer` | `rt::outer` (tensor tier), `rt::ext_outer` (promotion) | ✓ same-dtype (R4, all devices); ✓ promote (R4, cpu_serial + faer) | ✓ |
| `cross` | `rt::cross` / `rt::ext_cross` | ✓ core (R4, `#143`) | ✓ |
| `trace` | `rt::trace` (+ `trace_with_dtype`) — `diagonal` + `sum_axes` | ✓ core (R4) + shim (`#144`) | ✓ |
| `matrix_power` | `MatrixPowerAPI` (faer) | ✓ faer (R4, `#141`) | ✓ |
| `matrix_rank` | `MatrixRankAPI` (faer) — count of `svdvals > rtol * s_max` | ✓ faer (R4, this wave) | ✓ (gated by the faer SVD gap) |
| `vector_norm` | `VectorNormAPI` (faer) | ✓ faer (R5) | ✓ |
| `matrix_norm` | `MatrixNormAPI` (faer) | ✓ faer (R5) | ✓ |
| `qr` | `QrAPI` (faer) | ✓ faer (R6, `#147`) | ✓ |
| `slogdet` | `SLogDetAPI` (faer + blas) | ✓ faer landed in #133 | ✓ |
| `eig` | `EigAPI` (faer) | ✓ faer (R8, `#146`) | ✓ |
| `eigvals` | `EigvalsAPI` (faer) | ✓ faer (R8, `#146`) | ✓ |

Also relevant: `solve_symmetric` exists on blas but not on faer (**G-005**) —
array-API `solve` maps to `solve_general`, so this is tangential.

## 2. Rust-side work (register + request; never fix agent-side)

Ordered by failures-per-item. **Every rust-side work item (R1–R8) has landed** —
`cross`/`trace`/`matrix_rank` (R4), `qr` (R6), `eig`/`eigvals` (R8) included; see
**Progress**. The only residual `linalg` failures are the three value tests gated
by the shared faer SVD `NoConvergence` gap (`svd`/`pinv`/`matrix_rank`).

- [x] **R1. Batched (stacked) linalg** — **landed** (branch
      `261011/batched-linalg`, off main `39f83d0`). Every faer linalg entry now
      accepts `(..., M, N)` / `(..., M, M)`: `cholesky, det, eigh, eigvalsh,
      inv, pinv, solve, svd, svdvals` (+ the master `slogdet`). Implemented as a
      shared `faer_impl::batch` walk over per-matrix 2-D kernels (pre-allocated
      outputs, lockstep offsets), **not** an `iter_axes` stack loop; the solving
      entries keep an owned `b` in place (RSS-verified), and `solve` /
      generalized `eigh` also broadcast batch dims. **Measured `1303 / 46 / 33`
      → `1309 / 40 / 33`** (+6: `test_{cholesky,det,eigh,eigvalsh,inv,svdvals}`),
      0 regressions — see **Progress**. Residual `solve` / `pinv` / `svd` value
      nodes are gated by G-009 (mixed dtype) and the faer SVD `NoConvergence`
      gap, not by batching.
- [x] **R2. mixed-dtype `matmul`/`vecdot`/`tensordot`** (G-009) — **landed**
      (PR RESTGroup/rstsr#137, branch `261010/ext-matmul`): the `ext_linalg`
      family computes each operand pair in its promoted common dtype
      (`DTypePromoteAPI`), fused into the kernel inner loop, so `u8 @ u16`
      works. Gates `test_matmul` / `test_linalg_matmul` / `test_vecdot` /
      `test_linalg_vecdot` / `test_tensordot` / `test_linalg_tensordot`; all six
      now pass — see **Progress**.
- [x] **R3. `tensordot`** (tensor tier, `tensor/linalg/`) — **landed** (PR
      RESTGroup/rstsr#135): `rt::tensordot` (`axes` int or pair-of-lists) with
      serial + rayon device kernels and a view-only GEMM fast path; the shim
      exposes it top-level and on `_LinalgNamespace`. Its 4 `has_names` +
      signature nodes pass; the 2 `test_linalg` value nodes now fail only on
      R2/G-009.
- [x] **R4. small derivable ops** — **complete**: `outer` (#138), `trace`
      (rust-side, R4; shim hookup `#144`), `matrix_power` (#141), `cross`
      (#143), `matrix_rank` (this wave) all landed. The plan's shortcut
      (`outer` = broadcast mul + reshape) was **not** taken: the owner chose a
      real device-op family (`DeviceOuterAPI` / `DeviceExtOuterAPI` + kernels),
      matching `matmul`/`vecdot`; `trace` landed as planned (`diagonal` + sum);
      `matrix_rank` = count of `svdvals > rtol * s_max` (see **Progress
      (2026-10-12)**). All four `has_names` + signature nodes closed; `test_cross`
      / `test_trace` / `test_solve` closed, `test_matrix_rank` gated by the faer
      SVD gap.
- [x] **`matrix_power`** — **landed as `#141`** (branch `261011/matrix-power`,
      off `2e93deb`). As the earlier deferral note predicted, it belongs on the
      **`rstsr-linalg-traits`** side, not in rstsr-core: array-API draws `n`
      from −10..10 and requires invertible input for `n < 0`, i.e. the matrix
      inverse (rstsr-core's `rt::inv` is the *element-wise* reciprocal; the real
      `inv` lives in `rstsr-linalg-traits`, which depends on rstsr-core). Faer
      device only; composes the batched `inv_f` with the tensor-tier matmul;
      `n = 0` → identity, `n < 0` → invert once, general case by repeated
      squaring LSB→MSB — see **Progress**.
- [x] **R5. norms general `ord`** — **landed** (branch `261011/worktree`, PR
      RESTGroup/rstsr#142): `MatrixNormAPI` / `VectorNormAPI` on faer, sharing
      one `NormOrd` (`None`/`fro`/`nuc`/`inf`/`-inf`/numeric), composing the
      core reductions + the batched `svdvals` — see **Progress**.
- [x] **R6. `qr`** (**G-004**) — **landed** (`#147`): `QrAPI` + `faer_impl/qr.rs`
      (faer device only, via `MatRef::qr()`), a `QrMode` enum for the `mode`
      string, and the shim `linalg.qr`. The array-API grades only
      `reduced`/`complete` (no `'r'`/`'raw'`) — see **Progress**.
- [x] **R7. `slogdet` on faer** (**G-005**) — **done in #133** (`d7056ba`,
      "array-API linalg extension for rstsr_faer.api with n-dimensional
      slogdet"), already on `main` and so on this branch.
      `rstsr-linalg-traits/src/faer_impl/slogdet.rs` provides
      `impl SLogDetAPI<DeviceFaer> for Tr` (LU with partial pivoting in
      `faer_slogdet_ix2`) plus the n-dim batch walk `faer_impl_slogdet_f`;
      the shim exposes it as `linalg_slogdet` → `_LinalgNamespace.slogdet`,
      and `test_slogdet` passes (3/3). **The slogdet half of G-005 is
      resolved**; the other half (`solve_symmetric` on blas but not faer)
      still stands — `faer_impl/` has no `solve_symmetric.rs` — but array-API
      `solve` maps to `solve_general`, so it stays tangential.
- [x] **R8. `eig` / `eigvals`** — **landed** (`#146`): `EigAPI`/`EigvalsAPI` +
      `faer_impl/{eig,eigvals}.rs` (faer device only), **complex** output for
      every input. Not in the 2025.12 `linalg.__all__`, but `test_linalg.py`'s
      `test_eig`/`test_eigvals` (gated on `min_version("2025.12")` = our pin) now
      pass — see **Progress**.

## 3. Shim-side work (`crates-interop/rstsr-faer-py`) — done on the branch

Wrapper-only. S1–S4 landed on `261010/faer-py-linalg-init`, merged as `#133`
(`d7056ba`):

- [x] **S1. `xp.linalg` namespace** — a `_LinalgNamespace` object exposing the
      available names (13 at init, 14 with R3's `tensordot`); missing members
      stay absent (rendered as gaps, never stubbed). `api.py` §linalg.
- [x] **S2. top-level members** — `matmul`, `matrix_transpose`, `vecdot`, and
      `tensordot` (R3, landed — also added to `_LinalgNamespace`) on `xp`.
- [x] **S3. `__matmul__`** on `Array` (the `@` operator), delegating to
      `matmul`.
- [x] **S4. return structures** — `eigh` → namedtuple `(eigenvalues,
      eigenvectors)`, `svd` → `(U, S, Vh)`, `det`/`svdvals`/`eigvalsh` →
      arrays, `pinv` → array (rank dropped).
- [x] **S5. dtype/stack semantics** — **done**: promotion is wired
      (`matmul`/`vecdot`/`tensordot` route through the `ext_` entries — R2), so
      mixed dtypes work; batching landed rust-side (R1, no shim change).
      `eig`/`eigvals` complex output landed with R8 (`#146`).

## 4. Ordering

The namespace is now exposed (partial), so the skip→run flip has already
happened. Going forward:

1. ~~**R1 (batched linalg) first**~~ — **done** (2026-10-11, branch
   `261011/batched-linalg`; +6 nodes, no shim code). The remaining exposed
   value failures are the two genuine gaps, not the stack handling.
2. ~~Then the remaining members as they land~~ — **done**: R4's `cross` /
   `matrix_rank` (and the `trace` shim hookup), R6 (`qr`), R8 (`eig` /
   `eigvals`) have all landed. **Every `linalg` member is now present**; the only
   residual `linalg` failures are the three SVD-gated value tests.
3. Re-run per §5 after each landed item.

If instead the namespace should be **held** until the surface is more complete
(see D1), drop the `linalg` object and keep only the top-level trio + `@` —
that reverts the 49 skips and the `linalg` name nodes to their pre-init state.

## 5. Verification

Reuse the rstsr-faer-py-tests harness (skill `rstsr-faer-py-tests`):

```bash
cd <scratch>   # not inside a git checkout
FRESH=1 NO_EXPLAIN=1 CHUNKED=1 MODULE=rstsr_faer.api \
  /path/to/skills/rstsr-faer-py-tests/scripts/run.sh
```

- Build release at any opt-level (never the dev profile); record the wheel
  provenance as in `../2026-10-07-arrayapi-convergence/README.md`.
- **Recorded today:** `1249 / 51 / 82` (pre-init) → **`1287 / 62 / 33`**
  (init branch). The `+38` passed / `-49` skipped is the 49 activated `linalg`
  tests (16 pass / 33 fail); expect **0 regressions** in the node set outside
  the linalg/fft name family.
- **Recorded (tensordot / R3):** `1287 / 62 / 33` → **`1294 / 55 / 33`**. The 4
  `tensordot` has_names/signature nodes flip to pass; the 2 `test_linalg` value
  nodes now run and fail only on G-009; 0 regressions.
- **Recorded (batched linalg / R1):** `1303 / 46 / 33` → `1308 / 41 / 33`
  (batch wave) → **`1309 / 40 / 33`** (det wave), 0 regressions, node set
  **test-for-test identical**, on `261011/batched-linalg` (off main `39f83d0`).
  Flipped: `test_{cholesky,det,eigh,eigvalsh,inv,svdvals}`. The pre-R1 plan
  predicted ≈ +9 (`{cholesky,det,eigh,eigvalsh,inv,pinv,solve,svd,svdvals}`);
  it landed **+6** because `solve` is gated by G-009 (mixed dtype, not
  batching), and `pinv`/`svd` by the faer SVD `NoConvergence` gap. Confirm
  per-item counts against §0 as work lands (the suite is hypothesis-based;
  expect ±2 wobble).
- **Recorded (mixed-dtype linalg / R2):** `1294 / 55 / 33` → `1296 / 53 / 33`
  (matmul) → **`1300 / 49 / 33`** (vecdot + tensordot), 0 regressions, on the
  `261010/ext-matmul` branch (post-`#135` base, so **no R1 batching** — the two
  branches are not yet combined). Flipped: `test_matmul`,
  `test_linalg_matmul`, `test_vecdot`, `test_linalg_vecdot`, `test_tensordot`,
  `test_linalg_tensordot`. The residual `linalg` value failures are `solve`
  (G-009 on the solve path) and `pinv`/`svd` (faer SVD `NoConvergence`), plus
  the not-yet-implemented members.
- **Recorded (outer / R4):** `1300 / 49 / 33` → **`1303 / 46 / 33`**, 0
  regressions, on `261010/linalg-outer` (off the post-`#137` main `f03c3ca`;
  merged as `#138`). Flipped: `test_outer`, `test_has_names[linalg-outer]`,
  `test_extension_func_signature[linalg.outer]`. The residual `linalg` value
  failures are unchanged (`solve` = G-009; `pinv`/`svd` = faer SVD
  `NoConvergence`) plus the members still absent.
- **Recorded (matrix_power / `#141`):** `1309 / 40 / 33` → **`1312 / 37 / 33`**,
  0 regressions (flipped: `test_has_names[linalg-matrix_power]`,
  `test_extension_func_signature[linalg.matrix_power]`,
  `test_linalg.py::test_matrix_power`). `#140` (batched linalg) is the
  `39f83d0` → `1309 / 40 / 33` step above, now merged.
- **Recorded (norms / R5):** `1312 / 37 / 33` → **`1318 / 31 / 33`**, 0
  regressions, on `261011/worktree` (off main `bbc6a19`, PR
  RESTGroup/rstsr#142). Flipped: the `matrix_norm`/`vector_norm` name +
  signature nodes and `test_matrix_norm` / `test_vector_norm`. Gates: faer tests
  50/50; fmt/clippy clean. Re-run after the `/code-review high` fixes: unchanged.
  The residual `linalg` failures are exactly the 17 in §0.
- **Recorded (cross / R4, `#143`):** `1318 / 31 / 33` → **`1321 / 28 / 33`**, 0
  regressions. Flipped: `test_has_names[linalg-cross]`,
  `test_extension_func_signature[linalg.cross]`, `test_cross`.
- **Recorded (trace shim + solve promotion, `#144`):** `1321 / 28 / 33` →
  **`1325 / 24 / 33`**, 0 regressions. Flipped:
  `test_has_names[linalg-trace]`, `test_extension_func_signature[linalg.trace]`,
  `test_trace`, and `test_solve` (mixed-dtype promotion in the shim closes
  G-009's value node).
- **Recorded (matrix_rank / R4, this wave):** `1325 / 24 / 33` →
  **`1327 / 22 / 33`**, 0 regressions, on `261012/matrix-rank` (off main
  `06aa7e5`). Flipped: `test_has_names[linalg-matrix_rank]`,
  `test_extension_func_signature[linalg.matrix_rank]`. `test_matrix_rank` stays
  red on the faer SVD `NoConvergence` gap (see the Progress note). The residual
  `linalg` failures are exactly the 8 in §0. Re-run after the `MatrixRankRtol`
  enum refactor: **`1327 / 22 / 33` again, the non-passed node set byte-identical**
  (linalg-traits faer tests 58/58).
- **Recorded (`eig`/`eigvals` `#146`, `qr` `#147`):** `1327 / 22 / 33` →
  **`1332 / 17 / 33`**, 0 regressions. Flipped: `test_eig`, `test_eigvals`,
  `test_qr`, `test_has_names[linalg-qr]`,
  `test_extension_func_signature[linalg.qr]`. The residual `linalg` failures are
  exactly the 3 in §0 (all faer SVD `NoConvergence`); `linalg`-traits faer tests
  68/68; fmt/clippy clean. Re-run after the `qr` review fix: unchanged
  **`1332 / 17 / 33`**.

- [x] **D1. Implement vs. scope-out** — resolved: implement. The shim-side
      init is on the branch; the remaining `linalg` work is rust-side.
- [ ] **D2. Namespace now vs. gated** — the init exposes a *partial*
      `xp.linalg`, which converts the 49 `linalg` skips into graded tests
      (16 pass / 33 fail today). Keep it exposed and drive the failures down
      via R1–R8, or gate `xp.linalg` until the surface is more complete? (The
      owner's call; the branch makes either a one-line change.)
- [x] **D3. Batching location** — resolved: **inside the faer device ops** (R1).
      A shared `faer_impl::batch` walk over the existing per-matrix 2-D kernels
      (pre-allocated outputs, lockstep offsets), not a tensor-tier loop and not
      the earlier recursive `linalg_util`. `Out` became `IxD` for the entries
      whose output rank was previously fixed (`solve*`, generalized `eigh`,
      `det`); the blas column is untouched (its `solve` stays 2-D, out of scope).
- [ ] **D4. `ord` coverage for norms** — full array-API `ord` table vs. the
      subset rstsr can express; determines R5's scope.
- [ ] **D5. Where the symmetric/triangular solvers surface** — array-API
      `solve` is general; decide whether `solve_symmetric`/`solve_triangular`
      stay rust-internal or get shim names (not required by the suite).

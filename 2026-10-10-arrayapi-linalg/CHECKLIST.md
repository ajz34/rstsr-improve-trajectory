# linalg convergence checklist — rstsr_faer.api × array-api-tests (2026-10-10)

The last remaining block of the array-API convergence. With `fft` declared out
of scope for this version, the *only* thing standing between rstsr_faer.api and
the NumPy band is the **`linalg` extension** (the array-API `xp.linalg`
namespace, its 23 member functions, and the four top-level linear-algebra
functions). This directory is the work plan for it; the frozen status read-out
and the other (now-drained) buckets live in
`../2026-10-07-arrayapi-convergence/`; register ids (G-nnn) refer to
`../2026-10-04-rstsr-faer-py/GAP-REGISTER.md`.

**Current: R2 merged; `outer`/`ext_outer` open on a new branch** (suite
`array-api-tests@6c0b59f`, API `2025.12`, module `rstsr_faer.api`):

- `261010/ext-matmul` (R2, mixed-dtype linalg, PR RESTGroup/rstsr#137) — now
  **merged** to main as `f03c3ca`: **`1300 / 49 / 33`** of 1382.
- `261010/linalg-outer` (R4, `outer`/`ext_outer`), off the post-`#137` main
  `f03c3ca`: **`1303 / 46 / 33`** of 1382.

The shim-side init is merged (`#133`, `d7056ba`); it exposed 13 of the 23
members and flipped the 49 `linalg` skips into graded tests. The remaining
failures are `linalg` plus 14 out-of-scope `fft` nodes; the 33 skips are 28
`fft` + the 5 backend-independent `test_remainder`. `tensordot` (R3) landed
after the init — see **Progress**; `1287 / 62 / 33` → `1294 / 55 / 33` (R3);
R2 → `1300 / 49 / 33`; R4 (`outer`) → `1303 / 46 / 33`.

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
  gap** — it is now the single biggest lever.
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

Same branch. `matrix_power` was dropped (see the R4 item); this is the other
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
- **Not acted on**: the reviewer's `ext_outer`-driver duplication and the
  `outer_from` `Ix2` pin (deliberate — the device op takes `Layout<Ix2>`; a
  dim-generic form would need `into_dim` in the entry). Its CRLF claim about
  `numpy_coverage.csv` was **false** (`git show` diff is 1 insertion; 281 → 282
  CRLF lines, zero bare-LF).

## 0. The red map this checklist must clear

**Init-time failures (62) = `linalg` 48 + `fft` 14** — the groups below. R3 has
since removed the two `tensordot` has_names/signature rows and turned
`test_tensordot` / `test_linalg_tensordot` into G-009 failures, and R2 has since
flipped the six `matmul` / `vecdot` / `tensordot` value nodes to pass (see
**Progress**); re-derive the group columns on the next full run (the suite is
hypothesis-based; expect ±2 wobble).

`linalg` 48:

| group | nodes | what it checks |
|---|---|---|
| `test_has_names[linalg-*]` | 10 | the members still absent from `xp.linalg` (9 after R4) |
| `test_has_names[linear_algebra-tensordot]` | 1 | top-level `tensordot` absent |
| `test_signatures.py::test_extension_func_signature[linalg.*]` | 10 | the same members (9 after R4) |
| `test_signatures.py::test_func_signature[tensordot]` | 1 | top-level `tensordot` signature |
| `test_linalg.py` (name absent) | 12 | `test_{cross,eig,eigvals,matrix_norm,matrix_power,matrix_rank,outer,qr,slogdet,trace,vector_norm,tensordot}` (11 after R4) |
| `test_linalg.py` (exposed, value-failing) | 13 | `test_{cholesky,det,eigh,eigvalsh,inv,linalg_matmul,linalg_vecdot,matmul,pinv,solve,svd,svdvals,vecdot}` |

The 10 absent `linalg` members: `cross, matrix_norm, matrix_power,
matrix_rank, outer, qr, slogdet, tensordot, trace, vector_norm` (9 after R4,
which landed `outer`).

`test_linalg.py` **passing** today (4): `test_diagonal`,
`test_linalg_matrix_transpose`, `test_matrix_transpose`, `test_vecdot_conj`.

**Skips (33)** — 28 `fft` (out of scope) + 5 `test_remainder` (upstream
`@pytest.mark.skip("flaky")`). No `linalg` skips remain.

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
| `cholesky` | `CholeskyAPI` (faer) | ◐ 2-D only | ✓ |
| `det` | `DetAPI` (faer) | ◐ 2-D only | ✓ |
| `eigh` | `EighAPI` (faer) | ◐ 2-D only | ✓ |
| `eigvalsh` | `EigvalshAPI` (faer) | ◐ 2-D only | ✓ |
| `inv` | `InvAPI` (faer) | ◐ 2-D only | ✓ |
| `pinv` | `PinvAPI` (faer) | ◐ 2-D only | ✓ |
| `solve` | `SolveGeneralAPI` (faer) | ◐ 2-D only | ✓ |
| `svd` | `SVDAPI` (faer) | ◐ 2-D only | ✓ |
| `svdvals` | `SVDvalsAPI` (faer) | ◐ 2-D only | ✓ |
| `tensordot` | `rt::tensordot` (tensor tier), `rt::ext_tensordot` (promotion) | ✓ same-dtype (R3); ✓ promote (R2) | ✓ |
| `outer` | `rt::outer` (tensor tier), `rt::ext_outer` (promotion) | ✓ same-dtype (R4, all devices); ✓ promote (R4, cpu_serial + faer) | ✓ |
| `cross` | — (3-vector cross) | ✗ new | ✗ |
| `trace` | `rt::trace` (+ `trace_with_dtype`) — `diagonal` + `sum_axes` | ✓ core (R4); no shim yet | ✗ |
| `matrix_power` | — (repeated `matmul`) | ✗ new; **negative `n` needs an inverse ⇒ deferred** | ✗ |
| `matrix_rank` | — (via `svdvals`) | ✗ new | ✗ |
| `vector_norm` | `l2_norm` family (ord=2 only) | ◐ general `ord` missing | ✗ |
| `matrix_norm` | `l2_norm` family (ord=2 only) | ◐ general `ord` missing | ✗ |
| `qr` | — | ✗ **G-004** | ✗ |
| `slogdet` | `SLogDetAPI` (faer + blas) | ✓ faer landed in #133 | ✓ |
| `eig` | — (general, non-symmetric) | ✗ new (tested; not in 2025.12 `__all__`) | ✗ |
| `eigvals` | — (general, non-symmetric) | ✗ new (ibid.) | ✗ |

Also relevant: `solve_symmetric` exists on blas but not on faer (**G-005**) —
array-API `solve` maps to `solve_general`, so this is tangential.

## 2. Rust-side work (register + request; never fix agent-side)

Ordered by failures-per-item. **R1 is now the dominant lever** — it alone
gates 13 of the 20 graded members the shim already exposes.

- [ ] **R1. Batched (stacked) linalg** — every faer factorization today asserts
      `ndim == 2`. The array-API contract is "a matrix *or a stack of matrices*"
      (`(..., M, M)`), and the value tests exercise stacks immediately. Give
      the faer linalg entries (and the tensor-tier `solve`/`pinv` route) a
      leading-axis loop — either iterate the batch in the device op or map over
      `iter_axes` and stack. Cheapest, largest payoff: unblocks `cholesky, det,
      eigh, eigvalsh, inv, pinv, solve, svd, svdvals` at once.
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
- [~] **R4. small derivable ops** `outer` **landed**, `trace` **landed** (see
      Progress; `matrix_power` **dropped** — see below) —
      the plan's shortcut (`outer` = broadcast mul + reshape) was **not** taken:
      the owner chose a real device-op family (`DeviceOuterAPI` /
      `DeviceExtOuterAPI` + kernels), matching `matmul`/`vecdot`. `trace`
      landed as planned (`diagonal` + sum); `cross` and `matrix_rank` remain
      (`matrix_rank` = count of `svdvals > tol`). Low risk; each is a
      `test_linalg` + a `has_names` node.
- **[dropped] `matrix_power`.** array-API draws `n` from −10..10 and requires
  invertible input for `n < 0`, i.e. **the matrix inverse** — which does not
  exist in rstsr-core (`rt::inv` there is element-wise reciprocal; the real
  `inv`/`solve` live in `rstsr-linalg-traits`, which *depends on* rstsr-core).
  The owner chose to drop it rather than place it in the linalg layer. Revisit
  if a `rstsr-linalg-traits`-side entry is ever wanted (it would also need the
  `n ≥ 0` fast path: `1` → clone, `2/3` → products, `4` → `A2·A2`, `>4` →
  repeated squaring with a memo list).
- [ ] **R5. norms general `ord`** — today only the `l2_norm` family exists.
      `test_vector_norm` / `test_matrix_norm` grade every `ord` (`inf`, `-inf`,
      `0`, `1`, `2`, `-1`, `-2`, `fro`, `nuc`, …).
- [ ] **R6. `qr`** (**G-004**): neither `faer_impl` nor `blas_impl` has a QR
      trait. Needs a `QRAPI` + both device columns (`geqrf`/`orgqr`; faer
      `qr::no_pivoting`). `test_qr` grades mode (`reduced`/`complete`/`r`).
- [ ] **R7. `slogdet` on faer** (**G-005**): present on blas, absent on the
      DeviceFaer column the shim uses (port the blas routine or derive from
      `det` + sign). `test_slogdet` grades the `(sign, logabsdet)` pair.
- [ ] **R8. `eig` / `eigvals`**: general (non-symmetric) eigenproblem, complex
      output. Not in the 2025.12 `linalg.__all__`, but `test_linalg.py` carries
      `test_eig`/`test_eigvals` gated on `min_version("2025.12")` = our pin, so
      they run and fail. Lowest priority (names not required for the surface).

## 3. Shim-side work (`crates-interop/rstsr-faer-py`) — done on the branch

Wrapper-only. S1–S4 landed on `261010/faer-py-linalg-init` (uncommitted):

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
- [ ] **S5. dtype/stack semantics** — half done: promotion is wired
      (`matmul`/`vecdot`/`tensordot` route through the `ext_` entries — R2), so
      mixed dtypes work; batching is still absent (that is R1). `eig`/`eigvals`
      complex promotion is moot until R8.

## 4. Ordering

The namespace is now exposed (partial), so the skip→run flip has already
happened. Going forward:

1. **R1 (batched linalg) first** — it is the largest single move and needs no
   new shim code; the 13 exposed-but-failing value tests start passing as soon
   as the rust entries accept stacks.
2. Then R4–R8 as members land; each new member is a shim one-liner plus a
   `_LinalgNamespace` entry (add the name only when its rust entry is real, so a
   new gap never becomes a fresh failure). R2 (mixed-dtype products) and R3
   (`tensordot`) are already done.
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
- Expected after **R1** lands: `test_{cholesky,det,eigh,eigvalsh,inv,pinv,
  solve,svd,svdvals}` move to passed (≈ +9), the `matrix_norm`-family blockers
  remain. Confirm per-item counts against §0 as work lands (the suite is
  hypothesis-based; expect ±2 wobble).
- **Recorded (mixed-dtype linalg / R2):** `1294 / 55 / 33` → `1296 / 53 / 33`
  (matmul) → **`1300 / 49 / 33`** (vecdot + tensordot), 0 regressions, on the
  `261010/ext-matmul` branch (post-`#135` base, so **no R1 batching** — the two
  branches are not yet combined). Flipped: `test_matmul`,
  `test_linalg_matmul`, `test_vecdot`, `test_linalg_vecdot`, `test_tensordot`,
  `test_linalg_tensordot`. The residual `linalg` value failures are `solve`
  (G-009 on the solve path) and `pinv`/`svd` (faer SVD `NoConvergence`), plus
  the not-yet-implemented members.
- **Recorded (outer / R4):** `1300 / 49 / 33` → **`1303 / 46 / 33`**, 0
  regressions, on `261010/linalg-outer` (off the post-`#137` main `f03c3ca`).
  Flipped: `test_outer`, `test_has_names[linalg-outer]`,
  `test_extension_func_signature[linalg.outer]`. The residual `linalg` value
  failures are unchanged (`solve` = G-009; `pinv`/`svd` = faer SVD
  `NoConvergence`) plus the members still absent.

## 6. Decisions

- [x] **D1. Implement vs. scope-out** — resolved: implement. The shim-side
      init is on the branch; the remaining `linalg` work is rust-side.
- [ ] **D2. Namespace now vs. gated** — the init exposes a *partial*
      `xp.linalg`, which converts the 49 `linalg` skips into graded tests
      (16 pass / 33 fail today). Keep it exposed and drive the failures down
      via R1–R8, or gate `xp.linalg` until the surface is more complete? (The
      owner's call; the branch makes either a one-line change.)
- [ ] **D3. Batching location** — implement stacked linalg inside the faer
      device ops (R1), or as a tensor-tier batch loop over the 2-D kernels?
      Affects every linalg entry's signature and the blas column too.
- [ ] **D4. `ord` coverage for norms** — full array-API `ord` table vs. the
      subset rstsr can express; determines R5's scope.
- [ ] **D5. Where the symmetric/triangular solvers surface** — array-API
      `solve` is general; decide whether `solve_symmetric`/`solve_triangular`
      stay rust-internal or get shim names (not required by the suite).

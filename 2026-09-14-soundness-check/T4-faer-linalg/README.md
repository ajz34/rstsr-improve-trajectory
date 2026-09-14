# T4 — DeviceFaer + linalg correctness audit

Subtask of the 2026-09-14 soundness-check campaign.
Base: rstsr `acfa93e` (master), worktree `/home/a/rstsr_pack/tmp/snd-faer`.
Environment: 9950X3D (16C), cargo 1.97.1, rust-toolchain stable, faer 0.22.6,
faer-ext 0.6.0, rayon (device pool), default features (`row_major`,
`aligned_alloc`, `faer`, `faer_as_default`).

**Patches** (apply from repo root, `git apply <patch>`):

| patch | contents |
|---|---|
| `fix-01-naive-beta0-uninit-read.patch` | **[BUG-fixed]** beta=0 uninit-output reads in all 6 naive matmul/inner-dot kernels (`rstsr-native-impl/src/cpu_serial/matmul_naive.rs`, `cpu_rayon/matmul_naive.rs`) + `TC: Zero` bound propagation in `rstsr-core/src/device_cpu_serial/linalg/matmul.rs` |
| `fix-02-faer-linalg-error-paths.patch` | **[BUG-fixed]** `faer::set_global_parallelism` leak on error paths (9 fns in `rstsr-linalg-traits/src/faer_impl/`); **[BUG-fixed]** non-square / 0×0 input guards turning faer-internal panics into proper `rstsr` errors (and numpy-compatible empty results for 0×0 eigh/eigvalsh) |
| `safety-comments-and-tests.patch` | SAFETY commentary for all 25 `unsafe` tokens in `device_faer/`; new test suites (`rstsr-core/src/device_faer/matmul.rs` in-file `#[cfg(test)]`, `rstsr-linalg-traits/tests/test_faer_edge/`) |

**Running the tests**

```sh
# linalg-traits tests need generated npy resources (one-time; see
# rstsr-test-manifest/readme.md):
conda activate torch && (cd rstsr-test-manifest/resources && python gen_rand_vec.py)

cargo test -p rstsr-core --lib              # 118 pass (incl. 8 new matmul edge tests)
cargo test -p rstsr-linalg-traits --features faer   # 42 pass (24 old + 18 new edge tests)
```

## Surface map — what DeviceFaer actually implements

| functionality | where it lives | notes |
|---|---|---|
| elementwise / reductions / assignment / creation / adv-indexing | `rstsr-core/src/feature_rayon/auto_impl/*` via **symlinks** `device_faer/rayon_auto_impl/` (`DeviceRayonAutoImpl` = type alias of `DeviceFaer`) | shared with rayon CPU device; kernels in `rstsr-native-impl/src/cpu_rayon/` |
| matmul (`DeviceMatMulAPI`, `%`, `rt::matmul*`) | `device_faer/matmul.rs` + `matmul_impl.rs` | f32/f64/c32/c64 → faer gemm/syrk; all other dtypes → naive `gemm_ix2_naive_cpu_rayon`; rule-1 (vec·vec) **always** naive `inner_dot_naive_cpu_rayon`; rules 3–7 via `layout_matmul_dyn_row_major_with_lc` batch iteration (parallel outer when `n_task > 4 * nthreads`); col-major default order handled by reverse-axes + swap inside `DeviceFaer::matmul` |
| `DeviceGEMMAPI/SYMMAPI/SYRKAPI/HERKAPI/GEMVAPI` | **not implemented for DeviceFaer** (only `DeviceCpuSerial`/BLAS devices) | standalone `gemm()`/`syrk()` style APIs are unavailable on DeviceFaer; gemm/syrk internals exist in `matmul_impl.rs` but only the syrk fast-path uses them |
| linalg: cholesky, det, eigh, eigvalsh, inv, pinv, solve_general, solve_triangular, svd, svdvals | `rstsr-linalg-traits/src/faer_impl/*` (feature `faer`) | all are **faer-native drivers** (SVD/LU/LLT/EVD), not BLAS fallbacks; user-facing via `rt::linalg::*` from the `rstsr` meta crate |
| linalg: **slogdet, solve_symmetric** | only in `blas_impl` (`DeviceBLAS`) | **not available on DeviceFaer** — a compile-time gap on the faer-as-default build without a BLAS crate |
| vecdot (`DeviceVecdotAPI`), norm | rayon auto-impl + generic reduction kernels | vecdot → `vecdot_naive_cpu_rayon` (shared with T3 campaign work) |

## Findings

### [BUG-fixed] Naive kernels read the uninitialized output when `beta == 0` (NaN/Inf propagation)

- `rstsr-native-impl/src/cpu_serial/matmul_naive.rs` — `gemm_naive_cpu_serial` (L109),
  `gemv_naive_cpu_serial` (L154), `gevm_naive_cpu_serial` (L195), `inner_dot_naive_cpu_serial` (L233, pre-fix numbering)
- `rstsr-native-impl/src/cpu_rayon/matmul_naive.rs` — `gemm_ix2_naive_cpu_rayon` (L42), `inner_dot_naive_cpu_rayon` (L91)

The tensor-level matmul allocates the output via `empty` (`uninitialized_vec`).
BLAS GEMM semantics say `C` **need not be initialized when `beta == 0`** (the
`a % b` operator always passes `beta = 0`). The naive kernels computed
`c = c*beta + a@b*alpha`, reading the uninitialized buffer; a coincidental NaN
or Inf bit pattern in recycled heap memory survives `garbage * 0.0` and poisons
the result. **Empirically caught**: a plain `(31,2) @ (2,5)` product on
`DeviceCpuSerial` produced NaN at `c[29,2]` (faer gave the correct 529) when
the fresh output allocation reused pages previously holding NaNs. The same
kernels serve DeviceFaer as fallback (non-BLAS dtypes) and for every 1-D `·`
inner product.

Fix: `if beta.is_zero() { TC::zero() } else { beta * c }` in all six kernels
(requires adding the `num::Zero` bound to the four serial kernels — the rayon
sibling already had it). Regression test
`test_gemm_beta_zero_nan_canary_output` pre-fills the output with NaN canaries
and asserts full overwrite (deterministic; no heap-state dependence).

### [BUG-fixed] `faer::set_global_parallelism` leaked on error paths

`rstsr-linalg-traits/src/faer_impl/{inv,solve_general,eigh,eigvalsh,cholesky,svd,svdvals,pinv}.rs`
(set→compute-with-`?`→restore pattern). A failing faer call (e.g. cholesky on
a non-SPD matrix, SVD non-convergence) early-returned through `?` **without
restoring the previous global parallelism**, leaving process-wide faer state
mutated after any error. Fixed by restoring before rethrowing (compute-then-
restore-then-`map_err`); `faer_impl_generalized_eigh_f` (three fallible sites)
is wrapped in a closure so one restore covers all paths. Note the underlying
global-mutable-state design is itself not thread-safe (two concurrent calls on
different pools race) — HAZARD-documented below.

### [BUG-fixed] Non-square systems panicked inside faer instead of raising errors

`faer_impl_{inv,solve_general,cholesky,solve_triangular,eigh}` +
`faer_impl_generalized_eigh_f` (which now also checks `b`'s squareness/shape).
Previously e.g. `solve(A[2,3], b[2,1])` died on faer's internal
`assert!(nrows == ncols)`; numpy raises `LinAlgError`. All now return
`rstsr` `InvalidLayout` errors. Verified by
`test_faer_edge::edge_f64::test_non_square_errors`.

### [BUG-fixed] `eigh`/`eigvalsh` on 0×0 matrices panicked inside faer's EVD

numpy returns empty results for empty systems; faer 0.22.6 panics at
`linalg/evd/mod.rs:112`. All other 0×0 drivers (det/solve/inv/cholesky/svdvals)
already behaved numpy-compatibly (`det(∅) = 1` etc. — verified). Added `n == 0`
early-returns (empty eigenvalue/vector tensors) in `faer_impl_standard_eigh_f`,
`faer_impl_eigvalsh_f`, `faer_impl_generalized_eigh_f`; covered for f64 and c64
(`edge_f64::test_0x0_systems`, `edge_c64::test_svdvals_and_0x0`).

### [HAZARD-documented] `Mat`→`Tensor` owned conversion deallocates with a mismatched layout

`device_faer/conversion.rs` `IntoRSTSR for Mat<T>` (L73). faer's owned `Mat`
allocates via `alloc::alloc` with over-alignment and a padded row capacity;
reconstructing a `Vec` with `len = cap = upper_bound` (≤ true allocation) means
the later `Drop` deallocates with `Layout{size: upper_bound*size, align:
align_of::<T>()}` — a formal allocator-contract violation. In practice safe on
glibc (free ignores size/align) and consistent with rstsr's own
`uninitialized_vec` convention, but it is UB under Miri and layout-sensitive
allocators. Call sites: `cholesky` output wraps it transiently before
`into_contig` copies out (cholesky returns the copy); generalized eigh's `U()`
same. A sound fix (keep the faer allocation inside a custom `Data` repr, or
always copy) is left to the owner — the practical exposure is low.

### [HAZARD-documented] Singular systems return non-finite values silently

faer's SVD-based `solve`/`inv` do not detect singularity: `det` is fine, but
`solve(S, b)`/`inv(S)` for singular `S` return `±inf`/NaN (recip of zero
singular value). numpy/scipy raise `LinAlgError`. Pinned by
`edge_f64::test_solve_singular_current_contract` so a future contract change is
deliberate. **Owner judgment needed**: either document (faer semantics:
SVD pseudo-solve) or add a zero-singular-value check raising an error.

### [HAZARD-documented] faer global-parallelism is process-global state

`set_global_parallelism` races between concurrent calls on different
`DeviceFaer` pools (perf nondeterminism only, no unsoundness). A per-call
`Par` argument (like `solve_triangular` already uses) would be the cleaner
long-term design.

### [HAZARD-documented] Strided (broadcast stride-0) output for matmul is unvalidated

`rt::matmul_from`/`matmul_with_output` accept any `TensorViewMut` output; a
stride-0 (broadcast) output would give nondeterministic garbage (last-writer
wins per element). Rust's borrow rules make reaching this state through the
normal API hard, and numpy errors out; input validation only, not pursued.

### OK-verified (no action)

- **rstsr→faer stride mapping is transposition-free**: `IntoFaer for
  TensorView` maps `shape [m,n]`, `stride[0]→faer row_stride`, `stride[1]→faer
  col_stride` — matches faer's definition (offset per step along axis 0 / axis
  1). Same mapping back in `IntoRSTSR`. No silent transposition.
- Col-major default order: `DeviceFaer::matmul` reverses axes and swaps
  operands; verified numerically against DeviceCpuSerial and a manual
  `device.matmul` probe (2-D, transposed 2-D, batched-with-trailing-batch
  cases). Under col-major conventions the batch axes are *trailing* and
  contracted axes differ for ≥3-D operands, so col-major matmul expressions are
  **not** expression-compatible with numpy for rules 5–7 (documented in the
  col-major test comments; not a bug — convention).
- syrk fast-path detection (`equal_ptr && equal_shape && equal_stride`) is
  value-correct: matching ptr/shape/reversed-strides forces the second operand
  to be exactly the transpose view of the first. `A@Aᵀ`, `Aᵀ@A` verified
  numerically incl. odd sizes; syrk symmetrization is race-free (each upper
  element written once, lower triangle read-only).
- Edge shapes: 0×0, [0,n], [m,0] (K=0), 1×1, [1,n]·[n,1], odd/prime sizes
  (17×19, 31×2), huge-K-free; batch paths (parallel-outer 111 tasks,
  sequential-outer, single-thread pool) — all match DeviceCpuSerial.
- Strided/offset/transposed/f-order/broadcast-stride-0 **inputs** to matmul and
  to det/inv/eigvalsh (incl. a stride-2 slice through faer).
- alpha/beta GEMM path (`matmul_from`) incl. strided outputs.
- dtype coverage: f32/f64/c32/c64 through faer; i32 (naive fallback) exact.
- Rayon interplay: `ParIterRSTSR::split_at` recomputes chunk-boundary offsets
  from the layout (`unravel_index_f`), so no incremental drift; exercised by
  odd-shape batch tests (111-task parallel outer, 1-thread pool). Reductions
  are shared code already exercised by T2/T4' campaigns.
- faer 0.22 `MatRef/MatMut::from_raw_parts*` accepts stride 0 (used by
  broadcast inputs) and arbitrary positive strides; single-allocation/
  alignment/initialized preconditions hold at every call site (see SAFETY
  comments).

## SAFETY commentary added (patch `safety-comments-and-tests.patch`)

- `device_faer/conversion.rs`: 12 tokens — ptr arithmetic in-bounds (2),
  `MatRef/MatMut::from_raw_parts*` validity + stride-convention note (2),
  `Vec::from_raw_parts` ×4 (ManuallyDrop never-dealloc ×3; owned-Mat caveat
  block, see HAZARD), `new_unchecked` ×4 (layout in-bounds by construction).
- `device_faer/matmul_impl.rs`: 13 tokens — gemm a/b/c validity, aliasing
  contract and beta≠0 initialization note (3+3 closures), `transmute` to
  `MaybeUninit` ×2, syrk a/at/c ×3, symmetrize race-freedom ×3.
- `device_faer/matmul.rs`: 3 blocks — `set_offset` from valid rest-layout
  iteration ×2, per-task `c` slice reconstruction + disjoint-write argument.

(`get_index_mut_ptr`'s `unwrap` in `device.rs` is index-bounded by callers;
left as-is.)

## Tests added

In-crate (`cargo test -p rstsr-core --lib`, suite total 118 pass):

- `device_faer::matmul::test_gemm_edge_shapes_row_major` — 30+ compatible
  shape pairs incl. 0-axes, 1×1, primes; faer vs DeviceCpuSerial.
- `::test_gemm_transposed_and_strided_operands` — `A@Aᵀ`, `Aᵀ@A` (syrk paths),
  offset views, stride-2 slices, broadcast stride-0 operand.
- `::test_gemm_batch_parallel_and_serial_paths` — rules 5–7, parallel-outer
  (111 tasks) vs sequential-outer vs 1-thread device, batch broadcasting.
- `::test_gemm_alpha_beta_gemm_from` — `matmul_from` with α/β≠0, contiguous +
  strided outputs.
- `::test_gemm_col_major_default_order` — col-major order values vs serial
  row-major, incl. batched trailing-batch case.
- `::test_inner_dot_edge_shapes` — n ∈ {0,1,2,3,17,64,65,129}, strided views,
  0-dim output with α/β.
- `::test_gemm_dtype_coverage` — f32/c32/c64 vs serial; i32 naive exact match.
- `::test_gemm_beta_zero_nan_canary_output` — regression test for the beta=0
  uninit-read fix (all six kernels, NaN canaries).

`rstsr-linalg-traits/tests/test_faer_edge/` (new; `cargo test -p
rstsr-linalg-traits --features faer`, suite total 42 pass):

- `edge_f64.rs` (14 tests): 1×1 and 0×0 systems; non-square → Err;
  numpy-derived constants for solve/inv/det (`X`, `INV_A`, `DET_A`),
  eigh/eigvalsh (`EIGH_W`), cholesky (`CHOL_L`, L·Lᵀ=C, upper=lowerᵀ, non-SPD
  → Err), svd (`SVD_S`, thin/full shapes, reconstruction), pinv (`PINV_R2`,
  rank 2, Moore-Penrose property), triangular solve (`TRI_X`), generalized
  eigh itype=1 (`GEN_W`, Av=λBv property); strided/f-order/transposed inputs;
  solve with strided rhs; col-major default order; singular-solve contract pin.
- `edge_c64.rs` (4 tests): det/inv/solve (hand-computed), hermitian eigh
  (λ ∈ {1,4} + Av=λBv), hermitian cholesky, svdvals + 0×0 guard on c64.

## Efficiency notes (not chased)

1. `inv` uses `svd().inverse()` — SVD is the most expensive factorization;
   `faer::linalg::solvers::Lu` (partial pivoting) would match LAPACK `getri`
   cost. Same for `solve_general` (LU would do for square non-singular use).
2. `gemm_with_syrk_faer`'s symmetrization builds a **fresh rayon pool per
   call** and ignores the device pool (oversubscription risk under nesting);
   could reuse `pool.install`.
3. Rule-1 vec·vec always falls back to `inner_dot_naive_cpu_rayon` — never
   faer-accelerated; fine for large n (memory-bound) but odd for small n where
   a serial loop without rayon task overhead would win (T3 campaign has data).
4. `faer_impl_solve_general_f` computes a full SVD even when the caller just
   wants one solve; the faer LU path is ~10× cheaper per solve.
5. `into_faer` on already-contiguous inputs could use faer's
   `from_slice/view` fast paths to skip stride bookkeeping — marginal.
6. `det` of c64 via `determinant()` is fine, but `slogdet` (absent on
   DeviceFaer) would avoid overflow for large matrices — adding it on faer's
   LU would close the slogdet coverage gap too.

## Unresolved items needing owner judgment

1. Singular solve/inv contract (HAZARD above): document vs raise.
2. Owned `Mat`→`Vec` deallocation-contract UB (HAZARD above): custom `Data`
   repr vs always-copy.
3. `slogdet`/`solve_symmetric` missing on DeviceFaer (coverage gap): implement
   on faer (LU-based slogdet is straightforward) or document as
   BLAS-device-only.
4. `faer::set_global_parallelism` global-state design (thread-safety).
5. Col-major ≥3-D matmul expression incompatibility with numpy (rules 5–7
   differ semantically) — worth a doc note in the book, not a code change.

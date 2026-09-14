# rstsr soundness check — consolidated report

- **Date**: 2026-09-14; **rstsr base**: `acfa93e` (master)
- **Organization**: [PLAN.md](./PLAN.md); four parallel subtasks, each with its own
  detached worktree under `/home/a/rstsr_pack/tmp/snd-*` (removed after consolidation).
- **Subtask reports** (read these for expanded discussion; this file is the cross-task
  digest):
  - [T1-unsafe-audit/README.md](./T1-unsafe-audit/README.md) — 540 `unsafe` occurrences audited
  - [T2-col-major/README.md](./T2-col-major/README.md) — col-major contract feature wired + tested
  - [T3-layout-theory/README.md](./T3-layout-theory/README.md) — layout/index-math theory audit
  - [T4-faer-linalg/README.md](./T4-faer-linalg/README.md) — DeviceFaer surface + linalg audit

## TL;DR

**13 real bugs found and fixed** (all with regression tests), **~460 `unsafe` sites
given written SAFETY justifications** (303/513 unsafe-bearing code lines now have one
within 6 lines, vs ~32 lines total before), the **`col_major` contract feature is
wired and tested for the first time** (new 66-test col track + full 302-test body
green under both contracts), and the layout math / faer conversion boundaries were
verified correct with only documented hazards remaining. All fixes compose: the
combined patch is green on both the row-major and col-major feature configurations.

## Verification of the combined tree (`combined-all.patch` applied to `acfa93e`)

| Suite | Command | Result |
|---|---|---|
| rstsr-common, row | `cargo test -p rstsr-common` | 40 unit + 4 doc, green |
| rstsr-core lib, row | `cargo test -p rstsr-core --lib` | 128 pass, 3 ignored (pre-existing) |
| entry_row_cpu | `cargo test -p rstsr-core --test entry_row_cpu` | 302/302 (unchanged from base) |
| linalg-traits, faer | `cargo test -p rstsr-linalg-traits --features faer` | 42/42 (24 old + 18 new) |
| rstsr-common, col | `--no-default-features --features "col_major,std"` | 40 unit + 4 doc, green |
| rstsr-core lib, col | `--no-default-features --features "col_major,aligned_alloc,faer,faer_as_default,std" --lib` | 128/128 |
| entry_col_cpu (new) | same features, `--test entry_col_cpu` | 368/368 (302 shared body + 66 new col tests) |

Workspace `cargo check --workspace --all-targets` green (see §Integration).

## The 13 fixed bugs

| # | Location | Bug | Trigger class |
|---|---|---|---|
| T1-1 | `rstsr-core/src/tensor/creation.rs` `full_f` | allocated `layout.size()` instead of `bounds_index().1` — an offset/negative-stride layout reads out of bounds of the fresh storage (sibling constructors were already correct) | user-provided layout + `rt::full` |
| T1-2 | `rstsr-core/src/tensor/{iterator_elem,iterator_axes}.rs` | `iter()`/`indexed_iter()`/`axes_iter()` on **owned** tensors `transmute`d the `&self` borrow to a free impl lifetime — iterator could outlive the tensor → use-after-free (empirically confirmed) | any owned-tensor iterator escape |
| T1-3 | `rstsr-common/src/alloc_vec.rs` `aligned_uninitialized_vec` | wrapping `size * size_of::<T>()` → tiny allocation reinterpreted as huge `Vec` | absurd element count |
| T1-4 | `rstsr-core/src/storage/data.rs` | `unsafe impl Send … where C: Send` too loose: `&C` needs `C: Sync`, `Arc<C>` needs Send+Sync — e.g. `DataRef<'_, Cell<i32>>` could cross threads | shared handle + interior mutability |
| T2-1 | `rstsr-core/src/device_faer/matmul_impl.rs` | the only faer GEMM correctness test was `#[cfg(not(col_major))]`-silenced — col builds (where faer is the **default device**) had zero matmul coverage; col twin added (expects −78+282i, derived) | col-major builds only |
| T3-1 | `rstsr-common/src/layout/reshape.rs:141` | `reshape([0,-1])` panics divide-by-zero; numpy raises ValueError; also converts isize-product overflow to a clean error | 0-size + `-1` reshape |
| T3-2 | `rstsr-common/src/layout/rearrangement.rs` (`fn_b`) | `Order::A`/`B` buffer-order translation panics (`shape[0]`) on 0-dim layouts | scalar-layout translate |
| T3-3 | `rstsr-common/src/layout/rearrangement.rs:257` | multi-layout `Order::A` checked `c_contig`/`f_contig` where intent was `c_prefer`/`f_prefer` — prefer-but-not-contiguous arrays silently fall to cargo-feature default (unary twin was correct) | iteration-order/intent |
| T3-4 | `IndexedIterLayout` + 4 core impls | `next_back` returned the **front** index — every `(index, element)` from `.rev()` was mismatched (element correct, index wrong) | public API, unexercised |
| T3-5 | `rstsr-core/src/tensor/iterator_axes.rs` ×4 | only the first axis checked for negative values (`axes_iter([0,-5])` → OOB panic not `InvalidValue`); empty axes (`axes_iter(())`) underflowed `len()-1` | malformed axes input |
| T4-1 | 6 naive matmul/inner-dot kernels (`rstsr-native-impl/src/{cpu_serial,cpu_rayon}/matmul_naive.rs`) | beta=0 **read uninitialized output** (violates BLAS "C need not be initialized when beta=0"); NaN/Inf heap garbage propagated — empirically caught: `(31,2)@(2,5)` on DeviceCpuSerial returned NaN where faer was correct | any matmul with beta=0 path |
| T4-2 | `rstsr-linalg-traits/src/faer_impl/*` (9 fns) | `faer::set_global_parallelism` leaked on all error paths — global state corruption after any failed linalg call | error paths |
| T4-3 | `rstsr-linalg-traits/src/faer_impl/*` | non-square systems (inv/solve/cholesky/solve_triangular/eigh) panicked inside faer asserts instead of returning rstsr errors; eigh/eigvalsh on 0×0 panicked in faer EVD → now numpy-compatible empty results | malformed inputs, 0×0 |

## Patch inventory & integration protocol

| File | Content | Nature |
|---|---|---|
| `combined-all.patch` | **everything**, incl. two manual overlap resolutions (see below) | 90 files, +3364/−154 — **recommended landing artifact** |
| `T1-unsafe-audit/safety-comments.patch` | ~210 distinct SAFETY justifications, 61 files, comment-only | safe, mechanical |
| `T1-unsafe-audit/fixes.patch` | T1-1…T1-4 + regression tests (4 files carry adjacent comment hunks) | behavioral |
| `T2-col-major/col-major-wiring-and-tests.patch` | `entry_col_cpu.rs` + `[[test]]` gate + new `tests/col_func/` (66 tests) | additive |
| `T2-col-major/fixes.patch` | T2-1 col twin test | behavioral |
| `T3-layout-theory/fixes.patch` | T3-1…T3-5 + 10 in-crate regression tests | behavioral |
| `T4-faer-linalg/fix-01-naive-beta0-uninit-read.patch` | T4-1 | behavioral |
| `T4-faer-linalg/fix-02-faer-linalg-error-paths.patch` | T4-2, T4-3 | behavioral |
| `T4-faer-linalg/safety-comments-and-tests.patch` | ~25 SAFETY comments in `device_faer/*` + new `rstsr-linalg-traits/tests/test_faer_edge/` (18 tests) | mixed |
| `T4-faer-linalg/gen_expected_constants.py` | numpy reference constants for the linalg edge tests | tooling |

**Recommended**: apply `combined-all.patch` onto `acfa93e` — this exact tree is what
was verified (all suites above green, both feature contracts).

**Thematic alternative** (review/land per theme): apply in order
T3-fixes → T1-fixes → T4-fix-01 → T4-fix-02 → T2-fixes → T2-wiring → T1-comments →
T4-comments. Three overlap points need resolution (as done in `combined-all.patch`):

1. `rstsr-core/src/tensor/iterator_elem.rs` `IndexedIterVecMut::next_back` — T3's
   semantic fix (`index_end()` after `next_back`) and T1's SAFETY comment on the
   `transmute` were **combined**.
2. `device_faer/conversion.rs` + `matmul_impl.rs` — T1 and T4 both commented the same
   sites; **T4's (deeper) comments kept**.
3. `rstsr-core/src/tensor/iterator_elem.rs` `test_axes_iter_correctness` (T1's
   regression test) asserted row-major logical content, which fails under the
   `col_major` feature; replaced with an order-independent expectation derived from
   logical indexing (`t[[i, j]]`), green under both contracts. Only present in
   `combined-all.patch`.

Note for CI: row↔col invocations alternate feature unification and recompile the
dependency subtree (~50 s warm each way); the npy resources for
`rstsr-linalg-traits` tests are generated one-time via
`rstsr-test-manifest/resources/gen_rand_vec.py` (conda env `torch`).

## Findings needing owner judgment (not patched)

From T1 (§4 of its README):
1. **Write-through-`as_ptr()` pattern** in all rayon kernels — `&mut`-derived pointers
   cast to `*mut` and captured by rayon tasks; disjoint writes, but formally Stacked
   Borrows UB. Suggested: hoist an `AtomicPtr` per kernel.
2. Mutable custom-layout stride-0 axes can create aliasing `&mut` edge cases.
3. `flags.rs` static-mut default-order flag (set-once pattern).
4. faer `Mat` → owned-Tensor conversion: dealloc layout mismatch (formal UB, matches
   rstsr's own `uninitialized_vec` convention).
5. `Layout::size()` doc says "cached" but recomputes; silent wrap on overflow.

From T2 (systemic): the numpy-parity body pins `set_default_order(RowMajor)`
per test, so col-mode runs of the shared body exercise row semantics — col behavior
coverage deliberately lives in the new `col_func/` track. Owner may want a
contract note in ADR-0002 (its "col_major body" reservation is now realized).

From T3: overflow `debug_assert` hardening for `bounds_index`/`check_strides`
(unreachable below 2^63 elements); empty-axes semantics choice (yield-1-view vs
error — currently panics pre-fix, `InvalidValue` post-fix); dyn-vs-fixed
`stride_contig` disagree on zero dims (`[1,2,0]` vs `[1,0,0]`, benign today).

From T4: singular solve/inv silently return ±inf (faer SVD contract — document vs
raise); `slogdet`/`solve_symmetric` are BLAS-only, missing on DeviceFaer; col-major
≥3-D matmul expression semantics diverge from numpy (documented; pinned by tests).

## Efficiency quick wins (S4 — noted, not chased)

- `Layout::size()` recomputes the shape product per call despite doc claiming a
  cache; hot loops (`reduce_*`, prefer-checks, iterators) re-multiply. (T1+T3)
- `Layout::check_strides` heap-allocates two `Vec`s + sort on every `Layout::new`;
  a small-vec for `ndim ≤ 8` would remove churn on small-tensor creation. (T1+T3)
- `gemm_with_syrk_faer` builds a **fresh rayon pool per call** instead of
  `pool.install` on the device pool (oversubscription risk). (T4)
- `inv` implemented as `svd().inverse()` — LU/partial-pivoting would match LAPACK
  `getri` cost; same for square `solve_general`. (T4)
- Rule-1 vec·vec always takes the naive fallback, never faer. (T4)
- Maintenance twins: `IterLayoutColMajor`/`RowMajor` (~500-line copy-paste),
  `device_cpu_serial/operators` vs `feature_rayon/auto_impl` mirrors. (T1)

## Verified clean (highlights)

- rstsr→faer 2-D stride mapping is transposition-free (stride[0]→row_stride,
  stride[1]→col_stride); col-major device matmul numerically correct (T4).
- `attempt_nocopy_reshape` diffed line-by-line against numpy's shape.c — faithful,
  and safer on the `ni==0` edge numpy itself reads out of bounds. (T3)
- matmul broadcast/ellipsis rules 1–7 correct in both orders; iterator odometer
  math (negative/zero strides, 0-dim, empty) correct; `dim_narrow` slice clamping
  matches numpy in all probed cases. (T3)
- Julia-style first-axis broadcasting under col_major is the enforced contract and
  stride-exact; argmax/argmin flat index stays row-major even under col_major
  (matches docs). (T2+T3)

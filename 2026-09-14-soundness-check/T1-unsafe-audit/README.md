# T1 — Unsafe-soundness audit of rstsr (soundness aspect S2)

- **rstsr commit**: `acfa93e` (master), audited in worktree `/home/a/rstsr_pack/tmp/snd-unsafe`
- **Date**: 2026-09-14
- **Scope**: every `unsafe` usage in `rstsr-common/src`, `rstsr-core/src`,
  `rstsr-native-impl/src`, `rstsr-dtype-traits/src`, `rstsr-linalg-traits/src`
  (`rstsr-core/src/device_faer/rayon_auto_impl/*` are symlinks to
  `rstsr-core/src/feature_rayon/auto_impl/*`; each physical site counted once).
- **Deliverables**: `safety-comments.patch` (comment-only),
  `fixes.patch` (behavioral fixes + regression tests), this README.
- **Verification**: `cargo check --workspace` green; `cargo test -p rstsr-core
  --release` 114+302+2+183 pass; `cargo test -p rstsr-common --release` 36 pass;
  `cargo test -p rstsr-native-impl --release` 4 pass. Full diff
  `/tmp/t1-all.patch` = 3649 lines; both patches reverse-apply cleanly against
  the worktree.

## 1. Method

1. `grep -rn "unsafe"` per crate → site checklist (file:line, 540 token
   occurrences at start; `unsafe fn` decls and `unsafe impl Send/Sync` included).
2. For each site (or copy-paste group): read the surrounding code path and trace
   the actual invariant chain (layout validation → offset arithmetic → pointer
   use) instead of assuming it.
3. Classify: OK (write concrete `// SAFETY:` line), UNCLEAR (flag, do not
   invent justification), BUG (invariant genuinely fails → minimal fix + test).
4. Special attention to zero-size dims, stride-0 broadcast inputs, empty
   slices, computed-offset `get_unchecked`-style access, Send/Sync impls, and
   allocation code.

## 2. Site statistics

Raw `unsafe` token occurrences at audit start: **540**
(rstsr-common 57, rstsr-core 338, rstsr-native-impl 142, rstsr-dtype-traits 3,
rstsr-linalg-traits 0). 513 are on code lines (rest are doc/comment mentions).

| Classification | Sites (approx.) | Action |
|---|---|---|
| OK — invariant traced, comment written | ~460 | `// SAFETY:` added (~210 distinct justification blocks, 343 comment lines) |
| OK — already documented by owners (`# Safety` sections, existing `safety:` comments) | ~30 | left as-is |
| Test-code sites (`#[cfg(test)]`) | ~14 | not commented (no production impact) |
| UNCLEAR / design-level, flagged for owner review | 7 (see §4) | flagged only |
| Real bugs | 4 (see §3) + 1 post-audit find (BUG 5, from the §4.7 follow-up test) | fixed + regression tests |

After the audit, 303/513 unsafe-bearing code lines have a SAFETY justification
within a 6-line window (the remainder are lines inside groups whose
justification sits at the group head — e.g. `duplicate_item` macro closures —
or `unsafe fn` definitions whose contract is in the `# Safety` doc section).
Before the audit only ~32 lines mentioned Safety/SAFETY anywhere.

## 3. Real bugs found and fixed (in `fixes.patch`)

### BUG 1 — `full()` with a user-supplied `Layout` allocates too little storage (OOB)
- **File**: `rstsr-core/src/tensor/creation.rs`, `FullAPI for (Layout<D>, T, &B)` (~line 1044).
- **Trigger**: `full((layout, fill))` where `layout` has a non-zero offset or
  negative strides, e.g. `Layout::new(vec![3], vec![1], 100)` or
  `Layout::new(vec![3], vec![-1], 3)`. The documented overload
  `full((layout: Layout<D>, fill_value, device))` — "use the exact layout" —
  accepts such validated layouts.
- **Why it fails**: the code allocated `layout.size()` elements, but the
  layout's addressable upper bound is `bounds_index().1` (= offset + span),
  which can far exceed `size()`. The wrapped tensor's element access then reads
  (and the semantics, write) outside the freshly allocated `Vec` — heap OOB.
  Notably `empty`/`zeros`/`ones`/`uninit` all used `bounds_index()` correctly;
  only `full` used `size()`.
- **Fix**: `let (_, idx_max) = layout.bounds_index()?;` (same as the sibling
  constructors).
- **Tests**: `test_full_with_offset_layout` (offset 100, storage must be >= 103
  and all values equal `fill`) and `test_full_with_negative_stride_layout`.

### BUG 2 — `iter()` / `indexed_iter()` / `axes_iter()` on owned tensors: use-after-free via lifetime `transmute`
- **Files**: `rstsr-core/src/tensor/iterator_elem.rs` (`iter_with_order_f`,
  `indexed_iter_with_order_f`), `rstsr-core/src/tensor/iterator_axes.rs`
  (`axes_iter_with_order_f`, `indexed_axes_iter_with_order_f`).
- **Trigger**:
  ```rust
  let it = { let t = arange(6); t.iter() };   // compiles before the fix
  let v: Vec<_> = it.cloned().collect();      // reads freed memory
  ```
  Verified empirically: printed `[403706488, 6, 2, 3, 4, 5]` instead of
  `[0,1,2,3,4,5]`; `axes_iter` variant printed garbage as well.
- **Why it fails**: the impl blocks declare a free lifetime `'a` (only bound:
  `B: 'a`) unrelated to `&self`. For owned tensors `R = DataOwned<Vec<T>>`
  carries no lifetime, so nothing ties `'a` to the tensor. The constructors
  `transmute`d the internally borrowed `&[T]`/`TensorView<'_>` to `'a`,
  producing an iterator that can outlive the tensor and dangle. (The `_mut`
  variants are safe: they take `&'a mut self`, which really ties the borrow.)
- **Fix**: return the iterators tied to `&self` (`IterVecView<'_, …>`,
  `IterAxesView<'_, …>`, …) and delete the transmutes. The fix correctly
  surfaced 3 pre-existing in-crate tests that relied on the unsound escape
  (`a.t().iter()` on a dropped temporary); those were updated to bind the view
  first. **API note**: `let it = a.view().iter();` patterns must now bind the
  view (`let v = a.view(); let it = v.iter();`) — same policy as
  `ndarray::ArrayBase::iter`; the escape hatch was unsound for owned/arc/cow
  tensors, so it cannot be kept.
- **Tests**: `test_iter_correctness`, `test_axes_iter_correctness`
  (behavioral guards; the UAF pattern itself now fails to compile, which the
  patch demonstrates).

### BUG 3 — allocation-size overflow in `aligned_uninitialized_vec` (tiny alloc + huge `Vec`)
- **File**: `rstsr-common/src/alloc_vec.rs` (~line 91).
- **Trigger**: `uninitialized_vec::<u64>(size)` with
  `size * 8` overflowing `usize` (e.g. `usize::MAX/4 + 1` elements, reachable
  from `rt::empty` with an absurd shape under the `aligned_alloc` feature).
- **Why it fails**: `aligned_alloc(size * sizeof, alignment)` computed the byte
  count with wrapping multiply (release mode); a wrapped (possibly tiny)
  allocation was then reinterpreted via `Vec::from_raw_parts(_, size, size)` —
  out-of-bounds from first use. (`unaligned_uninitialized_vec` is safe: it uses
  `try_reserve_exact`, which checks internally.)
- **Fix**: `size.checked_mul(sizeof)` → `RuntimeError` on overflow.
- **Test**: `test_aligned_alloc_overflow` (rstsr-common).

### BUG 4 — `DataRef`/`DataCow`/`DataArc`/`DataReference` `Send`/`Sync` bounds too loose
- **File**: `rstsr-core/src/storage/data.rs` (lines 44–53).
- **Trigger**: `DataRef<'_, C>::from(&cell)` with `C: Send + !Sync` (e.g.
  `Cell<i32>`), then send the (public) `TensorView` to another thread.
- **Why it fails**: a `&C` is `Send` only if `C: Sync`, and `Arc<C>` is
  `Send`/`Sync` only if `C: Send + Sync`. The impls declared these wrappers
  `Send where C: Send`, so two threads could mutate a `Cell` behind nominally
  shared references — a data race (UB) through the public generic API.
- **Fix**: tightened bounds to `C: Send + Sync` wherever a variant can expose a
  shared reference or `Arc` (`DataRef: Send`, `DataCow: Send`, `DataArc:
  Send+Sync`, `DataReference: Send+Sync`). `DataOwned`/`DataMut: Send where C:
  Send` were already correct. One downstream bound needed updating:
  `IntoParallelIterator` for the axes iterators in
  `rstsr-core/src/feature_rayon/par_iter.rs` now requires `B::Raw: Sync` (which
  is exactly the semantic requirement for shipping views to other threads).
- No regression test (the fix is a compile-time bound; misuse now fails to
  compile).

### BUG 5 — allocating `rt::matmul` read its uninitialized output (found *after* the audit)

Not found during the original audit — surfaced 2026-09-16 by the §4.7 follow-up
integration test (`tests/allocatable_dtype.rs`, `BigInt` dtype), then verified
by hand.

- **Files**: `rstsr-core/src/tensor/linalg/matmul.rs` (`op_refa_refb_matmul`,
  ~line 286), `rstsr-native-impl/src/cpu_serial/matmul_naive.rs` (4 kernels),
  `rstsr-native-impl/src/cpu_rayon/matmul_naive.rs` (2 kernels).
- **Trigger**: any allocating `rt::matmul`/`rt::matmul_f`/`a.matmul(&b)`. The
  wrapper allocated its output with `unsafe { empty(...) }` (plain `TC`, the
  `uninitialized_vec` raw path), but the naive kernels scale the existing output
  first — `c[idx_c] = beta * c[idx_c]` unconditionally, even at `beta = 0`.
- **Why it fails**: reading uninitialized memory. For allocatable dtypes
  (admitted by the public `TC: Zero + One` bound) this reads and *drops* a
  garbage value — UB; empirically a 2×2 `BigInt` matmul aborted with
  `free(): invalid pointer` after heap churn. For `f64`, `0.0 * NaN-garbage`
  silently contaminates the result. It also violated the BLAS convention that
  at `beta = 0` the output is *not read* (the very convention the audit's BLAS
  buffer argument rests on). `vecdot`'s wrapper was checked and is clean
  (`uninit_impl` + beta-free kernel); BLAS-device gemm staging is clean;
  DeviceFaer already branched on `beta == 0`.
- **Fix (2026-09-16, owner-approved; final design after an intermediate
  `full`-zero-fill version)**: two-function dispatch on `beta`. The trait
  gained `DeviceMatMulAPI::matmul_uninit` — write-only `c = alpha * (a @ b)`
  into `MaybeUninit` storage, never reading `c` — and the allocating wrapper
  allocates via `uninit_impl`, calls it, and finalizes with one `assume_init`
  (same shape as the vecdot wrapper). `beta`-scaling (`matmul`,
  `matmul_from`, `matmul_with_output`) keeps caller-initialized `c`; the six
  naive kernels additionally skip the beta-scaling read at `beta.is_zero()`.
  POD dtypes ride BLAS/faer at `beta = 0` (non-read convention); other dtypes
  run new write-only naive kernels (cpu_serial: gemm/gemv/gevm/inner-dot +
  dispatcher; cpu_rayon: gemm/inner-dot + dispatcher, the BLAS crates'
  exotic fallback). Implemented for all 7 `DeviceMatMulAPI` impls.
- **Tests**: `test_matmul_allocating_exact` (allocating `rt::matmul`, method
  form, and batched branch, exact `BigInt` values) in
  `rstsr-core/tests/allocatable_dtype.rs`. Full suites green
  (core 125+8+302+2+185, common 40+4, workspace check).
- The former follow-up option (`MaybeUninit` fresh-output path) was
  implemented the same day per owner direction — see the fix above; a
  half-finished earlier attempt (device-crate side only) had been reverted
  and was rebuilt complete.

## 4. UNCLEAR / design-level findings — need owner review (not patched)

1. **Write-through-`as_ptr()` pattern in all rayon kernels**
   (`rstsr-native-impl/src/cpu_rayon/{op_with_func,assignment,vecdot,transpose,adv_indexing,matmul_naive,reduction}.rs`,
   ~30 sites): output-slice pointers are derived via `c.as_ptr()` (a *shared*
   reborrow of the `&mut` slice parameter, so a read-only provenance) and then
   **written** through after a `as *mut` cast. Writes are disjoint per rayon
   task (disjoint layout offsets), so results are correct today, but this is UB
   under Stacked Borrows (Miri flags it) and relies on compilers not exploiting
   `&`-based noalias. The codebase itself already uses the sound pattern where
   possible (`AtomicPtr::new(a.as_mut_ptr())` in `cpu_rayon/op_tri.rs` and
   `cpu_rayon/reduction.rs` lines 749/899). Suggested remediation: hoist
   `AtomicPtr::new(c.as_mut_ptr())` per function (raw pointers are `!Sync`, which
   is why the pattern exists) and `load()` inside the closure. Each flagged site
   carries a NOTE comment; touching ~30 hot-loop sites was judged beyond a
   "minimal fix" for this audit.
   **Status 2026-09-15: remediated.** All 36 native-impl sites plus 4 further
   sites in `rstsr-sci-traits/src/distance/native_impl.rs` (`cdist_rayon`/
   `cdist_weighted_rayon`, same pattern, found when re-grepping the workspace)
   were converted to the `AtomicPtr` hoist on branch `260915-unsafe-soundness-3`
   (uncommitted at time of writing). A-B benchmark: no regression on any
   affected kernel; cdist family −24%, in-place blocked-2D −5%, naive matmul
   −2%, everything else within noise; geometric mean 0.968 — see
   `../../2026-09-15-atomicptr-hoist-ab/README.md`.
2. **Multiple live `&mut` views from one iterator**
   (`iterator_axes.rs` `IterAxesMut`/`IndexedIterAxesMut`, `iterator_elem.rs`
   `split_at`): items are produced by lifetime-rewriting `transmute`s from a
   single `&mut`. Successive views are element-wise disjoint **provided the
   iterated axes have non-zero strides**. Edge case: a stride-0 (broadcast)
   axis in a *mutable* tensor — constructible via
   `asarray((&mut buf, layout))` with a custom stride-0 layout, since
   `check_strides(true)` skips zero strides — would make `axes_iter_mut` yield
   aliasing `&mut`s to the same element. Consider either rejecting stride-0
   axes in mutable-view constructors or building items from raw pointers with
   per-item provenance (ndarray-style).
   **Status 2026-09-16: resolved — capability model.** Holding a `TensorMut`
   with a stride-0 layout stays legal (it is inert); every *write path* now
   rejects `layout.is_broadcasted()` — the same predicate `assign`/`fill`
   already enforced. Gated: the `add_assign` family (tensor and scalar
   impls), `mapi_f`/`mapi_fnmut_f`, the matmul output driver,
   `iter_mut`/`indexed_iter_mut`, and `axes_iter_mut`/`indexed_axes_iter_mut`
   (guard on the *iterated* axes only — items keep non-iterated broadcast
   axes as inert views, and writes through them are caught by the item-level
   gates). `index_mut` needs no gate (single-element `&mut` is
   borrow-checked). Unary owned in-place ops fall back to a fresh output
   instead of erroring, matching the binary-op reuse policy. Committed as
   `6ad5e84` on `260915-unsafe-soundness-3` (9 files, +254/−8, all suites
   green).
3. **`flags.rs` `static mut` defaults** (`ChangeableDefault`):
   `get_default()`/`change_default()` racing would be a data race (UB). The
   unsafe-fn contract ("set at init time before threads") is now documented in
   a comment; a `AtomicU8`-backed or `OnceLock` representation would remove the
   hazard entirely.
   **Resolved 2026-09-16** on branch `260915-unsafe-soundness-3`: the macro
   (single instantiation, `TensorIterOrder`) was replaced by an
   `AtomicU8`-backed static with `Relaxed` ordering (`core` primitive, so
   `no_std` is unaffected — rstsr-common is `#![cfg_attr(not(test), no_std)]`);
   `change_default` is now a safe fn; `#[repr(u8)]` added to
   `TensorIterOrder` for the discriminant round-trip. Round-trip regression
   test added in `flags.rs`.
4. **RESOLVED 2026-09-16 — faer `Mat` → owned `Tensor` ownership transfer**
   (`device_faer/conversion.rs::IntoRSTSR for Mat`): the original code
   `mem::forget(self)` + `Vec::from_raw_parts(...)` re-homed faer's allocation
   into a `Vec`. Verified against the faer 0.22.6 source: the dealloc-layout
   mismatch happened on **every** numeric conversion, not hypothetically —
   `align_for` over-aligns power-of-two-sized, drop-free element types to
   `max(align, 64)` and pads row capacity to the alignment multiple, while the
   `Vec` deallocs with `align_of::<T>` and the logical length (e.g. a 5×1
   `Mat<f64>`: alloc (64 B, align 64) vs dealloc (40 B, align 8); UB by the
   GlobalAlloc contract, benign only because glibc's `free` ignores the
   layout). Resolution (owner decision, audit "option 2"): `into_rstsr` now
   **copies** the logical elements column-wise (`Mat::col_as_slice`; safe code,
   new `T: Clone` bound) into a fresh contiguous column-major buffer (stride
   `[1, nrows]`, not faer's padded strides) and lets faer drop its own
   allocation. The impl docstring documents the copy behavior, the reason, and
   the zero-copy alternative (`mat.as_ref().into_rstsr()` → non-owning
   `TensorView`); the reference conversions (`MatRef`/`ColRef`/`MatMut` —
   ManuallyDrop, never freed, already sound) got one-line ownership docs, and
   `IntoRSTSR` is now exported via prelude `rstsr_traits` (facade users
   previously could not name it). Tests: `test_mat_owned_into_rstsr`
   (incl. padded 5×1 case) + doctest. The custom allocator-handle alternative
   (audit "option 1") was not taken.
5. **`Layout::size()` doc/impl mismatch** (`rstsr-common/src/layout/layoutbase.rs`):
   doc says "uses cached size" but the product is recomputed (and could silently
   wrap for shapes whose product exceeds `usize`, e.g. `[2^32, 2^32]`) on every
   call. Allocation and iteration both use the same wrapped value, so this is
   wrong-results, not OOB — but a `checked_mul` product (or caching at
   construction) would be cheap insurance; `reshape`'s `quick_check` already
   does the checked version (gh-7455).
6. **`axes_iter([])` (empty axes list) panics** (`iterator_axes.rs`, 4 sites):
   `for i in 0..axes_check.len() - 1` underflows `0usize - 1` in release and
   then panics on the slice index — should yield a single whole-tensor view
   (NumPy semantics) or a clean error. Panic, not UB.
   **Resolved 2026-09-16** (rstsr branch `260915-unsafe-soundness-3`, commit
   `58a62c5`): duplicate-check rewritten as an adjacent-pairs `windows(2)`
   scan at all 4 constructor sites; no new code path needed — the existing
   0-d `layout_axes` / full `layout_inner` machinery already yields exactly
   one whole-tensor view, matching NumPy `ndindex()` (product over an empty
   axes set is 1; `()` and `vec![]` both spell "no axes" via
   `From<()> for AxesIndex`). Regression tests `test_axes_iter_empty_axes` /
   `test_axes_iter_mut_empty_axes`; verified under row_major and
   col_major+rayon feature sets.
7. **`uninitialized_vec` family** (`rstsr-common/src/alloc_vec.rs`): the
   `set_len`-on-uninitialized pattern is technically UB until each element is
   written (already documented by the owners); all in-crate callers initialize
   via ops/`assign*` before any read. A `Vec<MaybeUninit<T>>`-based API would
   be the clean fix (partially exists via `uninit_impl`).
   **Status 2026-09-17: remediated.** The family now carries a written safety
   contract (`rstsr-common/src/alloc_vec_contract.md`, rustdoc-included on all
   three functions). The allocating matmul was rerouted through a write-only
   `DeviceMatMulAPI::matmul_uninit` on `MaybeUninit` storage (no `beta`-scaling
   read of uninitialized values; all naive kernels gained a `beta.is_zero()`
   guard); sci-traits distance kernels allocate `Vec<MaybeUninit<M::Out>>` and
   re-view after full initialization. Guarded by `allocatable_dtype.rs` (BigInt
   exact values, DropGuard `created == dropped`) and in-crate cdist tests.
   Follow-up (same day): the last internal generic-`T` `empty` users, `diag` /
   `concatenate` (`creation_from_tensor.rs`), were migrated to `uninit_impl` +
   `assign_uninit` + `assume_init_impl` with **no trait-bound changes**.
   Motivation: `assign` writes with drop-of-old semantics (`*c = a.clone()`),
   so over `empty` storage it dropped uninitialized memory as `T` on *every*
   call for non-POD types (not merely on unwind); `assign_uninit` writes
   (`ci.write(ai.clone())`). rstsr-core non-test code now contains no internal
   `empty` users. Guarded by `test_diag_concat_allocatable_exact` and
   `test_diag_concat_drop_guard`.
   **Resolution 2026-09-16 (owner: Option A — accept and document, not
   migrate)**: written up as ADR-0008 (rstsr-book). The contract
   (`rstsr-common/src/alloc_vec_contract.md`, included via `include_str!` in
   the rustdoc of all three functions) blesses exactly three instantiations:
   `Vec<MaybeUninit<T>>` via `uninit_impl`+`assume_init_impl` for all generic
   code; statically-POD FFI buffers fully defined by the BLAS/LAPACK callee
   (`beta = 0` outputs, workspaces, operand packs; per-file NOTE comments
   were tried and removed the same day — owner prefers the central contract
   only); and by-contract-unspecified
   `empty`/`empty_like` (stay `pub unsafe fn`). Allocatable `Drop` types are
   forbidden as plain `T`; guarded by a new standalone rstsr-core integration
   test (`tests/allocatable_dtype.rs`) running a `BigInt` dtype through the
   safe API surface with exact-value assertions.

## 5. Efficiency notes (not chased)

- `IterLayoutColMajor`/`IterLayoutRowMajor` are ~500-line copy-paste twins; a
  macro or axis-order flag would halve maintenance (same for the
  `device_cpu_serial/operators` vs `feature_rayon/auto_impl` kernel mirrors —
  e.g. the 39-token `op_binary_arithmetic.rs` exists twice).
- `Layout::size()` recomputes the shape product per call (see §4.5); hot loops
  that call it repeatedly (e.g. `reduce_*`) pay a re-multiply each time.
- `Layout::check_strides` allocates two `Vec`s and sorts on every `Layout::new`
  — measurable on many small tensor creations.
  **Status 2026-09-16: fixed** (same commit as §4.2, `6ad5e84`): insertion sort
  into an 8-pair stack buffer (heap fallback for higher dimensionality), no
  allocation in the common case, and checked cumulative-span arithmetic
  (overflow raises `InvalidLayout` instead of wrapping in release mode —
  ties into §4.5). Semantics unchanged. Informal timing of `Layout::new`:
  3-D 33.5 → 8.0 ns/call, 10-D 75.9 → 46.4 ns. Possible follow-up (not done):
  cached-validity flag to skip the double check (`Layout::new` +
  `TensorAny::new_f` both call it) — informal numbers suggest the rewrite
  already removed most of the cost.
- `IterAxesView::next` clones the view struct (storage + device handle) per
  step; fine for axis iteration, but worth knowing before using it inner-loop.
- `cpu_rayon/op_with_func.rs` materializes `c.as_ptr()` **per task**; an
  `AtomicPtr` hoist (the §4.1 fix) also saves one branch-free reload per task.

## 6. Files touched

- `fixes.patch` (6 files): `rstsr-common/src/alloc_vec.rs`,
  `rstsr-core/src/storage/data.rs`, `rstsr-core/src/feature_rayon/par_iter.rs`,
  `rstsr-core/src/tensor/creation.rs`, `rstsr-core/src/tensor/iterator_axes.rs`,
  `rstsr-core/src/tensor/iterator_elem.rs`. Note: `creation.rs`,
  `iterator_axes.rs`, `iterator_elem.rs` and `alloc_vec.rs` mix fix hunks with
  new SAFETY comment hunks (comments immediately adjacent to changed lines);
  everything else in those diffs is comment/test-only.
- `safety-comments.patch` (61 files): comment-only additions across
  rstsr-common (layout machinery, pack_array, flags), rstsr-core (tensor,
  storage, operators, device_cpu_serial, feature_rayon, device_faer),
  rstsr-native-impl (cpu_serial, cpu_rayon), rstsr-dtype-traits.

## 7. Reproduce

```sh
git -C /home/a/rstsr_pack/tmp/snd-unsafe diff > /tmp/t1-all.patch   # 3649 lines
cd /home/a/rstsr_pack/tmp/snd-unsafe
git apply --check --reverse ../rstsr-improve-trajectory/2026-09-14-soundness-check/T1-unsafe-audit/fixes.patch
git apply --check --reverse ../rstsr-improve-trajectory/2026-09-14-soundness-check/T1-unsafe-audit/safety-comments.patch
cargo check --workspace && cargo test -p rstsr-core --release
```

## 8. Full-workspace recheck (2026-09-17)

Recheck base: branch `260915-unsafe-soundness-3` at `31ccac4` (clean tree);
R1/R2 fixes below committed as `fb45e78` on the same branch. The recheck
extended coverage to everything T1 never audited — `rstsr-blas-traits`
(~42 sites), `rstsr-tblis` (2), the five `crates-device/*` (~44 each),
`rstsr-sci-traits` follow-up, plus a pattern re-scan of the T1 crates to
confirm the remediations — via four parallel audit agents with all findings
hand-verified against source. All ten branch commits (b700fc8..31ccac4)
re-verified sound by direct diff review.

### Fixed in the recheck (rstsr commit `fb45e78`, 14 files, +158/−44)

- **R1 — §4.1-class residue in the never-audited crates.** The batched
  broadcast gemm "parallel outer" branch fabricated a per-task full-length
  `&mut [TC]` from `c.as_ptr()` (shared reborrow) in
  `rstsr-core/src/device_faer/matmul.rs` and all five
  `crates-device/*/src/matmul.rs`; the syrk lower-triangle write-back wrote
  through `c.as_ptr().add(idx) as *mut` in all five
  `crates-device/*/src/matmul_impl.rs`. Both reachable from `matmul` and
  `matmul_uninit` on multithreaded pools. Fixed with the §4.1 AtomicPtr
  hoist of `as_mut_ptr()`; the device_faer site carried a **wrong T1-era
  SAFETY comment** (argued disjointness only — provenance was the defect),
  rewritten. Residual caveat: per-task overlapping `&mut` reborrows from one
  unique base are Tree-Borrows-sound / Stacked-Borrows-theoretical — same
  acceptance class as §4.1; full raw-pointer plumbing through the leaf
  staging paths deferred.
- **R2 — §4.2 gate gaps.** Five public write entry points accepted
  broadcast output layouts without the gate:
  `op_mutc_refa_refb` (backs all 11 `*_with_output` ops), the three
  `op_with_func` drivers (`op_mutc_refa_refb_func`, `op_muta_refb_func`,
  `op_muta_func`), and `vecdot_from_f`. Under rayon, stride-0 outputs handed
  overlapping offsets to different threads (data race). All five now carry
  the `is_broadcasted()` assert.
- Tests: broadcast-rejection next to each gate;
  `test_matmul_rule7_broadcast_parallel_outer` (DeviceFaer, 64 batches —
  deterministically crosses the parallel-outer threshold; `matmul` and
  `matmul_from` with beta ≠ 0). Core lib 129 pass (default and `--features
  faer`); workspace check green. Device-crate tests not runnable here (no
  system cblas link) — compile-checked only.

### Open findings (owner review pending — not fixed)

- **R3 — rstsr-blas-traits / rstsr-tblis bugs** (never audited before):
  1. BLAS3 wrappers (`gemm.rs`/`syhemm.rs`/`trsm.rs`) pass allocation-base
     `raw().as_ptr()` ignoring `layout.offset()` and hard-code `ldc/ldb = m`
     — silent wrong results on offset/padded views (in-bounds, no UB; the
     LAPACK wrappers in the same crate do it correctly with offset-aware
     `as_ptr`/`as_mut_ptr` + `ld(order)`). In-tree caller
     (`rstsr-linalg-traits/src/ref_impl_blas.rs`, pinv/trsm) feeds benign
     contiguous inputs, so latent there; the public builder API is affected.
  2. `getri` never validates `ipiv.len() >= n` — OOB read (and possibly
     write via garbage pivots) reachable from the safe builder API.
  3. `getrf` allocates `ipiv[n]` but LAPACK defines only `min(m,n)` entries;
     `ipiv -= 1` reads the uninitialized tail for wide matrices (m < n) —
     violates the alloc_vec contract's "callee fully defines the buffer".
  4. tblis einsum output is written through a shared-derived pointer
     (`einsum_impl.rs` `as_ptr().offset(..) as *mut T`) — same provenance
     class as R1.
  5. Minors: gesvd order selection inverted vs gesdd (forces a transpose
     copy; perf only); gesvd `superb` length `minmn - 1` underflows for
     empty matrices; `usize as blas_int` truncation ≥ 2^31 unguarded;
     blis/aocl/kml `threading.rs` mutate vendor-global thread state without
     synchronization (openblas is Mutex-guarded, mkl thread-local).
- **R4 — notes:** `aligned_uninitialized_vec` (aligned_alloc feature)
  deallocates with `Layout::array::<T>` while the allocation is 64-aligned —
  dealloc-layout mismatch, same class as the faer fix of §4.4 (benign under
  glibc); the `Raw<T> → Raw<MaybeUninit<T>>` handle transmutes assume
  layout identity (true for all current devices — a static assertion would
  pin it); stale `REVIEWME` marker in `op_binary_arithmetic.rs`.

### Trajectory-independence check (owner requirement, 2026-09-17)

rstsr / rstsr-book / rstsr-agents **working trees are clean** — zero
references to this repository or its contents in any file type (including
defensive comments). Git *history* does carry references: `b700fc8`'s
message cites this repo's `2026-09-15-atomicptr-hoist-ab` bench with
numbers, and `6554364`/`1bc2c39`/`215ca02` say "T1 unsafe-audit". Branch
unpushed (rewritable); master's #104 already carries "T1 unsafe-soundness
audit" wording, so the owner's isolation bar evidently tolerates
audit-provenance wording but was not explicitly ruled on — flagged, no
action taken. The new commit `fb45e78` was written without any reference to
this repository.

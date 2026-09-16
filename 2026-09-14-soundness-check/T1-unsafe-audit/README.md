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
| Real bugs | 4 (see §3) | fixed + regression tests |

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

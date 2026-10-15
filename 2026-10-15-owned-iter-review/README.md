# Review of the "view-only iteration API" decision (PR #103, commit 6649b8b)

Task: review whether the 2026-09-15 decision that "we must use view to
produce iterator" (all `iter*` methods defined on `TensorView` only, owned
tensors forced through `.view()`) still stands, or whether it can be
alleviated without giving up the soundness guarantees it bought.

## The decision under review

Commit `6649b8b` (PR #103, 2026-09-15), following T1-unsafe-audit BUG 2:

- **Before**: `iter`/`indexed_iter`/`axes_iter` were defined on a *generic*
  block with a free lifetime `'a` (only bound: `B: 'a`) and `transmute`d the
  internally borrowed `&[T]` to `'a`. On owned tensors nothing tied `'a` to
  the tensor, so `let it = { let t = arange(6); t.iter() };` compiled and
  read freed memory (UAF, empirically verified: `[403706488, 6, 2, 3, 4, 5]`).
- **The fix chosen**: view-only API — `iter*` on `TensorView` carrying the
  view's inner lifetime; `iter_mut*` on `TensorMut` (consuming self); owned
  tensors must call `.view()` first.

## Review verdict

**The decision's premise — "`&Tensor` receivers are unsound" — was never
true; only the transmute was.** Evidence:

1. The audit's *first* fix (2026-09-14 patch) was to tie the iterators to
   `&self` (`IterVecView<'_, …>`) — sound, ndarray-style — and it compiled
   and passed the suite. The view-only step was an *additional* design choice
   (to keep `a.t().iter()` one-liners sound with inner-lifetime iterators),
   not a soundness necessity.
2. With a borrow-tied return, `iter(&self)` on owned/arc/cow is exactly
   `ndarray::ArrayBase::iter` (`fn iter(&self) -> Iter<'_, A, S, D>`) — a
   decade-proven sound design. For owned data `a.view()`'s inner lifetime IS
   the `&self` borrow, so the two designs give identical lifetimes there.
3. E0592 (verified by probe in `probe-e0592/probe.rs`): a generic
   `TensorAny` block cannot carry `iter` alongside the view block. The
   view-only choice avoided the per-alias boilerplate — an ergonomic/
   maintenance tradeoff, again not soundness.

## The alleviation, implemented and tested (branch `261015/owned-iter`)

Per-ownership-alias inherent blocks delegating to `view()`/`view_mut()` —
zero unsafe, zero new types:

- `Tensor::iter` / `indexed_iter` / `axes_iter` / `indexed_axes_iter`
  (`&self`); `Tensor::iter_mut` / `indexed_iter_mut` / `axes_iter_mut` /
  `indexed_axes_iter_mut` (`&mut self`).
- `TensorArc`: same immutable set + `iter_mut`/`axes_iter_mut` (COW:
  `Arc::make_mut` clones when shared — asserted in tests).
- `TensorCow`: immutable set only (DataCow lacks DataMutAPI; `view_mut`
  does not exist on it).

Verification (all on branch `261015/owned-iter`, rstsr @ `e765416`):

| Gate | Result |
|--|--|
| lib tests row-major | 161/161 (2 new tests: elem + axes direct-form/COW) |
| lib tests col-major | 161/161 (new test cfg-branched for order-dependent content) |
| doctests | 356/357 — the 1 failure (`tensor_from_nested`) fails identically on main |
| entry_row_cpu | 657/657 |
| allocatable_dtype / tensor_sum | 10/10, 2/2 |
| clippy `-D warnings` | clean |
| **miri** (all 16 iterator tests) | **green — and it caught a real pre-existing bug (next section)** |
| workspace check + default features | clean |

The UAF pattern stays rejected at compile time, now by borrow checking —
same rule as the anchor's compile_fail doctest on `TensorView::iter`
(`rt::arange(...).view().iter()` on a temporary; the direct form obeys the
identical borrowck rule). The view forms keep their inner-lifetime
semantics (`a.t().iter()` one-liners unchanged).

## Bonus finding: pre-existing UB in the ManuallyDrop path, fixed

Running miri over the iterator tests (the user remembered miri being
involved at the time; T1's miri flags were for the rayon/uninit findings,
not iter) surfaced **real UB on main**, in code PR #103 itself added:

- `test_iter_on_manually_drop_backed_view` fails under miri on **main**
  (verified via `git stash`): `asarray((&mut [T], …))` fabricates its Vec
  from `input.as_ptr()` — a shared reborrow (SharedReadOnly provenance
  under Stacked Borrows) — and `DataMut::into_slice_mut` later re-derives a
  `&mut` slice from it via `from_raw_parts_mut`. Miri:
  *"trying to retag from … for Unique permission … but that tag only
  grants SharedReadOnly"*.
- Fix (in this branch): the two `&mut [T]` sites in `asarray.rs` now use
  `input.as_mut_ptr()` so the fabricated Vec carries the `&mut` provenance.
  The two `&[T]` → `DataRef` sites stay `as_ptr` (shared is correct there).
- After the fix: miri green on all iterator + asarray tests.

This is a strong argument for the review's premise: the view-only decision
did not actually achieve miri-clean iteration on main — a miri run over
these tests was the missing evidence at the time.

## Files

- rstsr (branch `261015/owned-iter`, uncommitted):
  `rstsr-core/src/tensor/iterator_elem.rs`, `iterator_axes.rs`,
  `asarray.rs` (as_mut_ptr provenance fix).
- `probe-e0592/probe.rs`: the E0592 probe (generic block + view block both
  defining `iter` → duplicate definitions).

## Conclusion

The view-only decision is **overturned as a user-facing restriction** (kept
as the lifetime backbone): direct `iter*` on owned/arc/cow is sound, cheap
to provide, and the UAF pattern is still compile-rejected. The view forms
remain the general entry (inner lifetime, `a.t().iter()`). Recommendation:
adopt the alleviation upstream after maintainer review; also land the
as_mut_ptr provenance fix regardless of the iterator decision.

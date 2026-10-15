---
name: rstsr-owned-iter-review
description: 2026-10-15 review of the view-only iteration decision (PR #103) — &Tensor was never unsound; direct owned iter added on 261015/owned-iter; miri caught pre-existing as_ptr provenance UB (fixed)
metadata:
  type: project
---

Review of the 2026-09-15 view-only iteration decision (PR #103, `6649b8b`):

- The UAF (T1 BUG 2) came from the free-lifetime `transmute`, NOT from
  `&Tensor` receivers. The audit's first fix (tie to `&self`) was sound;
  view-only was an additional ergonomics choice (inner-lifetime iterators
  for `a.t().iter()`), never a soundness requirement.
- Alleviation implemented on rstsr branch `261015/owned-iter` (uncommitted):
  per-alias inherent `iter`/`indexed_iter` (+`iter_mut` on Tensor/TensorArc,
  immutable-only on TensorCow — DataCow lacks DataMutAPI) delegating to
  `view()`/`view_mut()`. E0592 forbids a generic TensorAny block (probe in
  the task dir). UAF pattern still compile-rejected (new compile_fail
  doctest on `Tensor::iter`).
- **miri caught real pre-existing UB on main**: `asarray((&mut [T], …))`
  fabricated its Vec via `input.as_ptr()` (SharedReadOnly provenance) while
  `into_slice_mut` re-derives `&mut` — Stacked Borrows violation. Fixed in
  the same branch by `as_mut_ptr()` at the two `&mut [T]` sites; the `&[T]`
  → DataRef sites stay `as_ptr`.
- Gates: lib 161/161 both orders, doctests 356/357 (1 pre-existing main
  failure `tensor_from_nested`), entry_row_cpu 657/657, clippy clean,
  miri green (16 iterator tests).
- The user recalled miri being involved at the time; T1's miri flags were
  about rayon/uninit findings — the iter tests had no miri evidence before
  this review.

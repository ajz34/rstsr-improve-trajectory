---
name: blas-layout-flag-logical-values-pitfall
description: "BLAS test pitfalls (lstsq in-place test, 2026-11-26): asarray with the other layout flag on the same bytes is a different logical matrix; derive off-order copies with to_contig_f(order); allclose raises DeviceMismatch across two DeviceBLAS instances; asarray(&[T]) is borrowed; how to assert LAPACKE write-back parity"
metadata:
  type: feedback
---

Writing `test_lstsq_inplace` cost three red/green cycles on test-side
mistakes, none of them a code bug. Check these first when an off-order or
cross-device comparison is "wrong":

- `rt::asarray((v, [m, n].f(), &dev))` with `v` built in C order stores the
  same bytes column-major, so the logical matrix differs from the
  reference's. Build an off-order copy with `a.to_contig_f(order)`, or
  re-index explicitly (`w[j*rows + i] = v[i*cols + j]`) when the tensor must
  live on another device instance.
- `rt::allclose` on tensors of two `DeviceBLAS` instances (default device vs
  `set_default_order(ColMajor)`) raises `DeviceMismatch`; compare elementwise.
- `rt::asarray((&v[..], ..))` is a borrowed (`DataRef`) tensor; pass
  `v.to_vec()` where an owned `Tensor` is required.
- A binding shadowed by an in-place (destroyed) input is no longer the
  logical reference.
- `duplicate_item` table cells must be bracketed `[..]`, substitution
  expressions included (`duplicate-item-identifier-traps`).

Pattern for LAPACKE-parity of a destroyed input: compare the destroyed `a`
of a row-major in-place call with what a direct column-major call of the
same logical matrix leaves (bit-identical on one LAPACK; red before the
write-back, green after). Do not assert "contents changed" — LAPACK leaves
destroyed contents unspecified.

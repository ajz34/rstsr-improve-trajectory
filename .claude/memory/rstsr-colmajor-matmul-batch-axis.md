---
name: rstsr-colmajor-matmul-batch-axis
description: rstsr matmul axes follow the device default order: last two axes under row-major, first two under column-major, so a batch of [n,k] is [p,n,k] resp. [n,k,p]; into_shape keeps the given shape, only strides follow the order
metadata:
  type: project
---

Under a column-major device the matmul axes are the **first two** axes of an operand and the
batch axes trail (rstsr-common `layout/matmul.rs` module doc; the CUDA `resolve_geom` reverses
all layouts and swaps the operands). A batch of `[n, k]` matrices is `[p, n, k]` on a row-major
device and `[n, k, p]` on a column-major one; the transpose of the right operand is
`transpose([0, 2, 1])` resp. `transpose([1, 0, 2])`.

- `rt::asarray(data, &dev).into_shape(s)` keeps the logical shape `s`; only the strides follow
  the device default order. Building `[p, n, k]` on a column-major device gives a batch of
  `[k, p]` matrices and the matmul is refused with a shape mismatch inside `resolve_geom`.
- Feeding data: position of logical index `(i0, ..)` is row-major under `C`, `i0 + i1*n0 + ...`
  under `F`.
- The CPU serial naive matmul refuses column-major inputs (`sc[0] != sa[0]`): build references
  row-major and compare through `to_contig(RowMajor)`.
- Rank-0 shape checks: `assert!(shape().is_empty())`; `assert_eq!(shape(), &[][..])` does not
  infer (rust-analyzer E0284).
- Cost me several iterations in `2026-11-26-cuda-matmul-forms` B2 (`B2-REPORT.md`).
- Recorded in rstsr-agents `rules/code-concepts.md` and the `rstsr-cuda` matmul module doc
  (uncommitted edits as of 2026-11-26).

---
name: cuda-assign-copy-semantics
description: rstsr assign semantics = positional zip in the device default order; CUDA copy kernels must decompose each side by its own shape, and the linear-copy fast path must gate on dtype+shape+order (equal-size dtypes are not memcpy-safe)
metadata:
  type: project
---

`assign_arbitary` on any rstsr device means: if both layouts are
contiguous in the device's default order, zip the buffers linearly;
otherwise zip the two layouts' iteration sequences **position by
position** in the device's default order. Shapes and ranks may differ
(reshape-copy, flattening a transposed view), so on CUDA each operand
must be decomposed by its *own* shape (`rstsr_iter_offset`), not by the
output shape.

The linear `memcpy_dtod` fast path is valid only when: same dtype
(`TypeId`, not equal `size_of`), same shape, both layouts contiguous in
the device's default order, and full buffers with zero offsets.
`i64→f64`, `i32↔f32`, `bool↔i8` are equal-size and silently produce
denormals/garbage if byte-copied.

**Why:** the first kernel round decomposed sources by the output shape,
which is correct for same-shape elementwise ops but reads out of bounds
for rank-changing assigns; the memcpy gate checked element size instead
of dtype identity.

**How to apply:** any new copy/assign/cast kernel keeps the per-side
shape+order decomposition; any new memcpy shortcut must include the
TypeId and shape equality checks. Regression tests live in
`crates-device/rstsr-cuda/tests/test_kernels.rs`
(`assign_reshape_*`, `assign_same_size_cast_pairs`). Related:
[[cuda-write-target-ordering]], [[cuda-kernel-nvrtc-grill]].

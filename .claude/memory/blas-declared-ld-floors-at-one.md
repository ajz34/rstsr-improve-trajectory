---
name: blas-declared-ld-floors-at-one
 description: cuBLAS rejects ld = 0 even for an empty matrix (ld >= max(1, rows)) - rstsr-cuda floors every declared leading dimension at 1 and refuses strides over i32::MAX instead of clamping (C5/C6)
metadata:
  type: project
---

From the 2026-11-26 review C5/C6, fixed on 261021/cuda (uncommitted):

- cuBLAS validates `ld >= max(1, rows)` **including empty matrices**: the `[0] @ [0]` dot lifts to
  a `1 x 0` operand whose natural `ld(0, 0) = 0` and dies with `INVALID_VALUE` ("GEMM_EX
  parameter number 12/10") on `cublasGemmEx` and the typed entries (cuBLASLt happens to tolerate
  0, which masked it on the default chain). The floor applies to every backend of this shape —
  expect the same rule in the CPU BLAS devices when their matmul derivations meet degenerate
  extents.
- The other half of the same rule: a stride that does not fit `i32` must be **refused**, never
  clamped to `i32::MAX` — the clamp addresses the wrong elements silently. Both are one guard
  line in `cublas_args::args` and `syrk::args` (`.max(1)` in `ld`, `stride > i32::MAX as isize`
  in the loop).
- Regression tests live at the unit level for the unreachable half (an extent-2 axis over 2^31
  elements is an 8.6 GiB `f16` operand — no GPU here can hold it): pure derivations need no
  hardware, see `empty_dots_floor_and_wide_strides_refuse` in both files.

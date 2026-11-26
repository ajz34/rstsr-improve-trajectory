---
name: cuda-matmul-broadcast-axis-wrong
 description: rstsr-cuda matmul (261021/cuda @ 67a0c79) gives silently wrong results for an operand broadcast along a matmul axis on all cuBLAS drivers and the syrk form; root is cublas_args (zero stride not refused, ld clamped up)
metadata:
  type: project
---

Found in the 2026-11-26 review (`2026-11-26-cuda-matmul-review/REVIEW.md`, C1), reproduced on the RTX 5070 Laptop:

- `b = asarray(([1.,2.,3.,4.], [1,4])).broadcast_to([3,4])`, `a.broadcast_to([3,2])` etc. (stride 0, extent > 1)
  -> every element wrong on cuBLASLt, cublasGemmEx, classic cuBLAS and the default chain, f64 and c64,
  both default orders; tiled and naive are correct but the chain never reaches them. Also `a @ a.t()` with
  `a` a column broadcast along k (syrk path): 1369/1369 wrong.
- Root: `src/matmul/cublas_args.rs` `args()` refuses only `stride < 0 && extent > 1` (comment: "an extent-1
  axis may legitimately carry stride 0"), and `ld()` = `stride.max(0).max(rows)`: lda = rows for a zero stride,
  cuBLAS reads out of bounds. Same clamp copied into `drivers/syrk.rs` `args()`.
- Pre-existing in `cublas_args.rs` (classic driver only at `c75aecd`); the new drivers inherit it through
  `cublas::prepare`. `matmul/mod.rs` promises zero strides are served.
- Fix: refuse `stride == 0 && extent > 1` in both `args()`; add the broadcast shapes to the layout matrix per
  cuBLAS driver (no test had them).
- Related low ones from the same probe: `ld()` yields 0 for `[0] @ [0]` (cuBLAS INVALID_VALUE on cublasEx and
  classic alone; chain fine via Lt) -> needs `.max(1)`; strides over i32::MAX are clamped, not refused.

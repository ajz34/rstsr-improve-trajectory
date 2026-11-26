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
- **FIXED 2026-11-26**, uncommitted on rstsr `261021/cuda` (9 files, +214/-23): both `args()`
  refuse `stride <= 0` with extent > 1 (`syrk.rs` keeps its unconditional negative-stride refusal
  and adds the zero-stride loop). Chain serves the shapes through tiled/naive; forced cuBLAS
  drivers refuse with "no enabled driver can serve". Regression: unit tests in `cublas_args.rs`
  and `syrk.rs` (`broadcast_matmul_axis_is_refused`) and hw `test_matmul_modes.rs::
  broadcast_along_matmul_axis` (chain correctness + forced refusal + the syrk form on a broadcast
  operand). Suite green; probe rerun 0 FAIL.

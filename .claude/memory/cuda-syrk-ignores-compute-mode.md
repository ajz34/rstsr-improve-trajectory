---
name: cuda-syrk-ignores-compute-mode
 description: rstsr-cuda `a @ a.t()` syrk form (261021/cuda) dispatches on plan type only, so it ignores the compute mode - 3.3x slower than gemm under EmulatedFixedPoint, Pedantic unapplied, mantissa-bits error bypassed
metadata:
  type: project
---

Found in the 2026-11-26 review (`2026-11-26-cuda-matmul-review/REVIEW.md`, C3), measured n = k = 1024 f64:

| flags | `a @ b` | `a @ a.t()` |
|---|---|---|
| `EmulatedFixedPoint`, `MatMulUseSyrk::Scientific` | 1.044 ms | 3.449 ms |
| `EmulatedFixedPoint`, `Disallow` | 1.045 ms | 1.010 ms |
| `Default` / `Pedantic`, `Scientific` | 6.419 ms | 3.448 ms |

- `syrk::run` matches `CublasPlan::F32(_) | F64(_)` only; the classic gemm path sets and restores the handle
  math mode, the syrk path does not, so native `Dsyrk` replaces an emulated gemm.
- `fixedpoint_mantissa_bits != 0`: `a @ b` returns the documented error, `a @ a.t()` returns a result.
- Fix: take the form only when the resolved mode is `Default` (and mantissa bits are 0), or set/restore the
  math mode inside the syrk call as `cublas::launch` does.

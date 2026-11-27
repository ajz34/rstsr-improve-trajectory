---
name: cuda-syrk-ignores-compute-mode
 description: FIXED on 261021/cuda (uncommitted) - cuBLAS ?syrk entries ignore a math mode that changes the arithmetic (TF32 syrk measures plain f32); rstsr-cuda now declines the form for those modes and keeps it for Default/Pedantic with the mode set/restored
metadata:
  type: project
---

Found in the 2026-11-26 review (`2026-11-26-cuda-matmul-review/REVIEW.md`, C3), fixed 2026-11-27.
Key measurement that decided the fix shape: **cuBLAS's `?syrk` entries do not apply a math mode
that changes the arithmetic** — `cublasSsyrk_v2` under `CUBLAS_TF32_TENSOR_OP_MATH` measures
`5.7e-7` max rel err (plain `f32`, identical to Default) while the classic `gemm` under the same
mode measures the TF32 scale. So "just set/restore the mode around the syrk" cannot honor a
down-converting mode.

Fix as implemented:

- `cublas.rs` exposes `Ask`/`ask_32f`/`ask_64f`/`bound` as `pub(super)`; the set/restore dance of
  `launch` moved into `pub(super) with_math_mode(blas, math, call)` used by both drivers.
- `syrk::run` computes the same ask (mantissa bits included, so the documented error fires
  identically from the form), then takes the form only for `Ask::Leave` or `Math(Pedantic)`
  (typed `?syrk` is exact arithmetic; Pedantic form 3.46 ms vs pedantic gemm 6.44 ms — keep it);
  other modes decline, the chain serves them on a gemm.
- Post-fix T3 (`results/t3_after_c3c4.txt`): EmulatedFixedPoint `a @ a.t()` 1.013 ms (was 3.449);
  `fixedpoint_mantissa_bits = 30` errors identically on `a @ b` and `a @ a.t()` (was silent).
- Regression: `syrk_honors_the_compute_mode` in `tests/test_matmul_modes.rs` (mantissa parity +
  forced-classic `a @ a.t()` under FastTF32 must measure TF32-level error, `> 1e-5`).

If a refactor ever extracts the syrk mirror out of `typed`, remember the mode is set only around
the syrk call, not the mirror (the mirror is an elementwise copy, mode-agnostic).

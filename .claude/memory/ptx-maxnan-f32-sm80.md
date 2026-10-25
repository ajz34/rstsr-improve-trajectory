---
name: ptx-maxnan-f32-sm80
description: "PTX min/max `.NaN` exists only for .f32 and requires .target sm_80+; f64 has no `.NaN` form, and unused device helpers are eliminated before ptxas."
metadata:
  type: reference
---

Verified 2026-10-25 on the RTX 5070 (sm_120, CUDA 13.x) while reviewing
rstsr-cuda:

- `max.NaN.f32` / `min.NaN.f32` need `.target sm_80` or higher. NVRTC
  happily emits them for `compute_75`; `ptxas -arch=sm_75` then fails with
  `Modifier '.NaN' requires .target sm_80 or higher`, i.e. the failure shows
  up at module load (driver JIT) on pre-Ampere devices.
- There is no `.NaN` form for `.f64`: `max.NaN.f64` is rejected by the
  driver JIT with `CUDA_ERROR_INVALID_PTX`, independent of the arch.
- `__device__` helpers that are never called are eliminated from NVRTC's
  PTX, so including an sm_80-only helper in a shared prelude does not break
  unrelated kernels; only kernels that actually call it fail.

**How to apply:** if an NaN-propagating f32 min/max must run on pre-Ampere,
use the select chain (`isnan` tests around plain `fmax`/`fmin`) or gate the
inline asm on `__CUDA_ARCH__ >= 800`; do not expect an f64 `.NaN` variant.

Related: [[cuda-kernel-nvrtc-grill]]

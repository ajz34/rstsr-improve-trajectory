---
name: syrk-f32-gate-thin-operands
description: cuBLAS f32 Ssyrk loses up to 2.3x against gemm on thin operands (k << n); a gate measured on square k=n operands is not enough; f64/c64 lose below n=256 because of the extra mirror launch
metadata:
  type: feedback
---

The `a @ a.t()` syrk form in rstsr-cuda was first gated on `n >= 2048` for `f32`, from a probe that
only timed square operands (`k = n`). The benchmark's `[4096, 64]` cell then showed the form 1.7x
slower than the gemm; a `(n, k)` sweep gave losses up to 2.3x.

- Settled by the maintainer: no size gate. `MatMulUseSyrk { Allow, Scientific, Disallow }` on
  `CudaRuntimeFlags` (`matmul_use_syrk`, default `Scientific` = `f64`/`c32`/`c64`; `Allow` adds
  `f32`). The maintainer prefers a type switch over measured thresholds.
- Lesson: when a form's win depends on arithmetic intensity, sweep **both** problem dimensions
  (thin and square), and sweep the small end too, before fixing a threshold; an extra launch shows
  up only at the small end.
- Sweep sources: `syrk-gate` part of `2026-11-26-cuda-matmul-forms/src/probe_forms.rs`
  (`SMALL=1` for the small end); tables in `results/b3_syrk_gate*.txt`.

---
name: rstsr-multithreaded-first-priority
description: Owner priority (2026-09-21) — rstsr is usually used multi-threaded; serial-only kernel improvements are not the main aim. Judge efficiency work on the rayon/faer path first.
metadata:
  type: feedback
---

At the variant-V wrap-up (2026-09-21) the owner said: "I guess that we
usually use this crate in multi-threading case, and serial-only improvement
is not the most aim."

**Why**: rstsr's realistic usage (and the default faer feature, which implies
rayon) runs the `DeviceFaer`/rayon paths; a serial-kernel win only matters
to single-threaded users or as the per-chunk inner loop of the rayon fold.

**How to apply**:
- When proposing or prioritizing efficiency experiments, target
  `cpu_rayon/*` kernels and the `feature_rayon`/`device_faer` wiring first,
  or serial changes that feed the rayon per-chunk fold.
- Benchmark the faer16 (multi-thread) cells as the headline, serial as
  secondary; use paired interleaved A/B and expect ±10–15% rayon lottery
  (only same-pass ratios are meaningful).
- Serial-only patches (like the T2' band walk) need explicit framing as
  such before investing integration effort.

Related: [[rstsr-sum-axis0-sequential-order]], [[rstsr-reductions-integration]].

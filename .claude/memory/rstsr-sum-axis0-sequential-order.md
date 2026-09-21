---
name: rstsr-sum-axis0-sequential-order
description: FLAGGED future experiment — reduce_axes branch-2 sequential-order walk (numpy-style row-outer) has ~2x headroom over variant V on large cells (numpy 0.47ms vs rstsr 1.17ms @2048² f64), but changes FP accumulation order → needs semantics decision first.
metadata:
  type: project
---

Flagged 2026-09-21 by the owner at variant-V integration ("flag this issue"):

**Opportunity**: `reduce_axes_cpu_serial` branch 2 (contiguous REMAINING
axis, e.g. `sum_axis0` row-major) still walks chunk-band OUTER / row INNER —
every 384 B slab (48-lane strip) jumps 16 KiB, so the HW prefetcher sees ~43
stride-streams per pass. NumPy accumulates row-outer full-width-inner (one
sequential stream) and reaches **0.47 ms** on 2048² f64 (~68 GB/s effective,
warm L3-class regime) vs rstsr variant V **1.17 ms** (~27 GB/s). Arithmetic
is ~6% of runtime; this is prefetch/strip-transition bound. A sequential
walk could plausibly recover ~2x on large cells (criterion numbers may
differ — my numpy spot-check was a separate warm process; validate in-criterion).

**Blocker**: swapping loop order changes `f_sum` fold order → floating-point
sums round differently (not bit-identical to current). min/max are
order-insensitive (values unchanged), sum/mean/var are not. This is a
SEMANTICS decision, not a kernel tweak — needs its own gate design (like the
T2' CURRENT-BEHAVIOR locks, but flipped deliberately) and a bench matrix.

**Design family**: same space as T1'/T4' blocked 2-D iteration (into_cast /
op_with_func). Note the faer16 (rayon) path is unaffected by serial kernel
order unless the rayon twin gets the same treatment — and see
[[rstsr-multithreaded-first-priority]]: owner weighs the multi-threaded path
first, so a future attempt should consider whether the rayon
`reduce_axes_cpu_rayon` branch-2 twin (CHUNK=64 iterator walk, untouched
since the phase-2 revert) is the better first target.

**Context**: variant V (band walk, order-preserving) captured the iterator
pathology (~1.5x); the sequential-order rewrite is the NEXT layer, worth
~2x more on large cells if the semantics gate can be cleared. Evidence:
`2026-09-09-reductions/results/integration260921/SUMMARY.md`.

---
name: rstsr-layout-size-cache-t1
description: T1 §4.5 Layout::size() cache question answered NO by MWE bench — recompute is sub-ns and rstsr calls size() once per op; doc "cached size" claim is false, fix the doc only
metadata:
  type: project
---

T1 audit §4.5 (`Layout::size()` recompute vs cached field) resolved by MWE
bench, 2026-09-16, in `2026-09-16-layout-size-cache-mwe/`: **do not cache**.

- Raw recompute cost: 0.36–0.68 ns/call (heap `Vec` shape, ndim 2–8);
  ≈0 ns for inline `[usize; N]` (hidden under loop floor ~0.18 ns). Cached
  load = floor everywhere.
- Grounding: every live `.size()` site in rstsr (grep at aa24643,
  cpu_serial/cpu_rayon/rstsr-core tensor files) is once-per-op; none per-task,
  per-step, or per-element. Measured kernel at that frequency: parity.
- Cached-field costs: +8 B per stored layout (56→64 Vec, 40/72→48/80 arr;
  `Option<Layout>` free via Vec's NonNull niche), +1.3–2.7% clone paths,
  construction ~1–2 ns SLOWER with ±6–9% transients (one product either way —
  caching moves it, then adds a field).
- Per-element `size()` callers (the only regime that pays, −33% bench) don't
  exist; if one appears, hoist `let size = l.size()` locally — layouts are
  immutable during kernels, so hoisting is always sound.
- Free follow-up kept: `Layout::size` doc says "uses cached size" — false,
  one-line doc fix in rstsr still pending. Unchecked product wrap (§4.5's
  paranoia half) is orthogonal: `checked_mul` recompute keeps struct unchanged.

Read this before proposing layout struct changes or size()-related opts.
Related: [[rstsr-soundness-t1-unsafe-audit]], [[rstsr-broadcast-write-gates-t1]].

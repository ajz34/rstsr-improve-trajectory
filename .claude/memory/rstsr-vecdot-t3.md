---
name: rstsr-vecdot-t3
description: T3' vecdot/inner_dot phase-1 findings at 386948be (branch anatomy correction, inner_dot % surface discovery, probe anchors, dispatch_simd/complex verdict, patch disjointness)
metadata:
  type: project
---

T3' (2026-09-09-vecdot) phase 1, base 386948be, no rstsr edits. Key facts:

- **T0/code-map branch mislabel CORRECTED**: batched `(m,k)·(m,k)` ax-1 hits
  vecdot **branch 1** (contiguous-SUMMED, `unrolled_binary_reduce`, one write
  per output — no RMW). The MaybeUninit-RMW branch 2 serves **axis-0
  contraction** (`vecdot(&a,&b,0)`), costing 962–1614 µs serial vs 444–465 µs
  for am1 on the same 2.1M-element data (native/portable). Branch probe by
  replicating vecdot.rs:44-68 with rstsr-common layout API
  (get_axes_composition + dim_split_axes; needs direct rstsr-common dep, not
  in rstsr-core prelude).
- **Per-row layout-dispatch machinery is the serial am1 cost**: branch-1 body
  alone (probe) runs 0.292–0.350 ms vs rstsr 0.444–0.465 ms (same math, plain
  slices). ~40 ns/row × 4096 rows. Same lesson as T2' (IterLayoutColMajor::next
  cost) — third confirmation.
- **inner_dot surface**: ONLY `%` on 1-D·1-D (rule (1,1,0) in
  device_faer/matmul.rs:114 / matmul_naive_cpu_serial). Under DeviceFaer it
  NEVER reaches faer for any dtype; alpha=1/beta=0 always on the `%` surface.
  inner_dot_naive_cpu_rayon has NO PARALLEL_SWITCH (10–30 µs rayon overhead
  at small n). Serial `%` = instruction flood (12.25 ins/elem, 4.1 ms @1e7 —
  slower than rt::vecdot's 2.9 ms). Latent hazard: both twins read
  `beta*c` from a fresh UNINIT output (0×NaN-bits → NaN); usually benign
  because pages are zero-filled. Flagged, not fixed.
- **Phase-2 design (in PLAN.md)**: A1 flat branch-1 path, A2 L1-budget local
  vacc for branch-2 (32 KiB byte budget), B1 serial inner_dot fast path under
  `stride1 && alpha==1 && beta==0` (8-lane reassociation accepted — `%` has
  NO cross-device bit-exact contract today: faer already re-associates),
  B2 rayon chunks + PARALLEL_SWITCH(512). Sequential folds are
  chain-latency-bound: probe D1 (plain zip fold) = D0 (current) = 4.0 ms;
  only 8-lane helps (2.8 ms).
- **dispatch_simd verdict #3**: SKIP (after T2' reductions and T6 argmax).
  Slice+ubr shapes beat `[T;8]` index-arithmetic lane arrays (probe A0 0.35/0.29
  vs A2 0.43/0.43 ms). **Complex decision (D6)**: complex NOT in v1 — c64 cost
  is arithmetic not dispatch; c64 fold doesn't autovectorize (A5); c64 lane
  array doesn't help (A6 ≈ A5, worse native); real c64 kernel would need
  interleaved-re/im intrinsics-class work.
- **Strided vecdot honest negative**: gather-bound (11.8 GB/s), 8-lane no-op,
  rstsr already 8–26× faster than numpy einsum 'ij,ji->i' (25–78 ms). Leave
  branch 3 alone.
- **Patch disjointness**: T3' files (cpu_{serial,rayon}/vecdot.rs +
  cpu_{serial,rayon}/matmul_naive.rs) overlap ZERO with the four existing
  patches (argmax, reductions, transpose-assign, elementwise);
  unrolled_binary_reduce untouched by all of them (grep-verified).
- faer16 criterion cells (axis0, am1-c64) have a ±8% transient band; serial
  cells ±2.7%. Gate faer16 on repeated runs.

**PHASE 2 DONE, D3 PASS** (2026-09-09-vecdot/proposed.patch, tree restored):
A1 flat branch-1 path + A2 32-KiB-vacc branch-2 (both bit-exact, fall-through
unchanged) → batched am1 serial −22% both cfgs (363/352 µs), axis0 serial
−64/−24% (538 µs), small −43%, odd −16..20%; B1 (serial % fast path, bounds
widened TC: Zero+One+PartialEq — PR-note like T6 ArgCmp) → % 1e7 serial
4.07→2.92 ms (−29%); B2 (PARALLEL_SWITCH 512 + par_chunks(8192) 8-lane, no
bound change) → % faer16 1e4–1e6 −73..−96%, small-n 10.4→0.41 µs; faer16 %
1e7 wall −1.3% ONLY (DRAM-bound, honest miss; core-time ins/el 11.76→2.95).
Gates: 1-D f64 ±2%, faer16 am1 improved, strided untouched path within band.
Now at numpy-einsum parity (6.5% of 32-FLOP/cyc peak); serial am1 still
~2.5× behind faer16 wall = per-core load bandwidth, not dispatch.
**Machine lesson**: single full-suite criterion passes carry ±5–8% faer16
and up to +67% single-cell codegen/thermal transients (f32 1e7, c64 cells
flipped between clean builds) — ALWAYS re-run suspect cells 3× before
believing a regression; medians of trials decided every gate here.

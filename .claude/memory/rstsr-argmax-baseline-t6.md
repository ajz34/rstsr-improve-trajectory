---
name: rstsr-argmax-baseline-t6
description: T6 argmax/argmin COMPLETE (2026-09-09, 386948be) — baseline 93 ms @1e7 serial → candidate 1.5-1.8 ms (50-60x), 280→3.7 ins/elem, D3 PASS, proposed.patch ready; API break + NaN-lane lessons inside
metadata: type: project
---

T6 argmax/argmin (2026-09-09, 386948be, dir `2026-09-09-argmax-argmin/`;
phase 1 baseline + phase 2 candidate COMPLETE, proposed.patch produced,
D3 PASS, ../rstsr reset clean):

Baseline → candidate (criterion medians):
- 1e7 f64 argmax serial: 92.6 ms → 1.82 ms portable (50.8x), 93.7 → 1.55 ms
  native (60.3x). faer16: 11.9 → 308 µs (38.6x), 12.4 → 251 µs (49.6x).
  argmin ≈ same. f32 up to 97x. 2-D whole 2048²: 50-79x.
- perf: 280.3 → 3.7 ins/elem, 51.5 → 0.9 cyc/elem, 0.85 → ~49 GB/s
  (L3-assisted; L1d-miss 0.1% → 12.1% = now memory-exposed). numpy 1.34 ms
  context: candidate serial 1.55 ms is within 16% of numpy at 1e7.
- Gates: small 1-D/2-D 5-73x faster; strided t-view (untouched fallback)
  paired same-session A/B ≈ +1-1.7% worst (within D3 ≤3%); fast path held
  `#[inline(never)]` because inlining it cost the fallback +2.3-3.8% native
  (codegen layout sensitivity of the closure fold).
- rstsr-native-impl API break: (Fcomp, Feq) closures → `ArgCmp{Min,Max}` enum
  on the reduce_*_arg_* pub fns (tensor-level API unchanged) — flag in PR.
- Gotcha for correctness tests: broadcast-row argmax does NOT return 0 unless
  data is all-equal — the max sits in row 0 at the row's max column (duplicates
  below are equal, never strictly greater). My first gate draft got this wrong.

---
name: rstsr-transpose-assign-t1
description: T1' phase-1 findings — transpose B-variant 108 ins/elem iterator pathology; dormant blocked kernel directly measured 2.7x serial / 2.2x rayon; loop-orientation swap is 3x worse; plan = route 2-D orderchange inside both assign families; ndarray t().to_owned() preserves f-order.
metadata:
  type: project
---

T1' transpose-assign (2026-09-09, dir `2026-09-09-transpose-assign`, rstsr
386948be). PHASE 1 ONLY (baseline + probes + PLAN.md); rstsr tree untouched;
phase 2 awaits G1.

- **Baseline (2048² f64)**: A idiom 23.2 ms serial native / 3.33 faer16;
  B reuse `c.assign(&a.t())` 17.2 / 1.45; C=A+MALLOC recovers rider
  (26% serial / 56% faer16 — T7 reproduced). perf B serial: 107.8 ins/elem,
  22.4 cyc/elem, IPC 4.8, 2.9% L1d-miss, ~0 faults. Gaps vs ndarray
  `.t().to_owned()`: 52× small 64², 29× medium, 33× odd, ~4× large.
- **Dormant kernel measured directly** (benches/kernels_probe.rs calls
  `orderchange_out_c2r/r2c_ix2_cpu_*` on raw slices+Layouts — kernels are
  `pub` via `rstsr_core::prelude_dev`): 5.70 ms serial native vs 15.24
  generic (2.7×), 0.542 rayon16 vs 1.18 (2.2×), odd 1000×777 0.296 ms/97.9 µs.
  **Loop-orientation probes: in-tree orientation (strided reads, contiguous
  write-combined stores inner) is RIGHT — swapped = 3.1× slower.**
  Raw-pointer stores vs bounds checks: ≈0-5%, not a lever.
- **Two assign kernel families**: `assign_arbitary_*` (A path,
  change_layout_f, C-order translate) vs `assign_*` (B path, tensor assign,
  K-order greedy) — phase-2 patch must touch BOTH (both in
  `cpu_{serial,rayon}/assignment.rs`) or B stays slow. Wiring costs ~1.9 ms
  on B (17.16 tensor vs 15.24 bare kernel).
- **Kernel guards admit**: negative slow-axis strides (flip views — direct
  test PASS), zero slow-axis stride (broadcast row re-read, correct);
  reject: non-+1 fast axis (zero/negative/stride-2), ndim≠2, shape mismatch.
  r2c/c2r orientation: `a.t().to_contig(RowMajor)` is the c2r case
  (la stride [1,n], lc stride [m,1]; ldc = row count — hand-building [1,n]
  output layout is INVALID, overlaps).
- **Plan (PLAN.md Option A)**: guard in both families' else-branch, ndim==2 +
  fast-axis strides exactly +1, no size floor (serial), keep rayon 16·64²
  fallback. Expected B: 17.2→~7-8 ms (wiring included), faer16 1.45→~0.8.
  dispatch_simd skipped (§5; gather pattern). Elementwise patch interaction:
  DISJOINT (T4' = op_with_func.rs; T1' = assignment.rs[/transpose.rs]) —
  patches compose in any order.
- **ndarray 0.16 quirk**: `a.t().to_owned()` PRESERVES f-order layout
  (`is_standard_layout()==false`, `as_slice()==None`, content = logical
  transpose) — compare via `.iter()`, not `as_slice().unwrap()`.
- Fall-through canaries baselined in `assign_gates`: sliced fast-axis-2
  assign large 6.40/8.75 ms serial native (sliced / sliced_t), contig assign
  large 1.37 ms, small 1.04 µs — phase-2 gates.

PHASE 2 COMPLETE (2026-09-09, post-G1). D3 PASS; proposed.patch captured
(4 files: cpu_{serial,rayon}/{assignment,transpose}.rs, +434/-28); rstsr
restored clean at 386948b; `git apply --check` verified.

- **Dtype decision (G1 item 1)**: chose `into_cast()` generalization over
  TypeId guard — identity-inlined for same dtype (same contract as the
  existing promote assign families), keeps the open dtype set fast-pathed,
  no unsafe transmute; cross-dtype casts route correctly as a bonus.
- **Result (B reuse, primary)**: large 2.86×/2.94× serial (nat/por),
  2.65×/2.68× faer16; odd ~9.9×; oddT ~8-8.8× serial; medium 2.6-3.8×;
  small 64² 6.2-6.5× (14.1→2.2 µs); A large 2.2×; f32 2.9×; r2c orientation
  B 2.9×; bcast slow-axis-0 B 8.4× serial. perf B serial: 107.8→13.2
  ins/elem, 4.0→11.1 GB/s, L1d-miss 2.9%→49.9% (instruction mountain →
  memory-bound = correct regime).
- **Wiring detail that mattered**: fully-contig pairs in family 2 never see
  the guard — `translate_to_col_major(K)` + greedy normalizes them into the
  contig branch (ndim_of_f_contig analysis). Gate canaries: contig/sliced
  all within noise (two >3% single-suite cells dissected with re-runs:
  sliced_t faer16-nat +6.1% not reproduced; contig faer16-por mean +1.4%,
  no mechanism).
- **Gotcha**: ndarray `.t().to_owned()` keeps f-order (as_slice None) —
  compare via `.iter()`; f-contig [m,n] storage order is out[i + j*m] =
  a[i][j], i.e. want[p] = av[(p%m)*n + p/m] (NOT av).
- criterion filter substring gotcha: "contig_large" also matches
  "to_fcontig_large" ids (regex contains-match).

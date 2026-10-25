# GRILL R1 — 03-elementwise: scope and benchmark guidelines

Round-1 questions for the owner, asked 2026-10-23 before PLAN.md is
written. Context established by the grounding pass (same day):

- Baseline branch `261023/cuda-review-fixes` @ `b44cbef` (rounds
  01-device-basic + 02-kernel-nvrtc + review fixes).
- Implemented on CUDA today: add/sub/mul/div (+scalars), 6 comparisons,
  21 float transcendentals, assign/fill/creation/format, all via the
  runtime-strided flat kernel (broadcast = stride 0).
- Missing vs the CPU surface: rem, bitwise (bitor/bitand/bitxor/shl/shr,
  incl. bool), 10 compound assigns, promoting binary family (atan2,
  copysign, floor_divide, hypot, log_add_exp, pow, maximum, minimum,
  nextafter, ext_*), non-transcendental unary (abs, sign, square, floor,
  ceil, round, trunc, neg, not, positive, ext_neg, conj, real, imag,
  is_nan/is_inf/is_finite, signbit), where, clip, *_with_output forms.
- Perf snapshot (f64, RTX 5070 Laptop, this day): contig add 1.01x cupy,
  sin 1.01x, to_contig(T) 1.3x, fill 1.6x, arange 1.0x — but
  A + B.T (1000x1000) is 2.3x cupy (69 vs 30 us), already past the 2x
  hard gate; the D9 T1/T2 ladder + dim-squash + fast divmod is the
  intended cure (agent's technical domain).

## Q1 — dtype coverage for the op families

Complex transcendentals (csin/cexp/... on c32/c64) need an own complex
libm prelude in kernel source. Options: (a) real dtypes complete
(bool/ints/f16/bf16/f32/f64) with complex keeping arith+assign only;
(b) full complex incl. transcendentals this round; (c) f32/f64 + ints
only. Agent recommendation: (a).

## Q2 — benchmark matrix breadth (all f64)

The five named scenarios (A+B.T, A+A.T, 3-D swapaxes, outer-cross,
both broadcast variants, sliced windows) plus contig unary/binary,
assign/reshape-of-strided, compound-assign. At how many sizes?
(a) ~20 cells: each scenario at two working-set sizes (~8 MB and
~48 MB) + one contig-add size ladder 1e3..1.6e7 for the
bandwidth/overhead curve; (b) lean ~10 cells, one canonical size each;
(c) full sweep, 3+ sizes per scenario. Agent recommendation: (a).

## Q3 — small-tensor host-overhead cells

Sizes 1e3-1e5 elements are enqueue/launch-bound (per-op host cost),
a real workload class but not in the named focus list. Gate them like
the rest, report informationally, or skip? Agent recommendation:
informational appendix (measured, outside the 1.2x/2x gates).

## Q4 — measurement protocol on the shared GPU

The GPU also runs the desktop session (~12% util during grounding).
(a) median-of-3 repeats, caveat documented, no user action;
(b) same, plus owner provides an idle window for the final tables;
(c) additionally attempt clock locking (nvidia-smi -lgc, needs sudo;
fallback (a)). Agent recommendation: (a).

## Settled without asking (recorded for the plan)

- Same branch `261023/cuda-review-fixes`; no auto git commit; no PR.
- Report + baselines live here (03-elementwise-kernel/); the in-repo
  bench (`rstsr-cuda/benches/`) stays free of library comparisons.
- cupy 14.2.0 (conda `ruaa-dev`) and torch 2.13.0+cu132 (`rest-torch`)
  as baselines; numpy reference for correctness where convenient.
- Gates: per-cell rstsr/cupy <= 1.2x target, <= 2x hard with a written
  investigation (what was tried, why it did not close the gap); geomean
  reported alongside.
- Type promotion (ext_*, int-sin -> f64, mixed where) implemented and
  spot-tested, not benchmarked.
- No rstsr-core changes unless a blocker is found; surfaced to the
  owner before any core edit.
- kernels_ops.rs split into a kernels/ directory (per-file size limit);
  code size otherwise unrestricted this round.
- CPU-side improvement suggestions go in the report (not implemented).

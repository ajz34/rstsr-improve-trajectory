---
name: rstsr-nanarg-semantics
description: nanargmin/nanargmax shipped (free); NumPy first-NaN-wins for plain argmin/argmax rejected by measurement — do not re-attempt without reading this.
metadata:
  type: project
---

2026-09-11, follow-up to [[rstsr-argmax-integration]]. `nanargmin`/`nanargmax`
landed in `../rstsr` commit `9c42b1f` (branch `260910-core-efficiency`, local):
NumPy nanarg semantics (skip NaN anywhere; all-NaN slice → `InvalidValue`
"All-NaN slice encountered"), `ArgCmp::{NanMin,NanMax}` kernels, rt:: families,
OpNanArg*API traits. **Free on NaN-free input**: the seed loop exits at element
0, then the plain 8-lane scan runs unchanged — nanarg needs no dedicated fast
path.

Plain argmin/argmax keep NaN-skipping semantics (NaN never wins; NaN@first
position poisons to that index; all-NaN → 0), documented as a deliberate NumPy
divergence. Measured reasons (kernel level, n=1e7 f64, Zen5, sweeps in
`2026-09-09-argmax-argmin/nan-scan-variants/`): the committed plain scan
auto-vectorizes (AVX-512 vmax/select); making NaN win needs an unordered-aware
update (`!(x <= best)` = extra parity flag check, NumPy's own scalar trick,
+5…+18% large / +73…+100% small) or a NaN pre-pass (a second pass = a full DRAM
pass, ~+100%; block-tiled per-block-call version +180…+340%; per-8-block check
+55…+80% — the interleaved check de-vectorizes the loop). Fused-no-early-return
is semantically broken (a winning NaN is washed out by later lane updates —
`!(x <= NaN)` is always true — so the index is lost). NumPy pays the same class
of cost and is still ~4× slower than rstsr at 1e7. Do not re-attempt
first-NaN-wins for plain args without a hand-written SIMD kernel (which the
campaign rejected as "dispatch_simd: do not add").

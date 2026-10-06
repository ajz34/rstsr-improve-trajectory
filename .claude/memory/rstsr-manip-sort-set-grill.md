---
name: rstsr-manip-sort-set-grill
description: 2026-10-06 grill closed — manip/sort/set wave (17 fns) decisions; implementation awaits user go
metadata:
  type: project
---

Grill for the manip/sort/set wave closed 2026-10-06 (4 rounds, task dir
`2026-10-06-manip-sort-set/`, DECISIONS.md there is authoritative). Scope:
repeat/roll/tile, sort/argsort(+custom), searchsorted, nonzero, isin,
unique_*×4, take_along_axis, diff. Implementation NOT yet started — awaits
explicit user go.

Key decisions and facts to remember:

- 3-tier for sort/argsort/searchsorted/unique/isin/nonzero/take_along_axis;
  device-independent composition for repeat/roll/tile/diff. O(1)-aux kernel
  discipline (exceptions registered: isin O(m) temp, unique_all seen-bits).
- `ExtSortCmp` (suffix-less, dtype-traits): NaN-last total order, Equal on
  `==` (±0 collapse), bool included, complex lexicographic included (unique
  needs it; suite draws complex for test_unique_*).
- `unique_*` dual algorithm (user's R4 call): naive `Clone + PartialEq`
  general bound + fast sorted path substituted via TypeId dispatch +
  from_raw_parts reinterpretation (DeviceFaer matmul pattern,
  device_faer/matmul.rs:53-79 template). No specialization on stable Rust.
- All new ops take `impl TensorViewAPI` (idiom for data-consuming compute
  fns). `AxisIndex<isize>` new in rstsr-common, no retrofit.
- C-order flat contracts iterate `IndexedIterLayout::new(la, RowMajor)`;
  `reshape(-1)` is NOT C-order on col-major devices (order_semantics.md).
  Output layout decided by high tier via `new_contig(None,
  device.default_order())`. Tier-2 axes-reversal exists (matmul/tri) but
  unused this wave.
- Workflow grants: auto-commit allowed this wave (explicit session grant),
  NO push / NO gh pr; per-stage review once by DeepSeek/deepseek-flash
  (haiku slot) high, final review max; impl subagents glm-5.3-flash
  (sonnet slot) max effort; router = claude-code-router.
- Suite facts: searchsorted dtype must equal default_dtypes()["indexing"];
  stable=True is sort default; unique_all inverse_indices keeps x shape,
  indices = first-occurrence flat C-order, NaNs distinct, ±0 merge; roll's
  `shift` must be named pos-or-kw; capabilities values never read by suite.

---
name: rstsr-reduction-apis-grill
description: 2026-10-05 grill closed for rstsr-core reduction API enhancement (keepdims/with_args/with_dtype, norm, cumulative, astype, custom reduce); decisions in 2026-10-05-reduction-apis/DECISIONS.md; no-input-sized-intermediates directive.
metadata:
  type: project
---

Grill for rstsr-core reduction APIs closed 2026-10-05 (4 subagent fact passes;
R1–R3 + DECISIONS.md in `2026-10-05-reduction-apis/`). Key owner directives:

- NO breaking dtype changes: existing sum/mean/etc keep `TOut = T`; new
  additive `*_with_args` (ReduceArgs{axes,keepdims}, axes IS a struct field,
  plural intentional) and `*_with_dtype::<TOut>` (scope: sum/prod/mean/var/std/
  cumulative_* only). NormArgs{ord,axes,keepdims}, NormOrd enum; SVD-ords
  deferred. Cumulative follows array-api strictly (axis=None only 1-D,
  include_initial, no reverse).
- **Reductions must not materialize cast/pre-op tensors**: reuse the
  TI/TS/TO closure kernels (`reduce_{all,axes}_cpu_serial[_rayon]`), cast
  inside the fold `F: Fn(TS, TI) -> TS`; memory O(1) per output element.
  This supersedes any "cast-then-reduce via astype" composition idea.
- `astype` → TensorCow (reshape-like: view iff T==TOut) + `into_astype`,
  macro-enumerated safe impl pairs, implemented this stage; NOT the
  with_dtype mechanism.
- Process: 3 PRs / 3 branches (`261005/reduction-withargs-dtype`,
  `261005/norm-custom-reduce`, `261005/cumulative`), HARD STOP after each for
  manual review. faer-py widening defaults (uint8→uint64) = shim dispatch to
  with_dtype instantiations (W3, wrapper-only).
- Q9: fresh subagent drafts `rstsr-agents/skills/rstsr-code-style/` at
  wrap-up; skill must carry the reference-not-normative notice (style evolves;
  on divergence ask the user before updating).

Related: [[rstsr-faer-py-w2]], [[rstsr-arrayapi-compliance-notes]],
[[rstsr-reductions-t2p]].

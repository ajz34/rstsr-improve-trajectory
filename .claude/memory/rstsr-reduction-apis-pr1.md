---
name: rstsr-reduction-apis-pr1
description: PR1 of reduction-apis implemented and under owner review (ReduceArgs/VarArgs + with_args, with_dtype via slim axes-only dtype trait; astype POSTPONED with artifacts in postponed-astype/); uncommitted working tree on branch 261005/reduction-withargs-dtype.
metadata:
  type: project
---

PR1 of the 2026-10-05 reduction-apis task is implemented in the rstsr working
tree (branch `261005/reduction-withargs-dtype`, NO commits per owner rule),
owner review round 1 received and applied 2026-10-05. A previous attempt
(closed PR RESTGroup/rstsr#109) is discarded; its record was removed.

Current state after review:

- `ReduceArgs { axes, keepdims }` / `VarArgs { axes, keepdims, correction }`
  (axes IS a struct field; From-overloads via `impl_reduce_args_from_int!` —
  isize direct `Val` path because std blanket TryFrom gives Infallible).
- `<fam>_with_args[_f]` for all families incl. arg\*/all/any/unraveled (macro
  marker-gated `[with_args]`), bool-sum; keepdims = `dim_insert` at normalized
  axes in ASCENDING order (descending insert is out-of-range when trailing
  axes are reduced — caught by test).
- var/std correction: algebraic rescale `M/(M-c)` on the output (O(output), no
  device-trait change), NaN when `M - c <= 0`; correction stays `Option<f64>`
  (owner accepted; documented on the field).
- `OpReduceDtypeAPI` slimmed per review: ONE method per family
  (`*_axes_dtype`); whole-array path routes all axes through the axes kernel.
  Element-cast inside the fold closure (D5), no input-sized intermediates.
- **astype POSTPONED by owner review** ("too much impl_tensor_cast"): full
  implementation + tests + reapply notes in
  `2026-10-05-reduction-apis/postponed-astype/`; alternatives listed there
  (sealed trait / runtime dtype enum dispatch).
- Turbofish lesson: free fns with >1 generic cannot take partial turbofish —
  method forms (`a.sum_with_dtype::<u64>(args)`) are the ergonomic spellings.
- Gates: entry_row_cpu 317, lib faer 133, doc default-features 188/0, clippy
  clean serial/faer/default/workspace, nightly rustfmt.

Next: owner commits/PRs PR1; PR2 (norm + custom reduce) and PR3 (cumulative)
wait on review.

Related: [[rstsr-reduction-apis-grill]], [[rstsr-faer-py-w2]].

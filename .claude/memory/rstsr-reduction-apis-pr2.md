---
name: rstsr-reduction-apis-pr2
description: PR2 trimmed to CUSTOM REDUCE ONLY (owner moved norm to future rstsr-sci-traits; prototype preserved at postponed-norm/); branch 261005/norm-custom-reduce, uncommitted, awaiting owner commit go.
metadata:
  type: project
---

PR2 of the 2026-10-05 reduction-apis task: branch `261005/norm-custom-reduce`
(based on main f736987 = merged PR1 #110), working tree uncommitted, awaiting
the owner's commit go (per-repo rule: no auto commit in rstsr). FINAL SCOPE
(D15): custom reduce only - the norm family was REVERTED from rstsr-core and
handed to future rstsr-sci-traits; full prototype preserved at
`2026-10-05-reduction-apis/postponed-norm/` (README + diff + tests).

Kept surface (D9): `reduce_all`/`reduce_axes`/`reduce_with_args` (+`_f`;
reduce_axes takes `(x, axes, ...)`; with_args takes `(x, args, ...)`) over
`OpReduceCustomAPI<T, TS, TO, D>` device closures (init/fold/combine/finalize,
Send+Sync, TS: Clone; combine must be associative, documented). No tensor
methods (expert-level). Implemented in operators/reduction.rs (trait),
device_cpu_serial + feature_rayon/auto_impl (impls; the file is SYMLINKED
into device_faer/rayon_auto_impl - one source, two cfg contexts, rayon
kernels take `Option<&ThreadPool>`), tensor/reduction.rs (rt fns; keepdims
via reduce_layout_keepdims reuse), prelude reduce_* exports.

Doc layout (owner-directed, follows api-doc-conventions §2): full docstring
on the NON-fallible `reduce_all`/`reduce_axes`/`reduce_with_args` (each with
a `# Common reductions expressed via this function` table on reduce_all -
sum/prod/mean/var/max/l2_norm/count_nonzero/all-any as fold spellings;
GFM-table pipes inside code spans MUST be `\|`-escaped), `_f` twins are
two-line variants linking back. Skill edit: api-doc-conventions anchor-
selection wording generalized from "returning a view" to any plain-form
return type (scalar/owned included) - anchor is about failure mode.

Gates (all green, custom-only): entry_row_cpu 329, faer lib 134, doc 189,
clippy 0 (default/rayon/faer feature sets), nightly fmt. Tests:
test_reduce_custom.rs (parity vs sum, multi-accumulator mean, keepdims,
Lp-norm-vs-44-cbrt, bool/string-like accumulators, zero-size), doc_draft
test_reduce_custom (P3 example, output 2.571281590658235 = 17^(1/3)),
DeviceFaer runtime test in tensor/reduction.rs test module.

Norm history (why it left + settled design for the sci-traits wave): see
D13/D14/D15 in DECISIONS.md and `postponed-norm/README.md`. Key review
rounds: enforced ord + ReduceArgs reuse + string ords (D13); flatten attempt
retracted after the L1 divergence table (norm([[1,2],[3,4]],1) = 10 vs NumPy
6) - NumPy vector/matrix convention stands (D14); norm reverted from core
entirely + nuc-exclusion preference (D15). array_api_standard.md stays
untouched (owner: array-api linalg does not track norm there).

Review-process gotchas (2026-10-05): the owner's STAGED snapshot can predate
the working tree (partial staging) - reset-to-HEAD + rebuild was needed;
review must use the working tree. Files staged = owner-reviewed.

Next: owner commit go -> PR (pr-writeup style, fork ajz34 ->
RESTGroup/rstsr). Then PR3 `261005/cumulative` (D8). Fresh subagent drafts
`rstsr-code-style` skill at task wrap-up (Q9).

Related: [[rstsr-reduction-apis-pr1]], [[rstsr-reduction-apis-grill]].

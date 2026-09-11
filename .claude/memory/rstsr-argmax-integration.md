---
name: rstsr-argmax-integration
description: Patch 1 (argmax/argmin) integrated into rstsr 091f3e2 after review; what differs from proposed.patch and what is still open.
metadata:
  type: project
---

Campaign patch 1 (argmax/argmin, T6) was integrated into `../rstsr` branch
`260910-core-efficiency` as commit `091f3e2` (2026-09-11; with the nanarg follow-up merged upstream via PR RESTGroup/rstsr#100, squash `e835173`).
Differs from the experiment's `proposed.patch`: (1) general closure-based
`reduce_*_arg_*` API restored verbatim (kept for future non-standard
arg-reductions), specialized kernels renamed `*_arg_cmp_*` — net vs
`386948be` is publicly additive, the earlier API-break note is superseded;
(2) clippy clean (allowed `eq_op` on the generic `x == x` NaN check — owner
chose `==` over `ExtNum::is_nan` to avoid bound widening; `while_let_on_iterator`
fixed via `for..by_ref`); (3) `unravel_c_order` helper replaced by
rstsr-common `DimShapeAPI::unravel_index_c`. Verified: gates 110+290 both
configs, 3 paired criterion passes vs refA, no stable regression (small/faer
cells swing ±5–20% between same-binary reruns — trust only replicated deltas).
Key correction: NumPy argmax/argmin return the FIRST NaN at any position;
rstsr skips mid-stream NaN — README's old "coincides with numpy" claim held
only for NaN-at-front/all-NaN. nanargmin/nanargmax proposal:
`2026-09-09-argmax-argmin/NANARG-PROPOSAL.md`. Patches 2–5 pending;
`combined_all_five.patch` is stale vs reviewed patch 1. Campaign status:
[[rstsr-argmax-baseline-t6]], [[rstsr-compose-smoke-t8]].

---
name: where-op-grill
description: rt::where grill closed 2026-10-06 - all decisions, Q8 fallback, proposal location
metadata:
  type: project
---

`rt::where` design grill closed 2026-10-06 (`2026-10-06-where-op/`). All 10
recommendations accepted: **`r#where`** raw ident (first in codebase) +
`where_f`; 3-arg only (1-arg = nonzero, separate); Rust-only scope (faer-py
follow-up, G-037); strict bool cond; common-family promotion
(`DTypePromoteAPI`, output `TA::Res`, cond excluded, no arithmetic bound);
3-way broadcast + fresh alloc + `_f`/panic pair; **first 4-layout kernel
family** (`dispatch_4`/`_par_4`, `op_mutc_refa_refb_refc_func`, bridge
trait, hand-written `OpWhereAPI` in new `op_quaternary_common.rs`; `clip`
reuses later); scalar x/y house-strong, cond tensor-only — **user
contingency: fall back to tensor-only v1 if scalar trait impls get too
complicated**; test transfer list per Q9; rstsr branch `261006/rt-where`,
no rstsr commits without instruction. Full plan: `PROPOSAL.md`; decisions:
`DECISIONS.md`; facts: `FACTS-numpy-where.md`, `FACTS-rstsr-where.md`.
IMPLEMENTED 2026-10-06 on rstsr branch `261006/rt-where`, UNCOMMITTED (no
auto-commit): 4-layout kernel family in rstsr-common (`layout_col_major_dim_dispatch_4`
+ `_par_4`), native serial/rayon kernels with blocked-2d 4-layout macro, bridge
trait, OpWhereAPI (serial + auto_impl, appended to op_ternary_common/op_with_func
files to respect per-file symlinks), tensor layer `tensor/operators/op_where.rs`
(IxD-intermediary 3-way broadcast — DimMaxAPI has no generic-projection impls),
prelude `r#where`/`where_f`/`TensorWhereAPI`, docs status rows, 8 parity tests +
doc_draft module, tracking CSV (11 rows) + 2 numpy_differences entries. Scalar
variants bound `num::Num` (coherence: DTypePromoteAPI blanket same-type impl
would collide with the tensor impl) — bool scalars unsupported at scalar
positions, no fallback needed. Verified: 351/351 serial+faer entry_row_cpu,
196 doctests, clippy clean, rustdoc clean (fn.where.html), openblas + col_major
compile. Gotchas: single-axis slicing has no 1-tuple TryFrom (use `&[expr.into()]`);
`slice!(0, None, -2)` == numpy `[0::-2]` not `[::-2]`; to_vec is 1-D only.

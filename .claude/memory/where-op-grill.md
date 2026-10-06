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
Implementation awaits user go.

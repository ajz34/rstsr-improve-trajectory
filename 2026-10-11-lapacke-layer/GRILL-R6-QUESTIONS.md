# Grill R6 — correction round: driver tier must BE the LAPACKE equivalent (2026-10-11)

## Trigger (user correction, agent transcribed)

Part-1 implementation hardcoded `ColMajor` in the builders ("builders pass
ColMajor; kept for trait stability"), leaving the trait's `order: FlagOrder`
dead. User: this is total drift — `GESDDDriverAPI::driver_gesdd` must
**recover the behavior of lapacke, including the `order: FlagOrder`**. The
session is named LAPACKE transformation for this reason. Agent tasked to
audit the ADR and skill for the same problem.

## Agent audit (drift points found)

1. **ADR-0009 §3**: describes the two-tier vendor file but omits the layout
   3-way branch (the essence of LAPACKE's `_work` tier); "workspace … live in
   the builder layer" conflates the generic builder with the vendor trait
   impl.
2. **ADR-0009 §1**: "cloned into a fresh F-order buffer" — wrong under the
   hybrid output policy; the clone target is the *call's* order (F on
   col-major-default devices, C on row-major-default ones).
3. **Skill §3**: skeleton shows no layout branch; no statement that the
   driver handles both orders.
4. **Code**: builders hardcode ColMajor; drivers' row-major branch deleted.

Root cause: agent misread R2-Q1's "lapack is col-major-only" (a statement
about the raw Fortran library, hence the *need* for a lapacke-like layer)
as "our drivers may be col-major-only".

## Corrected tier mapping (LAPACKE ↔ rstsr)

| LAPACKE | rstsr tier | Responsibilities |
|---|---|---|
| `lapacke_xxx` (high level) | vendor trait impl (`driver_xxx`) | formula-workspace alloc *before* query (gesdd iwork, complex rwork); query (`lwork=-1`) → alloc work (clamps) → forward; both calls through the thunk |
| `lapacke_xxx_work` | thunk (`pub unsafe fn xxx_work`-shaped block) | **layout 3-way**: ColMajor → direct FFI, caller lds; RowMajor → template lds `MAX(1,rows)`, transposed temps (orderchange kernels), call, transpose out; bridge failure → −1011 (never panic); NULL for unused u/vt; the `lwork==-1` query short-circuit per branch (query lds therefore always LAPACKE-correct) |
| caller of LAPACKE | generic builder (`src/lapack_svd/`) | policy: `order = device.default_order()` (hybrid output policy); `overwritable_convert_with_order(a, order)` (overwrite routines clone-or-consume); fresh outputs allocated in `order`; zero-size synthesis; error mapping (`map_lapack_info`) |

Unchanged from prior rounds: workspace policy table, hardening set
(MAX(1,·), lwork==0→1, checked conversions, f32 nextafter, formulas),
sentinel −1010/−1011 kept at driver tier + mapped to MemoryError by the
builder, superb smuggling, zero-size identity synthesis, ipiv (Part 2).

Open sub-question (was parked, now live because the row-major branch is
restored): bridge kernels stay `_cpu_serial` for this campaign (the
multi-threaded-transpose policy remains the parked open flag), with
`.unwrap()` replaced by −1011 returns. Per-device ambient-threaded bridging
(R3-Q1 (ii)) becomes a separate follow-up once the parked flag is settled.

❓ **Q1 - Confirm the corrected tier mapping and audit fixes** (rewrite:
builders pass `order`; drivers restored to the LAPACKE 3-way with the
hardening; ADR §1/§3 and skill §3 amended accordingly)?

➡️ Recommend confirm.

❓ **Q2 - Bridge kernels this campaign**: keep `_cpu_serial` + −1011 error
returns now (MT-transpose policy stays parked), deferring R3-Q1 (ii)
per-device ambient threading to a follow-up?

➡️ Recommend yes.

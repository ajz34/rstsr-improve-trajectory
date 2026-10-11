# Grill R4 — LAPACKE translation layer (session 2026-10-11)

## R3 answers (agent-transcribed from the on-screen session)

- **Q1 bridge threading**: (ii) confirmed — reuse per-device orderchange infra
  (ambient threading). Notice: this is already the elementwise-op behavior,
  but a NEW deviation vs existing LAPACKE-translation code (`driver_impl/lapack/
  eigh/syev.rs` uses the `_cpu_serial` kernel); retrofit replaces those calls.
- **Q2 vendor shape**: trait `driver_xxx` signatures stay untouched. The
  `_work`-equivalent (e.g. `dgesvd_work`) becomes an auxiliary free `pub fn`
  in the same file — public for reachability across the vendor-symlink setup,
  but internal-purpose (documented as such, not exported to prelude).
- **Q3 ipiv fix**: confirmed.
- **Q4 parity test**: features `lapack` and `lapacke` are mutually exclusive
  (`#[cfg(not(feature = "lapacke"))] pub mod lapack;` — verified). So: no
  both-features module; run the existing driver suite under BOTH configs
  separately (both green ⇒ good), numpy/scipy as correctness oracle. lapacke
  config usually passes with zero code effort; rare lapacke bugs possible.
- **Q5 zero-size**: confirmed; user's original SVD is poor at these values,
  current space setting too conservative/memory-costly — agent to improve.
- **Q6 triangle copies**: (ii) confirmed. Rationale added: dsy/dtr trans costs
  no less memory than dge trans, just adds footprint; O(N²) ≪ O(N³) main
  computation for most LAPACK functions.
- **Q7 output order**: hybrid (a) confirmed — fresh outputs device-default
  order; in-place/overwritable outputs reuse the input buffer and its F order;
  view-only f-prefer inputs always pass zero-copy.
- **Q8 NaN check**: user rethinking. Constraint: any nan-check toggle must be
  user-controllable (allow/disallow), and that means trait changes. Library
  is not widely used; users call high-level functions, not raw drivers; so
  driver-trait changes are feasible after all. User asks: if the trait-fn
  restriction breaks, what new feature/argument shape do I recommend?

## Agent's recommendation for the R3-Q8 trait-evolution question

**Default stance: no trait change now; if one is ever made, grow the trait,
don't fork it.** Concretely:

1. **Now (this campaign): no nan check anywhere.** Deliberate divergence
   (numpy precedent), hazard documented in ADR-0009. Zero trait churn while
   the layer's shape is still settling.
2. **If/when adopted, the mechanism is a knob on the existing args groups,
   not new methods and not method-level flags.** The codebase already has the
   pattern: argument-group structs (`RepeatArgs`, `SortArgs`, ...) supply
   overloads via `From`/`TryFrom` so one signature covers every call shape —
   and "when `()` is a valid overload, `None` must be too" is already an
   established rule. So: a `LapackNanCheck` (or `FiniteCheck`) enum
   (`Auto/On/Off`) living in the existing args struct of each driver (e.g.
   `GESVD_` builder / future `DriverArgs`), defaulting to `Auto` (= off in
   release, on in debug) or `On`; the builder reads it before the FFI call
   and runs a finiteness scan over f-prefer *views* only where it matters.
   Old code compiles unchanged (`()`/`None` → default).
   This mirrors LAPACKE's own design exactly: compile-time removal +
   runtime flag (`LAPACKE_DISABLE_NAN_CHECK` + `LAPACKE_get_nancheck`), one
   mechanism, not per-routine flags.
   For the high-level (linalg-traits) tier: same enum inside the existing
   `with_args`-style group, or a device-global toggle, not a new method.
3. **What NOT to do**: per-routine `*_checked` method twins, boolean params
   appended to existing signatures (breaks every call site + violates the
   tuple-group convention), or a global `static` without a user knob.

---

## R4 frontier

❓ **Q1 - NaN-check decision**: Accept the recommendation above — (a) no
nan check in this campaign (ADR records divergence + hazard); (b) future
adoption shape = enum knob in the args group (`Auto/On/Off`, `()`/`None` →
default), never method twins or appended booleans? Or do you want the knob
already in this campaign (debug-on only)?

➡️ Recommend: (a) + (b) as the recorded future path; no knob this campaign.

❓ **Q2 - Skill & ADR contents confirm**: ADR-0009 (next number verified):
(1) the layer's purpose (LAPACKE-equivalent for rstsr BLAS devices; scipy
layer as secondary reference); (2) input taxonomy f-prefer/c-prefer/
fully-non-contiguous with f-contig/c-contig as special cases (ld equality
not guaranteed); (3) input/output/bridge policies (clone-else, zero-copy
f-prefer views, hybrid output order, uplo-flip NOT adopted — we clone);
(4) two-tier vendor file shape (trait `driver_xxx` + free `pub fn
xxx_work`); (5) workspace policy table (per-routine query/formula, clamps,
nextafter, MAX(1,·)); (6) error mapping (raw info at thunk; builder maps);
(7) deliberate divergences (no nan check now + callee hazard record; no
triangle-only bridges; identity sub-blocks at zero size); (8) OpenBLAS-first
validation; lapack↔lapacke mutual exclusion. Skill `lapacke-translation`:
the recipe (step-by-step, per-routine checklist, skeleton, test matrix incl.
3-case layouts, gap-notice rule "if the recipe lacks coverage → stop and
report"), ADR-0009 referenced for rationale. Both in rstsr-agents /
rstsr-book respectively — confirm the split and content lists?

➡️ Recommend confirm.

❓ **Q3 - Retrofit scope of the FIRST review part**: R2-Q11 says small
parts, stop for review each. Part 1 = gesvd + gesdd only (driver files:
`driver_impl/lapack/svd/{gesvd,gesdd}.rs` real+complex blocks, high-level
`src/lapack_svd/{gesvd,gesdd}.rs`, shared util changes: output-order rule,
error mapping, zero-size identity synthesis, workspace hardening, ipiv is
not in SVD), plus driver tests (f64 + c64) with 3-case layout inputs and
zero-size cases, run on openblas under both feature configs. Confirm as
Part 1, with Part 2+ (other routines family-by-family: solve family ipiv +
error mapping; syev/syevd/sygv/sygvd; getrf/getri/potrf; blas3 untouched)
queued behind review?

➡️ Recommend confirm.

❓ **Q4 - KML override handling**: KML ships a local override of
`tests/driver_impl/f64/gesvd.rs` (documenting a KML non-unique-U quirk).
The retrofit changes the shape SVD tests exercise. Keep the override as-is
(adjust content only if compilation breaks), or fold it into the shared
test module with a `#[cfg]`/doc note? (KML is untested locally, low
priority per R2-Q10.)

➡️ Recommend: keep as-is, adjust only for compilation; note in the skill
that per-vendor test overrides live in the vendor crate.

❡ *(no Q5+ this round — everything else settled or blocked on Q1–Q4)*

---

*Remaining after R4: execution (skill+ADR drafts → Part 1 retrofit on a
rstsr branch → tests → local matrix) per R2-Q11, each part stopping for
review.*

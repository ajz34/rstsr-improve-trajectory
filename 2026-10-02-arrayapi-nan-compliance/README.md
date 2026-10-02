# 2026-10-02 — array API NaN compliance for the min/max family

- **rstsr base**: master `ca325a7` (PR #106 merged).
- **Type**: semantics-compliance change (owner-directed), efficiency impact
  recorded honestly but **not gated** — owner ruling: *array API compliance
  wins over efficiency, "no matter efficiency"* (2026-10-02; first stated
  2026-09-21 during the reverted reductions integration, see
  [[rstsr-reductions-integration]]).
- **Status**: implementation in progress in the rstsr working tree
  (uncommitted; owner reviews/commits). This directory records the decision
  trail, spec facts, and (later) paired A/B numbers.

## Owner decisions (this session, 2026-10-02)

1. Where rstsr semantics diverge from the Python array API standard, make
   them comply regardless of measured cost. Formalizes the 260921 ruling.
2. Scope: min/max family only, now. A systematic whole-standard conformance
   audit (and building an array-API test mechanism) is explicitly postponed
   — rstsr has no systematic array-api test harness today and building one
   is non-trivial.
3. Plain `argmin`/`argmax` adopt NumPy first-NaN-wins anyway (spec leaves
   arg NaN unspecified; owner chose plain-op family consistency:
   `argmin(x)` = index of `min(x)`). This SUPERSEDES the 2026-09-11
   measurement-based rejection ([[rstsr-nanarg-semantics]]) — the rejection
   was efficiency-motivated; the directive removes efficiency from the
   gate. nanargmin/nanargmax (PR #100) unchanged.
4. New `nanmin`/`nanmax` public reductions (NumPy value semantics: skip
   NaN, all-NaN → NaN returned, NumPy's RuntimeWarning has no Rust analogue;
   empty → error like NumPy). The standard has NO nan_* functions (verified
   across all versions) — these are NumPy-parity additions.

## Spec facts (verified against ~/Git-Others/array-api @ ff497ed8, 2025.12)

- Reduction `min`/`max`: NaN MUST propagate; empty is
  implementation-defined (rstsr's `InvalidValue` raise conforms, kept);
  signed-zero ties either value allowed; dtype preserved.
- Elementwise `minimum`/`maximum`: either operand NaN → result NaN.
- `clip`: unimplemented in rstsr — out of scope.
- `argmin`/`argmax`: NaN unspecified by the standard; only first-occurrence
  tie-break is mandatory (rstsr already conforms).
- `f64::minimum`/`maximum` (IEEE 754-2019) still `#[unstable]`
  (`float_minimum_maximum`, issue #91079) on stable 1.98.1 and nightly
  1.100.0 → hand-rolled `is_nan` select forms are required.

## Campaign linkage (supersessions)

- **Patch 4 (T2' reductions) A2 closure edit is superseded**: the
  strict-compare NaN-skipping min/max closures from
  `2026-09-09-reductions/proposed.patch` will NOT land as proposed; the
  wiring closures get the propagating primitive instead. The A band-walk
  sum kernel (serial `size_mc > 1` branch) is unaffected and still queued
  for a post-compliance integration pass.
- The 260921 integration's `ARRAYAPI-CHECK.md` audit
  (`2026-09-09-reductions/results/integration260921/`) anticipated exactly
  this change, including "main suites have no min/max NaN tests today
  (clean to change)".
- **#100 plain-arg NaN decision superseded** (see decision 3). The
  nan-scan-variants sweeps
  (`2026-09-09-argmax-argmin/nan-scan-variants/`) remain the cost reference:
  in-loop unordered-aware update +5…+18% large / +73…+100% small; pre-pass
  forms ~+100% and worse; fused-no-early-return semantically broken. The
  implementation uses an incumbent-freeze 8-lane form (NaN freezes the lane
  with the first-NaN index; non-NaN strict compares unchanged) — expected
  in the cheapest measured class; paired A/B numbers to be appended below.

## Measured cost (to be appended)

- [ ] min/max reduction propagating fold vs old skipping fold (paired A/B,
  serial + faer16, portable + native)
- [ ] argmin/argmax first-NaN-wins vs #100 kernels (paired A/B)
- [ ] elementwise minimum/maximum propagating vs skipping

## Implementation notes (filled as work proceeds)

- rstsr working tree only — no commits in rstsr from the agent side.

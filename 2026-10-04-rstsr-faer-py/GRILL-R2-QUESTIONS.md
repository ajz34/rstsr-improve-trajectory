# Grill round 2 — questions (2026-10-04)

## Background — R1 as answered (AGENT-AUTHORED transcription of the user's on-screen answers; the authoritative record is the user's own `GRILL-R1-ANSWERS.md`)

Fact-base correction from the user: the repo claims MOST support, not full —
`rstsr-core/src/docs/array_api_standard.md` is a per-feature fulfillment
table (Y/C/P/D/T/blank, blank = future), and the README's "full support"
line sits under "What is supposed to be supported in near future?" (side
finding: that future list is stale — moveaxis is listed but implemented).

- **Q1/Q10 converged**: A-first; Phase 1/2 stay rust-side future work;
  virtually no rstsr-core edits in this task (sole exception: "dumb" little
  fixes, guardrail → Q5 below); the task = implement a numpy-like /
  array-api-compatible Python surface so the official suite can grade rstsr.
- **Q2/Q4/Q6/Q8 accepted** as recommended (boundary rule; DeviceFaer +
  13 canonical dtypes; Python-class-first; native skip/xfail + issues +
  register).
- **Q3 accepted with override**: DLPack exchange is IN SCOPE (implementable
  via merged `rt::dlpack`), not a deferred gap. Pins unchanged: 2025.12,
  suite `6c0b59f` + submodule `5f847a3`.
- **Q5 accepted with addendum**: never published to PyPI/conda, but
  packaging-ready (wheel builds locally).
- **Q7 accepted with redirection**: rstsr repo stays clean — package code
  only; harness, pins, reports, register live in this task dir; repos
  clone to `~/Git-Others`.
- **Q9 rejected for this task**: no CI; revisited after the rust-side work
  concludes.

## Frontier

**Q1 — Build order.** (a) Skeleton-first: namespace skeleton (module +
`__array_namespace_info__` + dtype objects + `asarray` + the Python `Array`
protocol object incl. `__dlpack_device__` + a vertical slice of ~5 ops),
then run the suite once — the import-time contract (hypothesis
`make_strategies_namespace`, info namespace, device probes) is the only
part that can fail *architecturally*; meet it in hours, not after weeks of
shim labor — then grow the surface category-by-category (creation →
elementwise → manipulation → statistical → searching/sorting → linalg)
with a suite run per category. (b) Full surface first, single first run.
➡️ (a); per-category runs also build the register incrementally.

**Q2 — DLPack timing and route.** Route: Rust-side pyo3 `PyCapsule`
(name `dltensor_versioned`, rename-on-consume, Rust destructor — pattern
proven in the dlpack prototype's `demo-ffi`) wrapping `rt::dlpack` export;
import via `from_dlpack_versioned_f` (foreign buffers copy-only per the
merged crate's policy); both directions (test_dlpack round-trips through
xp's own capsule). Timing: with the skeleton (right after the first red
map) vs late-M1.
➡️ Route as stated, timing early: `__dlpack_device__` is already
import-time-required, the incremental cost is small, and it exercises
`rt::dlpack` from Python for the first time — high signal.

**Q3 — Register vs the in-repo fulfillment table.** Every register entry
cross-references: table row ↔ suite outcome ↔ rstsr issue. Discrepancies in
BOTH directions are findings (table-Y failing = overclaim; blank/D = expected
gap; stale entries like moveaxis). The table's C-status column (`%` =
matmul) is Rust-syntax-specific and invisible to the Python suite — the
shim maps `__mod__` → `rt::rem`; the register records Python-side
conformance only. M1 deliverable additionally includes a REPORT-ONLY patch
proposal updating the table (no doc PR from this task).
➡️ Adopt cross-reference scheme + report-only table diff.

**Q4 — Which Rust surface the shim binds.** (a) The facade crate `rstsr`
(`rt::` function surface, faer-as-default device feature if available) —
the same surface the fulfillment table documents, so shim code mirrors
documented usage. (b) `rstsr-core` directly — more control, bypasses the
documented mapping.
➡️ (a); explicit-device core calls only where facade overload tuples make
marshalling awkward. Exact feature spelling verified at implementation.

**Q5 — "Dumb little fix" guardrail.** Proposed test: allowed = local fixes
with NO public-API and NO semantic change (missing trait bound/impl,
panic-path→error, obvious bugs in the NEW glue code itself, pyo3-crate-local
issues); each fix = isolated commit on the task branch + register line
marked `fixed-in-task`; anything touching op semantics, promotion, or
existing public API ⇒ gap issue, no matter how small. Gut size cap ~50
lines.
➡️ Adopt. When in doubt, gap-issue it.

## Notes (stated, object if wrong)

- Default dtypes declared via `default_dtypes()`: int64 / float64 /
  complex128 (`asarray(int()/float()/complex())` probing reports these;
  the suite adapts expectations to the declaration).
- Suite entry: `ARRAY_API_TESTS_MODULE` dotted path or exec-snippet
  importing `rstsr_faer.api` as `xp`.
- In-place dunders exist Rust-side (table lists the `+=` family as Y); the
  shim exposes them; `__pos__`/`__ifloordiv__`/`__ipow__` are table-D ⇒
  expected Python-side gaps.

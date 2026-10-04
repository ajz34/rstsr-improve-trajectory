# Grill round 3 — questions (2026-10-04)

## Background — R2 as answered (AGENT-AUTHORED transcription of the user's
answers; authoritative record: `GRILL-R2-ANSWERS.md`)

- **Q5 OVERRIDE (stricter than proposed)**: disable ANY fix in rust-side
  code — full stop. Rust-side problems are reported (register / dedicated
  location) and fixed only by the user or on explicit permission. This
  supersedes R1's "dumb little fixes" exception: zero rstsr-core edits.
- **Q1/2/3/4**: proceed as recommended; adjust details when implementing.
- **Process**: implementation proceeds step by step, with the user
  reviewing at certain points. The agent may commit the user's answers
  files verbatim.

## Proposed implementation plan (the last grill subject)

- **S0 — Harness before any wrapper code.** Task-dir harness adapted from
  `numpy-compliance/`: setup.sh (overlay venv on conda `torch`,
  `git submodule update --init` in `~/Git-Others/array-api-tests`, install
  pytest-json-report / hypothesis / ndindex), run.sh + summarize. Gate:
  reproduce the NumPy 2.5.1 baseline locally (~1382 items, the 6 known
  failure clusters) — proves the harness before any Rust exists.
- **S1 — Skeleton + first red map.** Branch `261004/rstsr-faer-py` in
  rstsr; `crates-interop/rstsr-faer-py` (pyo3 + maturin + abi3-py310,
  workspace member); `rstsr_faer.api` namespace skeleton
  (`__array_namespace_info__`, dtype objects, Device singleton, `asarray`
  from scalars/nested lists, Python `Array` protocol object incl.
  `__dlpack_device__`, ~5-op vertical slice); first suite run against
  rstsr; gap register v0. → **Review point 1**.
- **S2 — DLPack, early.** `__dlpack__`/`from_dlpack` via pyo3 PyCapsule
  over `rt::dlpack` (import copy-only); test_dlpack resolved. Light review,
  folded into the S1→S3 cycle.
- **S3 — Surface categories.** creation → elementwise → manipulation →
  statistical → searching/sorting → linalg; one suite run per category;
  register grown; skips.txt/xfails.txt maintained alongside. → **Review
  after each category (skippable)**.
- **S4 — M1 close.** Final honestly-red report + complete register +
  report-only fulfillment-table diff proposal + issue batch; decision
  record + memory updates in this task dir. → **Final review**.

## Frontier (last two decisions)

**Q1 — Review granularity.** (a) Review gates as placed above: mandatory
after S1 (first red map) and at S4 (close), optional per-category gates in
S3 you may skip ad hoc. (b) Coarser: only S1 and S4.
➡️ (a); you keep the ability to wave through any per-category gate.

**Q2 — Gap issues on RESTGroup/rstsr (outward-facing).** Issues are public
artifacts on the upstream repo. (a) Register accumulates during work;
issues batched at review points, each batch opened only on your explicit
go (or you open them yourself). (b) Agent files issues as gaps are
confirmed during S3.
➡️ (a) — batches at review points on your explicit go; nothing public
without your say-so.

## Notes

- Zero rust-side edits is now absolute; discovered problems land in the
  register marked `rust-side`, with fixes by you or on your permission.
- After these two answers the frontier is empty; the agent writes the
  decision record (DECISIONS.md) summarizing all rounds and the grill
  closes on your confirmation. Implementation starts only then.

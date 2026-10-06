# 2026-10-06-where-op

Design grilling for implementing `rt::where` (NumPy `np.where` 3-arg form) in
rstsr-core and related device implementations.

- **Against rstsr**: `2ce2afd` (main, clean). Implementation would go to a
  fresh rstsr branch (`261006/rt-where`); rstsr itself is not auto-committed.
- **Motivation**: faer-py gap register G-037 (`where` absent; owner ruled
  rust-side, no shim) — see `2026-10-04-rstsr-faer-py/GAP-REGISTER.md`;
  `where` is in the array-api standard.
- **Method**: grilling per repo convention — `GRILL-INIT-PROMPT.md`,
  `GRILL-R<N>-QUESTIONS.md` (agent), `GRILL-R<N>-ANSWERS.md` (user-owned).
- **Status**: fact-finds converged (2026-10-06) — `FACTS-numpy-where.md`,
  `FACTS-rstsr-where.md`. Consolidated **R1** (full frontier, 10 questions)
  asked on screen + this directory; awaiting user answers
  (`GRILL-R1-ANSWERS.md` or screen). Next: DECISIONS.md + rstsr
  implementation plan once answers land.

Not a benchmark task: no environment/numbers section planned unless the
design calls for one (e.g. kernel variant comparison) — then this README
grows the usual baseline/post-change record.

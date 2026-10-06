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
- **Status**: R1 asked (on screen + this directory); fact-finding sub-agents
  (rstsr op mechanics, NumPy/array-api semantics + test surface) running;
  R2 recomputes the frontier once they converge and R1 is answered.

Not a benchmark task: no environment/numbers section planned unless the
design calls for one (e.g. kernel variant comparison) — then this README
grows the usual baseline/post-change record.

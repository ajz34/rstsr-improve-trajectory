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
- **Status**: **grill closed 2026-10-06** — all 10 R1 recommendations
  accepted (Q8 with a tensor-only fallback contingency). Records:
  `GRILL-INIT-PROMPT.md`, `GRILL-R1-QUESTIONS.md`, `FACTS-numpy-where.md`,
  `FACTS-rstsr-where.md`, `DECISIONS.md`, `PROPOSAL.md`. Next: implement in
  rstsr on branch `261006/rt-where` on user go (no rstsr commits without
  instruction).

Not a benchmark task: no environment/numbers section planned unless the
design calls for one (e.g. kernel variant comparison) — then this README
grows the usual baseline/post-change record.

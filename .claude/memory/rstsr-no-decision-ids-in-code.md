---
name: rstsr-no-decision-ids-in-code
description: "rstsr source code must not reference this repo's decision ids (D29, P0, patch-N) — comments have to stand alone"
metadata:
  type: feedback
---

Comments, docstrings, test names and commit messages in the rstsr repos must not
reference ids minted in `rstsr-improve-trajectory` (`D29`, `P0`/`P3`,
`patch-1`, "scoped rule", the grill round files). Those ids mean nothing to a
reader of the published crate and rot as soon as this repo's numbering moves.

State the rule itself in the code instead. Example from the GEMV dispatch work:
`/// for vector-shaped products (D29, scoped rule)` had to become
`/// for vector-shaped products`, with the reasoning left in the batch-loop
/// comment ("vector-shaped inner product is memory-bound: iterate batches in
/// the main thread …"). Cite decisions only in this repo's `DECISIONS.md` and in
/// the handoff text.

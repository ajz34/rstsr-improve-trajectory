# Grilling initial prompt — rt::where implementation proposal

Date: 2026-10-06
Target repo: rstsr (pinned at `2ce2afd`, main) — implementation would land in
`rstsr-core` + device crates on a fresh branch; **no auto-commit** there.
This repo (rstsr-improve-trajectory) holds the proposal/decision records and
follows its own CLAUDE.md (auto-commit allowed).

Prompt (verbatim, via `/grill-me`):

> You are going to propose the way of implementing rt::where in rstsr-core
> and related implementations.
> Notice rstsr-improve-trajectory and rstsr-agents. rstsr-improve-trajectory
> follow different claude and commit rules, and rstsr is not auto-committed.

Scope context brought in from prior records:

- `2026-10-04-rstsr-faer-py/GAP-REGISTER.md` G-037: `where` absent from
  rstsr; owner ruled **no shim-side implementation** — rust-side batch item.
  First blocker of test_getitem/test_setitem via `ph.assert_array_elements`.
- Related gaps in the same register: G-038 (boolean-mask indexing needs
  whole-tensor gather/scatter or nonzero), G-012/G-029 (searching/set
  surface), G-051 (`capabilities()` overclaims boolean indexing).
- `where` is in the array-api standard, so `rstsr_faer.api.where` is an
  eventual consumer (faer-py conformance).

Grilling method: design tree worked in rounds; each round's questions are
written to `GRILL-R<N>-QUESTIONS.md`; the user answers on screen or by
editing `GRILL-R<N>-ANSWERS.md` (answers files are user-owned).

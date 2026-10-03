# dlpack FFI family (rstsr-ffi)

- **Date**: 2026-10-03
- **Target repos**: `dlpack-ffi` (redesign + generation script), `rstsr-agents` (new update skill) — per the round-1 Q7 answer, *not* a new member of the rstsr-ffi workspace
- **Base**: rstsr-ffi master `2731f7c` ("rstsr-aocl-ffi: update to AOCL 5.3.0"); rstsr master `f179c46` (context only — this task does not modify rstsr)
- **Origin prompt**: [initial-prompt.md](./initial-prompt.md)
- **Status**: design **frozen** 2026-10-03 — rounds 1–2 answered ([round-1-answers.md](./round-1-answers.md), [round-2-answers.md](./round-2-answers.md)); no open design questions; implementation not started.

## Mission

Redesign and upgrade the `RESTGroup/dlpack-ffi` crate (dmlc/dlpack bindings): raw types-only
C-ABI surface whose API survives DLPack minor-version enum additions without breaking
(bindgen `NewType` style, vendored v1.3 header, generator script, provenance), plus an
`update-ffi-dlpack` skill (check/upgrade on upstream release) in the `update-ffi-blas` style.

## Files

- [initial-prompt.md](./initial-prompt.md) — the user's verbatim prompt and session follow-ups.
- [round-1-questions.md](./round-1-questions.md) — design questions Q1–Q9 with recommendations,
  the condensed fact base, and evidence quotes (answered; kept as the question record).
- [round-1-answers.md](./round-1-answers.md) — the answers, resolved decisions, and the Q7 reshape
  (task moves into the dlpack-ffi repo).
- [round-2-questions.md](./round-2-questions.md) — round-2 questions (version, identity, upgrade
  checklist, pack wiring) with the dlpack-ffi repo-anatomy digest as appendix.
- [round-2-answers.md](./round-2-answers.md) — round-2 answers, the dlpack-ffi purity constraint,
  and the closing checklist.

Design is closed as of 2026-10-03; next is a `PLAN.md` for the implementation.

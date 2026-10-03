# dlpack FFI family (rstsr-ffi)

- **Date**: 2026-10-03
- **Target repos**: `rstsr-ffi` (new crate + generation scripts), `rstsr-agents` (new update skill)
- **Base**: rstsr-ffi master `2731f7c` ("rstsr-aocl-ffi: update to AOCL 5.3.0"); rstsr master `f179c46` (context only — this task does not modify rstsr)
- **Origin prompt**: [initial-prompt.md](./initial-prompt.md)
- **Status**: round-1 **answered** 2026-10-03 ([round-1-answers.md](./round-1-answers.md)); task relocated to the `RESTGroup/dlpack-ffi` repo (Q7); round 2 open; no implementation started.

## Mission

Add a new `rstsr-ffi` family member binding DLPack (dmlc/dlpack): a types-only C-ABI crate
(`rstsr-dlpack-ffi`) whose API survives DLPack minor-version enum additions without breaking,
plus an `update-ffi-dlpack` skill (check/upgrade on upstream release) in the `update-ffi-blas` style.

## Files

- [initial-prompt.md](./initial-prompt.md) — the user's verbatim prompt and session follow-ups.
- [round-1-questions.md](./round-1-questions.md) — design questions Q1–Q9 with recommendations,
  the condensed fact base, and evidence quotes (answered; kept as the question record).
- [round-1-answers.md](./round-1-answers.md) — the answers, resolved decisions, and the Q7 reshape
  (task moves into the dlpack-ffi repo).

Answers will be folded into a round-2 pass (if the tree is not yet closed) and eventually a `PLAN.md`
for the implementation.

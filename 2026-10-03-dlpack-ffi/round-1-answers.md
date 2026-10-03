# Round-1 answers — dlpack FFI family

- **Date**: 2026-10-03 (answered in session, same day the questions were posed)
- **Questions**: [round-1-questions.md](./round-1-questions.md)

## Verbatim answers

- **Q1**: Raw only. This is what rstsr-ffi is for.
- **Q2**: Well (a). I'm not familiar to transparent, and nomicon's explanation seems good to me.
- **Q7**: Well I even forgetted this repo. This is a good timing and important notice. We will use dlpack-ffi repo for this task, not in rstsr-ffi repo.
- **Q3/4/5/6/8/9**: Your recommendation.

## Decisions as resolved

| Q | Decision |
|---|----------|
| Q1 | Raw-only crate (no safe wrappers); no rstsr-core changes in this task. |
| Q2 | bindgen `NewType` enum style; `DLDataType.code` stays the header's `u8` + consts; existing struct layouts frozen; legacy `DLManagedTensor` exposed but documented deprecated; names mirror the C header exactly. |
| Q3 | Family-conformant pipeline: per-crate generator script + pinned bindgen CLI, output committed, no `build.rs`, no CI drift job; the update-time gate lives in the new skill. |
| Q4 | New sibling skill `update-ffi-dlpack` in rstsr-agents (check + update modes); upstream checkout path recorded in `CLAUDE.local.md`, never hardcoded. |
| Q5 | Verification: (a) ABI/layout assertions + (b) round-trip/deleter-exactly-once test + (d) synthetic forward-compat test as crate tests; (c) numpy `from_dlpack` interop as an ignored, skill-documented manual check. No CI policy change. |
| Q6 | Export-first Rust-side interop substrate ((a)+(b) jointly); the rstsr-core consumer task stays separate/future; the dropped `from_dlpack` record is revisited then. |
| Q7 | **Revised from the recommendation**: the task moves **into the existing `RESTGroup/dlpack-ffi` repo** — not a new crate in rstsr-ffi, and no deprecation of the old crate. The repo/crate itself is redesigned and upgraded. |
| Q8 | Pin the **v1.3** release header (not HEAD). |
| Q9 | Whole-header mirror, including legacy structs and the v1.2+ exchange-API types. |

## Consequences of the Q7 reshape

- **Deliverable**: redesign + upgrade of `RESTGroup/dlpack-ffi` (regenerate with the `NewType` enum style; vendored header at v1.3; generator script; provenance/bookkeeping), **plus** skill `update-ffi-dlpack` in rstsr-agents.
- **rstsr-ffi stays untouched**; no readme-table row there. The "settled conventions" list in [round-1-questions.md](./round-1-questions.md) is superseded where it assumed a new `rstsr-dlpack-ffi` member in the rstsr-ffi workspace.
- The one-time API break (rustified enums → `NewType`) re-versions the crate — version/identity decided in round 2.
- The existing crate's published history: v1.2.0 (2025-12-03), single hand-vendored `src/lib.rs`, bindgen 0.72.1, exhaustive `#[repr(u32)]` enums ending at `kDLTrn=18`; no in-pack consumers were found.

## Round 2

Posed in session (version of the redesigned release, package identity, restructuring scope, pack wiring); results to be appended here or in a follow-up file once answered.

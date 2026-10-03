# Round-1 design questions — dlpack FFI family

- **Date**: 2026-10-03
- **Origin prompt**: [initial-prompt.md](./initial-prompt.md)
- **Status**: **OPEN** — posed at the end of the 2026-10-03 grilling session; written here before any answers arrived.
- **How to answer**: by question number, free text, partial answers fine. Q1's recommendation was revised mid-session (repo policy, noted in Q1); the versions below are the consolidated set and supersede everything said earlier in the session.

## Fact base (condensed)

Gathered on 2026-10-03 by four background agents (rstsr-ffi crate anatomy + bindgen internals, `update-ffi-blas` skill digest, DLPack upstream history/policy, pack-wide `dlpack` grep). Raw reports were not archived; this is the digest. Key quotes in the appendix.

1. **DLPack upstream**: latest release **v1.3** (2026-01-24). Unreleased HEAD (`94485e2`, 2026-08-12) carries three enum values with minor still 3: `kDLTPU=19`, `kDLTPUHost=20`, `kDLAscend=21`. Policy is explicit: major = ABI data-layout change (`DLManagedTensorVersioned`); minor = "the ABI is kept the same… addition of enumeration values". No published struct field has ever been reordered or removed (`DLTensor` frozen since v0.1, `DLManagedTensor` since v0.2, the versioned structs since v1.0rc). The header is types-only (no inline functions); v1.2 added the C Exchange API (five fn-pointer typedefs + `DLPackExchangeAPI{,Header}` structs) and made non-NULL `strides` mandatory for producers (behavioral, not ABI). Legacy `DLManagedTensor` is documented deprecated yet coexists.
2. **Local checkout**: `~/Git-Others/dlpack` already exists (created 2026-10-03; HEAD `94485e2`). Pack-wide, the only historical DLPack trace is rstsr-core's deliberate "dropped feature" record for `from_dlpack` (`array_api_standard.md`, status D, Python-interop rationale); rstsr-core has **no** foreign-buffer/custom-deleter storage (`deleter` = 0 hits; devices hardwire `Raw = Vec<T>`; `TensorArc` = `Arc<Vec<T>>`).
3. **rstsr-ffi family**: 7 crates, **no `build.rs` anywhere**; bindgen is a pinned CLI driven by per-crate `scripts/*.py`, output committed, scratch gitignored; CI is **fmt + clippy only** (no tests run, no drift gate); the readme states "No safe wrappers" and "We will probably only implement the latest version".
4. **bindgen 0.73.2**: enum styles `rust{non_exhaustive}` / `newtype` / `consts` / `moduleconsts`. `rust_non_exhaustive` adds **only** the attribute — no catch-all — and bindgen documents the UB on unknown values and recommends `newtype` instead.
5. **Prior art**: kornia's `dlpack-rs` (forward-compatible consts; LP64 size asserts 48/64/80); **RESTGroup's own `dlpack-ffi`** on crates.io (published 2025-12-03; single hand-vendored `src/lib.rs`, bindgen 0.72.1, pinned minor 1.2, exhaustive `#[repr(u32)]` enums stopping at `kDLTrn=18`).
6. **Skill conventions**: `SKILL.md` procedure + `references/` per distribution + checker script; local paths live in `AGENTS.local.md` / `CLAUDE.local.md`, never hardcoded.

## Questions

### Q1 — Deliverable boundary

Options:

- **(a)** `rstsr-dlpack-ffi` **raw-only** — bindgen'd types + hand-written `lib.rs`, per the repo's "No safe wrappers" policy;
- **(b)** raw + a small safe RAII/deleter layer in the same crate (an explicit **policy exception** — justified only because the ownership protocol *is* DLPack's substance);
- **(c)** raw crate here + separate safe facade crate elsewhere.

Rider: **no rstsr-core changes** in this task. Note: no `dynamic_loading` feature is possible (nothing to load — the family's marquee feature is N/A).

➡️ Revised recommendation: **(a)**. The safe wrapper's natural home is the future consumer (rstsr-core), where the storage plumbing has to be built anyway.

### Q2 — Enum representation + struct contract

Raw-layer enum styles:

- **(a)** bindgen `NewType` — `#[repr(transparent)] struct DLDeviceType(pub c_int)` + associated consts; unknown values representable by construction; **bindgen's own documented remedy**;
- **(b)** `Consts` — plain `c_int` fields + global consts; simplest, in-family precedent (lapack's `blas`/`lapack` modules); weakest typing;
- **(c)** `rustified_non_exhaustive` — **rejected**: no catch-all, documented UB on unknown values, which DLPack minor releases *will* send.

Struct contract: freeze existing layouts; upstream additions arrive as new items; legacy `DLManagedTensor` exposed but documented deprecated; names mirror the C header exactly. Q1 coupling: with raw-only, ergonomic `#[non_exhaustive]` enums leave this crate and belong to the future consumer.

➡️ **(a)**, with `DLDataType.code` kept as the header's `u8` + consts (the header *intends* unknown codes to flow there).

### Q3 — Generator pipeline + gate

Options:

- **(a)** family-conformant: per-crate `scripts/gen_dlpack.py`, pinned bindgen CLI, output committed, scratch gitignored, no `build.rs`;
- **(b)** `build.rs` calling bindgen (needs libclang for every consumer; invisible diffs);
- **(c)** hybrid, feature-gated.

Rider: the family has **no CI drift job** and CI would not run tests anyway — follow that; keep the "correctness at update time" gate in the skill (Q4), do **not** add a CI job.

➡️ **(a)**, single generated module (family-style `ffi_*` name) + hand-written `lib.rs`; `--allowlist-file` on the whole vendored header, as the family does.

### Q4 — Skill organization

Options:

- **(a)** new sibling skill **`update-ffi-dlpack`** with **check** and **update** modes. Check = `git fetch --tags` in `~/Git-Others/dlpack`, diff `include/dlpack/dlpack.h` vs the pinned tag, classify (version macros / new enum values / new structs / behavioral notes like v1.2's strides rule); update = refresh `header/`, regenerate, provenance, changelog;
- **(b)** add `references/dlpack.md` under `update-ffi-blas` (the name lies; mechanics differ — no symbols, no library);
- **(c)** rename/generalize to `update-ffi` with per-distribution references (touches CLAUDE.md references).

➡️ **(a)**, registered in the rstsr-agents README tree; the `~/Git-Others/dlpack` path recorded in `CLAUDE.local.md` under Resources per convention.

### Q5 — Verification

Combine:

- **(a)** ABI/layout assertions (kornia's 48/64/80 are prior art; cfg-gate per pointer width);
- **(b)** round-trip behavioral test — build a `DLManagedTensor` over a Rust buffer, consume, deleter called exactly once;
- **(c)** real interop test via Python `numpy.from_dlpack` (pure Python) — the family CI runs **no tests at all**, so this can only be an `#[ignore]`d manual/documented test unless we change CI policy;
- **(d)** synthetic forward-compat test — hand-add an unknown device-type value to a copy of the header, prove the raw layer absorbs it (directly validates Q2).

➡️ **(a)+(b)+(d)** as crate tests run by the skill's verify step and locally; **(c)** as an ignored, skill-documented manual check; no CI policy change unless you want one.

### Q6 — Driving use case (what is the consumer?)

- **(a)** Rust-side interop substrate, export-first (rstsr tensors handed to other DLPack-speaking runtimes, C/Rust libraries);
- **(b)** enabler for a later rstsr-core task that reopens the dropped `from_dlpack` in Rust form;
- **(c)** standby/completeness — no consumer yet;
- **(d)** other (name it).

This no longer gates Q1 (raw-only is use-case-neutral), but it determines whether a follow-up consumer task is scheduled and how the old crate's deprecation is worded.

➡️ **(a)+(b) jointly**: declare export-first Rust interop as the driver; consumer work stays a separate future task; revisit the "dropped" record then.

### Q7 — Relationship to pre-existing `RESTGroup/dlpack-ffi`

- **(a)** **supersede + deprecate**: `rstsr-dlpack-ffi` becomes the maintained binding; old repo gets a deprecation README + final release pointing here;
- **(b)** supersede *and* keep API-shape compatibility so downstream code ports mechanically;
- **(c)** independent; **(d)** treat as a spike — ignore/delist.

➡️ **(a)**, stating the concrete reason in the deprecation note: its exhaustive `#[repr(u32)]` enums break on exactly the minor-version enum additions the new design absorbs.

### Q8 — Upstream pin

- **(a)** vendor the **v1.3 release tag** (family rule: newest release, never branch tip);
- **(b)** vendor HEAD (has `kDLTPU`/`kDLTPUHost`/`kDLAscend` early, but they are unreleased and could still be renumbered);
- **(c)** v1.3 + additionally define the three HEAD consts marked "unreleased".

➡️ **(a)** — the skill's check mode exists precisely to catch their official release.

### Q9 — Header surface

- **(a)** whole-header mirror: core structs, version macros, all flags, **legacy** `DLManagedTensor` (marked deprecated), **and** the v1.2+ exchange-API fn-pointer typedefs/structs as plain types;
- **(b)** curated subset (skip the exchange API until a consumer needs it);
- **(c)** v1-only (drop legacy structs).

➡️ **(a)** — the family binds whole headers via `--allowlist-file`, and (c) would cut off producers still emitting the legacy struct during the transition.

## Conventions treated as settled unless objected

All family-standard: crate name `rstsr-dlpack-ffi`, initial version `0.1.0`, workspace member + inherited edition/license; `header/dlpack.h` vendored at the pinned tag with tag+commit provenance recorded in the crate readme + root readme table; `#![no_std]`-friendly (pure types); branch `YYMMDD/…` pushed to the `ajz34` fork, PR against `RESTGroup/rstsr-ffi`; no commits without explicit instruction.

## Appendix — evidence highlights

- `include/dlpack/dlpack.h`, `DLPackVersion` docstring: *"A change in minor version indicates that we have added new code, such as a new device type, but the ABI is kept the same."* … *"Minor version updates indicate the addition of enumeration values."* … on major mismatch: *"the consumer must call the deleter (and it is safe to do so). It is not safe to access any other fields as the memory layout will have changed."*
- Same header, legacy struct: *"This data structure is used as Legacy DLManagedTensor in DLPack exchange and is deprecated after DLPack v0.8. Use DLManagedTensorVersioned instead. This data structure may get renamed or deleted in future versions."*
- Same header, `DLTensor.strides`: *"Before DLPack v1.2, strides can be NULL to indicate contiguous data. This is not allowed in DLPack v1.2 and later."*
- bindgen 0.73.2, `Builder::rustified_non_exhaustive_enum` doc: *"Use this with caution, creating an instance of a Rust `enum` with an invalid value will cause undefined behaviour, even if it's tagged with `#[non_exhaustive]`. To avoid this, use the `Builder::newtype_enum` style instead."*
- `rstsr-ffi/readme.md`: *"This project just perform FFI bindings. No safe wrappers"*; *"We will probably only implement the latest version."*

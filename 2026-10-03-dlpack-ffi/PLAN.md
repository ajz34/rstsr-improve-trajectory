# Plan: dlpack-ffi v1.3 redesign

- **Date**: 2026-10-03
- **Origin prompt**: [initial-prompt.md](./initial-prompt.md); decisions: [round-1-answers.md](./round-1-answers.md), [round-2-answers.md](./round-2-answers.md)
- **Bases**: dlpack-ffi `255a74c` (main, clean, tag v1.2.0); upstream dmlc/dlpack tag `v1.3` = `84d107b` (2026-01-24); design frozen 2026-10-03
- **G1**: passed by the owner ("You can continue"); **G2**: independent review agent over the finished diff before handoff

## Mission

Redesign the `RESTGroup/dlpack-ffi` crate for the DLPack v1.3 header, switching enum
representation to bindgen's **newtype** style so that future DLPack minor releases (which add
enum values) no longer break the API; repair the generation pipeline; add ABI/ownership tests;
add the `update-ffi-dlpack` skill (check + update modes) in rstsr-agents.

## Settled design (rounds 1–2)

| Area | Decision |
|---|---|
| Shape | Raw-only types binding; no safe wrappers; no rstsr-core work |
| Enums | bindgen `newtype` style; `DLDataType.code` stays `u8`; legacy struct + exchange-API types kept (whole-header mirror) |
| Header | v1.3 vendored; provenance (tag + commit) recorded in the readme |
| Pipeline | Repair existing `scripts/bindgen.py`; committed output; no `build.rs`; no CI drift job (checker lives in the skill) |
| Tests | ABI layout asserts + deleter round-trip + unknown-value openness (Q5 a/b/d); numpy interop ignored/manual (Q5 c) |
| Release | **1.3.0**; crate version tracks DLPack; no semver promise; adjustments = patch bumps |
| Purity | No "rstsr"/"rest" references anywhere in dlpack-ffi (content, docs, code, metadata) |
| Visibility | rstsr-agents (skill + ecosystem map) and rstsr-improve-trajectory may reference the package |
| Identity | Package/repo name stays `dlpack-ffi` |

## Work breakdown

### W1 — dlpack-ffi (branch `261003/v1.3`, uncommitted handoff)

1. **Header**: vendor `include/dlpack/dlpack.h` from upstream tag `v1.3` (`84d107b`).
2. **Script** (`scripts/bindgen.py`): `--default-enum-style rust` → `newtype`; fix stale
   `cargo fmt -p rstsr-dlpack-ffi` → `-p dlpack-ffi`; add post-processing only if the generated
   output needs it (derive parity on the newtype enums; `non_upper_case_globals` allow) — with
   an assert so silent non-match fails loudly.
3. **Regenerate** with bindgen 0.73.2 (repo previously generated with 0.72.1; stamp changes,
   record the upgrade in the readme changelog).
4. **Tests** (`tests/`): ABI sizes (`DLTensor` 48, `DLManagedTensor` 64,
   `DLManagedTensorVersioned` 80, `DLDeviceType` 4 on 64-bit; cfg-gated), deleter-called-exactly-once
   round-trip (legacy + versioned structs), unknown device-type/dtype-code openness. No new deps.
5. **LICENSE**: add Apache-2.0 (copied verbatim from a sibling repo).
6. **Readme rewrite**: provenance (v1.3 + commit), version policy (tracks DLPack, no semver),
   newtype rationale, changelog for 1.3.0; no rstsr references.
7. **Small cleanups**: `.gitignore` de-noise; edition 2018 → 2021 (keep `rust-version = "1.64.0"`);
   `Cargo.toml` version → 1.3.0 (+ `Cargo.lock` refreshed).

### W2 — rstsr-agents (uncommitted handoff)

- New skill `skills/update-ffi-dlpack/`:
  - `SKILL.md`: check mode (fetch tags in `~/Git-Others/dlpack`, diff header vs pinned tag,
    classify version-macro/enum/struct/doc changes) and update mode (refresh vendored header,
    regenerate, verify, provenance, version = DLPack version, purity rule, uncommitted handoff).
  - `scripts/check_dlpack_bindings.py`: version-const consistency (header vs `src/lib.rs` vs
    readme), newtype style presence (no `pub enum DLDeviceType`), `DLDataType.code: u8`, purity
    grep, optional `--upstream` byte-parity of the vendored header + provenance hash check.
- Register in the skills README tree; add a `dlpack-ffi` line to the ecosystem map in CLAUDE.md.

### W3 — verification (gate before handoff)

- `cargo fmt --all -- --check`, `cargo clippy --all-targets --all-features -- -D warnings`,
  `cargo test` (all green).
- Regeneration idempotence: re-run `scripts/bindgen.py` → `src/lib.rs` unchanged.
- Checker script green against `--upstream ~/Git-Others/dlpack --tag v1.3`.
- Purity grep (`rstsr`) over the dlpack-ffi tree: empty.
- **G2**: independent review agent audits the dlpack-ffi diff + skill files against this plan;
  findings fixed or recorded before handoff.

### W4 — handoff

- Leave both repos **uncommitted** (house rule; commits/pushes only on explicit instruction).
- Record `dlpack-ffi.patch` (diff incl. new files) in this directory; write the campaign
  standoff report; commit those in rstsr-improve-trajectory only.
- Human: review → commit → release via the existing release-plz `workflow_dispatch` (version
  already set to 1.3.0 by hand so release-plz publishes rather than recomputing a major).

## Risks / details to watch

- **Release-plz version computation**: release-plz derives versions from commits; mitigate by
  hand-setting 1.3.0 in Cargo.toml. Human dispatches the release.
- **Newtype emits**: watch for missing derives (old enums had `Hash/PartialEq/Eq`) and for
  `non_upper_case_globals` warnings on `kDL*` constants; fix in the script's post-processing with
  asserts, not by hand-editing the generated file.
- **MSRV**: tests must build with `rust-version = "1.64.0"` (no `offset_of!`, no new std APIs).
- **Purity grep**: match `rstsr` (word) — the word "rest" alone would false-positive on ordinary
  English; check the org name instead.

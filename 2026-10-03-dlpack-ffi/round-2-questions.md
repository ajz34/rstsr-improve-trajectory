# Round-2 design questions — dlpack-ffi repo redesign

- **Date**: 2026-10-03
- **Status**: **OPEN** — filed after the repo-anatomy report (appendix). R2-Q1 gained a tension note (the repo's own readme policy argues against my lean); R2-Q3 became a concrete upgrade checklist.
- **Context**: round 1 ([round-1-answers.md](./round-1-answers.md)) relocated the task into the existing `RESTGroup/dlpack-ffi` repo (Q7). Round 2 settles: release version, package identity, upgrade scope, pack wiring.

## Repo facts that matter here (full digest in the appendix)

- The repo **already follows most family conventions**: vendored `header/dlpack.h` (v1.2), `scripts/bindgen.py` (bindgen 0.72.1; `--allowlist-file`, `--default-enum-style rust`, `--no-layout-tests`, `--use-core`, `--merge-extern-blocks`), committed generated `src/lib.rs` (225 lines; nothing hand-written but 3 fixed lines), fmt/clippy CI, release-plz `workflow_dispatch` (the v1.2.0 tag was cut by the bot).
- **Known defects/tensions to fix**: `--default-enum-style rust` → exhaustive enums (the redesign's target); the script's final line `cargo fmt -p rstsr-dlpack-ffi` is a stale transplant (the package is `dlpack-ffi`); **no LICENSE file** (Cargo.toml declares Apache-2.0); the readme's policy line *"We will not add anything other than bindgen automatically generated code"* conflicts with adding tests; `.gitignore` carries copy-paste noise from rstsr-ffi.
- **Exposure**: crates.io has exactly one version (1.2.0, 108 downloads); **no consumers anywhere in the pack**; the checkout has **no agent wiring** (no `.claude`/`.agents`/`CLAUDE.md`/`AGENTS.md`).

## R2-Q1 — Version of the redesigned release

The crate is 1.2.0 and its readme states *"Version of this crate will follow the DLPack version"*. The redesign is a one-time Rust-API break (rustified enums → `NewType`); after it, future DLPack minor bumps are non-breaking by construction.

- **(a)** `2.0.0` — decouple crate semver from DLPack's; keeps `dlpack-ffi = "1.2"` users from silently receiving an API-breaking match-surface on `cargo update`; readme records "binds DLPack v1.3" (family precedent: `rstsr-lapack-ffi` 0.5.1 binds LAPACK 3.12.1). Costs: breaks the stated mnemonic; readme line must be rewritten.
- **(b)** `1.3.0` — keep the repo's own policy (crate version == DLPack version); the break is documented in the changelog only; the ~108-download user base is unknown, and a minor bump would upgrade them silently.

➡️ **(a) 2.0.0** — the tension is real (the repo's own readme argues for (b)), but a silent breaking minor bump is exactly the failure mode this redesign exists to prevent; pay the honest major once, then never again. If you prefer policy continuity over semver hygiene, (b) is defensible given near-zero adoption.

## R2-Q2 — Package identity

Keep the crates.io name `dlpack-ffi`, or rename to `rstsr-dlpack-ffi` (new package; old one deprecated/left)?

➡️ **Keep `dlpack-ffi`** — published identity, repo name, and the short generic name is a feature; the rstsr linkage lives in docs and the shared skill.

## R2-Q3 — Upgrade checklist (confirm, or strike items)

Given the anatomy, this is not a transplant but a **repair + upgrade of the existing pipeline**. Recommended items:

1. **Enums → `NewType`**: script flag `--default-enum-style newtype`; regenerate. The two enums (`DLDeviceType` 18 variants, `DLDataTypeCode` 18 variants, both currently `#[repr(u32)]` exhaustive) become `#[repr(transparent)]` newtypes + associated consts. `DLDataType.code` stays `u8`. (This is the one-time API break of R2-Q1.)
2. **Refresh `header/dlpack.h` to v1.3** (from `~/Git-Others/dlpack` at the v1.3 tag); record tag + commit hash as provenance in the readme (the repo has no provenance record today — only the bindgen stamp).
3. **Fix the script**: stale `cargo fmt -p rstsr-dlpack-ffi` → `-p dlpack-ffi`; keep the name `scripts/bindgen.py` and the existing invocation/post-processing otherwise.
4. **Add tests** (round-1 Q5): (a) layout assertions (`DLTensor` 48, `DLManagedTensor` 64, `DLManagedTensorVersioned` 80 on 64-bit; cfg-gated), (b) round-trip deleter-exactly-once, (d) synthetic unknown-enum-value test.
5. **Add `LICENSE`** (Apache-2.0) — currently declared in Cargo.toml only.
6. **Readme rewrite**: provenance (DLPack v1.3 + commit), changelog, and amend the "nothing but generated code" line to permit tests + generation tooling while keeping the no-hand-written-API-surface policy.
7. **Small cleanups**: `.gitignore` de-noise; optionally edition 2018 → 2021 (keep `rust-version = "1.64.0"`).

➡️ **All seven.** Open sub-point: whether CI stays fmt+clippy only (tests run locally / by the skill) — round-1 Q5 said yes, restated here for the record.

## R2-Q4 — Pack wiring

- **(a)** `dlpack-ffi` joins the pack: checkout already sits at `/home/a/rstsr_pack/dlpack-ffi`; add the `.claude`/`.agents`/`CLAUDE.md`/`AGENTS.md` symlinks (skill `agent-setup`), a line in the rstsr-agents ecosystem map, and register `update-ffi-dlpack` in its skills README tree;
- **(b)** no wiring: keep the checkout bare, skill referenced by path;
- **(c)** checkout stays, instruction-map edits only.

➡️ **(a)** — the checkout is already in the pack; only the wiring is missing.

---

## Appendix — dlpack-ffi anatomy digest (2026-10-03)

- **Checkout**: `/home/a/rstsr_pack/dlpack-ffi`, HEAD `255a74c` ("move workflow files", 2025-12-03, ajz34), tag `v1.2.0` on the same commit (cut by release-plz bot), remote `git@github.com:RESTGroup/dlpack-ffi.git`, clean tree.
- **History**: 2 commits, one author: `1dc0b0b` "dlpack v1.2" + `255a74c` "move workflow files" (pure rename of `workflows/` → `.github/workflows/`).
- **Tracked files** (11): `.github/workflows/{clippy,release-plz}.yml`, `.gitignore`, `Cargo.lock`, `Cargo.toml`, `header/dlpack.h`, `readme.md`, `rustfmt.toml`, `scripts/bindgen.py`, `src/lib.rs`.
- **`Cargo.toml`** (verbatim, 12 lines): name `dlpack-ffi`, version 1.2.0, edition 2018, `rust-version = "1.64.0"`, license Apache-2.0, empty `[dependencies]`, no `[lib]`/`[features]`.
- **`readme.md`** (verbatim, complete):
  > This is a minimal FFI binding to [DLPack](https://github.com/dmlc/dlpack). DLPack is an open in-memory tensor structure for sharing tensors among frameworks.
  >
  > Current FFI version is DLPack [v1.2](https://github.com/dmlc/dlpack/releases/tag/v1.2). Version of this crate will follow the DLPack version. We do not add any safe-wrappers or abstractions on top of the FFI layer.
  >
  > This crate is not official bindgen project. This project will only serve as a minimal binding layer for users who want to build their own safe abstractions on top of DLPack FFI. We will not add anything other than bindgen automatically generated code.
- **`src/lib.rs`**: 225 lines, 100% bindgen output + 3 fixed lines (`#![allow(non_camel_case_types)]`, `#![allow(non_snake_case)]`, `use core::ffi::*;`), bindgen stamp 0.72.1. Public surface: 5 consts (`DLPACK_MAJOR_VERSION=1`, `DLPACK_MINOR_VERSION=2`, 3 flag bitmasks), 8 structs (`DLPackVersion`, `DLDevice`, `DLDataType`, `DLTensor`, `DLManagedTensor`, `DLManagedTensorVersioned`, `DLPackExchangeAPIHeader`, `DLPackExchangeAPI`), 5 fn-pointer type aliases (the exchange API), 2 exhaustive `#[repr(u32)]` enums. Layout fields match the upstream v1.2 header exactly.
- **`scripts/bindgen.py`** (71 lines): copies `header/` → `tmp/`, runs the CLI invocation quoted above, post-processes (`replace("::core::ffi::", "")`, `replace("::core::option::", "")`), prepends the 3 lines, writes `src/lib.rs`, then the stale `cargo fmt -p rstsr-dlpack-ffi`.
- **CI**: `clippy.yml` (rustfmt check + clippy `-D warnings`, on push/PR); `release-plz.yml` (`workflow_dispatch` only, `release-plz/action@v0.5`). No test job, no drift job.
- **crates.io**: 1 version (1.2.0, 2025-12-03), 108 downloads (67 recent), 11.9 KB. **GitHub**: public, not a fork, 0 stars/issues/PRs, `licenseInfo: null` (no LICENSE file).
- **Consumers**: none in the pack (`grep` across all Cargo.tomls finds only the crate itself). rstsr-core's only dlpack references are the "dropped `from_dlpack`" notes (round-1 fact base).

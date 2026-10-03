# Report: dlpack-ffi v1.3 redesign — implementation

- **Date**: 2026-10-03
- **Plan**: [PLAN.md](./PLAN.md) (G1 passed); **G2**: independent review agent — no blockers/majors, findings fixed
- **Artifacts**: [dlpack-ffi.patch](./dlpack-ffi.patch), [rstsr-agents.patch](./rstsr-agents.patch)
- **State**: both target repos **uncommitted** (house rule) — awaiting human review / commit / release

## What changed

### dlpack-ffi (branch `261003/v1.3`; base `255a74c`; 7 files modified, plus `LICENSE` and `tests/`)

| Change | Detail |
|---|---|
| Header | vendored from upstream tag `v1.3` = `84d107b` (replaces v1.2) |
| Generator | `scripts/bindgen.py`: `--default-enum-style newtype`; added `#![allow(non_upper_case_globals)]`; fixed stale `cargo fmt -p rstsr-dlpack-ffi` -> `-p dlpack-ffi` |
| Generated | `src/lib.rs` regenerated with bindgen 0.73.2; `DLDeviceType` / `DLDataTypeCode` are now `#[repr(transparent)]` newtypes with associated `kDL*` constants — the one-time API break; future DLPack minor additions no longer break the API |
| Tests | `tests/abi.rs` (layout 48/64/80, newtype sizes, known values, unknown-value openness, derive parity) + `tests/ownership.rs` (legacy and versioned deleter round-trips) |
| Metadata | version 1.3.0 (tracks DLPack; no Rust semver), edition 2021 (rust-version 1.64.0 kept), `LICENSE` (Apache-2.0) added, readme rewritten (provenance, policy, changelog), `.gitignore` de-noised, `[lib] doctest = false` (generated docs embed C/C++ examples) |

### rstsr-agents

- New skill `update-ffi-dlpack`: `SKILL.md` (check + update modes) and
  `scripts/check_dlpack_bindings.py` (19 checks: version/const consistency, newtype-style
  invariants, all-34 enum-value parity vs the header, generator-version provenance, test-suite
  presence, purity, upstream byte parity).
- README skills-tree line; `AGENTS.md` ecosystem-map bullet; pack `CLAUDE.local.md` resources line.

## Verification (all green at final state)

| Gate | Command | Result |
|---|---|---|
| fmt | `cargo fmt --all -- --check` | exit 0 |
| clippy | `cargo clippy --all-targets --all-features -- -D warnings` | exit 0 |
| tests | `cargo test` | 7/7 (5 abi + 2 ownership) |
| regen idempotence | re-run `scripts/bindgen.py` | `src/lib.rs` md5 unchanged (`87315dba…`) |
| checker | `check_dlpack_bindings.py --repo … --upstream ~/Git-Others/dlpack --tag v1.3` | 19/19 ok |
| checker negative test | mutated `kDLWebGPU` value in a `/tmp` copy | FAIL as expected (exit 1) |
| purity | `grep -rin rstsr` over the tree | empty |
| header parity | `git show v1.3:include/dlpack/dlpack.h \| diff -` | empty |

## G2 findings and resolutions

- **F1 (minor)** SKILL.md check-mode `git tag` lacked `-C <upstream>` → fixed.
- **F2 (minor)** no guard against a future bindgen silently dropping newtype derives → added a
  compile-time trait assertion (`enum_derives_preserved`) in `tests/abi.rs`.
- **F3 (minor)** checker blind spots → added all-34 const-value parity vs the header
  (negative-tested), bindgen-version vs readme cross-check, tests-suite presence and
  `doctest = false` checks, precise purity label, clean git-failure handling, escaped regex.
- **F4/F5/F6 (nits)** over-claims / changelog omissions → fixed in `AGENTS.md`, `readme.md`.
- **F7 (accepted)** `[lib] doctest = false` is an extra (8th) change — empirically necessary
  (rustdoc tries to compile doxygen C++ examples), documented in `Cargo.toml` and the skill.
- **F8 (pre-existing, no action)** rustdoc warnings from doxygen text; ignored `debug/` scratch.
- **F9 (info)** W4 artifacts — this report and the two patches.

## Handoff (human)

1. Review: the two patch files, or `git diff` / `git status` in each checkout (both left
   uncommitted on purpose).
2. **dlpack-ffi**: commit on `261003/v1.3`, push fork-first (`ajz34` → PR against
   `RESTGroup/dlpack-ffi`); release 1.3.0 via the existing release-plz `workflow_dispatch`
   (version already set by hand so release-plz publishes rather than recomputing a major).
3. **rstsr-agents**: commit the skill + index/map edits (repo exempt from the branch rule).
4. Create the `ajz34/dlpack-ffi` fork first if it does not exist.

## Notes

- No rstsr-core changes; no rstsr-ffi changes; no CI changes; no new dependencies; no `build.rs`.
- Purity: dlpack-ffi contains zero "rstsr" references (tracked + untracked); `RESTGroup` in the
  repository URL is the expected org identity.
- Benchmark/experiment character: this campaign is a binding-maintenance task, so no benchmark
  numbers apply; the verification table above is the evidence record.

## Execution update (2026-10-03)

- **dlpack-ffi**: committed on `261003/v1.3` as `e71bea0` (vendor v1.3 header), `1f6adf9`
  (newtype regeneration), `8d58a61` (tests + metadata for 1.3.0); pushed to the new fork
  `ajz34/dlpack-ffi`; **PR opened: https://github.com/RESTGroup/dlpack-ffi/pull/1**.
- **rstsr-agents**: skill + index/map edits committed on `main` as `e3a61b6` (ahead of
  `origin/main` by 1, **not pushed** — push on instruction).
- The 1.3.0 release remains a human-dispatched release-plz `workflow_dispatch` after the PR merges.

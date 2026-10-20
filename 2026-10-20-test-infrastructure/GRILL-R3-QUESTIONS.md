# Grill R3 — test infrastructure redesign

## R2 answers (agent transcription of the user's GRILL-R2-ANSWERS.md, 2026-10-20)

- **Q18 (absorb)**: `test-conventions` is **deleted** — the name misleads on both
  field (rstsr-core only) and content (conventions now live at rule level). Its
  mechanics move somewhere; the `core-*` skills re-point. Target = Q25 below.
- **Q19 (a)**: `test_issues/` symlinked into device crates; per-device skips via
  `TESTCFG`; backend-specific bugs keep per-crate `tests/issues/`.
- **Q20**: benchmarks = future implementation; home `crates-bench/`, **multiple
  crates** with different names per scenario/device (not one `rstsr-bench`).
- **Q21 (good)**: per-device-crate `tests/README.md` records entry set +
  divergences.
- **Q22 (amended — 4 tiers)**: T4 added. T1/T2 GitHub-triggerable; part of T3 too.
  Workflow files renamed `tier<N>-<surface>.yml`; clippy/rustfmt stay outside the
  tiers (default, always). T2 is often not triggered — practically closer to T3
  than T1. Assignments:
  - **T1 (every PR)**: core row cpu (+lib+doctests), core row faer (new), core col
    cpu (**future**, assigned T1 now), openblas conformance + driver/linalg,
    facade cheatsheet, linalg-traits.
  - **T2 (scheduled/path-triggered)**: cpu-dlpack, openblas ARM.
  - **T3 (pre-release + partially GitHub-triggerable)**: no_std builds,
    rstsr-faer-py conformance, CUDA (probably; → T2 if a GPU-capable CI vendor
    like CircleCI appears).
  - **T4 (manual / big-version-change)**: benchmarks (crates-bench),
    mkl/blis/aocl/kml for now.
- **Q23 (okay)**: `tests/allocatable/` body; `tensor_sum.rs` deleted.
- **Q24 (okay + protocol)**: multiple commits on a **new branch**; **no
  auto-commit**; **stop after each task for review** before proceeding.

---

## Q25 — Where do the absorbed mechanics land?

`test-conventions` dies. Its mechanics (provenance format, `specify_test!`/
`TestCfg` usage, creation idioms, `assert_equal`, tracking-CSV hygiene, NumPy
comment pairing) go where?

- (a) **Into `core-test`** (the authoring skill): the rule keeps policy + routing
  only; `core-test` gains a "conventions" section; `core-issue-regression` and
  `core-numpy-sync` re-point from `test-conventions` to `core-test` (or directly
  to the rule for policy bits).
- (b) Into `rules/testing.md` itself (rule becomes policy + mechanics).

➡️ (a). Q2 settled the router pattern: the rule stays policy-level; mechanics are
needed exactly when authoring a core test, which is when `core-test` loads. (b)
would recreate the command-drift problem the router exists to avoid — and the
rule must stay rstsr-core-agnostic now that it governs all surfaces.

## Q26 — Skill gap list (final contents)

Proposed final skill roster changes, written at rule time (after refactor):

1. **New skill `device-test`** (name tentative): authoring/wiring tests in device
   crates — entry file + `TESTCFG`, symlink set (`core_func`, `test_utils`,
   `test_issues`), the `blas_tests_impl` driver/linalg bodies + SciPy validators,
   dedicated-file exception pattern (KML precedent) + the `tests/README.md` duty,
   `--test-threads=1` / `RSTSR_DEV=1` specifics (from `cargo-inst`).
2. **`core-col-major-transfer` (stub) deleted**: superseded by the new-entry
   guideline in the rule; if the col-major integration task (B5) later wants a
   dedicated skill, create it then, informed by that task.
3. **Benchmark skill: not created now** — one routing line in the rule points to
   `crates-bench/` (future) and to this repo's charter for experiments.
4. `cargo-inst`, `rstsr-faer-py-tests`, `api-cheatsheet-conventions`,
   `api-doc-conventions` (doc_draft parts), `core-test`, `core-issue-regression`,
   `core-numpy-sync`: unchanged (minus re-pointing per Q25).

- (a) Accept list.
- (b) Modify (say what — e.g. keep the col-major stub, different device-skill
  name, add an FFI-testing skill).

➡️ (a). The list matches the gaps the exploration actually found; nothing else
came back undocumented.

## Q27 — Glossary placement for the new terms

Today: `rstsr-agents/CONTEXT.md` is docs-only by its own header; testing terms
live in `rstsr-core/tests/CONTEXT.md`. The rule introduces ecosystem-level terms
(tier 1–4, entry binary kinds, bodies, conformance entry, dedicated-file
exception).

- (a) **Split by scope**: ecosystem-level testing terms (tiers, entry kinds,
  sharing mechanisms) go into `rstsr-agents/CONTEXT.md` (dropping its docs-only
  scoping); rstsr-core-specific terms (parity test, provenance, numpy identifier,
  `specify_test!`, ...) stay in `rstsr-core/tests/CONTEXT.md`; the rule links
  both.
- (b) Everything testing-related moves to `rstsr-agents/CONTEXT.md`.
- (c) Define terms inline in the rule only.

➡️ (a). The two audiences differ (any-repo agent vs core-test author); the core
glossary is already loaded by the authoring flow and sits next to the tests it
describes.

## Q28 — New-entry checklist shape (the Q6 "a+" deliverable)

The rule's "adding a new entry" section. Two entry kinds, one checklist each:

**Device entry** (`entry_row_<device>.rs`, e.g. faer now, CUDA later):
1. Real entry file: `type DeviceType` alias + `static TESTCFG` (device, order,
   skip/allow), `[[test]] required-features`, dev-deps (manifest loader for BLAS
   crates).
2. Symlink set: `core_func/`, `test_utils/`, `test_issues/` from rstsr-core;
   declare `mod` lines; BLAS crates additionally symlink `driver_impl/` +
   `linalg_func/` from `rstsr-blas-traits/src/blas_tests_impl/`.
3. Divergences: dedicated real files replace specific symlinks only when
   semantics genuinely differ (KML `gesvd.rs` precedent); each recorded in the
   crate's `tests/README.md`.
4. CI: assign tier; workflow file `tier<N>-<surface>.yml`; add to the tier ×
   device matrix in the rule.
5. Update the rule's surface table + `tests/README.md`.

**Order entry** (`entry_col_cpu.rs`-style, col-major integration later):
1. New body dir (e.g. `col_func/`) — order-specific semantics tests; the shared
   row body is NOT reused blindly since some functions genuinely differ by order
   (contract: `src/docs/order_semantics.md`).
2. Entry file with `col_major`-gated `required-features`; per-test order pinning
   where a test must neutralize the crate default.
3. Existing verified wiring patch (2026-09-14 campaign) is the reference
   implementation to adapt.
4. CI tier assignment (col cpu = T1 per R2) + matrix update.

- (a) Accept both checklists as the guideline's backbone.
- (b) Modify (say what's missing/wrong).

➡️ (a). It encodes everything this grill settled; the col-major branch matches
both your "second class, integrated very near" stance and the patch's actual
shape (separate `col_func` body, not a symlinked reuse).

## Q29 — Branch strategy for the refactor

- (a) **One branch** for B1–B4 (e.g. `261020/test-infra`), one commit per task,
   stop for your review after each task; merge when all batches are approved.
   rstsr-agents edits (rule, ADR, skills) happen on its `main` (allowed there),
   same stop-per-task protocol, no commits without your go.
- (b) Branch per batch, merged sequentially.

➡️ (a). B2's CI work and B3's body adjustments build on B1's renames; sequential
branches would add merge overhead without isolation benefit. Per-task review
stops give you the control you asked for.

---

## Answering

Edit `GRILL-R3-ANSWERS.md` (user-owned) or answer on screen. After this round the
frontier should be empty → closure writes `DECISIONS.md` in this directory
summarizing the shared understanding (no rstsr/rstsr-agents edits until you
confirm and give the go).

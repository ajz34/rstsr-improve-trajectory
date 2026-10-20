# DECISIONS — test infrastructure redesign (grill closure, 2026-10-20)

R4 answers (transcribed from screen, agent-authored record): **Q30 (a)** accepted
— renamed skills `test-core` / `test-core-issues` / `test-core-numpy-sync` /
`test-device`. **Q31 (b)** per user suggestion — merged testing glossary lives in
`rstsr-agents/CONTEXT.md`; the core file is migrated and deleted.

Status: shared understanding reached over R1–R4 (questions in `GRILL-R*-QUESTIONS.md`,
user answers in `GRILL-R*-ANSWERS.md`). **Execution awaits explicit user go.**

## D1 — The rule: `rstsr-agents/rules/testing.md`

- Policy-level authority (taxonomy + placement + process): what test categories
  exist, where each lives, what each must cover; regression mandate; CI tiering;
  benchmark policy. Router pattern to skills (like `rules/api-doc-conventions.md`);
  exact commands stay in skills. Rules rank above skills; skills may restate policy
  for their own context.
- Full-enumbership umbrella: every test surface gets at least a routing line (core
  matrix, device crates, `blas_tests_impl` bodies, member crates, facade cheatsheet,
  rstsr-faer-py conformance, rstsr-ffi verified-via-device-crates stance, no_std
  build checks, future benches/CUDA). Deep sections only for core, device, and
  new-entry onboarding.
- Escape hatch: cases the rule does not categorize → ask the human (no silent
  improvisation).
- Written LAST, after the refactor, describing the refactored reality.

## D2 — New-entry guidelines (the rule's backbone)

Two checklists:

**Device entry** (`entry_row_<device>.rs`): real entry file (`type DeviceType` +
`static TESTCFG` + `[[test]] required-features` + dev-deps); symlink set
(`core_func/`, `test_utils/`, `test_issues/`; BLAS crates additionally
`driver_impl/` + `linalg_func/` from `rstsr-blas-traits/src/blas_tests_impl/`);
dedicated real files may replace specific symlinks when semantics genuinely
differ (KML `gesvd.rs` precedent) and must be recorded in the crate's
`tests/README.md`; CI tier assignment + `tier<N>-<surface>.yml` workflow; update
the rule's tier × device matrix.

**Order entry** (e.g. `entry_col_cpu.rs`): separate order-specific body
(`col_func/`-style) — the shared row body is NOT blindly reused since some
functions genuinely differ by order (contract: `src/docs/order_semantics.md`);
`col_major`-gated required-features; per-test order pinning where needed; the
verified 2026-09-14 wiring patch (`2026-09-14-soundness-check/T2-col-major/`)
is the reference implementation.

Col-major status: **second class**, a known gap to be addressed very near — as a
separate task (B5) consuming this guideline, not as part of this grill's refactor.

## D3 — CI: tier × device matrix with context gating

- **T1 (every PR, GitHub)**: core row cpu (lib + doctests + integration), core
  row faer (new), core col cpu (assigned T1; not implemented yet), openblas
  conformance + driver/linalg, facade cheatsheet, linalg-traits.
- **T2 (scheduled/path-triggered, GitHub)**: cpu-dlpack (incl. col variant),
  openblas ARM. In practice rarely triggered — closer to T3 than T1.
- **T3 (pre-release; partially GitHub-triggerable)**: no_std builds,
  rstsr-faer-py conformance suite, CUDA (probably; moves to T2 if a GPU-capable
  CI vendor such as CircleCI appears).
- **T4 (manual / big-version-change)**: benchmarks (`crates-bench/`, future),
  mkl / blis / aocl / kml for now.
- Workflow files renamed `tier<N>-<surface>.yml`; clippy/rustfmt stay as-is
  (default, always, outside the tiers).
- Context gating: common PR → T1; PR touching a device crate's (or rstsr-ffi's)
  paths → that device's jobs also run; large-refactor/release PRs → `ci:full`
  label → all T2; release process → T3/T4 checklists. New devices pick their
  column at onboarding (D2 checklist).

## D4 — Regression-test mandate

Every bug fix adds a test, placed by category: core issues → `test_issues/`
(one test per ticket, issue-number citation — NumPy `test_regression.py` pattern);
backend-specific bugs → the device crate's own `tests/issues/`; API/edge cases →
`custom_*` modules in `core_func`. `test_issues/` is symlinked into device
crates so core issue repros run on every backend, with per-device skips via
`TESTCFG` where the bug is device-specific. Terminology: "NumPy parity" (not
"NumPy style") for transferred tests tracked by the CSV.

## D5 — Sharing policy

Sanctioned mechanisms, named in the rule: symlinked shared body, symlinked
device extras (`blas_tests_impl`), per-crate real entry files, dedicated-file
exceptions replacing individual symlinks. Every divergence from the shared body
is recorded in the owning crate's `tests/README.md` (which also documents the
crate's entry set). No `include!`.

## D6 — Body adjustments in rstsr-core

- `tests_core_row.rs` → `entry_row_<device>.rs` in all five device crates
  (uniform, ADR-consistent; `tests_driver_impl` / `tests_linalg_func` keep
  their names).
- `entry_row_faer` wired (TESTCFG skip list for legitimate faer divergences).
- `doc_draft` NOT symlinked into device crates (ADR-0002 amended to bless
  reality: `core_func` + `test_utils` only — doctests already cover docstring
  examples under default features, and faer is the default backend).
- `tensor_sum.rs` deleted. `allocatable_dtype` becomes `tests/allocatable/`
  body, declared by core entries only (device entries never see it;
  `num-bigint` stays a core-only dev-dep).
- No new test obligations for `rstsr-native-impl` / `rstsr-tblis` beyond
  documentation in the rule's surface table (indirect coverage as today).

## D7 — Benchmarks

Future implementation only. Home: `crates-bench/` with **multiple** non-published
crates per scenario/device (`publish = false`; criterion already a workspace
dep). Never CI-gated (T4). Ad-hoc experiments and numbers stay in
rstsr-improve-trajectory per its charter; the main repo receives reports inside
PRs only. No benchmark skill yet — the rule carries a routing line.

## D8 — Skills (rstsr-agents, after the refactor)

- `test-conventions` **deleted** (absorbed): misleading name and scope. Policy →
  the rule; authoring mechanics (provenance format, `specify_test!`/`TestCfg`
  usage, creation idioms, `assert_equal`, tracking-CSV hygiene) → `test-core`.
- Renames under the `test-` prefix: `core-test` → **`test-core`** (+ conventions
  section), `core-issue-regression` → **`test-core-issues`**,
  `core-numpy-sync` → **`test-core-numpy-sync`**; new **`test-device`** skill
  (device-crate entry wiring, symlink sets, `blas_tests_impl` bodies + SciPy
  validators, dedicated-file exception + `tests/README.md` duty,
  `--test-threads=1` / `RSTSR_DEV=1` specifics). Other skills keep their names.
- `core-col-major-transfer` stub **deleted** (superseded by D2; B5 creates its
  own skill if it needs one).
- Test relocations / detailed changes within these skills' scope are in-scope
  for the refactor.
- Gap list is maintained and updated when backends/categories are added.

## D9 — Glossary

Merged testing glossary in **`rstsr-agents/CONTEXT.md`** (user preference, R4
Q31b): it drops its docs-only scoping and gains the testing terms — ecosystem
level (tiers, entry kinds, sharing mechanisms, conformance entry,
dedicated-file exception) and the migrated core terms (parity test, provenance,
numpy identifier, `specify_test!`, ...). `rstsr-core/tests/CONTEXT.md` is
deleted after its references are re-pointed during the skill batch.

## D10 — ADRs

Structural decisions get ADRs (amend ADR-0002: entry naming, `doc_draft`
blessing, the `blas_tests_impl` third body, the dedicated-file exception
pattern + README duty). Rule = policy; ADRs = why.

## Execution protocol

One branch `261020/test-infra` in rstsr for B1–B4; one commit per task; **no
auto-commit** (main-repo policy); stop after each task for user review.
rstsr-agents edits on its `main` (branch rule exempt there), same no-auto-commit
and stop-per-task protocol.

Batch order:

1. **B1** mechanical: device entry renames + ADR-0002 amendment.
2. **B2** `entry_row_faer`: wiring + skip list + `tier1-*` CI job.
3. **B3** body adjustments: `test_issues/` symlinks; `tensor_sum.rs` deletion;
   `tests/allocatable/`; per-crate `tests/README.md`.
4. **B4** CI rework: tier matrix, workflow renames, gating.
5. **B5** (separate near-future task): col-major integration via the verified
   wiring patch, against the D2 guideline.
6. **Then**: `rules/testing.md` + D8/D9 skill and glossary changes in
   rstsr-agents, summarizing the refactored reality.

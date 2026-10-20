# Grill R1 — test infrastructure redesign

Round 1 consolidates all questions asked so far in-session (Q1–Q11 from the
first two rounds, Q12–Q17 from the fact rounds), updated with everything the
two exploration reports and the trajectory repo's records revealed. Recommended
answers marked ➡️; "yes" accepts the recommendation unless noted.

## Factual baseline (agent-authored, from repo exploration 2026-10-20)

Both maps below are settled facts, not questions.

**rstsr main repo** (`main` @ ff0f250, clean):

- Only `entry_row_cpu` exists. `entry_row_faer`, `entry_col_cpu`, `tests/col_major/` do NOT exist. CI's col-major job runs `--lib` only. Test bodies hard-set `set_default_order(RowMajor)` even in the row entry.
- Device crates (OpenBLAS/MKL/BLIS/AOCL/KML): `tests/core_func` + `tests/test_utils` are symlinks to rstsr-core; per-crate entry `tests_core_row.rs` (real file); `driver_impl/` + `linalg_func/` are symlinks into `rstsr-blas-traits/src/blas_tests_impl/` (shared BLAS test source lives in `src/`, with SciPy/NumPy validation scripts); one real-file exception (KML `gesvd.rs`); OpenBLAS-only extras (`tests/issues/`, `zero_size/`, `test_workable`, `trait_bound`); `doc_draft` NOT symlinked into device crates.
- CI: core row integration + doctests; col-major `--lib` only; OpenBLAS (openmp/pthread + ARM); facade cheatsheet; dlpack; linalg-traits; no_std build checks. NO CI for MKL/BLIS/AOCL/KML, no col-major integration, no faer entry.
- Benchmarks: none. `criterion` + `cpu-time` declared but unused; `tensor_sum.rs` does ad-hoc `Instant` timing prints.
- Sharing is entirely symlinks (435 committed, mode 120000); zero `include!`. No `rstest` in the repo. `rstsr-native-impl` and `rstsr-tblis` have no integration tests. `rstsr-ffi` is test-free by design (functionally exercised via the device crates).
- Tracking: `numpy_coverage.csv` (193 transferred / 19 partial / 8 todo / 43 not-applicable), `numpy_differences.md` (+`_resolved`), `doc_coverage.csv`; oracle `sync_numpy.py` (never edits tests).

**rstsr-agents** (`main` @ 15e3643):

- Test-related skills: `test-conventions` (the de facto core-test guideline: taxonomy, `specify_test!`/`TestCfg`, provenance, creation/comparison conventions), `core-test` (author parity+doc_draft+docstring as one unit), `core-issue-regression` (test_issues/issue_<n>.rs), `core-numpy-sync` (version-bump oracle workflow), `core-col-major-transfer` (explicit STUB), `cargo-inst` (all commands), `rstsr-faer-py-tests`, `api-cheatsheet-conventions` (test aspects), `api-doc-conventions` §5/§5.1 (doc_draft procedure).
- `rules/` has only `code-concepts.md` and `api-doc-conventions.md` (a router). No `rules/testing.md`.
- ADR-0002 (entry-binary matrix): shared body `mod`-included by entry binaries; device crates symlink the body; only `entry_row_cpu` wired. **Drift**: device entries named `tests_core_row.rs` not `entry_<device>.rs`; `doc_draft` not symlinked into device crates despite the ADR body including it. A third body (`driver_impl`/`linalg_func` from `blas_tests_impl`) is not in the ADR.
- Testing glossary already exists: `rstsr-core/tests/CONTEXT.md`.
- `plans/` is gitignored working material; the api-doc campaign (`plans/2026-09-api-doc-campaign.md`) is the precedent for campaign-style execution.

**Trajectory repo** (this repo, from memory):

- Col-major is FURTHER ALONG than the main repo shows: `entry_col_cpu` + `tests/col_func/` (66 tests) exist as a verified but unintegrated patch — `2026-09-14-soundness-check/T2-col-major/col-major-wiring-and-tests.patch` (against acfa93e). Col invocation needs `+std`; the col body pins RowMajor per-test where needed (self-neutralizing).
- Benchmarks already have a home by charter: experiments/numbers live in THIS repo (one directory per experiment); main rstsr receives reports only inside PRs. Criterion is the soft default.

---

## Q1 — Scope of `rules/testing.md`

What should the rule be the authority on?

- (a) Taxonomy + placement policy only: what categories exist, where each lives, what each must cover, which skill to load.
- (b) (a) + process rules: regression-test mandate, CI tiering policy, benchmark policy.
- (c) Full manual incl. exact commands.

➡️ (b). Policy document; commands live in skills and would drift if duplicated.

## Q2 — Relationship to existing test skills

- (a) Router pattern: the rule routes to skills (like `rules/api-doc-conventions.md` → skill).
- (b) Rule contains guidelines; skills keep only commands.
- (c) Merge test skills into the rule.

➡️ (a). Matches the proven api-doc structure; skills stay single-purpose.

## Q3 — Col-major policy

The trajectory repo already has a verified wiring patch (`entry_col_cpu` + `tests/col_func/`, 66 tests). What is the policy going forward?

- (a) First-class: every numpy-portable test runs both orders; all new core tests order-generic.
- (b) Integrate the existing patch; new core tests order-generic where cheap; full dual-order parity retrofit deferred + tracked.
- (c) Leave col-major unintegrated; document as known gap.

➡️ (b). The patch is sunk, verified cost — integrating it converts "absent" to "wired" for near-zero new design. Full retrofit remains tracked, not mandated.

## Q4 — Regression test policy

- (a) Every bug fix must add a test in the numpy-style suite (shared entry).
- (a-) Every bug fix must add a test, placed by category (core issue → `test_issues/`; device bug → device crate's tests; API/edge → custom module).
- (b) Recommended, not mandatory.

➡️ (a-). "Every fix gets a test" + placement by category (the `core-issue-regression` skill already encodes the core case).

## Q5 — Benchmark policy

This repo's charter already says experiments live here. What should the RULE say?

- (a) Defer entirely; stub section only.
- (b) Define the category (criterion, this repo as home, reports into PRs only) but mark "not yet implemented in rstsr"; never CI-gated.
- (c) rstsr gains a `benches/` now.

➡️ (b). Codify the existing charter split (experiments here; main repo gets reports via PR) instead of inventing a new location; no CI gating.

## Q6 — CUDA-readiness of the guideline

- (a) Device-agnostic core: categories defined device-neutrally; CUDA gets its own skill when it lands.
- (a+) (a) + pre-declared "what a new device must bring" checklist (conformance entry + symlink set, device-specific suites, CI tier assignment, skip-list policy).
- (b) CPU-specific rule now; CUDA section later.

➡️ (a+). Filling a checklist beats inventing structure under pressure; ADR-0002 already anticipated future devices.

## Q7 — Sharing policy formalization

- (a) The rule names sanctioned sharing mechanisms (symlinked body, symlinked device extras, per-crate real entry files, one real-file exception pattern like KML gesvd) and when each applies vs device-native tests.
- (b) Keep sharing ad-hoc; rule stays silent.

➡️ (a). 435 symlinks of unwritten convention is how structure rots; one section names mechanisms and decision criteria.

## Q8 — Skill-writing sequencing

- (a) This grill produces `rules/testing.md` design + a skill gap list; skills written in follow-up sessions, one per skill.
- (b) This session designs and writes everything.

➡️ (a). Policy design and operational skill authoring are different tasks.

## Q9 — What "better organized" must fix (multi-select)

- (a) Placement clarity for new tests.
- (b) Unwired matrix (faer entry, col entry missing from main).
- (c) Cross-crate scatter (device crates' undocumented "different strategy").
- (d) Intra-`tests/` structure of rstsr-core.

➡️ (b) + (c). Directly serve CUDA-readiness; (a) is mostly solved by `test-conventions`; (d) uncertain — flag if you disagree.

## Q10 — Rule-first or code-first sequencing

- (a) Rule first (this session settles the design); refactor in follow-ups.
- (c) Campaign plan (rule as skeleton, code batches execute against it).

➡️ (a), escalating to (c) once the refactor size is known (it is campaign-sized: faer entry, col integration, device entry renames, ADR amendments, CI tiering).

## Q11 — ADRs for structural changes

- (a) Structure decisions get ADRs (amend/supersede ADR-0002); rule = policy, ADRs = why.
- (b) All in `rules/testing.md`.

➡️ (a). Established pattern (12 ADRs, charter just restyled 2026-10-20).

## Q12 — Device-crate conformance scope (doc_draft in device crates?)

ADR-0002 says the body includes `doc_draft`; reality symlinks only `core_func` + `test_utils`.

- (a) Amend ADR-0002 to bless reality: `core_func` + `test_utils` only (docstrings already doctested under default features; faer IS the default backend, so doctests already run on faer).
- (b) Add `doc_draft` symlinks to all five device crates.

➡️ (a). doc_draft validates documentation output strings, not device numerics; 5× compile of 164 tests buys no signal.

## Q13 — Entry naming alignment

- (a) Rename `tests_core_row.rs` → `entry_row_<device>.rs` across the five device crates (uniform, ADR-consistent, greppable), keep `tests_driver_impl`/`tests_linalg_func` names for the BLAS bodies.
- (b) Amend ADR-0002 to bless current names.

➡️ (a). Mechanical 5-file rename restoring the ADR convention before more entries (faer, col, CUDA) land.

## Q14 — Wire `entry_row_faer` now?

Faer is the DEFAULT backend; zero integration coverage today.

- (a) Wire it in the refactor with a TESTCFG skip list for legitimate faer divergences; CI-gate it (pure Rust, no runner deps).
- (b) Design-only until a concrete faer bug.

➡️ (a). Highest-value missing cell; architecture already paid for it (one small file + skip list).

## Q15 — CI tiering policy

- (a) Tier 1 (every PR): today's set + faer entry. Tier 2 (scheduled/nightly): col-major integration (once integrated), BLIS/AOCL/MKL (runner-installable), ARM OpenBLAS. Tier 3 (local, pre-release obligation): KML (vendor availability), special hardware; guideline states these are obligations, not optional.
- (b) Keep CI ad-hoc.

➡️ (a). Pre-decides CUDA's answer (pure-Rust/self-contained → Tier 1 candidate; vendor-dependent → Tier 2/3) instead of per-device arguments.

## Q16 — Standalone binaries (`tensor_sum.rs`, `allocatable_dtype.rs`)

- (a) Keep standalone; document as a category ("special-harness tests"); `tensor_sum` timing prints migrate to benches when they exist.
- (b) Fold into the shared body.

➡️ (a). Non-POD dtype invariants and perf-flavored comparisons are foreign to the parity body.

## Q17 — Umbrella breadth of `rules/testing.md`

- (a) Full enumeration: every test surface (core matrix, device crates, `blas_tests_impl`, member crates, facade cheatsheet, rstsr-faer-py conformance, rstsr-ffi stance, no_std checks, future benches/CUDA) with one routing line each; deep sections only for core/device/new-device-onboarding.
- (b) Core + device + CUDA only.

➡️ (a). The umbrella's value: any agent landing on any crate knows which section governs. Also fixes the real gap: the device "different strategy" is documented nowhere today.

---

## Answering

Edit `GRILL-R1-ANSWERS.md` in this directory (any format: "Q1: b" per line, or prose per question), or answer on screen. Rounds 2+ will branch on these.

# Grill R2 — test infrastructure redesign

## R1 answers (agent transcription of the user's GRILL-R1-ANSWERS.md, 2026-10-20)

- **Q1 (b)**: all policy goes in `rules/testing.md`; rules rank above skills; skills may
  re-iterate policy for their own context.
- **Q2 (a)**: router pattern confirmed.
- **Q3**: col-major is **second class** — a known gap to be addressed "very near", but
  **not this grill's task**. This grill must set up the *guideline for adding a new
  entry* (col-major being the first consumer). Note: some rstsr functions genuinely
  behave differently row vs col (contract spec: `src/docs/order_semantics.md`).
- **Q4**: terminology corrected — "NumPy style" = **NumPy parity** (transferred tests,
  tracked by skills + CSV). Open question kept: issue regressions in `test_issues/`
  made available/testable on all backends, vs common unittests in the `custom` field.
- **Q5 (b)**: benchmark policy codified, never CI-gated; **location open** — user
  leans toward "a/some non-publish crates".
- **Q6 (a+)**: device-neutral rule + new-device checklist.
- **Q7 (a)**: sharing policy formalized; KML-style dedicated files are *allowed*
  (replacing a symlink with a dedicated file is legitimate); the divergence must be
  **recorded somewhere** (where = open).
- **Q8 (a, re-sequenced by answer)**: refactor tests FIRST, then finalize and
  summarize into rule + skills. (The grill still settles the design now; the rule
  document is written after the refactor.)
- **Q9 (b)+(c)**, plus: `test-conventions` may not cover everything and **overlaps
  the new rule** — candidate for refactor or absorption; agent to re-check placement
  clarity; the rule must leave room for uncategorized cases (ask the human).
- **Q10–Q13 (okay)**: ADRs for structural changes; bless `core_func`+`test_utils`
  (no `doc_draft`) in device crates; rename `tests_core_row.rs` →
  `entry_row_<device>.rs`; wire `entry_row_faer`.
- **Q14 (okay)** + **Q15**: CI tiering confirmed, but user wants a **tier × device
  matrix** and future ability to mute/configure tiers by CI context (common PR,
  feature-specific PR, large refactor, release, ...).
- **Q16**: `tensor_sum.rs` **discarded**; `allocatable_dtype` **integrated into the
  entry** (placement details open).
- **Q17 (a)**: gap list maintained, updated when backends/categories are added.

### Agent re-check: placement clarity status (requested in Q9)

`test-conventions` covers placement **inside `rstsr-core/tests/` well** (taxonomy,
provenance, tracking hygiene) but covers **nothing outside core**: device crates'
entry sets, `blas_tests_impl`, standalone binaries, facade cheatsheet, member-crate
tests, and the backend-divergence exception pattern are all undocumented. So
placement clarity is a real gap the umbrella rule must close; the overlap with the
rule is the core taxonomy/placement section specifically.

---

## Q18 — Fate of `test-conventions` (from Q9)

The rule (authority) and `test-conventions` (skill) will overlap on taxonomy and
placement. What happens to the skill?

- (a) **Split by altitude**: taxonomy/placement/process policy moves up into
  `rules/testing.md`; `test-conventions` keeps only rstsr-core authoring mechanics
  (provenance format, `specify_test!`/`TestCfg` usage, creation idioms,
  `assert_equal`, tracking-CSV hygiene) and defers to the rule for policy. Name kept.
- (b) **Absorb**: `test-conventions` is deleted; its mechanics move into the rule or
  into `core-test`; the `core-*` skills re-point.
- (c) Keep both as-is; rule links to skill for core details only.

➡️ (a). Q1 settled "rules higher than skills"; mechanics that only matter while
writing a core test are skill material (loaded on demand), policy is rule material
(always read). Deleting the skill (b) bloats the rule with command-adjacent detail
the router pattern exists to avoid.

## Q19 — Issue regressions across backends (from Q4)

Today `test_issues/` runs only on `entry_row_cpu`; device crates symlink
`core_func` + `test_utils` but not `test_issues`. Meanwhile `custom_*` modules in
`core_func` already run on every backend that symlinks the body. Where should
issue regressions live?

- (a) **Symlink `test_issues/` into device crates too**: every core issue
  regression runs on every backend (CPU serial, faer, OpenBLAS, ...), with
  per-device skips via `TESTCFG` where the bug is device-specific; backend-specific
  bugs keep the existing per-crate `tests/issues/` pattern (OpenBLAS precedent).
- (b) Move issue regressions into `core_func` custom modules (drops the
  one-test-per-ticket + issue-number citation pattern; mixes them into function
  files).
- (c) Status quo: core issues tested on CPU serial only.

➡️ (a). A core bug fixed on serial may still fail on faer/BLAS — running the
repro everywhere is exactly what catches device divergence; the citation pattern
(mirroring NumPy's `test_regression.py`) is worth keeping; cost is one symlink per
crate plus skip-list entries.

## Q20 — Benchmark home (from Q5)

Where do stable benchmarks live? (Ad-hoc experiments stay in this repo per its
charter either way.)

- (a) **New non-published workspace member** in rstsr (e.g. `rstsr-bench/` or
  `crates-bench/rstsr-bench`): criterion benches (dep already declared workspace-
  wide), device selection via features, `publish = false`. This repo remains the
  R&D playground; wins graduate into the bench crate.
- (b) This repo only; rstsr main never gains benches.
- (c) `benches/` inside existing crates (rstsr-core etc.).

➡️ (a). Matches your "non-publish crate" lean; keeps published crates free of
bench-only dev-deps and feature wiring; gives CI-local/manual pre-release runs a
stable, versioned target; criterion is already a workspace dependency.

## Q21 — Recording backend divergences (from Q7)

KML's dedicated `gesvd.rs` is legitimate; the *difference* needs a recorded home.
Where?

- (a) **Per-device-crate `tests/README.md`**: each backend crate documents its
  entry set (which bodies symlinked, which dedicated files) and its divergences in
  one file next to the tests; the device-testing skill and the rule reference it.
- (b) One central file in rstsr-agents (rule/skill appendix).
- (c) In-crate doc comments on the dedicated file only.

➡️ (a). The divergence is a property of the crate's test set, so it lives with the
crate (survives repo-local browsing, PR review of that crate); a central file in
another repo goes stale the moment one backend changes.

## Q22 — CI tier × device matrix + context gating (from Q15)

Concrete proposal (rows = surfaces, columns = when they run). Path-triggered means
the job is added to a PR when its paths change; `ci:full` label opts a PR into the
whole Tier 2 column.

| Surface | T1 every PR | T2 nightly + path-triggered | T3 pre-release (local, checklist) |
|---|---|---|---|
| core row cpu (`entry_row_cpu`) + lib + doctests | ✅ | | |
| core row faer (`entry_row_faer`, new) | ✅ | | |
| core col cpu (`entry_col_cpu`, future) | | ✅ | |
| openblas conformance + driver/linalg | ✅ | | |
| openblas ARM | | ✅ | |
| mkl / blis / aocl conformance | | ✅ path-triggered | |
| kml | | | ✅ (vendor availability) |
| facade cheatsheet (+parity script) | ✅ | | |
| cpu-dlpack (row + col variants) | ✅ | col variant → T2 | |
| linalg-traits (faer) | ✅ | | |
| rstsr-faer-py conformance suite | | ✅ | |
| no_std builds (build-only) | ✅ | | |
| benchmarks (`rstsr-bench`) | never CI | | ✅ numbers recorded |

Context gating (the "mute/config" you asked for): common PR → T1; PR touching
`crates-device/rstsr-<x>/**` (or `rstsr-ffi`) → that device's T2 job also runs;
large-refactor/release PRs → `ci:full` (all T2); release → T3 checklist in the
release skill. New device (CUDA) picks its column at onboarding via the Q6
checklist.

- (a) Accept this matrix + gating model as the rule's CI section (details
  adjustable during implementation).
- (b) Modify (say what).

➡️ (a). It pre-answers CUDA ("vendor-dependent → T2/T3"), keeps common PRs at
today's cost, and makes escalation explicit instead of ad-hoc.

## Q23 — `allocatable_dtype` integration (from Q16)

Non-POD dtype tests (`BigInt`, drop-count guards) can only run on CPU serial
(faer/BLAS are float-only). Placement:

- (a) **New top-level body dir `tests/allocatable/`** (own `mod.rs`), declared by
  the core entry binaries only — device entries never declare it, so no symlink,
  no skip-list churn; keeps `num-bigint` a core-only dev-dep. `tensor_sum.rs` is
  deleted outright (its ndarray cross-check value is superseded by parity tests;
  timing prints move to the future bench crate).
- (b) Fold into `core_func` as a custom module (would ride the symlink into device
  crates and need skips everywhere).

➡️ (a). The tests assert dtype/ownership invariants, not numpy parity; keeping
them out of the symlinked body avoids compile breakage on float-only backends
entirely.

## Q24 — Refactor batch order (from Q8: refactor first, rule/skills last)

Proposed execution order (each batch = one PR-sized unit against a branch):

1. **B1 mechanical alignment**: rename `tests_core_row.rs` → `entry_row_<dev>.rs`
   (5 crates); amend ADR-0002 (names, `doc_draft` blessing, document the
   `blas_tests_impl` third body, the dedicated-file exception pattern + Q21
   recording duty).
2. **B2 faer entry**: wire `entry_row_faer` + `TESTCFG` skip list + T1 CI job.
3. **B3 body adjustments**: symlink `test_issues/` into device crates (Q19);
   delete `tensor_sum.rs`; integrate `tests/allocatable/` (Q23); per-crate
   `tests/README.md` (Q21).
4. **B4 CI rework**: tier × device matrix, path filters, nightly workflow (Q22).
5. **B5 (separate near-future task, not this grill)**: col-major integration via
   the existing T2 wiring patch, executed against the new-entry guideline.
6. **Then**: write `rules/testing.md` + refactor skills (Q18) + skill gap list
   (device-testing skill, benchmark skill stub, `core-col-major-transfer` fate)
   describing the refactored reality.

- (a) Accept order.
- (b) Reorder (say how).

➡️ (a). Mechanical-first batches de-risk; the rule is written against the final
state per your Q8 answer; col-major stays a separate consumer of the guideline per
your Q3 answer.

---

## Answering

Edit `GRILL-R2-ANSWERS.md` (user-owned) or answer on screen.

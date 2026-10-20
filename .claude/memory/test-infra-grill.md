---
name: test-infra-grill
description: 2026-10-20 test-infrastructure grill closed (D1-D10); rule+refactor design settled, execution awaits go
metadata:
  type: project
---

Test infrastructure redesign grilled 2026-10-20 (4 rounds, DECISIONS.md in
`2026-10-20-test-infrastructure/`). Key settled points:

- New `rstsr-agents/rules/testing.md`: policy-level router (taxonomy/placement/process), written LAST after refactor; escape hatch = ask human on uncategorized cases.
- Col-major is second-class; separate near-future task (B5) integrates the verified wiring patch from [[rstsr-colmajor-soundness-t2]] (`entry_col_cpu` + `tests/col_func/`, 66 tests) against the new-entry guideline.
- CI becomes a tier × device matrix: T1 every PR (core row cpu/faer, col cpu future-T1, openblas, cheatsheet, linalg-traits); T2 cpu-dlpack, openblas ARM; T3 no_std, faer-py suite, CUDA (→T2 with a GPU CI vendor); T4 manual (benches, mkl/blis/aocl/kml). Workflows renamed `tier<N>-<surface>.yml`; clippy/rustfmt outside tiers.
- `test-conventions` skill DELETED (absorbed: policy→rule, mechanics→`test-core`); renames `test-core` / `test-core-issues` / `test-core-numpy-sync` + new `test-device`; `core-col-major-transfer` stub deleted.
- Glossary merged INTO `rstsr-agents/CONTEXT.md` (user chose b over my (a) recommendation); core tests/CONTEXT.md deleted at skill-batch time.
- Device entries renamed `entry_row_<device>.rs`; `doc_draft` NOT symlinked to devices (ADR-0002 amended); `test_issues/` symlinked INTO device crates (regressions run everywhere, TESTCFG skips); `tensor_sum.rs` deleted; `allocatable_dtype` → `tests/allocatable/` core-only body; benchmarks = future multi-crate `crates-bench/`, T4, never CI-gated.
- Execution: one branch `261020/test-infra`, batches B1 mechanical → B2 faer entry → B3 bodies → B4 CI → B5 col-major → rule+skills; per-task review stops, no auto-commit in main repos.

Exploration facts worth remembering: only `entry_row_cpu` wired in main;
435 symlinks carry the sharing (no include!); `blas_tests_impl` lives in
rstsr-blas-traits/src/ with SciPy validator scripts; KML gesvd.rs is the one
dedicated-file exception; MKL/BLIS/AOCL/KML have no CI at all.

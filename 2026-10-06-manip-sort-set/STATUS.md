# STATUS — wave COMPLETE (2026-10-07, incl. external-review round)

All stages delivered; branch `261006/manip-sort-set` in the main repo holds
the full wave (13 commits, base `e7cdc6a`), committed but **not pushed** (no
PR, per wave grants). See SUMMARY.md for the full record.

## External review round (2026-10-07, post-wave)

`code-review-findings/REVIEW-manip-sort-set.md` (15 findings + below-cut
list) addressed in rstsr @ `4444fbe`; dispositions in
`code-review-findings/RESPONSE-manip-sort-set.md`. Headline fixes: the five
crates-device crates were broken (E0433 `half` in a symlinked module);
take_along_axis read index views in storage order and rejected legal
broadcast indices; naive unique_all undercounted complex multiplicities;
ExtSortCmp complex ordering was not NumPy's (rewritten as the exact
arraytypes.c.src semantics, 0/2000 mismatches vs np.sort); isin now treats
NaN membership as value equality (NumPy parity). Gates all green; suite
test-for-test unchanged (1059/241/82).

## NumPy test-parity review round (2026-10-07)

Second review (scope: NumPy test parity) found 16 issues; all fixed in the
rstsr working tree (uncommitted — no rstsr auto-commit); dispositions in
`code-review-findings/RESPONSE-numpy-parity.md`. Headline: the wave's
sort/searching/set/indexing tests were tracked in neither `numpy_coverage.csv`
(204→250 rows) nor `sync_numpy.py`'s SURFACE (now 0 MISSING); `unique`/`isin`/
`nonzero`/`take_along_axis` were written as `custom_*` despite dedicated NumPy
classes (now `numpy_*` with provenance); `diff` had no core_func/doc_draft test
(new `test_diff.rs`); three docstrings asserted phantom NumPy deviations
(`tile`, `take_along_axis`, `unique_*`); and five provenance headers cited
nonexistent NumPy paths/classes. Entry tests 462→499, clippy 0, fmt clean.

## Final state

- **17 functions** implemented rstsr-core side and bound in rstsr-faer-py:
  repeat, roll, tile, sort, argsort, sort_custom, argsort_custom,
  searchsorted, nonzero, unique_values/counts/inverse/all, isin,
  take_along_axis, diff.
- **Gates green at close**: 460 entry tests, 243 doctests, clippy 0,
  fmt clean.
- **Suite**: 1014/286/82 → **1059/241/82** (+45, 0 regressions,
  test-for-test vs census `20261006-183350`).
- Both repos' trees clean: rstsr @ `e0bcb9f`, trajectory @ SUMMARY/STATUS
  commit.

## Procedure audit (vs standing directives)

- Implement (glm-5.3-flash, sonnet slot) → review once (deepseek-flash,
  haiku slot, high) → apply/justify → commit → next stage: followed for
  stages 1–8. Stage 7's diff review was bundled into the stage-8 round
  (single `e6ed99a` commit carried both).
- Reviewer never edited code; every finding noticed; two findings
  **justified-not-applied** with evidence (stage-8 roll-mismatch premise —
  refuted by NumPy probe; stage-7 bound-nit — refuted by compile error).
- No push, no `gh pr`, main repo only (rstsr-local-workspace untouched).
- Subagent model routing honored: sonnet slot = glm/glm-5.3-flash
  (implementation), haiku slot = deepseek/deepseek-flash (review).
- The user-cancelled final total-level review (directive amended mid-wave:
  "not going to do the total-level code review job; after all stages
  finishes, you can pause") was NOT run.

## Follow-ups parked (not this wave)

- Complex searchsorted ordering (registered follow-up; see SUMMARY).
- nonzero 0-d raise-vs-empty deviation (DECISIONS).
- test_concat/test_stack red via pre-existing G-009.
- Col-major entry test binary (ADR-0002) still unwired.

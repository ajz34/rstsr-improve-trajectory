---
name: array-api-tests-explain-phase
description: Red array-api-tests runs are dominated by hypothesis's explain phase (~93% on failure-heavy files); the NO_EXPLAIN=1 harness/skill knob drops it; partial-support WIP wheels make per-failure explain much more expensive than a uniform red state
metadata:
  type: project
---

Measured 2026-10-04 (rstsr-faer-py S2-WIP wheel): a red run's wall time is
hypothesis's *explain* phase — the post-failure "Draw N ... (or any other
generated value)" minimal-explanation blob — not the shim's ops (numpy-class:
reshape 0.5us, add 0.9us), not data generation, not the build profile.
`--hypothesis-show-statistics` on test_manipulation_functions.py: generate
1.3s + shrink 0.8s + explain 34.2s = 36.8s. Per-failure explain is 1-7s when
the failure is input-dependent (partial support, e.g. int-getitem works but
numpy-int/array indices raise) vs ~0.1s for trivial uniform failures
(missing-attribute asserts) — that is why a WIP wheel can be 10-25x slower
per failing test than the S1 release wheel, and why numpy's 42-failure full
run is 39s while a ~1000-failure run is minutes.

**Why:** a red conformance run has ~1000 failing hypothesis tests; the
explain phase runs per failure, so wall time ~= failures x explain cost.

**How to apply:** use `NO_EXPLAIN=1` for grading/red-map runs (harness
`run.sh` of 2026-10-04-rstsr-faer-py, and skill rstsr-faer-py-tests): the
bundled `no_explain.py` pytest plugin registers a child of the suite's
"array-api-tests" hypothesis profile with `Phase.explain` removed and loads
it in `pytest_configure` — loading it later (e.g. collection_modifyitems)
does not take effect. Counts and falsifying examples unchanged (verified:
302/998/82 both single-process and chunked). Keep explain ON when one
failure's explanation blob is the evidence (gap-register repros).
Related: [[rstsr-faer-py-grill]].

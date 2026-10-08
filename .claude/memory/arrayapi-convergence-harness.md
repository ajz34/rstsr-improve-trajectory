---
name: arrayapi-convergence-harness
description: "Exact env and locations for the rstsr_faer.api array-api-tests conformance runs (test interpreter, suite checkout, wheel build, scratch report dir) — the harness needs three paths that AGENTS.local.md does not carry."
metadata:
  type: reference
---

The conformance harness (skill `rstsr-faer-py-tests`) needs three per-developer
paths that are **not** in `AGENTS.local.md` (that file is absent in this rstsr
checkout); pass them inline every run:

- `TEST_PY=/home/a/miniconda3/envs/torch/bin/python3` — pytest 9.1.1,
  hypothesis 6.168, pytest-json-report, ndindex; `rstsr_faer` is installed here.
- `SUITE_DIR=/home/a/Git-Others/array-api-tests` — pin `6c0b59f`, spec submodule
  `5f847a3`.
- `maturin build --release -i "$TEST_PY" -o <dir>` must run **from the crate
  dir** `rstsr/crates-interop/rstsr-faer-py` (not the workspace root).

Run from a scratch dir (reports land in `./reports` there, never in a checkout);
canonical whole-suite run:

    cd <scratch> && SUITE_DIR=… TEST_PY=… MODULE=rstsr_faer.api CHUNKED=1 \
      FRESH=1 NO_EXPLAIN=1 bash .claude/skills/rstsr-faer-py-tests/scripts/run.sh

Wheel rebuild is ~2m20s at `CARGO_PROFILE_RELEASE_OPT_LEVEL=0` (dev iterations;
`CARGO_PROFILE_RELEASE_DEBUG=0 CARGO_BUILD_JOBS=2` for memory). Set
`TMPDIR=$HOME/.cache/tmp-cargo` for all heavy builds. Keep per-wave reports under
`~/.cache/rstsr-conf-*` for the node-for-node diff: extract each failing test's
`nodeid` from the chunk JSONs (`tests[].outcome in {failed,error}`) and set-diff
against the baseline stamp — a clean flip shows only the intended nodes, zero new
failures.

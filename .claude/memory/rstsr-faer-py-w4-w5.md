---
name: rstsr-faer-py-w4-w5
description: W4 (creation/manipulation) + W5 (searching/indexing) done, review round fixed 5 core bugs, PR #113 merged as c0ea36b; what remains missing and how the new bindings work.
metadata:
  type: project
---

W4/W5 (2026-10-06, branch `261006/faer-py-creation-manip`, worktree
`tmp/faer-py-w4`) completed the creation and manipulation categories and
bound the searching/indexing entries over rstsr's existing `rt::` surface.
Suite: 934/366/82 → **984/316/82** (W4, stamp `20261006-121144`) →
**996/304/82** (W5, stamp `20261006-131221`); the review round's fixes were
a test-level 0-flip diff (`20261006-132433` → `20261006-144402`). Details in
`2026-10-04-rstsr-faer-py/reports/SUMMARY-w4.md` / `SUMMARY-w5.md`; register
v6/v7 (G-060..G-068). **PR #113 merged by the owner 2026-10-06 as squash
`c0ea36b`** (13/13 CI green on the first run, no fix-up commits).

**Still missing (rust-side)**: `repeat`/`roll`/`tile`; searching/set
`where` (G-037), `nonzero`, `searchsorted`, `take_along_axis`, `isin`,
`unique_*`; `sort`/`argsort`. Declines registered: `eye`/`tril`/`triu` bool
(G-062), `linspace` non-float dtype (G-063).

**How the new bindings work** (see also [[rstsr-faer-py-w2]]): creation
Rust bindings are thin dtype dispatches (`linspace` on the `ComplexFloat`
kernel, `eye`/`tril`/`triu` on the `Num`-bound creation kernels); `*_like`
is pure Python marshalling over the existing entries; manipulation
materializes views with `into_owned` (handle model has no shared storage);
joins are same-dtype-only (G-009 decline, no shim promotion table).
Searching: rstsr index reductions return `usize` → `ops::idx_lift`
re-materializes as int64; `dispatch_t_index_ord!` vs `dispatch_t_index_zero!`
split by kernel bound; `count_nonzero` on bool routes through the
bool-specialized sum (`sum_bool`, owner-approved documented exception).

**Review round** (`/code-review max`, commit `4bce438`): five core bugs
surfaced and fixed rust-side, each with a regression test + a
`numpy_differences_resolved.md` entry — Layout::diagonal super-diagonal gate
(G-064), squeeze mixed negative axes (G-065), take empty indices (G-066),
argmax/argmin empty output (G-067), tril/triu rank-1 error class (G-068) —
plus the shim-side `meshgrid()` zero-vector fix; `api.py` is wrapper-only
again ([[rstsr-faer-py-wrapper-only]]).

**Dev loop**: wheel = `CARGO_PROFILE_RELEASE_OPT_LEVEL=0 maturin build
--release -i "$TEST_PY" -o <dir>` from the crate dir (opt 0 for iteration;
opt 3 for the record wheel); suite runs via the task harness
(`MODULE=rstsr_faer.api CHUNKED=1 NO_EXPLAIN=1 ./run.sh`); per-stamp chunk
reports + `COMPLIANCE-FULL-*.csv` live in the task's gitignored
`harness/reports/`. Reminder: `device_faer/rayon_auto_impl` files are
tracked symlinks to `feature_rayon/auto_impl` — the live tree compiles
under the `faer` feature ([[rstsr-arg-semantics]]).

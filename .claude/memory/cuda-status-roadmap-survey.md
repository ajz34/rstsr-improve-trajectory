---
name: cuda-status-roadmap-survey
description: 2026-11-24 survey of rstsr-cuda at 9c7d49a - neg/sort/mask/nonzero/tril/allclose missing, reduction dtype limits, 11 device + 26 op + 19 linalg gaps, streaming primitives done but allocator/DLPack/graphs/peer/facade missing; deliverable dir 2026-11-24-cuda-status-roadmap
metadata:
  type: project
---

# rstsr-cuda status/coverage/roadmap survey

2026-11-24. Read the whole `rstsr-cuda` crate at `261021/cuda` @ `9c7d49a`,
the trajectory's CUDA directories, and PR #4 (body, commits, owner comments).

## Deliverable

`2026-11-24-cuda-status-roadmap/`:

- `README.md` — verdict + master status table + top-10 next actions.
- `STATUS.md` — layer status, missing device/op/linalg traits, dtype matrix,
  streaming capability matrix, 11 known defects, stale scope assumptions.
- `COVERAGE.md` — function-level common-usage matrix (Y/P/N/B/X).
- `ROADMAP.md` — P0/P1/P2 checklist with acceptance criteria and PR sequence.
- `PR-4.md` — what PR #4 added that the trajectory never recorded + the
  owner's three review comments (accept-all, var/std rule, ReducePlan refusal,
  cuBLASLt/cuTENSOR direction).
- `EVIDENCE.md` — how every claim was checked, including compile probes.

## Verified facts worth remembering

- `-&a` fails: `DeviceCuda` does not implement `OpNegAPI` (E0600, probe).
- Plain `i32.sum()` and `Complex<f64>.sum()` fail (E0599); `ReduceAcc` is
  f32/f64 only. Integer sum needs `sum_dtype::<f64>()`. Bool `b.sum()` works
  (count kernel).
- Missing: 11 device traits, 26 op traits (+7 host-closure families),
  19 linalg entries. Sort/search/set/nonzero/mask/tril/allclose/neg absent.
- Streaming primitives are done (per-stream events, pinned edges, 3-stream
  bench 2.29×); allocator framework (D7), DLPack, graphs, peer copy, pinned
  pool, facade `rt::cuda`, GPU CI are absent.
- Tests: 183 total = 72 lib + 111 integration; hardware runs are skipped
  silently without a GPU.
- PR #4: 30 commits (`451c2a9..a4d717a`), merged `6d17d04`; body's "14
  commits" is stale; handle is per-stream, not per-ordinal.
- Owner directive: matmul/tensordot target = cuBLASLt + cuTENSOR wrappers,
  shaped like rstsr-blas/linalg-traits; current code is interim.

## Caution

`lib.rs` "Absent, and why" is wrong for `allclose`, `take_along_axis` and
`neg`; the root causes are `OpAllCloseAPI`, the core `Raw = Vec<usize>` gate,
and `OpNegAPI` respectively. Do not quote that section without checking.

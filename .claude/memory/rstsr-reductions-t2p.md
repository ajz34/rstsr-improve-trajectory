---
name: rstsr-reductions-t2p
description: T2' reductions phase-1 findings at 386948be — sum_axis0 native regression = iterator next() per CHUNK band (not closure codegen); min/max_all native 4.9x regression from f64::min in unrolled_reduce; order-fixup copy is dead code; broadcast-summed-axis size_s0 index bug; fixup copy moot.
metadata:
  type: project
---

T2' value reductions (2026-09-09-reductions, phase 1, base 386948be):

- **Branch map** (`reduce_axes_cpu_serial`, cpu_serial/reduction.rs:180-370):
  axis1 (contiguous summed axis) → branch 1 = `unrolled_reduce` inner
  (already at probe speed); axis0 (contiguous remaining axis) → branch 2 =
  CHUNK=48 bands with `it_scd.clone().for_each` re-walking summed axes per
  band → 88k `IterLayoutColMajor::next` calls/op at 2048². Branch 3 = plain
  fold. Same structure in the rayon twin (CHUNK=64).
- **Native regression mechanism (sum_axis0 1.43→1.99 ms)**: NOT closure
  codegen — the fold got faster under native; `next()` self-time exploded
  (~18→~79 cyc/call × 88k calls; perf w/ debuginfo). Removing the iterator
  (probe): 0.57 ms native / 0.82 ms portable.
- **min/max_all native pathology**: 1e7 min_all = 10.4 ms native vs 2.15
  portable (numpy 1.17). 8-lane `unrolled_reduce` with `f64::min`
  (minnum) defeats autovectorization under native; strict-compare form
  `if x < acc {acc = x}` vectorizes both configs (0.44 ms) and is
  semantics-equivalent on the ±MAX seed (NaN never passes strict compare).
  Fix = wiring-closure rewrite + `PartialOrd` bound on OpMin/OpMax impls.
- **Order-fixup copy (reduce_axes :357-367, rayon :317-327) is DEAD CODE**:
  requires `TensorIterOrder::default() != K` but default == K always.
- **Upstream bug (do not silently fix)**: reductions over stride-0 SUMMED
  axes mis-multiply — `size_s0` (cpu_serial/reduction.rs:226) indexes
  `lm.shape()` with `ls`-axis indices: sum_axes(0) of [1,n]→[m,n] broadcast
  gives n×v (numpy: m×v); mean n/m×v; min/max insensitive. Locked as
  CURRENT-BEHAVIOR in the gate.
- **NaN semantics of min/max** (locked): skip NaN (f64::min semantics,
  rstsr-dtype-traits ext_real.rs:70-85); all-NaN slice → seed value
  (f64::MAX for min, f64::MIN for max).
- Phase-2 plan: Option A (iterator-free band walk via precomputed offset
  scratch) + A2 (strict-compare min/max closures); composes with the
  argmax patch (function-level disjoint, hunks ≥ line 424 serial / 439
  rayon / 333+356 wiring; T2' stays ≤ 370/330/354).
- criterion + RUSTFLAGS configs must use per-config `CRITERION_HOME`
  (shared target/criterion clobbers results across configs).

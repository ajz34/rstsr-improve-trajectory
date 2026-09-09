# PLAN.md — T2' phase 2: value reductions (sum/min/max/mean/var/norm + sum_axes)

Status: phase 1 COMPLETE (this document is the G1 review package).
Base: rstsr `386948be819baa334b8da02232f3a1944e5447d5` — clean tree verified
before baselining; **phase 1 made no edits to rstsr** (`git -C ../rstsr
status --short` empty at start and end, HEAD `386948b`).
All phase-1 numbers: [results/tables.md](results/tables.md),
[results/perf/](results/perf/), [results/probe_manual_*.txt](results/),
[results/layout_probe.txt](results/layout_probe.txt).

## 1. What phase 1 established (evidence base)

### 1.1 Where the time goes — call-chain anatomy (file:line at 386948be)

Serial call chain for `a.sum_axes(axes)`:

```
rt::sum_axes / TensorAny::sum_axes
  rstsr-core/src/tensor/reduction.rs:116-125 ($fn_axes of trait_reduction!)
  → DeviceCpuSerial::sum_axes           device_cpu_serial/reduction.rs:23-36
      closures: f=|acc,x| acc+x, f_sum=|a,b| a+b  (lines 30-32; mean/var/norm/
      min/max differ only in init/f/f_out — same driver)
  → reduce_axes_cpu_serial              cpu_serial/reduction.rs:180-370
      branch selection by get_axes_composition (rstsr-common/src/layout/
      rearrangement.rs:336) on ls (summed) / lm (remaining):
      - branch 1 (size_sc>1, :231-256): summed axis is the f-contiguous one
        → per-remaining-position `unrolled_reduce` (:44-83, 8-lane, ndarray-
        style) over contiguous slabs.  SERVES axis1 (sum_axes(-1)) on
        row-major 2-D, and summed-axis-contiguous views.
      - branch 2 (size_mc>1, :257-305): remaining axis is contiguous
        → `vacc.chunks_mut(CHUNK=48)` bands; for EACH band, re-walks the
        summed axes via `it_scd.clone().for_each(|i_scd| ...)` (:289) —
        one `IterLayoutColMajor::next` per row PER BAND (88k next() calls
        per op at 2048²) — then a scalar clone-through-closure add
        `*acc = f(acc.clone(), x.clone())` (:292-294) per element.
        SERVES axis0 (sum_axes(0)) on row-major 2-D, 3-D axis0/[0,1],
        f-contig axis-1, t-view axis-1, broadcast-summed inputs.
      - branch 3 (:306-329): plain fold, nothing contiguous.
      - broadcast-remaining pass (:332-351) and finalize f_out (:298-304).
      - ORDER-FIXUP COPY (:357-367) is DEAD CODE in the default build:
        it requires `TensorIterOrder::default() != K`, but the probe shows
        `TensorIterOrder::default() == K` (layout_probe.txt line 1), so the
        extra `op_muta_refb_func` pass never fires — for any layout.
        Plan-T2 hypothesis (b) ("kill the final order-fixup copy") is
        therefore MOOT. (Verified for 2-D rm both axes, 3-D [0],[-1],[0,1],
        t-view, f-contig, broadcast — layout_probe.txt.)
  Rayon twin: reduce_axes_cpu_rayon     cpu_rayon/reduction.rs:109-330
      (same 3 branches; PARALLEL_SWITCH=1024 serial fallback :130;
      branch 2 = par_chunks_mut(CHUNK=64) :228 with the same per-band
      summed-axes re-walk :232; branch 1 = per-position par fold over
      it_sd with unrolled_reduce :180-198.)
  reduce_all (scalars): reduce_all_cpu_serial cpu_serial/reduction.rs:144-178
      → translate_to_col_major K + size_contig≥CONTIG_SWITCH(32) →
      `unrolled_reduce` over contiguous slabs via
      `layout_col_major_dim_dispatch_1` (single pass, no axes iterator).
      1-D/2-D contiguous full reductions take this with size_contig = full
      size (layout_probe.txt last line: 4194304).
```

Axis conventions (documented, doctest + runtime-checked in the gate):
`sum_axes(0)` on [m,n] → shape **[n]**; `sum_axes(-1)` → **[m]** (numpy
convention; output layout IxD row-major, `lo` from
`layout_for_array_copy(&lm, K)`).

### 1.2 perf evidence: why does native REGRESS sum_axis0?

`perf stat -d` on `profile_ops` (serial, 2048², 600/1500 iters;
results/perf/raw.txt, derived in results/perf/derived_summary.txt):

| op | config | ms/iter | GB/s | ins/elem | cyc/elem | IPC | L1d-miss% | faults/iter |
|---|---|---|---|---|---|---|---|---|
| sum_axis0 | portable | 1.383 | 24.0 | 4.45 | 1.87 | 2.38 | 9.6 | ~13.8 |
| sum_axis0 | native | 1.828 | 18.2 | 2.56 | 2.51 | 1.02 | 12.8 | ~13.8 |
| sum_axis1 | portable | 0.449 | 73.6 | 1.51 | 0.61 | 2.46 | 20.9 | ~5.5 |
| sum_axis1 | native | 0.439 | 75.2 | 0.76 | 0.60 | 1.26 | 20.6 | ~5.5 |
| sum_all | portable | 0.430 | 76.7 | 1.34 | 0.59 | 2.27 | 24.6 | ~5.5 |
| sum_all | native | 0.416 | 79.2 | 0.58 | 0.57 | 1.02 | 24.4 | ~5.5 |

- Page faults/iter ≈ 14 for ALL cases (output is one [2048] f64 vector =
  16 KiB ≈ 4 pages): **the T7 allocation rider is ~0 for reductions, as
  predicted** — no MALLOC column needed; this is the sanity row.
- Native is FASTER for sum_axis1 and sum_all (branch 1 / reduce_all: the
  `unrolled_reduce` 8-lane loop vectorizes BETTER under native — asm: native
  emits memory-operand `vaddpd` groups, portable separates loads; both
  128-bit wide for these loop bodies, zmm appears only in a variant of the
  reduce_all slab loop).
- Native is 1.32x SLOWER for sum_axis0 (branch 2). Mechanism, from
  `perf record` (debuginfo builds) + objdump of both binaries:
  - **NOT the accumulate loop's codegen.** The fold body got FASTER under
    native (fold-symbol self-time 0.652 s portable vs 0.357 s native; the
    native chunk loop is a 4×xmm memory-operand `vaddpd` unroll).
  - **The layout-iterator `IterLayoutColMajor::next` explodes.** Branch 2
    re-walks the summed axes once per 48-element band
    (`cpu_serial/reduction.rs:289`): 43 bands × 2048 rows = **88 064
    `next()` calls per op-iteration**. Self-time in `next()`:
    0.167 s of 0.840 s (portable, ≈18 cyc/call) vs **0.731 s of 1.108 s
    (native, ≈79 cyc/call)**. The `next()` machine code is
    codegen-equivalent in both binaries (jump-table on ndim, scalar index
    update; objdump side-by-side in results/perf/) — the per-call cost
    difference is a call-frequencies × microarchitecture effect (outlined
    call + indirect jump-table dispatch per 48 elements, latency-bound at
    IPC 1.02), not a missing vectorization in `f`.
  - Local probes (results/probe_manual_*.txt) pin it: the SAME band shape
    with explicit offset arithmetic and **no iterator** runs 0.570 ms
    (native) / 0.824 ms (portable) — i.e. the iterator machinery accounts
    for **1.28 ms of 1.85 ms (69%) native and 0.58 ms of 1.40 ms (41%)
    portable**.
- T0's "closure codegen fragility" explanation is thereby REVISED: the
  closure `f = |acc,x| acc+x` inlines fine even under native; the
  regression lives in the layout-iterator walk that branch 2 imposes per
  band. (The D8/ndarray 1.7-2.0× speedups under native remain real —
  hand-unrolled/monomorphized loops don't pay the iterator either.)

### 1.3 Probe anchors for the design (2048² f64, plain Vec, serial)

| probe shape | native ms | portable ms |
|---|---|---|
| axis0: current shape (vacc + CHUNK=48 bands, **no iterator**) | 0.570 | 0.824 |
| axis0: full vacc (16 KiB, L1d) + contiguous row adds | 0.689 | 0.615 |
| axis0: full vacc + chunks_exact(8) lane unroll | 0.632 | 0.918 |
| axis0: column walk (per-output, stride-n reads) | 17.28 | 17.30 |
| axis1: per-row 8-lane unrolled fold (= branch 1 shape) | 0.442 | 0.482 |
| axis1: per-row plain scalar fold | 1.494 | 1.475 |
| sum_all: 8-lane unrolled (= reduce_all shape) | 0.454 | 0.450 |
| sum_all: plain scalar fold | 1.525 | 1.527 |
| min_all: 8-lane unrolled via `f64::min` (= rstsr shape) | 3.160 | 0.902 |
| min_all: 8-lane unrolled via strict compare `if x < acc` | 0.443 | 0.468 |
| min_all: plain scalar fold (`f64::min`) | 0.490 | 1.649 |
| min_all: 16-lane strict-compare unroll | 0.380 | 0.433 |

Readings:
- **Write-in-output-order is the WRONG direction for axis reductions**: the
  column walk (touch outputs in order, read inputs stride-n) is 17 ms —
  20× worse. Memory-order row walks + accumulate are mandatory. Plan-T2
  hypothesis (b) reformulated: the order-fixup copy is already dead (§1.1);
  the actual axis0 fix is iterator elimination, not output ordering.
- The band structure itself is fine (0.57/0.82 ms without iterator);
  full-vacc row adds are equivalent within codegen noise, but bands keep
  the vacc working set bounded for LARGE remaining axes (size_mc ≫ L1),
  so bands must stay for generality.
- axis1 (branch 1) and sum_all (reduce_all) already run at their probe
  shapes (0.44-0.48 vs actual 0.446-0.456; 0.45 vs 0.424-0.437) — they are
  at the L3-assisted streaming ceiling for this loop family. LEAVE ALONE.
- The `[f64;8]` lane-accumulator probe (75 GB/s) is NOT a valid axis0 shape
  (it degenerates to a sum_all-like 8-chain fold — outputs don't
  materialize per column); it only demonstrates that wider accumulation
  helps register-resident folds. For real axis outputs, vacc must exist.
- **SECOND PATHOLOGY — min/max reduce_all under native.** rstsr's actual
  1-D `min_all`/`max_all` at 1e7: 2.15 ms portable, **10.42 ms native**
  (4.9× native regression; numpy 1.17 ms). Probe-level mechanism: the
  8-lane `unrolled_reduce` shape with `f = acc.ext_min(x)` (= `f64::min`,
  the NaN-skipping `minnum` intrinsic) does NOT auto-vectorize under native
  (3.16 ms probe) — while the plain scalar `f64::min` fold DOES (0.49 ms;
  vminpd) and the strict-compare form `if x < acc { acc = x }` vectorizes
  in BOTH shapes and BOTH configs (0.44 ms native / 0.47 portable; 16-lane:
  0.38/0.43). The strict-compare form is semantics-EQUIVALENT for the
  seeded fold: with the `±MAX` seed, NaN candidates never pass the strict
  comparison exactly as `f64::min` skips them, an all-NaN slice keeps the
  seed (gate-locked behavior), and ties keep the incumbent (same value).
  This alone should recover ~24× native (10.4 → ~0.45 ms) for min/max_all
  and help every min/max axis reduction through the same closures.

### 1.4 Semantics established (current behavior at 386948be — must be preserved)

Gate: examples/correctness.rs, 124 checks, ALL PASS under both configs
(results/correctness_{portable,native}.txt):

- **NaN**: sum/mean/var/norm poison (IEEE); **min/max SKIP NaN** —
  `ExtReal::ext_min/max` use `f64::min/max` (rstsr-dtype-traits/src/
  ext_real.rs:70-85, "handles NaNs according to IEEE 754-2008"); an
  all-NaN slice returns the seed: **f64::MAX for min, f64::MIN for max**.
- **Axis conventions**: sum_axes(0) [m,n] → [n]; verified incl. 3-D.
- **Broadcast reductions over a stride-0 SUMMED axis are WRONG at
  386948be** (upstream bug, kept as CURRENT-BEHAVIOR LOCK in the gate):
  `sum_axes(0)` of a [1,n]→[m,n] `broadcast_to` view yields **n×v** (numpy:
  m×v); mean yields n/m×v. Root cause: the broadcast multiplier `size_s0`
  (cpu_serial/reduction.rs:226) does `as0.iter().map(|&i| lm.shape()[i])`
  where `as0` holds indices of the SUMMED layout `ls` but indexes the
  REMAINING layout `lm`'s shape vector. min/max are insensitive
  (idempotent). Reductions over NON-broadcast axes of broadcast inputs are
  correct (the m0-pass at :332-351 duplicates correctly).
  **A phase-2 candidate must reproduce this behavior bit-for-bit; fixing it
  is a separate correctness decision for the human** (flag in the README;
  do not smuggle a semantic fix into a perf patch).
- f32 spot checks pass within tolerance (8-lane summation order).

## 2. Design options for phase 2 (laid out; one recommended)

Target: eliminate the per-band summed-axes iterator walk in branch 2
(axis0-class) while keeping everything else untouched.

### Option A — iterator-free band walk in branch 2 (RECOMMENDED)

Rewrite the inner evaluation of the `size_mc > 1` branch
(cpu_serial/reduction.rs:257-305; rayon twin :199-249) to walk the summed
axes ONCE per outer group instead of once per band:

1. Before the band loop, materialize the summed-axis offsets once into a
   scratch `Vec<usize>` of length `it_scd.len()` (axis0 at 2048²: 2048 × 8 B
   = 16 KiB, L1-resident; the vec is reused across bands — one allocation
   per outer-group visit, replacing 43 iterator clones + 88k `next()`s):
   `let scd_offsets: Vec<usize> = it_scd.map(|i_scd| i_md + i_scd - offset).collect();`
   (usize offsets ≥ 0 today — same contract the current code relies on
   when forming `idx_in`; `i_md`, `i_scd`, `offset` are the existing
   quantities, no new arithmetic.)
2. The band loop becomes two plain nested loops:
   ```rust
   for (i_chunk, vacc_chunk) in vacc.chunks_mut(CHUNK).enumerate() {
       let start = i_chunk * CHUNK;
       let nchunk = vacc_chunk.len();
       for &idx_in in &scd_offsets {
           let slc = &a[idx_in + start..idx_in + start + nchunk];
           vacc_chunk.iter_mut().zip(slc).for_each(|(acc, x)| {
               *acc = f(acc.clone(), x.clone());
           });
       }
   }
   ```
   The closure `f`, broadcast-duplication, and `f_out` finalize are
   UNCHANGED (bit-exact semantics, including the documented broadcast bug).
   Optionally accumulate with `*acc = f(...)` kept as-is; do NOT re-associate.
3. Same for the rayon twin (`par_chunks_mut(64)`; the scratch vec moves
   inside the `it_md.zip(it_od)` closure — per-task allocation, unchanged
   parallel semantics; `f` is `Send+Sync` as before).
- Guards: none new — this is a rewrite INSIDE the existing branch condition
  (`size_mc > 1`); branches 1/3, the broadcast passes, and finalize are
  untouched. isize contract: all indices are the existing
  `i_md + i_scd - offset` sums, formed exactly as today (usize; negative
  strides are handled upstream in the layout offsets — same as current
  code, which already indexes `a[idx_in + start..]`).
- Blast radius: ~25 lines serial + ~30 lines rayon, 2 files in
  rstsr-native-impl; no API/trait/feature changes; all dtypes and all
  `reduce_axes` ops (sum/min/max/mean/var/std/prod/norm/all/any/count)
  benefit through the shared driver.
- Expected (probe-anchored): sum_axis0 2048² serial 1.40 → ~0.6-0.8 ms
  portable, 1.85 → ~0.6-0.8 ms native (2-3×, beats portable); faer16
  sum_axis0 445 → ~250-350 µs (same restructure amortized over 16 threads);
  odd 1000×777 similar ratio. D3 ≥10% both configs exceeded ~4×.

### Option B — monomorphized 2-D single-axis fast path beside branch 2 (fallback)

ArgCmp-precedent style: keep branch 2 as-is; add `#[inline(never)]
fn reduce_axes_2d_contig_rem_cpu_serial<T, F>(a, la, vacc, f, ...)`
dispatched on guard `ndim == 2 && summed axes == 1 && remaining axes == 1
&& lm.stride() == [1]` (row-major axis0), writing directly with plain index
loops. Smallest possible codegen disturbance to other paths, but serves
only the exact 2-D case; duplicates finalize logic; leaves 3-D/odd-shaped
branch-2 cases on the slow path. Expected: same headline numbers as A for
the benched 2-D case.

### Option C — closure-layer enum (ArgCmp-style) for the value ops — REJECT

Unlike argmax, the value-op wiring closures (`|acc,x| acc+x` etc.) already
monomorphize and inline INTO the fold (perf shows inlined add loops; no
per-element indirect calls). An enum would add a dispatch inside the hot
loop with zero upside. The ArgCmp precedent does not transfer.

### Option A2 — strict-compare min/max wiring closures (RECOMMENDED, independent of A)

Replace the min/max accumulate closures in the four wiring impls
(`OpMinAPI`/`OpMaxAPI` × `device_cpu_serial/reduction.rs` +
`feature_rayon/auto_impl/reduction.rs`; each has min_all/min_axes/max_all/
max_axes with `f = |acc: T, x: T| acc.ext_min(x)` / `acc.ext_max(x)` and
`f_sum = |a, b| a.ext_min(b)` / `a.ext_max(b)`):

```rust
let f = |acc: T, x: T| if x < acc { x } else { acc };      // min
let f_sum = |acc1: T, acc2: T| if acc2 < acc1 { acc2 } else { acc1 };
// max symmetric with >
```

- Requires widening the impl bounds `T: ExtReal` → `T: ExtReal + PartialOrd`
  (every in-tree ExtReal type is PartialOrd; ints would silently keep
  `Ord::min` semantics — actually the strict-compare form IS Ord::min/Ord::max
  for ints except tie/order nuances that don't exist for values). Alternative
  without a bound change: add `ext_min`/`ext_max`-preserving helper — but
  that means editing rstsr-dtype-traits, outside D1 scope; prefer the bound.
- Semantics: equivalent by construction for both ints and floats
  (`if x < acc {x} else {acc}` == `Ord::min` for PartialOrd ints; == the
  NaN-skipping `f64::min` on a ±MAX-seeded accumulator for floats —
  argument in §1.3, locked by the gate).
- Impact: fixes the 4.9× native min/max_all regression (10.4 → ~0.45 ms
  expected), speeds up all min/max axis reductions (the closures feed
  `unrolled_reduce`/branch-2 fold identically). ~16 one-line closure
  replacements + 4 bound edits in 2 files.
- Risk: vectorized min changes nothing observable (min is
  order-independent for values; NaN behavior locked identical). Gates: the
  full correctness gate (NaN placements, all-NaN, both dtypes) must pass.

**Recommendation: A + A2, implemented as two separately-verifiable steps.**
A fixes the measured axis0 mechanism (iterator calls per band) for ALL
geometries that hit branch 2, is dtype/ops-generic; A2 is an
orthogonal, ~20-line wiring change that removes the min/max native
catastrophe. Its riskiest gate (axis1 = branch 1 codegen) is a disjoint
code block. If A trips the axis1 gate in a way re-tuning cannot fix, fall
back to B. A2's gates are the NaN/all-NaN correctness locks; if the
`PartialOrd` bound widening is judged unacceptable at G1, A2 can be
dropped independently (report the 4.9× native min/max_all regression as a
structural finding).

## 3. Interaction with the argmax patch (same files!)

The argmax patch (2026-09-09-argmax-argmin/proposed.patch, applies to clean
386948be) and a T2' patch touch the same four files. Function-level
disjointness (all line numbers = clean 386948be):

| file | argmax patch touches | T2' touches |
|---|---|---|
| cpu_serial/reduction.rs | `reduce_all_unraveled_arg_cpu_serial` (427-479 → split into fold fn + contig fast path), `reduce_axes_unraveled_arg_cpu_serial` (481-531), `reduce_all_arg_cpu_serial` (533-553), `reduce_axes_arg_cpu_serial` (555-582); adds `ArgCmp`/`arg_contig_*` at the `/* #region reduce unraveled axes */` marker (425) | `unrolled_reduce`/`unrolled_binary_reduce` (44-140, likely untouched), `reduce_all_cpu_serial` (144-178, untouched), `reduce_axes_cpu_serial` (180-370 ONLY), any new helper placed BEFORE line 374 (`/* #endregion */` of the reduce region) |
| cpu_rayon/reduction.rs | `reduce_all_unraveled_arg_cpu_rayon` (442-512), `reduce_axes_unraveled_arg_cpu_rayon` (514-577), `reduce_all_arg_cpu_rayon` (579-600), `reduce_axes_arg_cpu_rayon` (602-633) | `reduce_all_cpu_rayon` (20-106, untouched), `reduce_axes_cpu_rayon` (109-330 ONLY) |
| device_cpu_serial/reduction.rs | `OpArgMinAPI` (323-372), `OpArgMaxAPI` (374-423), `OpUnraveledArgMin/MaxAPI` (519-615) | `OpSum/Min/Max/Prod/Mean/Var/Std/L2NormAPI` impls (7-321 ONLY); no signature changes to any arg API |
| feature_rayon/auto_impl/reduction.rs (= device_faer symlink) | `OpArgMinAPI` (356-466), `OpArgMaxAPI` (468-570), `OpUnraveledArg*` (572-673) | `OpSum/Min/Max/Prod/Mean/Var/Std/L2NormAPI` impls (7-354 ONLY) |

- The argmax hunks in cpu_serial/reduction.rs begin at `@@ -424,7 +424,132`
  (context = the region marker at line 425); the earliest T2' edit ends by
  line 370. The rayon argmax hunks begin at `@@ -439,7`; T2' ends by line
  330. Wiring hunks: argmax starts at `@@ -333,40` / `@@ -368,42`; T2'
  stays ≤ line 354 (A2's min/max closure + bound edits live at lines
  39-116 serial / 43-129 rayon-auto-impl — inside the value-op impls the
  argmax patch never touches).
- **Verdict: the two patches COMPOSE.** No hunk overlaps or shared context
  lines; functions touched are disjoint; T2' changes no signatures so the
  `ArgCmp` API change is unaffected. Applying both with `git apply` (either
  order; the second applies with line offsets) succeeds. The only textual
  adjacency: T2' edits inside `reduce_axes_cpu_serial` end ~55 lines before
  the argmax region — no shared context. (Re-verified: any T2' helper must
  be inserted before line 374 serial / before line 332 rayon to keep the
  argmax hunks' context intact.)

## 4. Phase-2 bench matrix & accept criteria

Matrix = phase-1's, unchanged (benches/reduce.rs + benches/anchors_ndarray.rs),
re-run with `--save-baseline candidate` against the committed portable/native
baselines; per plan D3/D7:

- **Primary accept**: sum_axis0 (sum_axes(0)) at 2048×2048, serial device,
  improves ≥10% in BOTH configs — target ≤1.0 ms native (from 1.83; T0 saw
  1.90) AND ≤1.0 ms portable (from 1.38); stretch ~0.6-0.8 ms (probe
  anchor). Native must not be slower than portable after the change.
- **Primary accept A2**: 1-D min_all/max_all 1e7, serial native, improves
  ≥10% in BOTH configs (from 10.42 ms native / 2.15 ms portable; expected
  ~0.4-0.5 ms both — ~24× native); min_axis0 2048² expected to improve
  substantially too (2.93/2.24 ms, iterator + closure bound).
- **Secondary (report, not gate)**: odd 1000×777 axis0 both axes/directions;
  faer16 sum_axis0 (444.7 µs portable / 341.1 µs native — the rayon twin
  gets the same restructure; expected improvement, gated below).
- **Regression gates (no >2-3%, both configs, both devices)**:
  sum_axis1 (all sizes — branch-1 codegen must not shift),
  sum_all (all sizes + 1-D 1e7),
  small_64x64 (all ops), odd_1000x777 (all ops),
  faer16 sum_axis0 large (must IMPROVE or stay within noise),
  mean_axis1/var_axis1/norm_all/min_axis0/max_axis0 spots (medium+large),
  1-D 1e7 min/max/mean/var/norm spots (reduce_all paths untouched),
  f32 spots, ndarray anchors (context only).
- Correctness gate 100% PASS (both configs) BEFORE any perf claim —
  including the CURRENT-BEHAVIOR broadcast lock (§1.4) and NaN lock.
- Deliverables: proposed.patch (git diff of ../rstsr vs 386948be), README
  with before/after tables + perf stat -d on the new sum_axis0 (expect
  next()-share ≈ 0, ins/elem ≪ 4.45/2.56, IPC ≫ 1.02 native), updated
  memory + CONTEXT.md.

## 5. dispatch_simd assessment (plan §5: lane accumulators — only if batching insufficient)

**Not needed for T2' — batching alone is expected to clear D3 by ~4×.**
Evidence: the probe's plain row-add batching (no SIMD machinery, no lane
types) reaches 49-55 GB/s under BOTH configs at 2048² (vs 18-24 GB/s
today), already ≥ the 40 GB/s DRAM triad ceiling when L3-assisted. The
register-lane variant ([f64;8]) shows more headroom (75 GB/s) but only for
register-resident folds; real axis outputs need materialized vacc. Per plan
§5's order of attack, fixed-size batching wins here; a lightweight-simd
lane variant would add a feature + dependency + compile time for a gap the
measurements say is closed by Option A. Revisit only if phase-2 numbers
land <10% in either config (not expected).

## 6. Risks & fallbacks

| risk | mitigation / fallback |
|---|---|
| Branch-1/3 codegen shifts when branch-2 body changes (same function) | gates sum_axis1/sum_all/all sizes; if tripped, out-of-line the new band walk (`#[inline(never)]`, argmax-precedent) or fall back to Option B |
| A2 bound widening `T: ExtReal + PartialOrd` judged unacceptable | drop A2 independently; report the 4.9× native min/max_all regression as a structural finding (or request a dtype-traits exception) |
| A2 changes int min/max semantics? | no: `if x < acc {x} else {acc}` IS `Ord::min` for PartialOrd ints; gate covers integer…(gate is f64/f32 — add an i32 min/max spot in phase 2 if desired) |
| Scratch vec allocation cost at small sizes (e.g. 64×64: 64×8 B) | alloc is one malloc vs 43 iterator clones today; gate small_64x64; if measurable, threshold the restructure on `size_scd ≥ CONTIG_SWITCH` and keep the old path below |
| Rayon twin: per-task scratch vec on 32 par_chunks tasks | measured faer16 axis0 445 µs has ample margin; gate faer16 sum_axis0; scratch inside the closure is task-local (no races), `f` bounds unchanged |
| Broadcast bug (§1.4) accidentally "fixed" or further broken | the rewrite keeps the finalize path byte-identical; the gate's CURRENT-BEHAVIOR lock fails the build if anything shifts; bug fix reported separately to the human |
| Negative/odd strides into branch 2 (t-view axis-1, f-contig axis-1) | idx_in formation identical to today; correctness gate covers t-view and f-contig shapes (odd sizes incl.) |
| `size_scd` huge (summed axis ≫ L3, e.g. reduce 1e7×small) | scratch = size_scd × 8 B could blow memory where the old code streamed; guard: if `scd_offsets` would exceed a cap (e.g. 1 MiB), keep the old iterator walk (fall-through-unchanged) — added as a size guard in phase 2 |

## 7. Phase-2 execution checklist (after G1)

1. Re-verify `git -C ../rstsr status --short` empty, HEAD 386948be.
2. Implement A2 (min/max strict-compare closures + `PartialOrd` bounds,
   4 impls in 2 wiring files) — smallest, independently gated.
3. Gate (`examples/correctness`, both configs — NaN locks are the A2 gate)
   + bench reduce (1-D min/max cells) → verify A2 accept.
4. Implement Option A serial (cpu_serial/reduction.rs branch 2 only).
5. Gate + bench both configs vs saved baselines; gates.
6. If clean: rayon twin + re-run (faer16 columns), perf stat, probe reruns.
7. `git -C ../rstsr diff > proposed.patch`; `git -C ../rstsr checkout -- .`;
   README with results; leave rstsr clean.

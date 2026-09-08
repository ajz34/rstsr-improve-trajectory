# PLAN.md — T4' phase-2 edit plan (elementwise), written at end of phase 1

Base: rstsr `386948be819baa334b8da02232f3a1944e5447d5` (clean tree verified
before baselining; phase 1 made no edits). All numbers cited here are in
[results/tables.md](results/tables.md) and
[results/perf/derived_summary.txt](results/perf/derived_summary.txt).

## 1. What phase 1 established (the evidence base)

1. **The contiguous kernel is already AVX-512 auto-vectorized and
   memory-bound. Reuse-variant add contig 2048² runs at 0.83 ins/elem,
   IPC 0.27, 40.3% L1d misses (perf, native) — sub-scalar instruction count
   is only possible vectorized; `vaddpd %zmm` confirmed in the bench binary.
   It ties ndarray `zip_prealloc` at large (2.103 vs 2.132 ms native) and is
   within 3% at medium. Local probes reproducing the exact branch shape
   (index loop + generic `FnMut(&mut MaybeUninit<T>, &T, &T)` closure + bounds
   checks) tie with hand-zipped and `chunks_exact` shapes at both configs
   (probes table).** Consequences:
   - Hypothesis (a) chunks_exact slice zips: **moot** — nothing to kill.
   - Hypothesis (b) MaybeUninit/closure overhead: **moot** — the closure
     inlines cross-crate and the MaybeUninit interface costs nothing
     measurable (zip_generic_mu_closure ≈ zip_direct).
   - Hypothesis (c) dispatch_simd lane variant: **not needed** (see §5).
   - T0's "add_contig 12.9 ins/elem, no autovectorization" is **revised**:
     that profile measured the ALLOCATING variant; the instruction stream was
     dominated by page-fault handling + allocator work (this run: A = 12.79
     ins/elem with 8358 faults/iter vs B = 0.83 with 0). The A-vs-B wall-time
     gap is the T7 fault rider, not kernel codegen.
2. **The strided branch is instruction-bound AND pattern-bound: 161.6
   ins/elem, 34 branches/elem, IPC 5.17, 4.3 GB/s (23.2 ms reuse at 2048²).**
   Root causes, both fixable:
   - *Access pattern*: after `translate_to_col_major(..., K)` +
     `greedy_layout` (`rstsr-common/src/layout/rearrangement.rs:36`,
     stride-ascending sort), `a` and `c` are contiguous on the inner axis and
     `bᵀ` walks with stride 2048 (16 KiB hops). Raw-loop bound for that
     pattern: 16.6 ms (probe `bound_rowmajor`).
   - *Iterator machinery*: the else-branch drives three
     `IterLayoutColMajor::next` per element
     (`rstsr-common/src/layout/iterator.rs:290-297`: end check, multi-axis
     index/offset update, `try_into().unwrap()`), ~6.2 ms of pure overhead
     above the raw bound (23.2 − 16.6).
   - A 64×64 blocked kernel that reads `b` in tiles (b-tile = 64 cache lines,
     reused 64× in L1) removes both: **6.8 ms = 3.4× on the reuse variant**,
   - faer16 strided reuse is 2.14 ms (47 GB/s aggregate); a parallel blocked
     kernel targets ~0.5–0.8 ms.
3. **Broadcast reuse (B) is already good** (1.49 ms serial native; the
   `[1,n]` row stays L1-resident). It takes the CONTIG branch (greedy perm +
   `translate_to_col_major_with_contig` yields size_contig = 2048 ≥
   CONTIG_SWITCH=16, outer layout = 1-D, inner fixed-length loop).
4. **CONTIG_SWITCH interplay** (`rstsr-native-impl/src/cpu_serial/op_with_func.rs:8`):
   fully-contig layouts collapse to 0-dim layouts with size_contig = full
   size (single inner loop); partially-contig (broadcast) → row-length
   chunks; anything with no f-contig prefix (the strided case) →
   `size_contig = 0` → the iterator branch. Any strided fix must live in
   that else-branch (or beside it) and must not touch the contig branch.

## 2. Design options for the strided branch (laid out, one recommended)

### Option A — 2-D blocked fast path beside the else-branch (RECOMMENDED)

Add a blocked 2-D tile kernel used when, after translation, the layouts have
no common f-contig prefix (`size_contig < CONTIG_SWITCH`), `ndim == 2`,
`size ≥ 4096`, and no stride-0 axis among the tiled axes (broadcast safety;
zero-stride cases fall back to the existing iterator path).

- Shape (from probe `blocked64`, dtype-generic, closure unchanged):
  ```rust
  // lc', la', lb' are the translated IxD layouts (post greedy perm).
  const TILE: usize = 64; // sweep {32, 64, 128} once in phase 2
  for j0 in (0..n).step_by(TILE) {          // axis order = lc' greedy order:
      for i0 in (0..m).step_by(TILE) {      // c's fastest axis innermost
          for i in i0..min(i0+TILE, m) {
              for j in j0..min(j0+TILE, n) {
                  f(&mut c[oc + i*sc0 + j*sc1],
                      &a[oa + i*sa0 + j*sa1],
                      &b[ob + i*sb0 + j*sb1]);
              }
          }
      }
  }
  ```
  The inner loop compiles to direct index arithmetic on one bounds-checked
  base pointer each (or split at tile bounds); `f` inlines as today. All
  offsets are isize stride sums — correct for negative strides (greedy flips
  only the pivot layout; others may keep negative strides).
- Files touched:
  - `rstsr-native-impl/src/cpu_serial/op_with_func.rs:10-45`
    (`op_mutc_refa_refb_func_cpu_serial` — insert dispatch in/next to the
    else-branch at line 37-44). Mirror mechanically into
    `op_muta_refb_func_cpu_serial` (`:115-146`) if numbers hold.
  - `rstsr-native-impl/src/cpu_rayon/op_with_func.rs:13-80`
    (`op_mutc_refa_refb_func_cpu_rayon` — same dispatch below the
    PARALLEL_SWITCH guard at `:31-34`; parallelize the outer tile loop with
    rayon directly, mirroring the file's existing pool handling).
- Blast radius: ~40-60 lines in 2 files, no trait/API/type changes, no new
  crate features, dtype-generic (all ops of this driver family —
  add/sub/mul/div/rem/bitwise — benefit via the shared kernel).
- Expected (probe-anchored): serial large strided B 23.2 → ~7 ms native and
  22.8 → ~6.9 ms portable (~3.3×); A alloc-inclusive 28.4 → ~12 ms; faer16 B
  2.14 → ~0.5-0.8 ms. D3 (≥10% reuse, both configs) exceeded by >10× margin.

### Option B — monomorphized 2-D index loops (D8-style) as the default else-branch

Replace the iterator zip in the else-branch with ndim-dispatched explicit
loops (Ix1/Ix2/Ix3 monomorphizations computing offsets by stride adds, no
`IterLayoutColMajor`). Cuts the ~161 ins/elem machinery to ~20-30/elem but
keeps the 16 KiB-hop pattern: probe-anchored expectation ≈ the raw bound,
16.6 ms (1.4×), i.e. well under half of Option A's win. No runtime feature
flag needed (unlike D8, this is a code change, not a caller choice).

### Option C — dispatch_simd gather/lane variant for strided

Rejected without build: the strided pattern is a stride-n gather; fixed-lane
SIMD cannot vectorize it (strided probes show native ≈ portable), and §5's
rule is batching-first anyway. Zero expected win, nonzero compile-time cost.

**Recommendation: Option A**, with Option B as fallback if A trips a gate
regression it cannot be patched for (e.g. an unforeseen small-strided
interaction). Option B is strictly safe but leaves 2× on the table.

## 3. Semantics preservation (phase 2 correctness contract)

- **All dtypes stay on the generic path.** The tile kernel is dtype-generic
  (`T` + closure); no dtype special-casing anywhere. Integer/complex/half
  behavior unchanged; correctness gate adds i32 and Complex<f64> spots.
- **Order-independence**: elementwise `f` is applied once per output element
  with exactly one value per input element; no reductions, no FP
  re-association, so any visit order is bit-identical. Tiling only changes
  visit order.
- **Broadcast zero-stride**: layouts with stride-0 axes bypass the tile path
  (fallback to the existing iterator branch). Explicit zero-stride and
  auto-broadcast correctness cases already exist in
  `examples/correctness.rs` and stay.
- **Aliasing policy**: unchanged. Through safe rstsr API, `c` cannot alias
  `a`/`b` (owned `Vec` storage, no shared mutable views); the current kernel
  already assumes non-aliasing (uninitialized output write). Phase 2 adds no
  new `unsafe` beyond what the kernel already does (index arithmetic on raw
  slice offsets); if tile loops use split-at-bound subslices, that is the
  only new bounds assertion. Document the same assumption in a comment.
- **Negative strides / flip views**: stride arithmetic (isize) handles them;
  correctness gate gains `a.flip(0) + b` and `bt.t()` odd-size cases
  (rstsr's own test suite at `rstsr-core/src/tensor/operators/op_binary_arithmetic.rs:1136-1149`
  is the reference list).
- **Odd sizes**: `step_by` yields degenerate tail tiles; gate = 1000×777
  strided correctness + a new strided-odd perf gate.

## 4. Phase-2 bench matrix & accept criteria

Matrix = this crate's suites, re-run unchanged after the rstsr edit (path-dep
picks it up), both configs, both devices, f64+f32, A/B/C variants; plus:
- NEW gate: `add_strided_small_64x64-serial` (tile-path degenerate case) and
  `add_strided_odd_1000x777-serial` (tail tiles), added to
  `benches/elementwise.rs` in phase 2.
- Correctness additions: i32 + Complex<f64> spots, `flip()` negative-stride
  case (phase-2 edit of `examples/correctness.rs` only).

Accept (D3, per plan §2/D7):
- **Primary**: add strided large REUSE variant improves ≥10% in BOTH configs
  (target: ~3× — 23.2 → ≤8.0 ms serial native; 22.8 → ≤8.0 portable; faer16
  2.14 → ≤1.0 ms). Stretched target vs the 17.2 ms T7 bound: **beat the
  bound** (blocking removes the hop pattern the bound still pays); probe says
  6.8 ms is reachable.
- **Secondary**: alloc-inclusive A strided ≥25% (28.4 → ≤21 ms; expected
  ~12 ms with the rider shared); faer16 A ≥40%.
- **Gates (no >2-3% regression, both configs, reuse variant judged for
  large)**: add contig small/medium/odd (serial+faer16), add bcast large,
  mul/scale large, new small-strided gate, ndarray anchors unchanged.
- Correctness gate 100% PASS (both configs) incl. new dtype/negative-stride
  cases.
- Deliverables: proposed.patch (git diff of ../rstsr), README post-change
  table, perf stat -d on the new strided kernel (expect ins/elem ≪ 161.6 and
  task-clock GB/s ≫ 4.3), updated memory + CONTEXT.md.

## 5. dispatch_simd verdict (preliminary, evidence-based)

**Not proposed for T4'.** Evidence: the contiguous elementwise kernel already
auto-vectorizes to 0.83 ins/elem under plain `native` with no batching
rewrite (probes: chunks_exact8/zip_direct/idx-loop all tie); the strided
branch's cost is the gather pattern + iterator machinery, which fixed-lane
SIMD cannot address (strided probes: native ≈ portable). Plan §5's
batching-first rule is satisfied vacuously — batching exists already in
effect. dispatch_simd would add a feature, a dependency, and binary size for
no measurable win on this kernel family. Revisit only if T3 (vecdot
reductions) shows a need; the T4' diff contains no dispatch machinery.

## 6. Risks & fallbacks

| risk | mitigation / fallback |
|---|---|
| Small-strided regression from tile setup | tile loops are `step_by`-cheap; gate `add_strided_small` before/after; fallback: raise the `size ≥ 4096` dispatch floor, or shrink to Option B |
| Zero-stride strided case mis-tiled | explicit stride-0 guard → iterator fallback; correctness has both broadcast forms |
| Negative strides mis-indexed | isize stride sums (already how translated layouts work); add flip() correctness cases |
| faer16 tile parallelization overhead at medium | keep PARALLEL_SWITCH guard; tiles are ~32 KB → ≥64 tiles at 512², enough for 16 threads |
| Regression hides behind A-variant fault rider | judge on B (T7 rule), C column recorded as secondary |
| Odd tail tiles wrong | step_by degenerate tiles + 1000×777 strided correctness/perf gate |

## 7. Phase-2 execution checklist

1. `git -C ../rstsr status` clean + HEAD 386948be (re-verify).
2. Implement Option A (serial + rayon, `op_mutc_refa_refb` first).
3. Extend this crate's benches/correctness per §4.
4. Full re-run: `./reproduce.sh all`; tables + perf derived summary updated.
5. If gates pass: `git -C ../rstsr diff > proposed.patch` (do NOT commit);
   then `git -C ../rstsr checkout -- .` to restore.
6. If gates fail: revert rstsr, document honest negative, consider Option B.

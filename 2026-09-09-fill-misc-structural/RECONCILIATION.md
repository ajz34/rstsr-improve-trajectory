# RECONCILIATION — plan hypotheses vs outcomes (input for the final campaign summary)

Maps the original T1–T5 hypotheses
([../2026-09-08-plan-prompt/260908-plan-cpu-serial-efficiency.md](../2026-09-08-plan-prompt/260908-plan-cpu-serial-efficiency.md)
§6, and the code map §7 ranking) against what the accepted experiments
actually found. One line per hypothesis: **HIT** (won, patched), **MISS**
(theory wrong, honestly recorded), **MOOT** (target disappeared on closer
inspection), **SPUN OUT** (became its own task).

## T0 (harness) — re-ranking outcomes

- Code map rank #1 (vecdot contiguous-remaining RMW) → **scoped down then
  corrected**: T0 showed 1-D dot memory-bound; the batched case became the
  target; T3's branch probe further showed batched am1 hits branch 1
  (contiguous-SUMMED), not the RMW branch — the RMW branch serves axis-0
  contraction geometry (**branch attribution corrected**, T3' PLAN §1.2).
- Code map rank #6 argmax → **promoted to top tier** by T0 evidence
  (13× behind ndarray, 283 ins/elem) → T6, 60× (**SPUN OUT, biggest win**).
- T0's "add_contig 12.9 ins/elem, no autovectorization" → **MISS (alloc
  artifact)**: reuse-variant profiling shows 0.83 ins/elem, `vaddpd %zmm`
  (T4'). Lesson institutionalized by T7: profile kernel-only denominators.
- T0's allocation-rider observation → **SPUN OUT** as T7 (study) — the
  campaign's denominator discipline comes from it.

## T1 (transpose-assign)

- (a) route 2-D order-change through the dormant blocked kernel → **HIT**
  (2.65–2.94× large, ~9–10× odd/oddT, 6.4× small; both assign families).
- (b) contiguous assign via slice copy → **MOOT** (contiguous branch was
  already a zipped slice iterator; code map §2(a) wording oversold it).
- (c) tiled generic strided assign → subsumed by (a) for 2-D; 3-D+ falls
  through unchanged.
- (d) `dispatch_simd` lane variant → **skipped** (see dispatch_simd verdict
  in EDIT-GUIDE a).
- Interim idea "use `dispatch_dim_layout_iter` for +13/+98 %" → superseded:
  the blocked kernel gives 2.9×/9.7× on the same cells.

## T2 (reductions)

- (a) local accumulators instead of clone-through-closure CHUNK=48 →
  **PARTIAL HIT**: option A (iterator-free band walk) −16/−38 % sum_axis0;
  the ≤1.0 ms stretch target NOT met (per-group vacc alloc+finalize and
  driver overhead remain — recorded as the D3-accepted outcome).
- (b) write in output order to kill the final order-fixup copy → **MOOT**:
  T2's layout probe proved the order-fixup pass is **dead code** in the
  default build (`TensorIterOrder::default() == K`, so the extra
  `op_muta_refb_func` copy never fires — T2' PLAN §1.1).
- (c) verify `unrolled_reduce` autovectorization → **HIT with a twist**: it
  vectorizes; the blocker was the `minnum`-shaped min/max closure — A2
  strict-compare wiring (min_all 1e7 native −88 %) fixed it.
- (d) argmin/argmax fast path → **SPUN OUT** to T6 (composeability with the
  T2' patch proven both orders).
- Unplanned outcomes: rayon twin rewrite **REVERTED** (lottery discipline);
  stride-0 summed-axis bug documented, not fixed (BUG-NOTES a).

## T3 (vecdot)

- (a) local accumulator killing MaybeUninit RMW in contiguous-remaining →
  **HIT** (A2; ins/elem 3.17 → 0.90 on axis0).
- (b) lane accumulators via dispatch_simd → **skipped**; 8-lane
  `unrolled_binary_reduce` sufficed (A1: batched am1 −22 %).
- (c) `inner_dot` routed through contiguous unrolled path → **HIT** (B1/B2:
  % serial −29 %, faer16 ≤1e6 −73..−96 %; faer16 1e7 honest miss — DRAM-bound).
- (d) stride-based inner loop for the general branch → **negative, left
  alone** (branch-3 shapes have no anchor pressure; T3' PLAN §1.2).
- Latent uninit-beta hazard found and reported, not fixed (BUG-NOTES b).
- Complex in dispatch_simd (D6 decision) → **moot** with the feature
  rejected (c64 cost is arithmetic, not dispatch — probes A5/A6).

## T4 (elementwise)

- (a) contig branch → chunks_exact slice zips → **MOOT**: the contig branch
  already compiles to vectorized streaming (0.83 ins/elem); the plan's
  "index arithmetic + bounds checks" cost was not real post-inline.
- (b) reduce MaybeUninit/closure overhead → **MISS (theory revised)**: the
  closure+MaybeUninit layer costs nothing measurable when inlined (T4'
  probe; T5's fill kernel confirms 0.29 ins/elem) — the problem is API
  ergonomics, not speed (EDIT-GUIDE d).
- (c) dispatch_simd → **skipped** (a).
- (d) strided-branch blocking → **HIT, the real win** (blocked 64² tile
  path: 3.08× serial native large, 8.7× odd, 7–16× small; visit-order note
  recorded).

## T5 (fill/misc — this task)

- "fill_promote slice-fill" → **MOOT/HONEST-SKIP**: the kernel already sits
  at/above the raw write-only bound at every size ≥ medium (507 vs 544 µs
  native large); `slice::fill` would be code clarity, not performance
  (PLAN.md §2-i). The strided branch has ~2× headroom but no user surface
  reachable from the tensor API (PLAN §2-iii).
- "used by zeros/ones creation" → **MISS in the code map**: creation uses
  `vec![fill; len]`, never `fill_promote` (PLAN §1; EDIT-GUIDE b).
- "the 88 % rider on rt::full" → confirmed as **T7's domain**: env tunables
  4.59 → 0.62 ms; kernel exonerated (at the write floor, beats numpy's
  allocating fill once the rider is removed).
- Consolidation halves (EDIT-GUIDE/BUG-NOTES/reconciliation) → delivered.

## Campaign-level observations (for the final summary)

- Biggest wins came from structure, not SIMD: argmax 60×, transpose 2.7–9.9×,
  elementwise strided 3.1×, min/max −88 %, vecdot/inner_dot −22..−96 % —
  all plain-loop restructurings; zero features added, five patches compose
  pairwise (verified per-experiment).
- The single most transferable methodological result is T7's: separate the
  allocation/page-fault rider (glibc mmap class ≥ ~32 MiB outputs,
  ~4 ms/8193 faults) from kernel time; judge kernels on reuse denominators;
  remedies are user-side (reuse APIs, env tunables) before library-side.
- Remaining known gaps (documented, unpatched): sum_axis0 stretch (1.0 ms)
  not reached; faer16 large % DRAM-bound; reductions reuse path missing;
  strided-fill branch slow but unreachable; stride-0 reduce semantics bug
  and inner_dot beta hazard await maintainer PRs.
- Final phase (bookkeeping diffs for rstsr-book / rstsr-agents) still
  pending owner authorization: reuse-idiom page + MALLOC note (T7 EDIT-GUIDE
  Option 1, now including fill numbers from this task) and a
  benchmark-conventions skill (lottery/A-B/denominator lessons, EDIT-GUIDE
  g.8).

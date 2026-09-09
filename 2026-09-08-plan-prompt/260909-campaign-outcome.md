# Campaign outcome — rstsr CPU efficiency at `386948be`

- **Date**: 2026-09-09 (execution complete)
- **Plan**: [./260908-plan-cpu-serial-efficiency.md](./260908-plan-cpu-serial-efficiency.md)
- **Executed by**: main agent (this session) + one code agent + one review agent,
  two-gate cadence (G1 pre-implementation, G2 pre-acceptance) per task.
- **Baseline survey**: [../2026-09-09-bench-harness-baseline/](../2026-09-09-bench-harness-baseline/)

## Verdict at a glance

Five integration-ready patches are proposed (never applied; each `proposed.patch`
verifies with `git apply --check` against a fresh `386948be` and each pair is
file- or hunk-disjoint so they compose in any order):

| # | Experiment dir | Verdict | Headline (serial, large f64) |
|---|----------------|---------|------------------------------|
| 1 | `2026-09-09-argmax-argmin` | patch, G2 PASS | argmax/argmin 1e7: 92.6→1.82 ms **50.8×** portable, 93.7→1.55 **60.3×** native; faer16 12→0.31 ms; perf 280→3.7 ins/elem. Also fixes a latent rayon wrong-answer when NaN sits at a chunk start |
| 2 | `2026-09-09-elementwise` | patch, G2 PASS | strided add 2048²: 23.1→7.5 ms **3.1×** (blocked 64×64 tile path, guarded); strided odd 8.7×; A+MALLOC composed 4.0× |
| 3 | `2026-09-09-transpose-assign` | patch, G2 PASS | transpose copy 2048²: 17.2→6.0 ms **2.9×** (dormant blocked orderchange kernels routed into both assign families); odd 1000×777 ~**9.7×** |
| 4 | `2026-09-09-reductions` | patch, G2 PASS | sum_axis0 native −**32–38%** (native≤portable regression eliminated); min/max_all 1e7 native 10.4→1.23 ms **−88%** |
| 5 | `2026-09-09-vecdot` | patch, G2 PASS | batched vecdot axis-0 **−64%** (RMW eliminated), axis-1 −22%; `%` 1e7 serial −29%; `%` small-n faer16 −73…−96% |

Measurement-only: `2026-09-09-alloc-pagefault-study` (T7), and `2026-09-09-fill-misc-structural` (T5, honest-skip + campaign consolidation).

## Honest negatives and skipped levers (all evidence-backed)

- **`dispatch_simd` (the plan's central feature idea): NOT proposed.** All five
  kernel tasks reached their ceilings under both target configs via plain
  restructuring; measured lane-array shapes were slower than slice/unroll
  shapes. ADR-candidate text ("do not add") with per-task evidence table lives
  in T5's EDIT-GUIDE.md. The D6 complex decision (not in v1) is recorded in
  T3's README.
- **fill kernel: honest-skip** — `fill_promote` is not the creation path
  (`vec![]` is), and the clone loop already compiles to a vectorized broadcast
  at the write floor (<5% D3 class both configs).
- **faer16 `%` at 1e7: honest miss** (−1.3% wall; core-time ins/elem 11.76→2.95
  but the op is DRAM-bound). Reductions' ≤1.0 ms stretch for sum_axis0 also
  missed (driver overheads; D3 met).
- **Rayon-twin rewrites reverted twice on evidence** (reductions branch-2;
  smaller vecdot deviations kept): serial/rayon fold bodies legitimately
  diverge; recommendation is to share geometry predicates, not bodies.

## Cross-cutting findings (bigger than any single kernel)

1. **The allocation fault rider**: ≥32 MiB fresh outputs pay ~8193 zero-page
   faults ≈ 4 ms (sys>user) — 66–88% of wall for contig add/full, 25–57% for
   transpose copy. Recovery today without code change: reuse APIs
   (`op_mutc_refa_refb_func`, `assign`, `+=`, `fill`) or
   `MALLOC_MMAP_THRESHOLD_=67108864 MALLOC_TRIM_THRESHOLD_=134217728`
   (96–97%). THP `madvise` is blocked by rstsr's 64-B (non-page) alignment.
2. **Contiguous kernels were already fine**: elementwise contiguous is
   AVX-512-vectorized and memory-bound (0.83 ins/elem); T0's "12.9 ins/elem,
   no autovectorization" was allocator-instruction contamination of the
   allocating variant. All instruction floods lived in strided/batched/
   arg-reduction paths.
3. **Judge kernel wins on reuse-variant denominators**; single criterion runs
   on ~0.5 ms cells carry ±5% build-layout lottery — paired same-session A/B
   is the standard (methodology encoded in CONTEXT.md).

## For the human (integration queue)

1. **Five patches** above, each dir's README carrying its PR-notes: API-break
   notes (ArgCmp enum; inner_dot bound widening), visit-order/sign-bit notes,
   the faer16 `%` wall-miss framing, and the f32-portable codegen-lottery flag.
2. **Upstream bug reports** (T5 BUG-NOTES.md, all gate-verified):
   - stride-0 summed-axis reductions mis-multiply (`reduction.rs:226`,
     `lm.shape()` indexed with summed-axis indices; `[1,n]→[m,n]` sum gives
     n×v, numpy gives m×v);
   - inner_dot reads `beta*c` from uninit output (serial pre-loop ≈:231–233,
     rayon post-fold :91) — latent NaN hazard if β≠0 ever becomes reachable;
   - `dispatch_dim_layout_iter` f32-vecdot 0.57× regression caveat.
3. **Docs candidates** (not drafted as diffs — the plan's bookkeeping phase
   conditioned on user-facing features landing, which did not happen; content
   is ready in T7/T5 EDIT-GUIDE.md if wanted): reuse-idiom + MALLOC tunables
   guidance for rstsr-book; benchmark-conventions for rstsr-agents.

## Deviations from the plan (all sanctioned or documented)

- Reordering after T0 evidence (argmax first, alloc study inserted) — endorsed
  by the T0 review as the G1 review input.
- T6 (argmax) spun out of T2; T7 (alloc study) added — both per plan §6's
  extensibility clause.
- Code-map corrections: `device_faer/rayon_auto_impl` are **symlinks** to
  `feature_rayon/auto_impl` (not a dead duplicate); `fill_promote` is not the
  creation path; reductions "order-fixup copy" is dead code.
- Toolchain identity is unpinned across sessions (stable 1.97.1 recorded at
  several gates; rstsr's rust-toolchain.toml does not propagate along path
  deps) — cross-experiment absolute comparisons are drift-prone.

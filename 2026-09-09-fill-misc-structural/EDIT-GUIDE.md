# EDIT-GUIDE — campaign-wide structural guide (T5 consolidation, base 386948be)

Consolidates the structural findings of the six accepted campaign
experiments (T0 harness, T1' transpose-assign, T2' reductions, T3' vecdot,
T4' elementwise, T6 argmax, T7 alloc-pagefault) plus T5's fill study, into
concrete, ranked edit guidance for the rstsr maintainers. **Nothing here has
been applied**; per-experiment diffs live in each directory's
`proposed.patch`. Sections: (a)–(g) structural topics, then the ranked
"what the evidence supports" list. Corrections to the T0 code map are in
§(b).

---

## (a) `dispatch_simd` — final verdict: do NOT add the feature (ADR candidate)

Five kernel tasks each evaluated the plan §5 `dispatch_simd` design
(feature-gated fixed-lane `lightweight-simd` + TypeId runtime dispatch) and
each skipped it with per-task evidence; plain batching + in-tree unrolls
reached the memory/FLOP limits in both target configs every time:

| task | where SIMD was considered | what sufficed instead | evidence |
|---|---|---|---|
| T4' elementwise | contig + strided add/mul | contig already auto-vectorized (`vaddpd %zmm`, 0.83 ins/elem); strided is a gather no fixed lanes help; blocked 64² tile path 3.0× | T4' README phase-1 §1–2, PLAN §5 |
| T1' transpose | order-change copy | dormant 64×64 blocked kernel 2.65–2.94× large, ~9.7× odd; after: 13.2 ins/elem, memory-bound | T1' README phase-2 |
| T2' reductions | sum/min/max | 8-lane `unrolled_reduce` vectorizes once the `minnum`-shaped closure was replaced by strict-compare (min_all 1e7 native −88 %, 0.43 ins/elem) | T2' README mechanism |
| T3' vecdot/inner_dot | contraction | flat row loop + local accumulator + 8-lane `unrolled_binary_reduce`: am1 −22 %, % serial −29 %, 6.5 % of FMA peak; faer16 large is DRAM-bound (instruction collapse 4×, wall flat) | T3' README results |
| T6 argmax | scan | 8-lane scalar unroll: 60× serial native (0.85 → 49 GB/s) | T6 README perf |
| T5 fill | broadcast stores | kernel already a vectorized broadcast (0.29 ins/elem, at the write bound) | T5 tables |

**ADR-candidate text** (for an `rstsr` ADR, if the owner wants the decision
recorded):

> **ADR: no `dispatch_simd` feature (rejected design)**
> *Decision.* rstsr's CPU kernels do not get a feature-gated fixed-lane SIMD
> layer (`lightweight-simd`, TypeId dispatch over f32/f64, F64x8/F32x16) as
> considered in the 2026-09 efficiency campaign.
> *Rationale.* Every kernel the campaign examined (elementwise, assignment/
> transpose, value reductions, vecdot/inner_dot, argmax/argmin, fill) reached
> its memory-bandwidth or FLOP ceiling under both `portable` and `native`
> via plain restructuring — contiguous slice loops, 8-lane unrolled
> accumulators, blocked 2-D iteration — all of which LLVM auto-vectorizes
> under `-C target-cpu=native`. A fixed-lane layer would add a dependency,
> a feature matrix dimension, and per-kernel fallback code for zero measured
> gain in any examined cell; the one in-tree runtime dtype dispatch
> (`gemm_faer_ix2_dispatch`) remains the template should a future kernel ever
> show a measured ISA gap.
> *Consequences.* Kernels stay scalar-generic and autovectorization-dependent;
> contributors should benchmark under both `portable` and `native` configs
> (D7) before assuming AVX-512 helps; `dispatch_dim_layout_iter` remains an
> opt-in per-caller tool with the f32-vecdot caveat recorded in BUG-NOTES (c).

---

## (b) Code-map corrections (for anyone using `rstsr-386948b-code-map.md`)

1. **§5 "dead byte-identical copy" is wrong — it is ONE physical file.**
   `rstsr-core/src/device_faer/rayon_auto_impl/*` are **symlinks** to
   `rstsr-core/src/feature_rayon/auto_impl/*` (verified: e.g.
   `assignment.rs -> ../../feature_rayon/auto_impl/assignment.rs`). The live
   DeviceFaer wiring and the supposed "dead duplicate" are the same files;
   there is nothing to delete or deduplicate. Practical consequences:
   - edits to DeviceFaer non-gemm ops show up in `git diff` under the
     `feature_rayon/auto_impl/` path;
   - a patch touching `device_faer/rayon_auto_impl/` content paths still
     applies (git follows symlinks), but review tooling and `.gitattributes`
     filters see the `feature_rayon` path;
   - five campaign patches (T1' routing notes, T2' A2 closures, T6 ArgCmp
     wiring, T3/T4 twins) were made through the `feature_rayon` path.
2. **§2(e)/§7-item-8 "fill_promote used by zeros/ones creation" is wrong.**
   Creation (`rt::full/ones/zeros`) routes to device `*_impl` constructors →
   `vec![fill; len]` (`device_cpu_serial/creation.rs:17-75`), never through
   `fill_promote_cpu_serial`. The kernel serves only `c.fill(v)` (owned
   tensors ⇒ contiguous layouts), `eye`'s diagonal, and BLAS beta-zeroing.
   T5 measured the kernel at the write-bandwidth floor (no patch; PLAN.md).
3. Minor: §6 "no benches" — true at 386948be, but the campaign's T0 crate is
   the de-facto harness pattern for CPU benchmarking of this tree.

---

## (c) Serial/rayon kernel duplication vs shared blocking helpers

Facts at 386948be: `rstsr-native-impl/src/cpu_serial/` and `cpu_rayon/`
carry 9 near-mirrored files each; every rayon twin re-implements the serial
branch structure with a `PARALLEL_SWITCH` serial fallback. Campaign
experience with editing both twins:

| task | serial twin | rayon twin | note |
|---|---|---|---|
| T1' assign/transpose | blocked routing | blocked routing (same shape) | composed cleanly; rayon got debug_assert'd raw-pointer stores |
| T4' op_with_func | blocked 2-D iterator | same | visit-order note applies to serial only (rayon always order-free) |
| T2' reduction band walk | rewritten (−16/−38 %) | same rewrite tried and **REVERTED** (+10–40 % single-run readings; lottery) | risk containment: twins diverged on purpose |
| T3' vecdot A2 | single local accumulator | CHUNK-local per-band buffers (per-position vaccs would serialize the parallel split) | twins NEEDED different shapes |
| T6 argmax | 8-lane seeded scan | in-order partial combine | rayon semantics needed extra machinery to be serial-exact |

**Assessment (honest):** a shared-helper refactor (e.g. a common
blocked-2-D geometry/walker in `rstsr-common` used by both assign and
`op_with_func`) would remove ~200–300 lines of mechanical duplication
(T1' and T4' each hand-rolled their own tile walker with nearly identical
guards: ndim==2, fast-axis strides ±1, size floor, no zero-stride axes).
But the campaign's evidence cuts against deeper sharing:

1. The *fold/accumulate bodies* legitimately differ between twins (T3' A2,
   T6 combine logic) — abstraction there costs performance shape control.
2. Twin divergence was used as an experimental control (T2' revert); shared
   helpers would have forced the risky change onto both twins at once.
3. The documented ±5–11 % build-layout lottery means any refactor of hot
   iteration must be re-gated with back-to-back A/B per cell — the helper
   indirection risks perturbing codegen for cells it never touches (T6's
   `#[inline(never)]` lesson).
4. The duplicated guards (the actual bug-prone part) are small and now
   documented here and in the patches' doc comments.

**Recommendation:** share *geometry predicates* only — a small
`pub(crate)` helper computing "2-D order-change shape (m, n), fast-axis
strides +1, no broadcast axes, size ≥ floor" reused by assign, op, and any
future strided kernel — and keep the walkers/folds per-kernel. Do not merge
serial and rayon drivers behind a common abstraction.

---

## (d) `MaybeUninit` closure API ergonomics at the operator layer

What T4's anatomy actually found (revising the plan's hypothesis): the
`MaybeUninit` + generic-closure interface at the driver layer costs **zero
measurable performance** — cross-crate inlining erases it (probe:
`zip_generic_mu_closure` ≈ `zip_direct`; contig add 0.83 ins/elem; T5:
same for the fill kernel, 0.29 ins/elem). The problem is ergonomics and
discoverability, not speed: T7 found the single-pass reuse driver
(`op_mutc_refa_refb_func`) is public but practically undiscoverable
(`MaybeUninit` closures, `impl TensorViewMutAPI`, no `rt::` surface, no
tensor-method sugar), even though it recovers 66–77 % of large-add wall.

A safer/faster closure contract would therefore be *wrapping, not
replacing*:

1. **Tensor-method sugar over the existing driver** (T7 EDIT-GUIDE Option
   2a): `c.assign_with(&a, &b, |x, y| x + y)` (+ fallible `_f` twin, house
   style). Hides `MaybeUninit` + view plumbing; zero new kernel code; every
   number behind it is already proven.
2. **Visit-order contract documentation** (T4' PR-note): stateless
   per-element ops are bit-identical under the tiled visit order; a
   *stateful* `FnMut` closure passed to the low-level drivers must not
   assume layout order (the rayon path never guaranteed it). This belongs
   in the driver docs when the sugar lands.
3. **Kernel level**: keep `&mut [MaybeUninit<TC>]` + index arithmetic as
   today; the write-once coverage justification pattern (T1' MaybeUninit
   comment) is the right documentation style if more uninit kernels appear.

---

## (e) Reductions lack a reuse path (T7 finding, still true)

`sum_axes`/`mean_axes`/... allocate their output unconditionally; there is
no `sum_axes_into(&mut c, &a, axes)` / `reduce_into`. Measured impact is
nil for small outputs (reduction outputs are tiny — 0 % fault rider, T7
negative controls) — the value is **API consistency** with the other
reuse paths and enabling kernel-only benches for future reduction work
(T2's residual driver overhead could then be measured without the alloc).
Lowest priority of the API additions; T2's proposed.patch does not need it.

---

## (f) THP alignment blocker (T7 finding)

`madvise(MADV_HUGEPAGE)` hides the fault rider (0.68 vs 4.20 ms/iter for a
fresh 32 MiB fill; 16 huge faults vs 8192) — **but it cannot be applied to
rstsr's current buffers**: `aligned_alloc(bytes, 64)` (via glibc's memalign
trim path) returns 64-B-aligned, not page-aligned pointers → `EINVAL`
(madvise requires page-aligned addr; observed during T7 development). A THP
integration therefore = change `rstsr-common/src/alloc_vec.rs` alignment
(4 KiB minimum, 2 MiB for THP; 2 MiB alignment showed no measurable alloc
slowdown) + hint + fallback handling. System THP is `[madvise]` here but
often `never` on HPC — not portable guidance. The env-tunable alternative
(96–97 % recovery, user-side, zero code) dominates for now; a device-level
buffer pool (T7 EDIT-GUIDE Option 3) would deliver the same ceiling
by default at higher complexity.

---

## (g) Leftovers flagged across the campaign READMEs

1. **`arg_contig_seeded_cpu_serial` visibility** (T6 follow-up): should be
   `pub(crate)` in `rstsr-native-impl` (only `arg_contig_cpu_serial` needs
   crate-wide visibility); cosmetic, fold into any next reduction PR.
2. **fill/assignment `PARALLEL_SWITCH=16384`** (T5): parallel fill is 18 %
   slower than serial at 512² (16.4 vs 13.8 µs) and never faster at large
   (write bandwidth saturates per-core). Raising the threshold for the
   fill/assign family is a µs-class tuning, no patch.
3. **`rt::zeros` laziness is size-regime dependent** (T5, refines T0/T7):
   calloc-lazy zero pages only ≥ ~32 MiB outputs (2.3 µs at 2048² f64);
   below the glibc mmap cap, calloc = alloc + explicit memset — ties `full`
   at 2 MiB (13.3 µs, regular stores) and is **2.4× slower than `full` at
   6.2 MiB** (107 vs 45 µs; glibc's non-temporal memset runs at DRAM speed
   ~58 GB/s while `vec![v; n]`'s broadcast stays cache-resident). Absolute
   cost tiny → "leave zeros alone" stands, but docs should not claim
   unconditional laziness.
4. **faer16 creation is serial delegation by design** (T5):
   `DeviceFaer::full_impl/zeros_impl/ones_impl` call
   `DeviceCpuSerial::default().*_impl` (`feature_rayon/auto_impl/
   creation.rs:19,62,68`) — one allocating thread, no pool. Fine (the
   rider dominates anyway); worth a comment in-tree if anyone looks for
   parallel creation.
5. **Anchor caveats for future benchmarks** (from T1/T0): ndarray
   `.t().to_owned()` PRESERVES f-order in ndarray 0.16 (as_slice() == None)
   — not a rstr-comparable "transpose copy"; numpy `.T.copy()` was
   anomalously slow on this machine (99.8 ms) — context only (BUG-NOTES d).
6. **`dispatch_dim_layout_iter`** stays per-caller (T0 D8 + BUG-NOTES c);
   note most of its measured wins (odd transpose 1.98×, strided add 1.36×)
   are now superseded by the T1'/T4' kernel patches (9.7× / 3.1×).
7. **Toolchain identity is unpinned across the campaign's crates** (T5
   correction; committed evidence
   `results/toolchain_identity.txt` in this directory): rustup resolves the
   toolchain from the invoking directory tree only — the rstsr checkout's
   `rust-toolchain.toml` (`channel = "nightly"`, undated) does **not**
   propagate along cargo path deps, so every campaign crate (all siblings
   of rstsr) actually built with the machine's rustup default. Recorded at
   T5 gate time: stable `rustc 1.97.1 (8bab26f4f 2026-07-14)`; the earlier
   READMEs' "nightly via rust-toolchain.toml" attribution is corrected
   here. No drift was measured, but identity is unpinned: treat
   cross-experiment absolute comparisons as drift-prone (each experiment's
   internal A/B is self-consistent) and pin a dated toolchain before any
   cross-experiment meta-analysis.
8. **Benchmark-methodology lessons** (for rstsr-agents conventions skill,
   final phase): judge kernels on reuse-variant denominators (T7);
   back-to-back interleaved A/B for ≤5 % cells (T2'/T4'/T6 lottery);
   re-run before believing portable-config cells (±5–8 %); criterion
   in-session `change:` lines are not baseline comparisons (T6).

---

## What the evidence supports (ranked)

1. **Docs-only, do first** (T7 EDIT-GUIDE Option 1, unchanged): reuse-idiom
   page + large-array MALLOC note in rstsr-book. 66–97 % wins for
   adopting users at zero risk; now also covers `full` (T5: env-tuned
   `rt::full` 4.59 → 0.62 ms, beating numpy).
2. **Small API additions** (T7 Option 2): `assign_with` sugar (d.1);
   optionally `sum_axes_into` (e). Proven mechanisms, wrapping only.
3. **Kernel patches already captured** (per-experiment `proposed.patch`,
   compose in any order — verified pairwise): T6 argmax, T2' reductions,
   T1' transpose, T4' elementwise, T3' vecdot/inner_dot. T5 adds none.
4. **Do NOT**: add `dispatch_simd` (a); "fix" `rt::zeros` (g.3); add
   madvise without the alignment change (f); share serial/rayon fold
   internals beyond geometry predicates (c).
5. **Correct the code map** (b) before it misleads the next session:
   symlinks are one file; `fill_promote` is not the creation path.

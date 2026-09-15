# 2026-09-15 — AtomicPtr hoist A-B (T1 unsafe-audit §4.1 remediation)

- **rstsr commits**: before = `aa24643` (worktree `/home/a/rstsr_pack/tmp/aptr-before`),
  after = branch `260915-unsafe-soundness-3` working tree (uncommitted edits).
- **Change under test**: all 40 write-through-`as_ptr()` sites in
  `rstsr-native-impl/src/cpu_rayon/{op_with_func,assignment,reduction,vecdot,
  transpose,adv_indexing,matmul_naive}.rs` and
  `rstsr-sci-traits/src/distance/native_impl.rs` replaced by
  `AtomicPtr::new(x.as_mut_ptr())` hoisted before the parallel region +
  `load(Ordering::Relaxed)` at the old derivation point. Comment-only
  otherwise; task partitioning, addressing arithmetic, and visit order
  untouched. (Stacked-Borrows UB remediation; see
  `2026-09-14-soundness-check/T1-unsafe-audit/README.md` §4.1.)

## Environment

- AMD Ryzen 9 9950X3D, 16 threads visible to the pool (`nproc` = 16), AVX-512.
- rustup default stable (1.97.1 era); `cargo build --release`, default profile.
- Device pool pinned explicitly: `DeviceFaer::new(16)` (no `RAYON_NUM_THREADS`).
- Harness: dependency-free (`src/main.rs` shared verbatim by both crates;
  3 warmup + 30 timed runs per case, median reported; CSV to stdout).
  Criterion deliberately not used (single-crate vendoring overhead not worth
  it for a one-shot A-B).

## Cases

All through the public high-level API on `DeviceFaer` (whose rayon dispatch is
`feature_rayon/auto_impl` → the edited kernels):

| case | kernel exercised |
|---|---|
| add_contig_{2^20,2^24} | op_with_func `op_mutc_refa_refb` (fresh output) |
| add_owned_reuse_{2^20,2^24} | op_with_func `op_muta_refb` (reuse output; T7 reuse-denominator) |
| add_blocked2d_3layout_2048sq | blocked_2d_3layouts (fresh) |
| add_blocked2d_2layout_inplace_2048sq | blocked_2d_2layouts (reuse) |
| add_tallskinny_70x10000 | generic outer-parallel path |
| fill_2p24 | assignment `fill_promote` (warm buffer) |
| assign_perm3d | assignment generic branch (3-D permuted src) |
| transpose_assign_4096sq | transpose orderchange_out_r2c family |
| sum_contig_2p24 / sum_strided_axis0_2048sq | reduction reduce_axes |
| vecdot_2p22 | vecdot_naive_cpu_rayon |
| index_select_4096_of_8192x512 | adv_indexing index_select |
| matmul_naive_i64_256sq | matmul_naive gemm_ix2 (faer-device i64 fallback) |
| cdist_euclidean_512x3 / cdist_weighted_512x3 | sci-traits cdist_rayon / cdist_weighted_rayon |

## Method

6 interleaved rounds of (before-run, after-run) to decorrelate frequency/thermal
drift (T3' ±5–8% transient lesson). Raw output: `results-raw.txt`,
`results-raw2.txt`. Per-case medians over the 6 rounds:

## Results

| case | before (ns) | after (ns) | ratio | note |
|---|---:|---:|---:|---|
| add_contig_1048576 | 146 378 | 148 757 | 1.016 | ~115 µs case, ±10% round transients |
| add_contig_16777216 | 15 646 082 | 15 680 918 | 1.002 | fresh 128 MiB output; alloc lottery (T7) |
| add_owned_reuse_1048576 | 115 732 | 118 687 | 1.026 | round spread 0.99–1.16 |
| add_owned_reuse_16777216 | 5 695 978 | 5 671 293 | 0.996 | reuse denominator, tight |
| add_blocked2d_3layout_2048sq | 3 663 442 | 3 617 451 | 0.987 | |
| add_blocked2d_2layout_inplace_2048sq | 698 011 | 665 815 | 0.954 | consistent win |
| add_tallskinny_70x10000 | 218 428 | 227 779 | 1.043 | ±20% round transients both sides |
| fill_2p24 | 5 439 938 | 5 437 884 | 1.000 | |
| assign_perm3d | 6 677 648 | 6 589 462 | 0.987 | |
| transpose_assign_4096sq | 7 255 382 | 7 226 532 | 0.996 | |
| sum_contig_2p24 | 926 550 | 899 828 | 0.971 | |
| sum_strided_axis0_2048sq | 102 146 | 104 392 | 1.022 | |
| vecdot_2p22 | 883 658 | 889 379 | 1.006 | |
| index_select_4096_of_8192x512 | 2 322 920 | 2 323 552 | 1.000 | |
| matmul_naive_i64_256sq | 906 799 | 890 509 | 0.982 | consistent small win |
| cdist_euclidean_512x3 | 63 901 | 49 021 | 0.767 | never slower in any round |
| cdist_weighted_euclidean_512x3 | 75 131 | 56 951 | 0.758 | never slower in any round |

**Geometric mean ratio (after/before): 0.968.**

## Conclusion

- **No regression on any kernel.** The per-task-granularity cases (fill,
  index_select, transpose-assign, fresh/reuse add, sum, vecdot) sit at
  0.97–1.03, i.e. indistinguishable from noise — consistent with the theory
  that the swap is load-for-load at identical granularity.
- **Small consistent wins** where the old code derived the pointer inside an
  element loop from a capture the compiler could not register-allocate:
  in-place blocked-2D (−5%), naive matmul (−2%).
- **cdist family −24%**: the old inner loop re-derived `dists.as_ptr()` per
  element through two dependent loads; the per-`j_batch`-task hoist leaves a
  register-resident base pointer, removing that load from the k=3 element
  loop. The largest measured effect, and it aligns with audit §5's "per-task
  materialization" note.
- Short (~100 µs) and fresh-large-alloc cases show the documented ±10–20%
  run-to-run transients (T3'/T7); direction is not monotone across rounds, so
  they carry no signal beyond "no regression".
- **Verdict**: the §4.1 remediation is efficiency-neutral-or-better on every
  affected kernel; keep it. Correctness: rstsr-core rayon suite green
  (116+302+2+184), sci-traits green, workspace `cargo check --all-targets`
  green.

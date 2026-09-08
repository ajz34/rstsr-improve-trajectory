---
name: rstsr-elementwise-t4
description: T4' phase-1 findings — rstsr contig elementwise kernel is ALREADY AVX-512-vectorized (0.83 ins/elem; T0's 12.9 was an alloc/fault artifact); strided branch is 161.6 ins/elem and a 64x64 blocked kernel gives 3.4x; dispatch_simd not needed for elementwise.
metadata:
  type: project
---

T4' elementwise (2026-09-09, dir `2026-09-09-elementwise`, rstsr
386948be). Phase 1 baseline/investigation + phase 2 IMPLEMENTED (patch
captured in proposed.patch, +346 lines; rstsr tree restored clean; awaiting
G2):

- **Contig kernel already vectorized**: reuse-variant add 2048² = 0.83
  ins/elem, IPC 0.27, 43.7 GB/s native; ties ndarray zip_prealloc (2.10 vs
  2.13 ms). T0's "12.9 ins/elem no autovectorization" was an artifact of
  profiling the ALLOCATING variant (fault + allocator instructions dominate;
  8358 faults/iter). Cross-crate generic closure chain inlines fine — do not
  restructure the contig elementwise path.
- **Phase-2 win (D3 PASS)**: blocked [64,64] 2-D tile path in
  cpu_{serial,rayon}/op_with_func.rs for op_mutc_refa_refb + op_muta_refb.
  Strided reuse-B 2048²: 23.1→7.5 ms serial (3.1×, both configs), faer16
  2.2×; odd 1000×777 8.7×; small 64×64 7-16×; c+=bᵀ 2.8×; idiomatic A 2.3×;
  A+MALLOC-tunables 4.0×. ins/elem 161.6→19.0, GB/s 4.2→13.1. Gates pass;
  >3% cells dissected as build-layout lottery (clean tree itself spans
  124-157 µs on portable contig-odd B across rebuilds, A anti-correlated,
  A+B sum invariant — re-run before believing any single ±5% wobble).
- **dispatch_simd verdict for elementwise: not needed** — contig already
  auto-vectorizes; strided is a gather SIMD can't fix. Plain blocked
  indexing got 3×.
- **Guards that matter in the tile path**: all offsets isize, usize cast
  only after full sum (flip views have negative strides); stride-0 guard on
  ALL layouts; visit-order note for stateful user closures (bit-identical
  for stateless ops).
- Broadcast add reuse is fast already (contig branch, row L1-hot). f32 has
  no large-class rider (16 MiB < 32 MiB glibc cap). MALLOC tunables + kernel
  fixes compose (strided A: 28.4→7.06 ms = 4.0×).
- rstsr API quirks: asarray/zeros give IxD; broadcast_to([m,n]) array
  literal forces Ix2 (use vec![m,n] to stay IxD); asarray can't take [m,n]
  for non-square bt fixtures — shape the storage tensor [n,m] so .t() fits.

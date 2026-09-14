---
name: rstsr-elementwise-t4
description: T4' elementwise patch — blocked 64x64 2-D strided kernel (contig was ALREADY vectorized, 0.83 ins/elem); MERGED to restgroup/master 2026-09-14 as c08e44a (PR #101, squash) after owner review; tall-skinny rayon parallel-degree caveat documented upstream.
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

Integration status (2026-09-14, patch-2 cycle): applied clean on
`../rstsr` branch `260914-elementwise` (base e835173 = post-PR#100 master;
+346, then local rustfmt reflow of added lines → capture in
proposed-v2-post-fmt.patch; zero content change). Gates: lib 110 + entry
302 both configs, clippy -p rstsr-native-impl clean both, correctness 94/94
both. G1 caller scan: only out-of-family callers are reduction order-fixup
sites (dead under default order; disjoint-buffer copy — tiled order
harmless). Paired 3×refA/cand benches portable+native: NO stable
regression (54 cells ×2 configs), wins all reproduce (strided B serial
0.32×, faer16 0.46×, odd 0.11×, small 0.06×). Evidence
`2026-09-09-elementwise/results/integration260914/`. Patch is
working-tree-only in ../rstsr — NOT committed (main-repo no-auto-commit;
owner reviews first). Local workspace `fmt --check` noise on untouched
files is the known local-vs-CI rustfmt comment-wrap divergence — never
"fix" it in patches.

Owner review (2026-09-14, same day as integration; report
`2026-09-09-elementwise/review-260914.md`, test capture
`results/review260914/review-tests.patch`): verdict CORRECT. Soundness
arguments verified (broadcast guard exact via `d>1 && s==0`; isize offset
formula identical to IterLayoutColMajor; tiled coverage write-once; bounds
contract matches file norm). Adversarial kernel matrix (negative/interleaved
strides, offsets, TILE-boundary shapes; serial+rayon; 106 traced blocked-path
executions) + e2e all bit-exact vs layout-iterator reference. NEW finding:
rayon parallelism is over slow-axis tile bands only — tall-skinny 70×10000
measured ~1.5× SLOWER on 8 threads (2 bands); documented in
`blocked_2d_applicable_rayon` doc, fix sketch (flatten to one par_iter over
both tile axes) recorded, not implemented. Nits fixed: removed unneeded
`too_many_arguments` on the 2-layout rayon fn. Lesson: rstsr `.gitignore`
has `tmp*` — never prefix review/test files with `tmp_` if git must see
them (use e.g. `review_blocked2d_e2e.rs`). Review tests are applied-on-top /
reverted via the captured patch; round-trip verified byte-identical.

MERGED (2026-09-14): owner authorized commit+PR+merge; committed as 96ecd63
on 260914-elementwise, pushed to ajz34 fork, PR #101 to restgroup/rstsr —
all 12 CI checks pass first try (rustfmt/clippy/unittests incl. col-major +
pthread + no-std + doctests), squash-merged as c08e44a. Local rstsr branch
260914-elementwise is now redundant (pre-squash SHA). Open follow-ups:
tall-skinny rayon parallel degree (fix sketch in review-260914.md), scalar-
operand kernels (`*_numb_*`) still on generic path for strided 2-D.

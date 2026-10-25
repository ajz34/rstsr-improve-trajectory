# Elementwise kernel wave — status report (2026-10-24)

## What landed (rstsr repo, branch `261023/cuda-review-fixes`, uncommitted)

### Correctness
- All 5 `test_ops_wave` tests + 51 total `rstsr-cuda` tests green.
- Fixed the two remaining wave failures (mismatched operand shapes in the
  layout-matrix test; NaN-vs-NaN `assert_eq` on round/floor/ceil/trunc/square).
- **Broadcast-flat soundness fix**: `plan_operand` used to return `Flat` for a
  single stride-1 level regardless of extent (e.g. a `[2048,1]` operand
  broadcast into `[2048,2048]` planned flat -> OOB reads). Flat now requires
  the run to span exactly `n` (`sq[0].0 == n as u64`); broadcast operands keep
  a divmod level whose top quotient is discarded (the walk implements the
  broadcast). Unit tests added for both tiers.

### Performance (three structural changes)
1. **Narrow (u32) plan tier** (`ParamsN` / `_pn` entries): 32-bit magic
   division (`__umulhi`), i32 offset accumulation (wrapping, so negative
   strides are exact), selected when `n < 2^31` and every operand's stride
   magnitude and accumulated offset fit `i32` (checked host-side in
   `narrow_pack`, incl. stride-magnitude via wrapping-neg for negative
   strides). Marshals as a 68-byte `repr(C)` plan (4-byte aligned, no pad —
   the initial 72-byte padded host struct misaligned every subsequent kernel
   parameter and produced wrong indexing; caught by `test_kernels`).
   - Exhaustive-ish magic-div test for all d ≤ 2^14 + boundary numerators;
     narrow-pack eligibility tests (negative stride, oversized stride/divisor,
     i32 overflow rejection).
2. **Grid-stride ILP4 on all flat entries** (`rstsr_*_f`): each thread handles
   4 elements strided by the grid width (`g + I*w`), fully coalesced per
   instruction. First attempt (4 *consecutive* elements per thread) regressed
   3x — uncoalesced per-thread stride-4 access; strided-grid pattern is the
   correct shape. `FlatLaunch::elems_per_thread(4)` divides the grid.
3. **Lean both-scalar clip** (`rstsr_clip_ss_f/pn`): out-of-place
   `min(max(in, lo), hi)` without bound plans/kind dispatch. The dominant win
   was fixing expression *nesting*: the fused min stage textually embedded the
   max stage 4x, and with `isnan` select control flow the compiler cannot CSE
   it — dedup via the sequential `val` local cut clip 2.35 -> ~2.0 ms; a
   host-checked non-NaN-bounds fast path (`scalar_bytes_nan` inspecting the
   marshaled bytes; bounds cannot inject NaN, so only the x-guard is needed)
   plus single-instruction PTX `max.NaN.f32` helpers for f32 brought clip to
   1.03 ms (0.93x cupy). `max.NaN.f64` is NVRTC-accepted but the driver JIT
   rejects it (CUDA_ERROR_INVALID_PTX) on this driver — f64 keeps the
   2-select fallback (f64 max/min NaN handling documented in kernel.rs).
   Note `rstsr_clip2s_launch` (in-place) became dead code after this path
   landed; `clip2s_generic` likewise — cleanup pending.

### Bench harness (Workstream E)
- `benches/bench_elementwise.rs`: 19 cells (fill/add/add_t/add_t_self/
  add_swap3d/add_bcast_r/add_bcast_c/add_outer/add_slice/add_step/add_flip/
  where_tt/where_ts/clip/rem/sin/pow/copy_T/to_contig), median-of-R wall
  timing, sustained warm-up, `RSTSR_BENCH_ITER`/`RSTSR_BENCH_REPEAT` knobs.
- `bench_cupy.py` / `bench_torch.py` in this directory: mirror cells,
  flush-per-line, explicit arena free before the 2-D cells (the ~4 GiB
  3-D arena previously made later cells silently OOM -> truncated CSVs that
  misaligned column-paste comparisons).

### Current 3-way table (median-of-5 x 200 iters; same-day runs)

| cell | rstsr ms | cupy ms | torch ms | r/cupy | r/torch |
|---|---|---|---|---|---|
| fill | 0.329 | 0.341 | 0.147 | 0.97 | **2.25** |
| add | 0.436 | 0.458 | 0.419 | 0.95 | 1.04 |
| add_t | 1.553 | 1.513 | 1.530 | 1.03 | 1.01 |
| add_t_self | 1.574 | 1.505 | 1.558 | 1.05 | 1.01 |
| add_swap3d | 15.198 | 14.685 | 14.588 | 1.03 | 1.04 |
| add_bcast_r | 14.237 | 13.635 | 12.541 | 1.04 | 1.14 |
| add_bcast_c | 12.488 | 12.286 | 11.895 | 1.02 | 1.05 |
| add_outer | 0.467 | 0.381 | 0.383 | **1.23** | **1.22** |
| add_slice | 0.242 | 0.249 | 0.234 | 0.97 | 1.03 |
| add_step | 0.326 | 0.282 | 0.274 | 1.16 | 1.19 |
| add_flip | 0.238 | 0.228 | 0.217 | 1.05 | 1.10 |
| where_tt | 0.434 | 0.491 | 0.459 | 0.88 | 0.95 |
| where_ts | 0.104 | 0.401 | 0.425 | 0.26 | 0.24 |
| clip | 1.094 | 1.116 | 1.089 | 0.98 | 1.00 |
| rem | 3.562 | 3.333 | 2.807 | 1.07 | **1.27** |
| sin | 6.623 | 6.812 | 6.415 | 0.97 | 1.03 |
| pow | 31.440 | 31.476 | 30.498 | 1.00 | 1.03 |
| copy_T | 1.286 | 1.300 | 1.475 | 0.99 | 0.87 |
| to_contig | 1.529 | 1.470 | 1.490 | 1.04 | 1.03 |

17/19 cells inside the 1.2x soft gate. Remaining flags:

- **fill vs torch (2.25x)**: fill kernel itself is NOT the bottleneck —
  `fill_v` (16-B pair stores, ILP4) is selected and runs 0.33 ms; the
  vector-width experiments (32-B quad stores) made it *worse* (0.91 ms).
  In-place fill into an existing buffer times identically (0.329), ruling
  out allocation. `torch.full` reaches 0.147 ms. Suspect: torch's kernel
  uses wider per-thread vectors + its runtime is warm when measured (see
  clock caveat below). Needs a re-measure under locked clocks before more
  kernel surgery.
- **rem (1.27x)**, **add_outer (1.22x)**, **add_step (1.19x)**: marginal;
  likely same clock caveat.

### Measurement caveat (important)
The sudo clock lock (`nvidia-smi -pm 1 -lgc 210,210`) applied on 10-21 did
not survive the power failure; the GPU idles at P5/210 MHz and the short
benchmark kernels never raise it (observed 3-4% util, 210 MHz *during* runs).
This is why today's numbers move non-monotonically with kernel changes that
are provably fewer-instruction (e.g. the ILP4 consecutive-vs-strided
regression). All three harnesses now include a sustained warm-up, but a
proper comparison needs the lock re-applied:

```
sudo nvidia-smi -pm 1
sudo nvidia-smi -lgc 210,210
```

(or any fixed frequency pair; the absolute numbers shift, the ratios are
what matter, and they are only stable when both libraries run at the same
locked clock).

## Boosted-clock re-measurement (2026-10-24, Option D)

The `-lgc` clock lock is deprecated on this driver (595.91.07) and cannot
be applied; Option D (software warm-up) works: a 5-second sin/pow spin
before timing spins the GPU from P5/210 MHz to ~2780 MHz @ 55-80 W
(observed via nvidia-smi during runs). All three harnesses now carry that
warm-up. No revert of GPU state is needed (persistence mode is the only
setting that stuck; it is the recommended state anyway).

### 3-way table at boost clocks (median-of-5 x 200)

| cell | rstsr ms | cupy ms | torch ms | r/c | r/t |
|---|---|---|---|---|---|
| fill | 0.0931 | 0.0927 | 0.0139 | 1.00 | 6.71* |
| add | 0.3022 | 0.3016 | 0.2909 | 1.00 | 1.04 |
| add_t | 0.3696 | 0.3268 | 0.3225 | 1.13 | 1.15 |
| add_t_self | 0.2967 | 0.2676 | 0.2769 | 1.11 | 1.07 |
| add_swap3d | 9.535 | 9.530 | 9.536 | 1.00 | 1.00 |
| add_bcast_r | 6.535 | 6.481 | 6.495 | 1.01 | 1.01 |
| add_bcast_c | 6.553 | 6.482 | 6.506 | 1.01 | 1.01 |
| add_outer | 0.0932 | 0.0933 | 0.0933 | 1.00 | 1.00 |
| add_slice | 0.1333 | 0.1156 | 0.1167 | 1.15 | 1.14 |
| add_step | 0.1714 | 0.1787 | 0.1666 | 0.96 | 1.03 |
| add_flip | 0.1224 | 0.1166 | 0.1174 | 1.05 | 1.04 |
| where_tt | 0.2119 | 0.3137 | 0.3167 | 0.68 | 0.67 |
| where_ts | 0.0133 | 0.2150 | 0.2137 | 0.06 | 0.06 |
| clip | 0.1996 | 0.2097 | 0.2125 | 0.95 | 0.94 |
| rem | 0.2990 | 0.2994 | 0.3041 | 1.00 | 0.98 |
| sin | 0.4792 | 0.4881 | 0.4652 | 0.98 | 1.03 |
| pow | 2.2381 | 2.2499 | 2.1741 | 0.99 | 1.03 |
| copy_T | 0.2117 | 0.2007 | 0.2374 | 1.05 | 0.89 |
| to_contig | 0.3264 | 0.2437 | 0.2374 | 1.34 | 1.37 <<<

At real clocks every throttle-era flag collapsed: rem 1.27 -> 1.00/0.98,
add_outer 1.22 -> 1.00, add_step 1.19 -> 0.96, bcast cells ~1.01. The
throttled-clock numbers were systematically inflating instruction-heavy
kernels (ours) more than memset/library paths (theirs).

### fill vs torch (the 6.71x flag) — measured artifact
- Direct kernel probe at boost clocks: our `rt::zeros` (cudaMemsetAsync
  path) 0.0033 ms; our full kernel 0.0117 ms. torch's cell shows 0.0139 ms
  for the same 32 MiB — i.e. our fill KERNEL is at torch-kernel parity;
  the 6.7x came from the cell structure (our rt::full cell pays per-iter
  allocation churn inside the timed loop, torch's does not) plus L2
  residency (32 MiB fits Blackwell L2; repeated fills never reach DRAM).
- No kernel action needed; the honest fill comparison is zeros-vs-zeros
  (0.0033, memset) and kernel-vs-kernel (0.0117 vs 0.0139).

### Remaining gap: to_contig (1.34x)
Transposed-read gather is uncoalesced; cupy/torch presumably use a
shared-memory tiled transpose (cublas-geam style). Follow-up item, next
wave: tile the 2-D transpose case in smem inside `rstsr_copy_p/_pn`.

### Next steps
1. ~~Re-lock clocks~~ (deprecated on this driver; Option D warm-up adopted).
2. smem-tiled transpose for to_contig/copy_T (to_contig 1.34x -> ~1.0x).
3. B8/B9 polish: f16/bf16 coverage check, promotion spot-tests.
4. Owner note recorded: rstsr_c32/c64 refactor remains future work.

## Next steps
1. Re-lock clocks (needs sudo), re-run the 3-way table, re-evaluate the
   remaining >1.2x cells.
2. Dead-code cleanup: remove `clip2s_launch`/`clip2s_generic`/
   `CLIP2S_ENTRIES` (superseded by `clip2s_out_launch`), unused imports in
   `operators.rs`, `concat_expr`.
3. `launch_trace` feature: keep (documented eprintln of entry selection;
   zero-cost when off).
4. B8/B9 polish: f16/bf16 coverage check, promotion spot-tests.
5. Owner note recorded: rstsr_c32/c64 refactor remains future work.

## Tiled 2-D transpose (2026-10-24, later session)

Implemented the smem-tiled transpose copy (`rstsr_copy_transpose`): 32x32
tile with +1 padded row (bank-conflict-free), coalesced reads AND writes on
both sides; algorithm per the CUDA Samples `transpose.cu` diagonal-tile
kernel — the same tiling PyTorch's TensorIterator copy and cublas geam use
for 2-D transposes (attribution in the source comment).

Detection: `transpose2d_plan` recognizes 2-D same-shape pairs where one
operand is row-major contiguous and the other strides (1, cols) (pure
swapaxes), any offsets, same-dtype plain copies, n >= 4096. The kernel
always reads the CONTIGUOUS side; the launcher swaps pointer roles per
orientation (to_contig-of-swapped vs assign-into-swapped-view). An initial
double-transpose indexing bug (128x100 regression test) caught the role
assignment; fixed and covered by `new_ops_transpose_copy_tiled` (square,
prime margins 63x67, wide 128x100, large 2048x3500; both directions; f64 +
i64).

Result: to_contig 0.326 -> 0.199 ms (0.82x cupy, 0.84x torch; was 1.34x),
copy_T 0.212 -> 0.211 ms (0.99x cupy — the scatter direction benefits less
since it was already gather-shaped). Probe evidence: the tile kernel hits
L2-resident 12 us for a hot 2048^2 transpose (vs cupy 278 us in the same
probe); the bench-cell numbers are DRAM-cold and allocator-inclusive.

## complex<T> template assessment (as requested, not implemented)

Current: 52 device functions (26 ops x 2 widths) in the 620-line
rstsr_complex.cuh + 15 Rust-side cname-branch sites.

- Option (a) vendor cupy's thrust-derived template complex: battle-tested,
  but ~8-10k lines across 20 headers to splice (NVRTC include-path problem
  identical to ours), Apache-2.0/BSD attribution chain, and their catrig
  inverse-trig family is the SAME msun source we already translated.
- Option (b) minimal template<tyepname T> over our existing algorithms:
  620 -> ~450 lines, 52 -> 26 functions; Rust cname branches collapse to
  C++ overload resolution; CNAME strings/cache keys unchanged; PTX
  equivalent (struct layout byte-identical). One-time ~2-3 h rewrite.

Recommendation: (b), bounded to the elementwise surface; recorded as
follow-up. No effectness difference either way.

## complex<T> template rewrite (2026-10-24, owner-approved)

Implemented option (b): `rstsr_complex<T>` template in the CUDA_PRELUDE
(constructor + `rstsr_make_complex`), `rstsr_c32`/`rstsr_c64` as `using`
aliases — CNAME strings and cache keys unchanged, struct layout
byte-identical. rstsr_complex.cuh rewritten: 620 -> 456 lines, 52 -> 26
functions + one `rstsr_r*` real-math overload pair per libdevice call;
width-class constants (eps, ln2, pio2, sqrt_min, t_max, e_base) moved into
a `rstsr_cw<T>` trait specializations struct.

Rust-side cname branching collapsed:
- operators.rs `transc_expr`: one template call name (`rstsr_csin(x)`, no
  64-suffix logic), one `rstsr_make_complex((T)1.0, (T)0.0)` literal form;
  conj sites likewise.
- exprs.rs: rem/pow/sign/abs/round_even sites lose their branch pairs;
  a local `t_real` helper supplies the component C type for scalar
  literals.
- kernels_ops.rs: cast expressions and the linspace body use
  `rstsr_make_complex` unconditionally.

Latent bug fixed en route: `rstsr_cpow`/`rstsr_csign` were referenced by
exprs.rs but never defined in the old header (any complex sign/pow call
would have failed NVRTC at runtime). Both now defined in the template
header and covered by new parity tests (complex sign z/|z|, complex pow
exp(y log x) vs CPU; tolerance 1e-10/1e-12 rel).

Also noted: `rt::round` does not bind for complex at the tensor tier
(CPU ExtReal-only), so no cround comparison exists; the device-side
round_even template path supports it if the tier ever binds it.

All 5 suites green (incl. 17-op complex transcendental parity at 1e-10,
now exercising the template instantiations for both widths); clippy 0,
fmt clean; sin/pow/rem bench cells unchanged (0.48/2.24/0.30 ms).

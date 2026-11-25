# Tiled-kernel check (implementation step B1)

The tiled SIMT GEMM kernel (`crates-device/rstsr-cuda/src/kernels/gemm.rs`) before it is moved into
the new `matmul/drivers/tiled.rs`. Two questions: is it right for every layout it accepts, and is it
worth keeping in the default driver chain (`DECISIONS.md` D13)?

| | |
|---|---|
| rstsr | `261021/cuda` @ `9c7d49a` |
| GPU | NVIDIA GeForce RTX 5070 Laptop 8 GiB, sm_120; clocks idle 442 MHz at start, 2737 MHz / 46 W after the timing runs (`results/gpu_state.txt`) |
| files | kept **only here**: `src/test_matmul_tiled_check.rs` (goes to `rstsr-cuda/tests/`) and `src/bench_matmul_check.rs` (goes to `rstsr-cuda/examples/`); the rstsr tree carries no file from this step |
| raw output | `results/tiled_check_f64.txt` (f32 and f64 cases), `results/bench_{cublas,tiled}_3.txt` (third of three runs; runs 1-3 agree within a few percent) |

## 1. Correctness

Method: `src/test_matmul_tiled_check.rs` (placed at `rstsr-cuda/tests/` for the run, then removed), `RSTSR_CUDA_MATMUL=tiled`, operands integer-valued in
`[-4, 4]` so every product and partial sum is exact in f32 and f64; the CUDA result is compared
**exactly** against `DeviceCpuSerial`. Debug build (host code only; the kernel is the same). Each
case prints `ok` or `REFUSED <error>`; a mismatch would fail the test. Result: **18 tests passed,
0 mismatches, in both dtypes.**

| layout class | case | result |
|---|---|---|
| contiguous | 64³, 65³, 1x1x1, 1x37x1, 17x33x5, 130x70x129 (tile edges) | ok |
| transposed operand | `a0.t()`, `b0.t()`, both | ok |
| stepped operand, outer axis | `[::2, :]` on a, `[::2, :]` on b, both | ok |
| negative stride, non-unit axis | `a.flip(0)`, `b.flip(0)` | ok |
| broadcast with a unit-stride axis | a rows `[0, 1]`, b cols `[1, 0]`, both | ok |
| batch | `[p]`, `[p,q]`, stepped batch `[::2, m, k]`, batch block not leading (`[m,p,k]` transposed) | ok |
| batch broadcast | `[m,k] @ [p,k,n]`, `[p,m,k] @ [k,n]`, explicit `[1,m,k] @ [p,k,n]` | ok |
| column-major | batched, unbatched a / unbatched b, transposed a, transposed b | ok (6 cases) |
| strided output (`matmul_from`) | c stepped on n (beta != 0 and beta = 0), c flipped on m, c flipped on n | ok |

Refusals — raised as `UnImplemented` errors by name, never a wrong answer:

| layout class | case | where |
|---|---|---|
| no unit-stride axis in an operand | inner-axis step `[::2]` on the k or n axis; a fully broadcast `[1,1] -> [m,k]` operand (strides `[0, 0]`); **a unit stride of `-1`** (`transpose` then `flip`) | `gemm.rs:210` (`lane_axis`) |
| non-affine batch | two-axis broadcast batch `[p,1,m,k] @ [1,q,k,n]` (batch strides `[q*m*k, 0]`); transposed 4-D batch block | `gemm.rs:330` (`linear_step`) |

Findings beyond the two known refusals:

1. **A negative unit stride is refused.** `lane_axis` tests `s_k == 1`, not `|s_k| == 1`, so
   `a.transpose(1,0).flip(0)` (strides `[-1, m]`) — a layout the kernel's signed address arithmetic
   handles — is rejected. The cuBLAS argument check declines it too (`stride < 0 && extent > 1`).
   One-line fix in the tiled driver (`abs`); until then such a view needs the naive driver.
2. **Broadcast batches are often non-affine.** Any operand with two broadcast batch axes
   (`[p,1]` against `[1,q]`) is refused, although the matmul rule table accepts it and the CPU
   computes it. A one-axis broadcast batch is fine (step 0 is affine).
3. Not covered by this file: an m-contiguous (transposed) output `c`. rstsr has no mutable-view
   constructor for it in the test toolbox (`slice_mut`/`i_mut` only); cuBLAS declines `scn != 1`,
   and the tiled kernel writes `c` with signed strides, so it is expected to work but is unverified.

## 2. Timing against cuBLAS

`src/bench_matmul_check.rs` (placed at `rstsr-cuda/examples/` for the run, then removed), release build, wall time per call around a synchronized loop of
20 repetitions (after 2 warm-ups + sync), so launch overhead is included. Driver by
`RSTSR_CUDA_MATMUL`. Third run shown; see the stability note above.

| dtype | m x k x n | cuBLAS ms | cuBLAS GFLOP/s | tiled ms | tiled GFLOP/s | tiled / cuBLAS |
|---|---|---|---|---|---|---|
| f32 | 512 x 512 x 512 | 0.028 | 9676 | 0.062 | 4350 | 2.2x slower |
| f32 | 2048 x 2048 x 2048 | 1.083 | 15870 | 3.234 | 5312 | 3.0x slower |
| f32 | 65536 x 64 x 64 | 0.065 | 8279 | 0.110 | 4881 | 1.7x slower |
| f32 | 64 x 64 x 65536 | 0.056 | 9519 | 0.109 | 4885 | 1.9x slower |
| f32 | 65536 x 16 x 16 | 0.020 | 1714 | 0.027 | 1225 | 1.4x slower |
| f32 | 256 x 256 x 256, batch 128 | 0.466 | 9214 | 0.822 | 5221 | 1.8x slower |
| f32 | 32 x 32 x 32, batch 4096 | 0.132 | 2037 | 0.210 | 1277 | 1.6x slower |
| f32 | 17 x 33 x 5 | 0.0116 | — | 0.0053 | — | 2.2x faster (launch overhead: ~10 us vs ~5 us) |
| f64 | 512 x 512 x 512 | 0.887 | 303 | 0.782 | 344 | 1.13x faster |
| f64 | 2048 x 2048 x 2048 | 52.70 | 326 | 44.92 | 383 | 1.17x faster |
| f64 | 65536 x 64 x 64 | 1.704 | 315 | 1.453 | 370 | 1.17x faster |
| f64 | 256 x 256 x 256, batch 128 | 12.59 | 341 | 11.12 | 386 | 1.13x faster |
| f64 | 17 x 33 x 5 | 0.0124 | — | 0.0378 | — | 3x slower |

(GFLOP/s = `2 * batch * m * k * n / time`; the example printed this column 1000x too large in the
first runs — fixed before runs 1-3, which are the numbers here.)

Reading:

- **f32**: tiled reaches 4.4-5.3 TFLOP/s on large shapes (about a quarter of this GPU's non-tensor
  FP32 rate) and is **1.4-3.0x behind cuBLAS** everywhere except tiny shapes, where cuBLAS's ~10 us
  call overhead dominates. It is a credible fallback, not a competitor.
- **f64**: both drivers sit at 0.30-0.39 TFLOP/s, i.e. at the GeForce FP64 rate (1/64 of FP32), and
  **tiled is 13-17% faster** than typed `cublasDgemm` on this GPU. The f64 chain order
  Lt > Ex > cuBLAS > tiled is therefore not a performance statement on this card; it stays because
  the accelerated drivers carry the modes (emulated fixed-point FP64, Q3) and because a datacenter
  part reverses the ratio. Worth re-measuring once the Ex/Lt drivers exist.

## 2b. Experiment: lifting the lane refusal (kernel patched temporarily, then restored)

The lane choice only decides which tile axis the thread id enumerates, so the refusal should not be
a correctness need. Checked by patching `lane_axis` twice and rerunning the same test file
(`results/tiled_check_absstride.txt`, `results/tiled_check_nolane.txt`); the kernel was restored
afterwards and the rstsr tree carries no change.

| patch | formerly refused cases | result |
|---|---|---|
| `s == 1` -> `s.abs() == 1` in the two unit-stride tests | `transpose` + `flip` operand (unit stride `-1`) | **ok**, exact, f32 and f64; everything else unchanged |
| additionally, the final refusal -> `Ok(LaneAxis::K)` (arbitrary lane) | inner-axis step on a and b, fully broadcast `[1,1]` operand | **all ok**, exact; only the two-axis broadcast batch stays refused (non-affine batch) |

So every "no unit-stride axis" refusal and the negative-unit-stride case are address-correct in
this kernel with a two-line change; what is lost in the arbitrary-lane case is load coalescing on
the stepped operand, not correctness. A negative stride is *not* harder for this kernel: strides
enter as signed `rstsr_isz` and the layout offset already points at element 0 of the view. It is
harder only for the cuBLAS family (leading dimensions and `ld` must be positive there), which is a
reason for those drivers to decline, not for the tiled one.

## 3. Verdicts

- **Correctness**: no wrong answer in 70 accepted cases across f32/f64. The kernel is right for
  every layout it accepts, including negative strides on non-unit axes, zero strides with a
  unit-stride partner, non-leading and stepped batches, column-major and strided outputs.
- **The two refusals are restrictions, not correctness needs.** The kernel addresses with signed
  strides; `lane_axis` only picks the coalesced axis, and a non-affine batch only needs per-axis
  batch strides instead of one scalar step. Under D11/D13 the naive driver takes these layouts, so
  no fix is required for the new chain. Optional tiled follow-ups, in value order:
  (a) accept `|stride| == 1` in `lane_axis` (one line, removes finding 1);
  (b) fall back to an arbitrary lane when no axis is unit-stride (perf-only loss);
  (c) per-axis batch strides (removes the non-affine refusal).
  (a) and (b) are verified correct by the experiment in section 2b; (c) is untested.
  **Owner's call (D30): noted and deferred — none of (a)-(c) is done in this task.**
- **D13 confirmed**: tiled stays in the default chain for f32/f64, after the accelerated drivers and
  before naive. On this GPU it would even be the fastest f64 driver, but it has no compute modes.

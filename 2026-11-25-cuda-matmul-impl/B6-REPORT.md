# B6 — the type-triple layout matrix, `bench_matmul`, crate docs

In the rstsr working tree of `261021/cuda` @ `85ed399` (B5), **not committed** (awaits the
owner's go). The patch is `patches/b6-matrix-bench-docs.patch` (6 files; one new file,
`benches/bench_matmul.rs`).

| | |
|---|---|
| rstsr | `261021/cuda` @ `3508d3a` |
| GPU | NVIDIA GeForce RTX 5070 Laptop 8 GiB, sm_120, CUDA 13.2 toolkit, cuBLAS 13.4 runtime, cudarc 0.19.10 |
| bindings | two builds: `nvcc`-detected (13.2) and `CUDARC_CUDA_VERSION=12060` (the CI pin) |
| raw output | `results/b6_full_suite.txt`, `results/b6_full_suite_12060.txt`, `results/b6_clippy.txt`, `results/b6_doc.txt`, `results/b6_bench_after.txt`, `results/b6_bench_before_{cublas,tiled}.txt` |

## What was done

| decision | result |
|---|---|
| D26 (tests) | `test_matmul_modes::triple_layout_matrix`: the eight accelerated triples × six argument shapes (plain, strided batch, transposed right operand broadcast against the batch, transposed left operand, broadcast batch of `a`, stepped `a`) × two default orders × each serving cuBLAS driver alone, the naive kernel (the on-device oracle) and the chain — 464 cells — against the CPU `f64` product of the rounded operands, tolerance per triple. A new `test_matmul_naive::bf16_triples_on_naive`. |
| D27 (bench) | `benches/bench_matmul.rs` (`cargo bench --bench bench_matmul`): triple × shape × driver, median of 5 timed runs. The **before** numbers come from a worktree of `9c7d49a` running `src/bench_matmul_baseline.rs` (kept here, never in the rstsr tree). |
| docs | `matmul` module docs: the type-triple table, the mode resolution rule, mixed operands; crate docs: coverage row, a divergence bullet for mixed-dtype `matmul`, the bench and the matrix in the test section. |

## The matrix found a bug

`bf16` on the **naive kernel did not compile**: the kernel's epilogue hard-coded the `f16`
conversion helpers (`__half2float`, `__float2half`) for both half dtypes. Nothing forced a `bf16`
triple onto naive before (B5 ran it on Lt/Ex and on the chain), so it only showed here.
Fixed in `drivers/naive.rs` (the helpers of `kernels::exprs::{load, store}`, per dtype), with the
regression test above.

Two things that are **not** bugs, learned the hard way while writing the matrix:

- A `ColMajor` device puts the **batch axis last**: `[m, k, p]`, not `[p, m, k]`. Under ColMajor
  a row-major-shaped batched tensor is a different matrix (or an error), and the operands must be
  built in the device's order for the CPU reference to mean the same thing.
- A fresh `DeviceCuda` per cell is right for timing but wrong for sharing: tensors carry their
  device value, so operands made on another device value are fine to *read* but the reference
  must be built from the same bytes in the same order.

## Verification

**230 tests pass** under both binding levels (B5's 228 plus 2). Clippy clean in `rstsr-cuda`;
`cargo doc` clean.

## Numbers

In `README.md` ("Numbers"). The short version:

- **Before/after within ±4 %** on every cell the baseline could run (f32, f64, four shapes):
  the chain picks the same cuBLAS kernel. The small f32 shapes lose 2-4 % (launch-bound; one
  run, median of 5 — not established as real).
- **`f64`: the tiled kernel is 14-16 % faster than every cuBLAS driver here** (0.37 vs
  0.32 TF/s), and naive beats cuBLAS on 512³ (0.75 vs 0.89 ms): at a GeForce FP64 rate of 1/64
  these shapes are memory-bound. The chain order (`Lt > Ex > cuBLAS > tiled`) never reaches
  tiled for f64.
- **`c64`: cuBLASLt is 19 % faster than `cublasGemmEx`/classic at 2048³** (163 vs 203 ms).
- **Half uses the tensor cores**: `f16 -> f16` 43 TF/s at 2048³ (f32: 13), `f16 -> f32` 42.

## Open points for the owner

1. **Criterion** (D27) was not used: `rstsr-cuda`'s two benches are `harness = false` with a
   hand-timed median-of-R, and `bench_matmul` follows them (no new dev-dependency). Say if you
   want it on criterion after all.
2. **The chain order for `f64`.** With tiled 14-16 % ahead of cuBLAS on this GPU, either the
   table's f64 row puts `Tiled` before the cuBLAS drivers on such devices, or the owner forces
   it (`MatMulDrivers::TILED`). An FP64-capable GPU (A100/H100) would answer the other way.
3. **The naive kernel on small `f64`/`f32` shapes**: naive is within 1.2x of cuBLAS on f64 512³.
   Launch-bound shapes have no driver-specific advantage; nothing to do, noted.
4. The bench runs one process per triple/shape/driver; the cuBLASLt heuristic of the first call
   is excluded by the warmup. A "first call" column exists in `src/bench_matmul_driver_check.rs`.

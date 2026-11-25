# 2026-11-25 — CUDA matmul implementation (grill)

Design of the matmul implementation of `rstsr-cuda`: several drivers
(cuBLASLt > cuBLASEx > cuBLAS > tiled > naive), output-keyed compute modes, a device-level
`matmul_with_mode_inner`, tensor-level `*_mode` entries, and the flags refactor that replaces the
op-behaviour environment variables.

| | |
|---|---|
| rstsr | `261021/cuda` @ `9c7d49a` (B6 `3508d3a`) |
| machine | RTX 5070 Laptop 8 GiB, sm_120, driver 595.99.02, CUDA 13.2 |
| status | grill closed 2026-11-25; B1-B5 done (B2 `c75aecd`, B3 `493a42f`, B4 `554f956`, B5 `85ed399` committed in rstsr); B6 next; B6 in `PLAN.md` |

- `GRILL-INIT-PROMPT.md` — the task and the facts gathered before round 1.
- `GRILL-R1..R3-QUESTIONS.md` — the three rounds (round 2/3 files start with the previous round's
  answers and the fact checks they triggered).
- `DECISIONS.md` — D1-D28.
- `PLAN.md` — batches B1-B6 (all done).
- `TILED-CHECK.md` — implementation step B1: correctness matrix, timing vs cuBLAS, verdicts.
- `B2-REPORT.md` — implementation step B2: `src/matmul/` skeleton, the move, the naive driver, the flags refactor (f32/f64/integers/complex on naive); verification, numbers, open points for the owner.
- `B3-REPORT.md` — implementation step B3: compute modes, the `cublasGemmEx` driver, math modes on the classic handle, FP64 fixed-point emulation (8x on this GPU), the `build.rs` binding-level cfgs; numbers, open points for the owner.
- `B4-REPORT.md` — implementation step B4: the cuBLASLt driver (per-ordinal handle and heuristic cache, per-stream workspace), descriptor construction, alignment; speed and first-call numbers; the fixed-point-through-Lt probe.
- `B5-REPORT.md` — implementation step B5: complex and half triples on the cuBLAS drivers, mode resolution per triple, the tensor-level `*_mode` entries; `ext_matmul` withdrawn; the syrk question.
- `B6-REPORT.md` — implementation step B6: the type-triple layout matrix (which found and fixed a `bf16` compile failure of the naive kernel), the `bench_matmul` bench, crate docs; before/after numbers.
- `src/` — the check test, the timing examples (driver, mode, per-driver shapes), the Lt fixed-point probe and the trace example (copied into `rstsr-cuda/` only for a run); `patches/` — the B2-B6 patches of the rstsr tree; `results/` — raw output.

The existing tiled suite at `9c7d49a` also passes (5 tests, debug build, 1.27 s).

Reference facts used: `../2026-11-24-cuda-gemm-functionality` (which GEMM configurations this GPU
accepts), CuPy `8f7a35d16` and PyTorch `ff4f371` (promotion and compute-type policy, see round 3).

## Numbers (B6, `bench_matmul`, this GPU, `Default` compute mode)

Median wall time of `rt::matmul` per call, sync once per timed run; a fresh device per cell.
Raw: `results/b6_bench_after.txt` (and `b6_bench_before_{cublas,tiled}.txt`, from a worktree of
`9c7d49a`). Before = the code at the grill (`RSTSR_CUDA_MATMUL` unset: cuBLAS, else the tiled
kernel); after = the default chain.

### Before / after (the triples the baseline could run)

| triple | shape | before: cuBLAS (ms) | after: chain (ms) | before TFLOP/s | after TFLOP/s | before/after time |
|---|---|---|---|---|---|---|
| f32 -> f32 | square 2048 | 1.2986 | 1.2778 | 13.23 | 13.45 | 1.02x |
| f32 -> f32 | square 2048 (tiled forced) | 3.3255 | 3.3260 | 5.17 | 5.17 | 1.00x |
| f32 -> f32 | square 512 | 0.0304 | 0.0318 | 8.82 | 8.43 | 0.96x |
| f32 -> f32 | square 512 (tiled forced) | 0.0636 | 0.0645 | 4.22 | 4.16 | 0.99x |
| f32 -> f32 | tall-skinny 65536x64x64 | 0.0697 | 0.0714 | 7.70 | 7.52 | 0.98x |
| f32 -> f32 | tall-skinny 65536x64x64 (tiled forced) | 0.1139 | 0.1181 | 4.71 | 4.55 | 0.96x |
| f32 -> f32 | batch 256 x (256,256,256) | 1.0690 | 1.0740 | 8.04 | 8.00 | 1.00x |
| f32 -> f32 | batch 256 x (256,256,256) (tiled forced) | 1.7286 | 1.7085 | 4.97 | 5.03 | 1.01x |
| f64 -> f64 | square 2048 | 53.9472 | 53.5825 | 0.32 | 0.32 | 1.01x |
| f64 -> f64 | square 2048 (tiled forced) | 46.0943 | 45.8342 | 0.37 | 0.37 | 1.01x |
| f64 -> f64 | square 512 | 0.9002 | 0.8850 | 0.30 | 0.30 | 1.02x |
| f64 -> f64 | square 512 (tiled forced) | 0.7941 | 0.7866 | 0.34 | 0.34 | 1.01x |
| f64 -> f64 | tall-skinny 65536x64x64 | 1.7297 | 1.7020 | 0.31 | 0.32 | 1.02x |
| f64 -> f64 | tall-skinny 65536x64x64 (tiled forced) | 1.4810 | 1.4580 | 0.36 | 0.37 | 1.02x |
| f64 -> f64 | batch 256 x (256,256,256) | 25.5756 | 25.5906 | 0.34 | 0.34 | 1.00x |
| f64 -> f64 | batch 256 x (256,256,256) (tiled forced) | 22.9472 | 22.6765 | 0.37 | 0.38 | 1.01x |


### After: every triple, per driver

TFLOP/s (a complex product counted as four real ones). `chain` is the default driver set.

| triple | shape | chain | cublasLt | cublasEx | cublas | tiled | naive |
|---|---|---|---|---|---|---|---|
| f32 -> f32 | square 2048 | 13.45 | 12.90 | 12.75 | 12.56 | 5.17 | — |
| f32 -> f32 | square 512 | 8.43 | 8.38 | 8.40 | 8.38 | 4.16 | 1.10 |
| f32 -> f32 | tall-skinny 65536x64x64 | 7.52 | 7.44 | 7.63 | 7.25 | 4.55 | — |
| f32 -> f32 | batch 256 x (256,256,256) | 8.00 | 8.05 | 8.00 | 8.01 | 5.03 | — |
| f64 -> f64 | square 2048 | 0.32 | 0.32 | 0.32 | 0.32 | 0.37 | — |
| f64 -> f64 | square 512 | 0.30 | 0.30 | 0.30 | 0.30 | 0.34 | 0.36 |
| f64 -> f64 | tall-skinny 65536x64x64 | 0.32 | 0.32 | 0.32 | 0.32 | 0.37 | — |
| f64 -> f64 | batch 256 x (256,256,256) | 0.34 | 0.34 | 0.34 | 0.34 | 0.38 | — |
| c64 -> c64 | square 2048 | 0.42 | 0.42 | 0.34 | 0.34 | — | — |
| c64 -> c64 | square 512 | 0.37 | 0.37 | 0.37 | 0.37 | — | 0.22 |
| c64 -> c64 | tall-skinny 65536x64x64 | 0.40 | 0.40 | 0.32 | 0.33 | — | — |
| c64 -> c64 | batch 256 x (256,256,256) | 0.39 | 0.40 | 0.33 | 0.33 | — | — |
| f16 -> f16 | square 2048 | 42.39 | 42.77 | 41.73 | — | — | — |
| f16 -> f16 | square 512 | 25.48 | 25.43 | 25.40 | — | — | 1.22 |
| f16 -> f16 | tall-skinny 65536x64x64 | 16.84 | 16.73 | 16.27 | — | — | — |
| f16 -> f16 | batch 256 x (256,256,256) | 28.82 | 28.75 | 28.61 | — | — | — |
| f16 -> f32 | square 2048 | 41.81 | 39.87 | 39.61 | — | — | — |
| f16 -> f32 | square 512 | 25.28 | 25.39 | 25.46 | — | — | 1.21 |
| f16 -> f32 | tall-skinny 65536x64x64 | 14.16 | 14.04 | 13.51 | — | — | — |
| f16 -> f32 | batch 256 x (256,256,256) | 20.64 | 20.62 | 20.57 | — | — | — |

What they say:

- **Before/after is within ±4 %** on every cell: the chain resolves to the same cuBLAS kernel
  as the baseline (cuBLASLt picks it on this GPU); the small f32 shapes lose 2-4 % (launch-bound;
  the Lt lock and cache lookup, or run-to-run noise — single run, median of 5).
- **`f64`: the tiled kernel is 14-16 % faster than every cuBLAS driver on this GPU** (0.37 vs
  0.32 TF/s), and the naive kernel is faster than cuBLAS on 512³ (0.75 vs 0.89 ms): a GeForce
  FP64 rate of 1/64 makes these shapes memory-bound. The chain order never reaches tiled for f64.
- **`c64`: cuBLASLt is 19 % faster than `cublasGemmEx`/classic at 2048³** (163 vs 203 ms).
- **Half runs on the tensor cores**: `f16 -> f16` 43 TF/s at 2048³, against 13 for `f32`.

## What is left (after B6, 2026-11-25)

All six batches are committed in rstsr `261021/cuda` (`c75aecd`, `493a42f`, `554f956`, `85ed399`,
`3508d3a`); nothing is pushed and no PR is open. The open points of the batch reports, collected:

**Decisions for the owner**

1. **`f64` chain order** (`B6-REPORT.md` 2): the tiled kernel is 14-16 % faster than every cuBLAS
   driver on this GPU; the chain never reaches it. Reorder the `f64` row, or force `TILED`.
2. **`Acc16F` on a `bf16` output is silently the default** (`B5-REPORT.md` 1): D9's rule; the
   alternative is an error.
3. **Criterion or `harness = false`** (`B6-REPORT.md` 1): the bench follows the crate's style,
   not D27.
4. **Workspace per stream, not per ordinal** (`B4-REPORT.md` 1), the heuristic cache dropped whole
   at 1024 entries (2), and the first-call heuristic cost (3).
5. **Driver-level refusals vs D9's silent default** (`B3-REPORT.md` 4) and the Lt emulation probe
   (`B4-REPORT.md` 4, suggesting to close `B3-REPORT.md` 6 as "Lt does not serve
   `EmulatedFixedPoint`").
6. **CI pin `12060` costs real function** on a machine with newer bindings (`B3-REPORT.md` 3).

**Small code items, not in any batch**

- *Done 2026-11-26 (rstsr `80e626d`):* `a.matmul_with_mode(&b, mode).out::<TC>()`
  replaces the four-type turbofish of `B5-REPORT.md` 4; the output is always an `IxD` tensor.
- Inner dot (`[k] @ [k]`) is refused by every driver; the naive kernel could serve it
  (`B2-REPORT.md` 4).
- Tiled `lane_axis` with a stride of `-1` (D30, deferred by the owner; verified in `TILED-CHECK.md`).
- `trace_matmul` prints less than the removed `RSTSR_CUDA_DEBUG_GEMM` (no pointer bases, no
  cuBLAS arguments) (`B2-REPORT.md` 2.8).
- The crate's pre-existing files are not rustfmt-clean; the new code is formatted on its own
  (`B2-REPORT.md` 9).

**Follow-ups recorded, out of this wave**

- syrk for `a @ a.t()`: after B6, as a measured pre-driver step (`B5-REPORT.md`).
- `tensordot` (D29) with cuTENSOR; FP8 GEMM; a hand-written tensor-core kernel (D10).

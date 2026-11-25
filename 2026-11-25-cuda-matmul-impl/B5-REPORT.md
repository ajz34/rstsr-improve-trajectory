# B5 — half and complex triples, the tensor-level `*_mode` entries

Committed in rstsr `261021/cuda` as `85ed399`, on top of `554f956` (B4).
The patch is `patches/b5-half-complex-tensor-api.patch` (11 files, about +1100/-210; one new file,
`matmul/tensor.rs`).

| | |
|---|---|
| rstsr | `261021/cuda` @ `85ed399` |
| GPU | NVIDIA GeForce RTX 5070 Laptop 8 GiB, sm_120, CUDA 13.2 toolkit, cuBLAS 13.4 runtime, cudarc 0.19.10 |
| bindings | two builds: `nvcc`-detected (13.2) and `CUDARC_CUDA_VERSION=12060` (the CI pin) |
| raw output | `results/b5_full_suite.txt`, `results/b5_full_suite_12060.txt`, `results/b5_modes_release.txt`, `results/b5_clippy.txt`, `results/b5_doc.txt` |

## Scope change by the owner (before the batch)

- **D14 (`ext_matmul`) withdrawn**: no `DeviceExtMatMulAPI` is implemented. Mixed operands need no
  entry of their own: `matmul_from` / `matmul_with_output` carry no `Mul` bound, and the chain
  already takes every triple (the naive kernel casts each operand to the output type, no copy).
  This is tested (`tensor_level_modes`: `f32 @ f64 -> f64` through `matmul_from` on the full chain).
- syrk for `a @ a.t()`: asked whether now or after B5 — answer at the end of this report.

## What was done

| decision | result |
|---|---|
| D1 (dtype scope) | Accelerated triples: `f32`, `f64`, `c32`, `c64` (Lt, `cublasGemmEx`, classic), `f16 -> f16`, `f16 -> f32`, `bf16 -> bf16`, `bf16 -> f32` (Lt, `cublasGemmEx`). Everything else stays naive. |
| D18 (table) | `CublasPlan` is now the **problem**: `F32`, `F64`, `C32`, `C64`, `F16`, `F16F32`, `BF16`, `BF16F32`, `None`, each with its mode. `plan::<TA, TB, TC>` resolves it; `gemm_types()` gives the three `cudaDataType_t`s (`CublasGemmTypes`), `compute()` the compute type. |
| D9 (inapplicable modes) | Resolved **in `plan`**, before any driver: `Fast*` is `f32`-input only (complex inputs and half inputs with an `f32` output keep the default); `Acc16F` is `f16` only (a `bf16` output keeps the default); `EmulatedFixedPoint` is real `f64` only (a `c64` output keeps the default). `Pedantic` is served by every accelerated triple and never downgraded. |
| D15 (tensor level) | `matmul/tensor.rs`: extension trait `TensorCudaMatMulAPI<R, T, D>` on `TensorAny<R, T, DeviceCuda, D>` — `a.matmul_with_mode::<TB, TC, _, _>(&b, mode)` (allocating), `a.matmul_with_output_mode(&b, &mut c, mode)`, `c.matmul_from_mode(&a, &b, alpha, beta, mode)` — and the fallible free function `rstsr_cuda::matmul::matmul_with_mode_f`. `mode: <TC as CudaMatMulMode>::Mode`. They read nothing from the flags. Exported from `prelude_dev` and `matmul`. |
| open point "bf16 family" | Closed as the existing code had it: `bf16` uses `ComputeType16F` (`Default` = accumulate in `f32`), which is one field for both half outputs. |
| shared code | `Uniform<T>` became `Buffers<A, B, C>` (`Problem::retyped`, `uniform` kept as the same-type case); `prepare` is generic over the three storage types with per-operand element sizes (half operands, `f32` output) and keeps the geom-named pointer order under `ColMajor`. The classic entries gained `cublasCgemm_v2`/`cublasZgemm_v2` (+ strided batched). |

A fact not in the decisions: **`CUBLAS_COMPUTE_16F` takes `f16` scales**, not `f32` (the descriptor
creation answers `INVALID_VALUE` otherwise). `Acc16F` therefore scales in `f16` on Lt and Ex;
every other half triple scales in `f32`. The scales of a half output are converted from `TC` by
`drivers::scale` (unit-tested).

## Verification

**Whole crate, `--features faer`, `--test-threads=1`**, both binding levels: **228 tests pass**
(B4's 221 plus 5 host tests and 3 hardware tests; the existing ones unchanged). Clippy clean in
`rstsr-cuda`; `cargo doc` clean; `--all-features` check clean.

| target | tests | B5 |
|---|---|---|
| `--lib` | 101 | `plan_by_type_triple`, `modes_the_triple_cannot_serve_resolve_to_default`, `plan_storage_and_compute`, `scales_of_the_output_type`; `table_by_type_triple` covers the four half rows and complex; `key_tells_the_storage_types_apart` (Lt); `retyped_enums_keep_their_values` extended to the half and complex data types and `16F` |
| `test_matmul_modes` | 8 | `modes_complex`, `modes_half`, `tensor_level_modes` (new) |

`test_matmul_modes`, each cuBLAS driver **alone** then the chain (so a driver that silently declined
would fail):

| triple | modes | Lt | Ex | classic | max rel err |
|---|---|---|---|---|---|
| `c32` | `Default`, `Pedantic` | ok | ok | ok | 2.0e-7 |
| `c64` | `Default`, `Pedantic` | ok | ok | ok | 3.6e-16 |
| `f16 -> f16` | `Default`, `Pedantic` | ok | ok | — | 3.6e-4 |
| `f16 -> f16` | `Acc16F` | ok | ok | — | 6.4e-3 |
| `f16 -> f32` | `Default`, `Pedantic` | ok | ok | — | 5.2e-7 |
| `bf16 -> bf16` | `Default`, `Pedantic` (and `Acc16F` = default) | ok | ok | — | 3.0e-3 |
| `bf16 -> f32` | `Default`, `Pedantic` | ok | ok | — | 2.8e-7 |

The reference is the CPU `f64` product of the **rounded** operands, so these numbers are the
output rounding (and the accumulation, for `Acc16F`). Tolerances: 1e-2 for a half output, 5e-3
for `Acc16F`, 1e-5 for an `f32` output of half operands, 1e-5 / 1e-13 for `c32` / `c64`.

`tensor_level_modes`: `matmul_with_mode_f`, `matmul_with_mode` and `matmul_with_output_mode` agree
bit for bit (`f16` operands into `f32`), `matmul_from_mode` applies `alpha`/`beta`; the mode of the
call is the one run (`Pedantic` and `FastTF32` give different bits under the `Default` flags);
`rt::matmul` and `%` give the `f16` output, within the `f16` rounding of the `f32` output.

Under the 12060 bindings the classic entries have no math mode, so the classic driver is left out
of the complex `Pedantic` runs (it refuses, as `modes_32f` checks).

## Open points for the owner

1. **The 16F family of `bf16`**: `ComputeType16F::Acc16F` on a `bf16` output is silently the
   default (D9). The alternative is an error, as `Pedantic` on an unsupported triple is.
2. **`Acc16F` with half scales**: `alpha`/`beta` of an `f16` output are converted `f16 -> f32 ->
   f16`, exact. But the epilogue is evaluated in `f16` by cuBLAS under this mode, where the naive
   kernel evaluates it in `f32`.
3. **The candidate table row for `Acc16F`** lists Ex, and Ex serves it here (cuBLAS 13.4). The
   cuBLAS documentation of `cublasGemmEx` does not name `16F` for every version; an older runtime
   would decline (a status), and the chain would fall to naive with `CUBLAS_EX` alone — slow but
   exact. Nothing to change unless it is seen.
4. **`tensor.rs` names** (*resolved 2026-11-26*: `a.matmul_with_mode(&b, mode).out::<TC>()`, a pending
   product, names one type): `matmul_with_mode_f` is the only free function; the GEMM forms under a
   mode are methods only. `a.matmul_with_mode::<TB, TC, DB, DC>` names four types in the
   turbofish where only `TC` is needed in practice (the rest infer).
5. **Complex + `Pedantic` on the classic entries**: `CUBLAS_PEDANTIC_MATH` is accepted by
   `cublasCgemm`/`Zgemm` here and gives the same bits as `Default`. Whether it has an effect on
   complex entries is cuBLAS's business, not observable from this test.

## syrk for `a @ a.t()`

**After B5, as its own step** (B5 is done; this is not part of it). Reasons:

- It is not a driver-chain item: the chain sees a `Problem` with three *buffers*, and `a @ a.t()`
  shares one buffer for `A` and `B`. Detecting it is a check on the tensor-level entries that
  B5 just put in place (`ptr_equality` of the two operand storages and transposed layouts).
- The gain is bounded: `syrk` halves the FLOPs, but writes **one triangle only**, so the full
  result needs a second pass (a symmetrize kernel, or `CUBLAS_FILL_MODE` + a copy), and the
  tensor's other triangle must be filled. For large `n` this could still be up to ~1.8x; for the
  sizes where matmul is launch-bound it is a loss.
- cuBLASLt has no syrk, so it is a classic-cuBLAS-only path, below Lt in the chain — it would be
  a rewrite of the problem (GEMM -> SYRK + fill), best done as a *pre-driver* step, after the
  chain is complete (B6), and measured first.

Suggested order: B6 (tests, bench, docs) -> a 20-line probe (`cublasDsyrk` + symmetrize vs
`cublasDgemm` on the `bench_matmul` shapes) -> decide.

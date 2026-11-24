# 2026-11-24 — what GEMM configurations does CUDA actually accept here?

**Question.** cuBLAS, cuBLASLt, TF32, "other lower-precision approximations to
FP32/FP64", mixed dtypes, "all dtypes" — which of these combinations does the
library on this machine really run, and how accurate are they?

**Scope.** Functionality only. Every configuration is executed exactly once on
small matrices and checked against an exact reference; nothing is timed. (A
throughput study of the same matrix is a separate exercise.)

**Answer in one paragraph.** The core matrix is boringly healthy: f16, bf16,
f32, f64, complex f32/f64 and int8 all work through both the classic and the
Lt API, including the mixed `f16/bf16 → f32` and `f8 → f32/f16` forms and
strided-batched GEMM. The interesting part is the "approximate FP32/FP64"
layer: TF32 works and is worth ~1e-4 relative error; the BF16x9 FP32 emulation
is accepted by the API but **silently falls back to native f32 on this GPU**
(sm_120 is not in its supported list); the fixed-point FP64 emulation does
engage, but only through the classic handle API with the mantissa-bits knob
explicitly set — the Lt heuristic never selects it here. On the exotic side,
FP8 e4m3 works (with per-tensor scaling), FP8 e5m2 does not, FP4 does not, and
`A f16 + B bf16` does not.

## Environment

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Laptop, 8 GiB, **sm_120** (12.0) |
| Driver | 595.99.02 |
| CUDA runtime / toolkit | 13.2 |
| cuBLAS / cuBLASLt | 13.4.0 (`130400`) |
| CPU / OS | Ryzen 9 9955HX / Ubuntu 26.04.1 |

## Method

[`gemm_demo.cu`](gemm_demo.cu) runs 38 GEMM configurations plus one batched
case. For each:

- `m = n = k = 512` (256 for the complex cases), column-major, `alpha=1`,
  `beta=0`, deterministic uniform inputs in `[-1, 1]` (int8: `[-8, 7]`).
- The reference product is computed on the CPU in `double` (or `int64` for
  int8), column-major, same operands.
- Two error columns:
  - **err/exact** — relative Frobenius error against the product of the
    *original* inputs. This is what a user sees end-to-end; for f16/bf16/fp8 it
    includes the input quantization.
  - **err/stored** — relative error against the product of the values *actually
    stored* in the input dtype. This isolates the algorithm's own error from
    the storage quantization (both columns coincide for f32/f64).
  - **max|d|** — largest absolute elementwise deviation for the stored-operand
    reference.
- A configuration counts as accepted when the cuBLAS call returns
  `CUBLAS_STATUS_SUCCESS`; the error column is what proves the requested
  compute mode was really used (e.g. TF32 shows 2.6e-4, native f32 2.1e-7).

The companion probes in [`probes/`](probes) answer the follow-up questions
(does emulation engage? which algorithms advertise emulation?) and their raw
output is in [`results/`](results).

## Results

```
API       case                               result        err/exact     err/stored max|d|
--------------------------------------------------------------------------------------------------------
classic   cublasSgemm                        OK            2.10e-07      2.10e-07   1.00e-05
classic   cublasDgemm                        OK            3.62e-16      3.62e-16   2.49e-14
classic   cublasHgemm                        OK            8.82e-04      8.42e-04   6.82e-02
classic   cublasCgemm                        OK            4.07e-07      4.07e-07   3.00e-05
classic   cublasZgemm                        OK            9.28e-16      9.28e-16   6.39e-14
classic   Sgemm + TF32 math                  OK            2.61e-04      2.61e-04   9.71e-03
classic   Sgemm + BF16x9 math                OK            2.10e-07      2.10e-07   1.00e-05
classic   Dgemm + fixed-point math           OK            3.62e-16      3.62e-16   2.49e-14
classic   Dgemm fixed-point, 30 bits         OK            2.28e-09      2.28e-09   8.56e-08
classic   Dgemm fixed-point, 40 bits         OK            3.99e-14      3.99e-14   1.50e-12
classic   Dgemm fixed-point, 55 bits         OK            7.74e-16      7.74e-16   6.75e-14
GemmEx    f16 -> f32, compute 32F            OK            2.61e-04      6.44e-07   3.58e-05
GemmEx    bf16 -> f32, compute 32F           OK            2.09e-03      3.94e-07   2.16e-05
GemmEx    f16 -> f16, compute 16F            OK            8.82e-04      8.42e-04   6.82e-02
GemmEx    f32, compute TF32                  OK            2.61e-04      2.61e-04   9.71e-03
GemmEx    f64, compute 64F                   OK            3.62e-16      3.62e-16   2.49e-14
cuBLASLt  f32, compute 32F                   OK            2.10e-07      2.10e-07   1.00e-05
cuBLASLt  f32, compute TF32                  OK            2.61e-04      2.61e-04   9.71e-03
cuBLASLt  f32, compute 16BFX9 (eager)        OK            2.10e-07      2.10e-07   1.00e-05
cuBLASLt  f32, compute FAST_16F              OK            2.61e-04      2.61e-04   9.71e-03
cuBLASLt  f32, compute FAST_16BF             OK            2.61e-04      2.61e-04   9.71e-03
cuBLASLt  f64, compute 64F                   OK            3.62e-16      3.62e-16   2.49e-14
cuBLASLt  f64, compute fixed-point (eager)   OK            3.62e-16      3.62e-16   2.49e-14
cuBLASLt  f16 -> f32, compute 32F            OK            2.61e-04      6.44e-07   3.58e-05
cuBLASLt  f16 -> f16, compute 16F            OK            8.82e-04      8.42e-04   6.82e-02
cuBLASLt  f16 -> f16, compute 32F            OK            3.34e-04      2.08e-04   1.09e-02
cuBLASLt  bf16 -> f32, compute 32F           OK            2.09e-03      3.94e-07   2.16e-05
cuBLASLt  bf16 -> bf16, compute 32F          OK            2.66e-03      1.66e-03   1.16e-01
cuBLASLt  bf16 -> bf16, compute 16F          NOT_SUPPORTED
cuBLASLt  e4m3 -> f32, scale 0.5/2           OK            3.34e-02      0.00e+00   0.00e+00
cuBLASLt  e4m3 -> f16, no scale              OK            3.34e-02      2.08e-04   1.39e-02
cuBLASLt  e5m2 -> f32, no scale              NOT_SUPPORTED
cuBLASLt  e2m1 (fp4) -> f32                  INVALID_VALUE
cuBLASLt  f16 + bf16 mixed -> f32            INVALID_VALUE
cuBLASLt  c32 -> c32, compute 32F            OK            4.07e-07      4.07e-07   3.00e-05
cuBLASLt  c64 -> c64, compute 64F            OK            7.03e-16      7.03e-16   4.62e-14
GemmEx    i8 -> i32, compute 32I             OK            mismatches=0
cuBLASLt  i8 -> i32, compute 32I             OK            mismatches=0
classic   strided-batched f32 (b=4)          OK            4 independent 128^3 products
--------------------------------------------------------------------------------------------------------
accepted: 34   rejected: 4   (out of 38)
```

## Findings

### 1. Which dtypes work

| inputs → output | accepted | notes |
|---|---|---|
| f32 → f32 | yes | native, TF32, FAST_16F/16BF, BF16x9 (falls back, see §2) |
| f64 → f64 | yes | native, fixed-point emulation (classic only, see §3) |
| f16 → f16 / f32 | yes | 16F accumulate or 32F accumulate; both fine |
| bf16 → f32 / bf16 | yes | bf16 output needs 32F accumulate; `bf16×bf16→bf16` with 16F is **not supported** |
| f16 + bf16 mixed | **no** | `CUBLAS_STATUS_INVALID_VALUE`; A and B must have the same type |
| complex f32 / f64 | yes | `cublasCgemm`/`Zgemm` and Lt both |
| int8 → int32 | yes | exact (0 mismatches over 262144 elements) |
| fp8 e4m3 → f32 / f16 | yes | per-tensor scale pointers work (`A×0.5`, `B×2` verified) |
| fp8 e5m2 | **no** | `NOT_SUPPORTED` on this GPU/cuBLAS build |
| fp4 e2m1 | **no** | `INVALID_VALUE` |
| int32 inputs, f16 complex, … | no | not offered by cuBLAS |

So "all dtypes" in practice means: `f16, bf16, f32, f64, c32, c64, i8,
e4m3` — with e4m3 only through cuBLASLt.

### 2. FP32 compute modes

| mode | relative error | verdict |
|---|---|---|
| `CUBLAS_COMPUTE_32F` (pedantic) | 2.1e-7 | baseline |
| `CUBLAS_COMPUTE_32F_FAST_TF32` / `CUBLAS_TF32_TENSOR_OP_MATH` | 2.6e-4 | **works**, tensor cores, ~3 decimal digits |
| `CUBLAS_COMPUTE_32F_FAST_16F` / `FAST_16BF` | 2.6e-4 | works, but at this size the library effectively picks TF32, not f16/bf16 inputs |
| `CUBLAS_COMPUTE_32F_EMULATED_16BFX9` / `CUBLAS_FP32_EMULATED_BF16X9_MATH` | 2.1e-7, **bit-identical to native f32** | accepted but **not engaged** |

The BF16x9 result is not a subtle accuracy coincidence: 262144/262144 output
elements are bit-identical to plain `CUBLAS_COMPUTE_32F`, and forcing an
algorithm that advertises emulation support returns `CUBLAS_STATUS_NOT_SUPPORTED`.
NVIDIA's support table lists BF16x9 for compute capabilities **10.0 and 10.3
only**, so sm_120 (this laptop) is outside the supported set — the library
accepts the compute type and falls back silently.

### 3. FP64 compute modes

| mode | relative error | engaged? |
|---|---|---|
| `CUBLAS_COMPUTE_64F` | 3.6e-16 | native |
| fixed-point, default strategy / `PERFORMANT` | 3.6e-16 | **falls back** (bit-identical to native) |
| fixed-point, `EAGER` + dynamic mantissa | 7.6e-16 | engaged (only 19k/262k elements bit-identical) |
| fixed-point, `EAGER` + 30 mantissa bits | 2.3e-9 | engaged |
| fixed-point, `EAGER` + 40 mantissa bits | 4.0e-14 | engaged |
| fixed-point, `EAGER` + 55 mantissa bits | 7.7e-16 | engaged |

The fixed-point path emulates FP64 on int8 tensor cores by splitting each
double into fixed-point limbs; 30/40/55 mantissa bits is a clean accuracy knob
(~9, ~14, ~16 decimal digits). On this GPU the automatic dynamic-precision
(ADP) default dispatches to native FP64 unless the strategy is `EAGER`, and the
number of bits has to be pinned explicitly.

**cuBLASLt does not expose it here.** With compute type
`CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT`, even with an emulation descriptor
(`STRATEGY=EAGER`, `MANTISSA_CONTROL=FIXED`, 40/55 bits), the Lt heuristic
still chooses a non-emulation algorithm (`chosen emu_cap=0`) and produces
native-f64 results; forcing the one emulation-capable algorithm by hand gives
`CUBLAS_STATUS_INVALID_VALUE`. The working recipe is the classic handle API:

```c
cublasSetEmulationStrategy(h, CUBLAS_EMULATION_STRATEGY_EAGER);
cublasSetFixedPointEmulationMantissaControl(h, CUDA_EMULATION_MANTISSA_CONTROL_FIXED);
cublasSetFixedPointEmulationMaxMantissaBitCount(h, 40);
cublasSetMathMode(h, CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH);
cublasDgemm(...);
```

(raw numbers in `results/classic_emul.txt`, `results/lt64_fixed.txt`).

### 4. Mixed precision in the ordinary sense

The useful combinations are the input/output mixes, not A/B type mixes:

- `f16 × f16 → f32` (32F accumulate) — 2.6e-4, the standard "half storage,
  float math" mode.
- `bf16 × bf16 → f32` — 2.1e-3 (bf16's 8-bit mantissa dominates).
- `f8 e4m3 × e4m3 → f32` with per-tensor scales — 3.3e-2 for `U(-1,1)` data.
  Interesting detail: the **stored-operand** error was exactly zero, i.e. on
  this value range the tensor-core sum is exact in f32 (all products are
  multiples of 2⁻¹⁸ and the 512-term sum fits in 24 bits), so the entire 3.3 %
  is input quantization.
- `i8 × i8 → i32` — exact.
- `A` and `B` must share a type: `f16 + bf16` is rejected outright.

### 4.1 Cross-precision A/B — there is no `f32 × f64 → f64`

A and B must have the **same** dtype; only the output and the accumulate type
may differ. Five cross-precision pairs, all rejected:

| A | B | C / compute | `cublasGemmEx` | cuBLASLt |
|---|---|---|---|---|
| f32 | f64 | f64 / 64F | `NOT_SUPPORTED` (15) | `NOT_SUPPORTED` (15) |
| f64 | f32 | f64 / 64F | `NOT_SUPPORTED` (15) | `INVALID_VALUE` (7) |
| f16 | f32 | f32 / 32F | `NOT_SUPPORTED` (15) | `INVALID_VALUE` (7) |
| f32 | f16 | f32 / 32F | `NOT_SUPPORTED` (15) | `INVALID_VALUE` (7) |
| bf16 | f32 | f32 / 32F | `NOT_SUPPORTED` (15) | `INVALID_VALUE` (7) |

(`results/mixed_ab.txt`, `probes/probe_mixed_ab.cu`.) The fixed-point FP64
emulation path does not change this: it also requires both operands in f64.

The workaround is *exact*, not approximate: every f32 value is representable
in f64 (24-bit vs 53-bit significand), so upcasting A to f64 and calling
`cublasDgemm` computes the same real product with no rounding at all — the
only cost is a temporary f64 copy of A (2× memory, one extra pass). If that
temporary matters, a custom kernel that loads f32 A and converts to double in
registers halves A's traffic, but on this GPU native fp64 throughput (~1/64 of
fp32 on a GeForce part) dominates either way. The same holds for `f64 × f32`
and for f32 output (upcast, `Dgemm`, downcast).

### 5. cuBLAS vs cuBLASLt

Both routes reach the same kernels and accept the same core dtype set; the
classic typed entry points (`cublasSgemm`, `Dgemm`, `Hgemm`, `Cgemm`, `Zgemm`)
are thin wrappers over the same machinery. The practical differences found:

- **Compute-type control**: both have it (`cublasGemmEx` compute type, or
  `cublasSetMathMode` for the typed calls; `cublasLtMatmulDesc` for Lt).
- **Emulation**: on this GPU the classic handle path engages fixed-point FP64,
  the Lt path does not.
- **FP8**: only the Lt path (and `cublasGemmEx` in principle) supports the
  scale pointers.
- **Batching**: classic `cublasGemmStridedBatchedEx` works; Lt has grouped /
  pointer-array variants.

## Practical implications for `rstsr-cuda`

1. The dtype/compute-type matrix to expose is exactly the "accepted" column
   above; anything else should be rejected at the Rust type level rather than
   failing at runtime.
2. Keep f32 on `CUBLAS_COMPUTE_32F` by default; make TF32 an explicit opt-in
   (2.6e-4 vs 2.1e-7). Do not ship a "float32" mode that silently enables TF32.
3. Do not advertise BF16x9 on GeForce parts: the enum exists in CUDA 13 but
   the path falls back to native f32 on sm_120.
4. FP64 emulation is worth wiring only through the classic handle, with the
   mantissa-bit count exposed as an accuracy knob (30/40/55 bits ≈ 1e-9 /
   1e-14 / 1e-16 here). It is not reachable through the Lt descriptor on this
   GPU.
5. `bf16 → bf16` must use 32F accumulation; `bf16` with 16F accumulate is
   `NOT_SUPPORTED`.
6. FP8 support means e4m3 only on this hardware; e5m2 and fp4 are rejected.

## Caveats

- One GPU, one CUDA/cuBLAS version, one matrix size (512). Errors depend on
  `K` and on the data distribution; the qualitative support matrix should be
  stable, the exact error constants are not.
- "Accepted" means the call succeeded; the error column is the evidence that
  the intended compute mode was used. TF32 and the emulation modes were
  additionally checked bitwise (`results/emulation_bitwise.txt`).
- The batched case uses `128³ × 4`; grouped/batched-with-pointer-arrays and
  epilogue variants were not exercised.

## Reproduce

```bash
make            # builds gemm_demo
make probes     # builds the four follow-up probes
./gemm_demo     # prints the table above
```

Raw logs: [`results/`](results). Probe sources: [`probes/`](probes).

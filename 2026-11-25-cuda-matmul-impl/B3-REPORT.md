# B3 — compute modes with the `cublasGemmEx` and classic drivers

Working tree of rstsr `261021/cuda` on top of `c75aecd`, **not committed** (owner's go needed). The
patch is `patches/b3-compute-modes-drivers.patch` (10 files, +979/-71).

| | |
|---|---|
| rstsr | `261021/cuda` @ `c75aecd` + B3 working tree |
| GPU | NVIDIA GeForce RTX 5070 Laptop 8 GiB, sm_120, CUDA 13.2 toolkit, cudarc 0.19.10 |
| bindings | two builds: `nvcc`-detected (13.2, all bindings) and `CUDARC_CUDA_VERSION=12060` (the CI pin, old bindings) |
| raw output | `results/b3_full_suite.txt`, `results/b3_full_suite_12060.txt`, `results/b3_modes_test.txt`, `results/b3_modes_release.txt`, `results/b3_clippy.txt`, `results/b3_check_doc.txt` |

## What was done

| decision | result |
|---|---|
| D2 (output-keyed modes) | `ComputeType32F::gemm_ex` / `ComputeType64F::gemm_ex` in `matmul/mode.rs` map a mode to the `cublasGemmEx` compute type. `drivers::plan::<TC>(mode)` resolves the mode for the output type by `TypeId` (`f32 -> F32(mode)`, `f64 -> F64(mode)`, anything else `None`): the place where a mode the triple cannot serve resolves to the family default once complex/half get drivers. |
| D3 (64F family, fixed-point emulation) | `EmulatedFixedPoint` is a **math mode of the classic handle** (`CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH`), as decided. `fixedpoint_mantissa_bits`: **cannot be honoured** — cudarc 0.19 binds no `cublasSetFixedPointEmulationMaxMantissaBitCount`. A non-zero value with the emulation asked for is an error naming the field (never a silent default); `0` keeps the library default. See open point 1. |
| D9 (inapplicable modes) | By construction for the family mix (a call reads its own field only). A driver that cannot express a mode **declines** and the chain falls through (classic has no `Fast16BF`; without the 13.0 bindings it has no fixed-point emulation; without the 12.9 bindings no math mode at all). `Pedantic` never resolves to a lower mode: the kernels that would receive it compute in the output precision anyway, so there is no triple on which it is an error today; if nothing can serve, the chain error names the drivers. |
| D18 / D5 (table) | `drivers::candidates` per `(triple, mode)`, as a stack `Chain` (no allocation): `f32`: Ex, cuBLAS, tiled, naive for every 32F mode; `f64`: same for `Default`/`Pedantic`, **cuBLAS, tiled, naive** for `EmulatedFixedPoint`; everything else naive. Lt is inserted in B4. |
| D24 (handle state) | The classic driver sets the math mode inside the handle lock, launches, restores the previous mode (`cublasGetMathMode` first, not "the default") and only then unlocks. A failure to set is a refusal (nothing enqueued, falls through); a failure to restore after the launch is an error. `Default` touches nothing. |
| new driver | `drivers/cublas_ex.rs`: `cublasGemmEx` / `cublasGemmStridedBatchedEx`, storage = output type (`CUDA_R_32F`/`CUDA_R_64F`), scales of the same type, `CUBLAS_GEMM_DEFAULT`. Shares `cublas::with_gemm_call` with the classic driver (argument derivation, ordering guards, folded pointers, handle lock by the callee). |

Shared refactor inside `drivers/cublas.rs`: the preparation of a `gemm` call (`with_gemm_call`) is one
function both drivers use; `launch` takes an optional `MathMode`; the `ask_*` functions are pure and
unit-tested (`Ask::{Leave, Math, No}`).

## The cudarc binding level (new: `build.rs`)

cudarc gates part of its cuBLAS bindings on the CUDA version it is built for: `cublasSetMathMode`
from **12.9**, `CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH` from **13.0**. A dependency's features are not
visible to `cfg`, so the crate now has a `build.rs` that resolves the version the way cudarc does
(`CUDARC_CUDA_VERSION`, else `nvcc --version`) and emits two cfgs: `cublas_math_mode`,
`cublas_fixedpoint_math`. Without them the code paths are compiled out and the classic driver
declines with a message naming the needed CUDA version; `cublasGemmEx` needs no gating.

Consequences: the CI pin `12060` builds and passes everything, but there the classic entries serve
`Default` only, and the fixed-point emulation is unreachable (no GPU on CI anyway). Locally with
`nvcc` (13.2) everything is on. Limitation, documented in `build.rs`: a `cudarc/cuda-XXXXX` feature
pinned by another crate of the build is not seen.

## Verification

**Whole crate, `--features faer`, `--test-threads=1`**, both binding levels: **215 tests pass**
(`results/b3_full_suite*.txt`), B2's 201 plus 10 host tests and 4 hardware tests.

| target | tests | new in B3 |
|---|---|---|
| `--lib` | 92 | mode -> compute type (2), plan by output type, table 32F / 64F / other triples, `Chain` bounds, classic `Ask` per 32F / 64F mode, unbound modes refuse (3) |
| `test_matmul_modes` (new, hardware) | 4 | see below |
| all B2 targets | 119 | unchanged |

`test_matmul_modes.rs` (`results/b3_modes_test.txt`):

- `modes_32f`: every 32F mode on `cublasEx`, on the default chain, and on the classic entries
  (forced), against the CPU `f64` product of the `f32` operands, 200x300x180. `Fast16BF` forced on
  classic alone is an error naming `bf16`.
- `modes_64f`: `Default`/`Pedantic` on both drivers and the chain; `EmulatedFixedPoint` on classic
  and the chain; Ex alone with the emulation is "no enabled driver is a candidate" (D18).
- `fixedpoint_mantissa_bits`: `30` + emulation -> error naming the field; `30` with `Default` -> no
  error (the field is read only by the emulation).
- `cublas_ex_batched_and_transposed`: a strided batch and a transposed right operand through the
  strided-batched / `OP_T` argument shapes.

Measured max relative error (vs CPU `f64`):

| mode | f32, 200x300x180 | f32, 2048³ vs device default | f64, 200x300x180 | f64, 2048³ vs device default |
|---|---|---|---|---|
| `Default` | 1.9e-7 | 0 | 4.3e-16 | 0 |
| `Pedantic` | 1.9e-7 | 0 (same bits) | 4.3e-16 | 0 (same bits) |
| `FastTF32` | 2.4e-4 | 2.8e-4 | — | — |
| `Fast16F` | 2.4e-4 | 2.8e-4 | — | — |
| `Fast16BF` | 2.4e-4 (Ex) | 2.8e-4 (Ex) | — | — |
| `EmulatedFixedPoint` | — | — | 4.3e-16 | 3.8e-15 |

Tolerances in the test: 1e-5 (exact f32), 1e-2 / 5e-2 (TF32 / bf16 class), 1e-13 (f64), 1e-9
(fixed point).

`cargo clippy --all-targets --features faer`: no warning in `rstsr-cuda` under either binding level
(three pre-existing in `rstsr-core`). `cargo check --all-features`, `cargo doc --no-deps`: clean.
New and changed files rustfmt-formatted (the crate's other files are not fmt-clean, as in B2).

**What the modes do on this GPU** (`src/bench_matmul_mode_check.rs`, release, 10 reps,
`results/b3_modes_release.txt`):

| | `Default` | `Pedantic` | `FastTF32` | `Fast16F` | `Fast16BF` | `EmulatedFixedPoint` |
|---|---|---|---|---|---|---|
| f32 2048³, `cublasEx` | 14.0 TF/s | 13.9 | 20.0 | 21.6 | 20.4 | not served (D18) |
| f32 2048³, classic | 13.5 | 13.9 | 19.3 | 20.5 | not served | — |
| f64 2048³, `cublasEx` | 0.32 | 0.32 | — | — | — | not served |
| f64 2048³, classic | 0.32 | 0.33 | — | — | — | **2.55 TF/s (8x)** |

- Every Fast mode lands on the **same bits** (2.75e-4): cuBLAS's header allows "half **or TF32**"
  and "bf16 **or TF32**", and on sm_120 it picks the TF32 path for all three. The modes are
  honoured (1.4-1.55x faster, different from `Default`), but `Fast16BF` is not less accurate than
  `FastTF32` here.
- `Pedantic` is bit-identical to `Default` on this GPU: the default does not use TF32 on sm_120.
- **The fixed-point emulation is the real win of the 64F family on this GeForce**: 0.32 -> 2.55
  TF/s at 3.8e-15, through the classic handle only, exactly the D18 pin.

Pitfall met while writing the check: a tensor carries its device value, so setting the mode on a
`with_flags` clone after creating the operands does nothing; the operands must be moved onto the
flagged device (`to_device`) or created on it.

## Bugs found on the way

- `tests/test_matmul_modes.rs` first checked the refusals with `f64` tensors for 32F modes (the
  mode read is the 64F field, so nothing was refused) and asserted a `cublasEx: cannot serve`
  message for the emulation, where the table makes Ex a non-candidate. Both were test mistakes,
  fixed; no driver bug surfaced.

## For the owner

1. **`fixedpoint_mantissa_bits` is not applicable under "no new FFI"**: cudarc 0.19 has no binding
   for `cublasSetFixedPointEmulationMaxMantissaBitCount` (nor the `MantissaBitOffset` /
   `MantissaControl` entries). Options: (a) keep as is (error on non-zero); (b) drop the field
   until cudarc binds it; (c) allow one hand-written `extern "C"` declaration (against D19); (d)
   an upstream PR to cudarc. I would take (a) now and (d) later.
2. **`build.rs` is new for the crate.** It only resolves the cudarc CUDA version into two cfgs; the
   alternative was to drop math-mode support whenever the bindings are old, which still needs the
   cfg. Say if you prefer another mechanism.
3. **CI pin `12060`** now costs real function on a machine that has both the 12.6 bindings and a
   GPU (no math modes, no emulation). The CI has no GPU, so today it costs nothing; raising the pin
   to `13000` would remove the split at the price of requiring a 13.x toolkit to build.
4. **Driver-level refusals vs D9's silent default**: a mode valid for the triple but not for a
   driver (classic + `Fast16BF`) makes that driver decline, and the chain falls to tiled/naive
   (full `f32`, i.e. more precise than asked) instead of the driver running with `Default`. I took
   "the driver says what it cannot" as the consistent rule; it only shows when Ex is disabled.
5. `Pedantic` on an unsupported triple being "an error" (D9) is subsumed: the kernels that could
   receive it compute in the output precision, so no triple needs the error; when no driver can
   serve, the chain error stands. No separate check was written.
6. In the 13.x API the emulation is also a compute type (`CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT`,
   available to Ex and Lt), so the D18 "classic only" pin could be relaxed in B4/B5 if Lt should
   serve it; not done.
7. New in-tree test file `tests/test_matmul_modes.rs` (4 tests), like B2's `test_matmul_naive.rs`.

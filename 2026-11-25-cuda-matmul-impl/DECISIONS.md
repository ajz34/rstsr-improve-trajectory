# Decisions — CUDA matmul implementation

Grilled 2026-11-25 against rstsr `261021/cuda` @ `9c7d49a`. Questions and the facts behind each
decision: `GRILL-R1..R3-QUESTIONS.md`, `GRILL-INIT-PROMPT.md`. Batch plan: `PLAN.md` (summary in D28).

Status: **decisions closed; all batches done (B6 = `3508d3a`, `B6-REPORT.md`).** B2-B4
are committed in rstsr as `c75aecd`, `493a42f`, `554f956` and `85ed399` (`B4-REPORT.md`, open
points: workspace per stream instead of per ordinal, first-call heuristic cost, Lt does not emulate on
this GPU).

## Scope

- **D1 — dtype scope.** In principle every `(TA, TB) -> TC` is accepted by the *naive* driver;
  cuBLASLt / cuBLASEx / cuBLAS are the accelerated drivers, used wherever they can serve.
  Accelerated coverage in this wave: `f32`, `f64`, `c32`, `c64`, `f16 -> f16`, `f16 -> f32`,
  `bf16 -> bf16`, `bf16 -> f32` (operands always the same dtype). FP8 is a separate decision
  (rstsr has no fp8 dtype). Integer and other triples work through naive.
- **D14 — promotion.** *Withdrawn by the owner, 2026-11-25, before B5:* no `ext_matmul` is
  implemented; "either we do not implement `ext_matmul`, or `matmul` itself becomes functional
  with the usual type casting". Mixed operands run through the entries without a `Mul` bound
  (`matmul_from`, `matmul_with_output`) on the naive kernel (in-kernel casts, no copy), naive
  only this wave. Fact base kept: CuPy materializes the promoted cast of the operands; PyTorch
  refuses mixed operand dtypes except a float output with half/bf16 inputs.

## Compute modes (output-keyed)

- **D2 — model.** One enum per output family, one field per family in the flags:
  `ComputeType32F` (exists), `ComputeType64F`, `ComputeType16F`. A crate-internal mapping from
  the output type selects the enum and resolves `(mode, TA, TB)` to a cuBLAS compute type and
  scale type. `f32`, `bf16` and f32-out-of-half share the 32F family.
- **D3 — 64F family.** `ComputeType64F { Default, Pedantic, EmulatedFixedPoint }`; the fixed-point
  mantissa bit count is a separate flags field (`fixedpoint_mantissa_bits`). BF16x9 is **not**
  added (silently falls back to native f32 on sm_120); noted as a possible later variant.
- **D9 — inapplicable modes.** A mode of an unrelated family is ignored. A mode in the right family
  but invalid for the type triple resolves to the family default silently (documented on the
  enum). `Pedantic` is never downgraded: pedantic on an unsupported triple is an error.

## Entry points

- **D4 — device-level.** `matmul_with_mode_inner` is `pub(crate)`, takes the mode explicitly as its
  last argument and a `MaybeUninit<TC>` output buffer; `DeviceMatMulAPI::matmul` passes the
  flag-derived mode and the caller's `beta`, `matmul_uninit` passes `beta = 0`.
- **D15 — tensor-level.** Extension trait `TensorCudaMatMulAPI` in `rstsr-cuda` (core cannot name
  CUDA modes): `c.matmul_from_mode(&a, &b, alpha, beta, mode)`,
  `a.matmul_with_mode::<TC, DB, DC>(&b, mode)` (allocating; `TC` by turbofish),
  `a.matmul_with_output_mode(&b, &mut c, mode)`. `mode: <TC as CudaMatMulMode>::Mode`, with a
  public marker trait `CudaMatMulMode` for `f32, f64, Complex<f32>, Complex<f64>, f16, bf16`.
- **f16 x f16 -> f32** needs no core change: `matmul_from`/`matmul_with_output` have no `Mul`
  bound, while `rt::matmul` / `&a % &b` infer `TC = f16` from `half`'s `Mul`.

## Drivers

- **D5 — selection.** A pure, host-testable static table of type triple + mode + layout class
  returns the ordered candidates; runtime fall-through only on launch failure or unloadable
  library.
- **D18 — table.**

  | call | mode | candidates in order |
  |---|---|---|
  | `f32 -> f32` | any 32F mode | Lt, Ex, cuBLAS, tiled, naive |
  | `f64 -> f64` | `Default`/`Pedantic` | Lt, Ex, cuBLAS, tiled, naive |
  | `f64 -> f64` | `EmulatedFixedPoint` | cuBLAS only, then tiled, naive |
  | `c32`/`c64` | `Default`/`Pedantic` | Lt, Ex, cuBLAS, naive (a `Fast*` mode and the fixed-point emulation resolve to the default for complex outputs) |
  | `f16 -> f16` | `Default` (32F acc) / `Pedantic` / `Acc16F` (16F acc, `f16` scales) | Lt, Ex, naive |
  | `f16 -> f32`, `bf16 -> f32`, `bf16 -> bf16` | `Default`/`Pedantic` (32F family for the `f32` outputs, 16F for `bf16`; `Fast*` and `Acc16F` resolve to the default) | Lt, Ex, naive |
  | mixed operands | — | naive only |

- **D13 — chain.** cuBLASLt > cuBLASEx > cuBLAS > tiled > naive. Tiled stays in the default chain
  for `f32`/`f64`; its refusal cases (no unit-stride operand axis, non-affine batch) now land in
  naive. `ComputeType16F::Default` is the 32F accumulate; `Acc16F` the opt-in.
- **D11 — naive driver.** A new universal kernel (`drivers/naive.rs`), elementwise style: one
  thread per output element, loop over `k`, full layout addressing (signed/zero strides, per-axis
  batch strides), templated `@TA@ @TB@ @TC@`; nothing is refused with `UnImplemented`.
- **D12 — accumulator.** Fourth template type `@TACC@`: `f32` for `f16`/`bf16` output, else `TC`;
  matches the cuBLAS default accumulation.
- **D23 — naive coverage.** Instantiated for every `CudaArith` type plus f16/bf16 (integer matmul
  thus works on CUDA).
- **D6 — cuBLASLt resources.** One Lt handle and one workspace buffer per ordinal (lazy, 32 MiB,
  stream set per call); heuristic cache keyed by the full problem descriptor; handle mutex per
  call. Use only cudarc's `cublaslt` layer — **no new FFI** (D19).
- **D24 — classic handle state.** The cuBLAS driver sets math mode / emulation knobs inside the
  handle lock, launches, and restores the defaults before unlocking.
- **D20 — tiled check.** (done, `TILED-CHECK.md`) Before any implementation: correctness vs CPU (f32/f64 layout matrix),
  timing vs cuBLAS, a verdict per refusal case; reported here (`TILED-CHECK.md`). A step, not a
  grilling item.

## Flags and environment

- **D16 — runtime flags.** `CudaRuntimeFlags` becomes `#[non_exhaustive]` with
  `compute_type_32f`, `compute_type_64f`, `compute_type_16f`, `fixedpoint_mantissa_bits`,
  `matmul_drivers`, `trace_matmul`. The op-behaviour environment variables
  (`RSTSR_CUDA_MATMUL`, `RSTSR_CUDA_DEBUG_GEMM`, `RSTSR_CUDA_COMPUTE_TYPE_32F`) are removed; tests
  set flags programmatically. Kernel-build variables (`ARCH`, `STRICT_FP`, `INCLUDE`, `CACHE*`)
  stay environment variables (process-global, part of the PTX cache key).
- **D17 — driver enable set.** `MatMulDrivers` bit set (`CUBLAS_LT`, `CUBLAS_EX`, `CUBLAS`,
  `TILED`, `NAIVE`; `all()`, `contains()`, `BitOr`). Filtering the static table by the enabled
  bits is the whole selection; forcing one driver = enabling only its bit; nothing able to serve
  = error naming the driver and the type triple. Default `all()`. (Consequence accepted: no
  "prefer X but do not fall through" other than enabling X alone.)
- **D25 — builder.** `DeviceCuda::with_flags(self, flags) -> Self`, immutable twin of
  `set_flags_override`.

## Layout

- **D8/D19 — files.** `crates-device/rstsr-cuda/src/matmul/`:

  ```
  mod.rs          docs + mod/re-export only (lightweight)
  mode.rs         ComputeType32F/64F/16F, CudaMatMulMode, mode -> cublas compute/scale type
  flags.rs        MatMulDrivers
  geom.rs         GemmGeom, resolve_geom, linear_step (moved)
  cublas_args.rs  CublasArgs, args() (moved)
  inner.rs        matmul_with_mode_inner + DeviceMatMulAPI / DeviceExtMatMulAPI impls
  tensor.rs       TensorCudaMatMulAPI (the `*_mode` entries)
  drivers/mod.rs  GemmDriver enum, the static table, the ordered dispatch loop
  drivers/cublas_lt.rs, cublas_ex.rs, cublas.rs, tiled.rs, naive.rs
  ```

  `CudaRuntimeFlags` stays in `src/flags.rs`; `ComputeType32F` moves to `matmul/mode.rs` and is
  re-exported from the old path. `src/kernels/{gemm,cublas_gemm}.rs` deleted after the move;
  `tensordot` stays in `src/linalg.rs` (untouched, see D29).

## Verification and delivery

- **D26 — tests.** Host-side lib tests (mode resolution, table, Lt descriptor construction); per
  type triple a hardware test running each driver alone vs CPU with tolerance per `(triple, mode)`
  (TF32 ~1e-3, half storage ~1e-2, emulation per mantissa bits, else ~1e-9); layout matrix for
  naive and tiled; naive as second on-device oracle.
- **D27 — bench.** criterion `bench_matmul` (square, tall-skinny, batched; f32, f64, c64,
  f16->f16, f16->f32), driver fixed per sample, before/after, numbers in this README.
  *Deviation (B6):* the crate's `harness = false` median-of-R style, no criterion dependency.
- **D28 — branch and batches.** Work **on the current head of `261021/cuda`** (no new branch).
  B1 tiled check -> B2 skeleton, move, naive, flags refactor (behaviour-preserving for f32/f64)
  -> B3 Ex + classic drivers with modes -> B4 Lt -> B5 half/complex, tensor API, `ext_matmul`
  -> B6 tests/bench/README. No commit or push in the rstsr repo without go per batch.
- **D10 — "native implementation".** No hand-written tensor-core kernel in this wave.
- **D30 — tiled `lane_axis` fixes noted, deferred** (owner, 2026-11-25): accept a unit stride of
  `-1` (`s.abs() == 1`), and optionally the arbitrary-lane fallback, are verified correct
  (`TILED-CHECK.md` 2b) but are **not** part of B2 or any batch here; they stay a recorded follow-up.
- **D29 — `tensordot` deferred entirely** (owner, 2026-11-25): it will be implemented with cuTENSOR.
  The interim `DeviceTensordotAPI` impl is left as it is (it follows `DeviceMatMulAPI<T, T, T, Ix2,
  Ix2, Ix2>` and so gets whatever the chain gives) and is not adapted, extended or reviewed here.

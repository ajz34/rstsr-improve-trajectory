# Round 1 questions — CUDA matmul implementation

Design tree root: *matmul on `DeviceCuda` with several drivers and output-keyed compute modes*.
Round 1 covers the trunk: dtype scope, the compute-type model, the internal function, driver
selection, module layout. Everything else (fallback details, tiled-kernel scope, tests, benches,
env knobs beyond Q7) hangs off these and waits for round 2.

Facts referred to below are in `GRILL-INIT-PROMPT.md`.

---

❓ **Q1** - **Dtype scope of this wave**: which `(A, B) -> C` combinations must work when this wave
is done?

- (a) `f32->f32`, `f64->f64`, `c32->c32`, `c64->c64`, plus `f16->f16`, `f16->f32`,
  `bf16->bf16`, `bf16->f32` (A and B always the same dtype).
- (b) (a) plus integer `i8->i32`.
- (c) (a) + (b) plus FP8 `e4m3` with scale pointers.

Mixed operand dtypes (`f16 x bf16`, `f32 x f64`) are rejected by cuBLAS and stay out in every
option.

➡️ (a). Integer and FP8 GEMM have no caller in the target workload, and FP8 drags in scale
pointers (so `alpha`/`beta` are no longer the whole scaling story). They slot into the same
driver table later without touching the design.

---

❓ **Q2** - **Compute-type model, keyed by output type**: how should the modes be typed?

- (a) **One enum per output family**, flags carry one field per family:
  `ComputeType32F` (exists), `ComputeType64F`, `ComputeType16F` (for f16 output: `Default`/`Pedantic`,
  and whether to accumulate in 16F or 32F). A crate-internal trait
  `MatMulMode<TOut>` (one impl per output dtype) maps the output type to its enum and resolves
  `(mode, TA, TB) -> cublas compute type + scale type`. `bf16` and `f32`-out-of-half share the 32F
  family: the mode is chosen by the **output** type, exactly as you stated.
- (b) One flat enum `MatMulCompute { F32(ComputeType32F), F64(ComputeType64F), .. }` in a single
  flags field.

➡️ (a). (b) makes one field carry a mode for a family the call does not use, which then has to be
ignored — the "silently does nothing" failure the current `compute_type_32f` already has.

---

❓ **Q3** - **Shape of the 64F family**: `ComputeType64F { Default, Pedantic, EmulatedFixedPoint }`
with the mantissa bit count (30/40/55 measured) as a **separate** integer field
(`fixedpoint_mantissa_bits`, applied only when the 64F mode is `EmulatedFixedPoint`), rather than
three enum variants? And BF16x9 (`32F_EMULATED_16BFX9`): add it to `ComputeType32F` at all, given
it silently falls back to native f32 on sm_120?

➡️ Yes to the enum + separate mantissa field (the bit count is a knob, not a mode). Do **not** add
BF16x9 now: it is datacenter-only (cc 10.0/10.3) and would be an enum variant that provably does
nothing here; record it as a possible later variant.

---

❓ **Q4** - **Internal function and its visibility**: one crate-level function

```rust
// src/matmul/mod.rs
pub fn matmul_with_mode<TA, TB, TC, DA, DB, DC>(
    device: &DeviceCuda, c: &mut CudaBuffer<MaybeUninit<TC>>, lc: &Layout<DC>,
    a: &CudaBuffer<TA>, la: &Layout<DA>, b: &CudaBuffer<TB>, lb: &Layout<DB>,
    alpha: TC, beta: TC, mode: <TC as MatMulModeInput>::Mode,
) -> Result<()>;
```

- (a) `pub(crate)`, reached only through `DeviceMatMulAPI` (mode from `device.flags()`).
- (b) `pub` (plus `rt::cuda::` free function later), so a caller can ask for TF32 on one call
  without pinning device flags — the "per-call compute type" the workload note preferred over a
  process-global flag.

`DeviceMatMulAPI::matmul` passes `device.flags()`-derived mode and a real `beta`; `matmul_uninit`
passes `beta = 0` so the single internal body serves both (it writes every element either way).

➡️ (b) as an **inherent method** `DeviceCuda::matmul_with_mode` (falls in line with
`with_stream`/`set_flags_override`, no new trait), re-exported in `funcs` later only if asked.
Taking the mode as the last argument by value; the output buffer is `MaybeUninit<TC>`-typed for
both paths.

---

❓ **Q5** - **Driver selection: static table, not runtime fall-through**: the priority
cuBLASLt > cuBLASEx > cuBLAS > naive is realized as a **pure function of the type triple + mode +
layout class** returning an ordered candidate list, with runtime fall-through only on *launch*
failure (`NOT_SUPPORTED`/`INVALID_VALUE`/status != success) and on "library not loadable"? In
particular the FP64 fixed-point emulation is pinned to the **cuBLAS (classic handle)** driver in
the table because Lt cannot reach it on this GPU, and FP8 would be pinned to Lt.

➡️ Yes. The table is host-testable without a GPU (same style as the existing `args()` unit
tests), and runtime fall-through stays as a safety net, not as the selection logic.

---

❓ **Q6** - **cuBLASLt resources**: one Lt handle + one workspace buffer per **ordinal** (lazily
allocated, e.g. 32 MiB, stream set per call), and a heuristic-result cache keyed by the full
problem descriptor (type triple, compute type, transposes, `m, n, k, batch, lda, ldb, ldc`)? The
handle mutex is taken per call like the cuBLAS one.

➡️ Yes. Lt requires a workspace for good algorithms; the heuristic query costs tens of
microseconds, which is significant next to a small GEMM, so it must be cached. Per-ordinal (not
per-stream) because the workspace is memory, not stream state.

---

❓ **Q7** - **Driver forcing knob**: generalize `RSTSR_CUDA_MATMUL=tiled` to
`RSTSR_CUDA_MATMUL_DRIVER=cublaslt|cublasex|cublas|tiled` (forces that driver, errors if it
cannot serve the combination) and **drop** the old variable name (crate not published on
crates.io under a stable API)? Keep `RSTSR_CUDA_DEBUG_GEMM` as is, extended to print the chosen
driver and resolved compute type.

➡️ Yes to both; the forced-driver error is what gives each driver hardware test coverage
(the same reason `=tiled` exists today).

---

❓ **Q8** - **Module layout**: a new directory `crates-device/rstsr-cuda/src/matmul/` holding
everything matmul: `mod.rs` (internal fn + `DeviceMatMulAPI` impl), `mode.rs` (compute-type
enums + resolution), `geom.rs` (the moved `resolve_geom`/`GemmGeom`), `cublas_args.rs` (moved
argument derivation), and `drivers/{cublas_lt.rs, cublas_ex.rs, cublas.rs, tiled.rs}`. The
compute-type enums move out of `src/flags.rs` into `matmul/mode.rs` (re-exported from the old
path so `rstsr_cuda::ComputeType32F` keeps working). `src/kernels/gemm.rs` and
`src/kernels/cublas_gemm.rs` are deleted after the move; `tensordot` stays in `src/linalg.rs`.

➡️ Yes.

---

❓ **Q9** - **Inapplicable mode policy**: when the device flags carry a mode that does not apply to
the call (e.g. `FastTF32` with an f16 output, or any 32F mode with f64 output), the call **ignores
the irrelevant family** (a 64F call never reads the 32F field) — but when a mode is in the right
family and yet invalid for the dtype triple (e.g. `Fast16F` on `f16 -> f32`, where inputs are
already f16), resolve it to the family default **silently**? The alternative is an error.

➡️ Silently resolve to the family default, and say so in the doc of the enum; a `pedantic`
request, however, is **never** downgraded (pedantic on an unsupported triple is an error).

---

❓ **Q10** - **Meaning of "naive / native implementation"**: "naive" = the existing tiled SIMT
kernel (`drivers/tiled.rs`), and "native implementation" = a future hand-written tensor-core
kernel that is **not** part of this wave (only a placeholder in the driver enum, if at all)?

➡️ Yes: tiled SIMT kernel is the naive driver; no tensor-core kernel now.

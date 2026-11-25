# Round 3 questions — CUDA matmul implementation

## Background (agent-authored): round 2 answers as given on screen

- **Q13** accepted. Checking the tiled kernel is **not** a grilling task, but must be done **before
  implementation**; results are reported in this repository.
- **Q14** accepted in principle, pending a fact check of how CuPy / PyTorch do it; the result is to
  be recorded and folded into the decision.
- **Q16** accepted; `CudaRuntimeFlags` should probably be `#[non_exhaustive]`.
- **Q19** accepted; **declare no new FFI** — use only what cudarc provides.
- **Q11, Q15, Q18**: recommendations accepted.
- **Q12, Q17**: consequences unclear; recommendation followed for now.

### Facts checked for this round (agent)

**A. Tiled kernel — existing hardware tests at `9c7d49a` (run today, debug build).**

```
cargo test -p rstsr-cuda --features faer --test test_matmul_tiled -- --test-threads=1
test tiled_batched_vs_cpu ... ok
test tiled_col_major_and_gemv_gevm ... ok
test tiled_gemm22_vs_cpu ... ok
test tiled_transposed_operands ... ok
test tiled_matmul_from_vs_cpu ... ok
5 passed; 0 failed (1.27 s)
```

So: five f64-vs-CPU cases pass. Not covered by any test today: f32, non-affine or broadcast-heavy
batches, negative/zero-stride operands, the no-unit-stride refusal, any timing.

**B. PyTorch (`ff4f371`, 2026-10-19) — `aten/src/ATen/native/cuda/Blas.cpp`,
`aten/src/ATen/cuda/detail/CublasLtUtils.h`, `aten/src/ATen/native/LinearAlgebra.cpp`.**

- **No dtype promotion.** `addmm` checks `mat1.scalar_type() == mat2.scalar_type()` and errors
  otherwise; `matmul` likewise requires equal operand dtypes. Mixed-dtype `matmul` is an error.
- **Output dtype may differ from the operands** in exactly one direction: float output with
  half/bf16 inputs (`is_float_output_with_half_input`, `out=` of f32 dtype).
- **Compute types** (`getCublasLtTypeInfo`): f64 → `CUBLAS_COMPUTE_64F`; f32 →
  `CUBLAS_COMPUTE_32F` by default, `32F_FAST_TF32` and `32F_EMULATED_16BFX9` only by opt-in global
  context flags; f16 → `CUBLAS_COMPUTE_32F` with `16F` accumulate only by opt-in
  (`allowFP16AccumulationCuBLAS`, cc >= 7); bf16 → `32F` only. No FP64 emulation.
- **Driver order**: cuBLASLt first, and on Lt failure the function **recurses with Lt disabled**
  and calls `cublasGemm[StridedBatched]Ex` — the same "Lt, then Ex" chain with runtime fall-through.
- Compute type is a property of the **operand/output type triple**, selected per call.

**C. CuPy (`8f7a35d16`, 2026-10-19) — `cupy/_core/_routines_linalg.pyx`, `cupy/_core/_gufuncs.py`,
`cupy/linalg/_product.py`.**

- **Promotion by casting the operands.** `cupy.matmul` is a gufunc; its loop is selected by
  `numpy.promote_types` of the operand dtypes and each operand is `astype(in_type, copy=False)` to
  the common dtype — a **materialized** cast copy when dtypes differ. Output dtype = promoted type.
- For the batched path (`ndim > 2`) with a promoted half/bf16 dtype it goes further: operands are
  cast to **f32** (`ascontiguousarray(a, f32)`), the GEMM runs in f32, and the result is cast back
  to the half output (`dtype` vs `ret_dtype` in the source). Integer types take a separate
  strided-batched integer tensordot kernel.
- **Compute type is keyed by dtype**: `compute_types = [TBD x6]` for
  `bfloat16, float16, float32, float64, int8, int32`, values `DEFAULT / PEDANTIC / FP16 / FP32 /
  FP64 / BF16 / TF32`, set by `cupy.cuda.cublas.set_compute_type(dtype, ...)` (TF32 also seeded by
  env `CUPY_TF32`). I.e. one compute-type slot **per output dtype family** — the same shape as the
  model accepted in round 1 Q2 (a).

**Consequence for Q14.** "Cast to the output type first" is what both do, but both **materialize**
the cast (CuPy always for promotion, PyTorch by refusing). Our in-kernel cast in the naive kernel
gives the same numbers without the copy; a materialized "promote-then-GEMM" path would be the way
to get mixed operands onto the accelerated drivers later.

**D. cudarc 0.19.10 `cublaslt` layer.** `result.rs` (handle, matrix layout, matmul desc,
preference, heuristic, `matmul`) and `safe.rs` (`CublasLt` bound to a stream) exist behind the
`cublaslt` cargo feature. Everything the Lt driver needs is reachable without new `extern`
declarations; the emulation knobs for the classic handle are in `cublas::sys`.

---

❓ **Q20** - **Tiled-kernel check: placement and content**: do it as the **first step before any
implementation**, inside this directory (`TILED-CHECK.md` + a test file), and report: (1)
correctness vs CPU for f32 and f64 over a layout matrix (contiguous, transposed operand, stepped
operand, negative stride, zero stride, broadcast batch, non-affine batch (refusal), col-major),
(2) timing vs cuBLAS for square/skinny/batch shapes, (3) a verdict per refusal case. The two
refusals are **not fixed** in that check; they are removed by the new chain (naive takes them).

➡️ Yes. It is evidence for the decision "tiled stays in the default chain" (Q13), so it belongs
with the matmul work, but it is run as its own step.

---

❓ **Q21** - **Q12 consequence, restated — accumulator of the naive kernel**: with `TACC = f32` for
a half output, each thread adds `k` products in f32 and rounds **once** at the end; without it, the
`k` additions happen in f16 (11-bit significand), so the error grows roughly like `sqrt(k) * 2^-11`
— about `1e-2` relative at `k = 4096`, versus `4e-4` (one rounding) with f32 accumulation. cuBLAS's
default for `f16 -> f16` is the 32F accumulate, so without `TACC` the fallback would be up to ~50x
less accurate than the driver it replaces; for f32/f64/complex `TACC == TC` and nothing changes.
Cost: one more type name in the template and in the kernel-cache key.

➡️ Keep `TACC`.

---

❓ **Q22** - **Q17 consequence, restated — enable set vs forced driver**: with a bit set, one flag
value gives both uses: *disable to compare* (`all() - CUBLAS_LT`) and *force one*
(`MatMulDrivers::TILED` alone). Two things follow that you should be aware of: (1) fall-through
among the **enabled** drivers is always automatic — there is no way to say "use Lt and error if the
heuristic fails" other than enabling Lt alone, which then errors on refusal as specified; (2) an
empty set, or a set whose members cannot serve the triple, is an error. The alternative is two
fields (`matmul_driver: Option<GemmDriver>` to force + a disable mask), which can express "prefer
Lt, never use Ex" but is two concepts to document.

➡️ Keep the single enable set (Q17 as written).

---

❓ **Q23** - **Type coverage of the naive driver and `ext_matmul` routing**: the naive kernel is
instantiated for every `CudaArith` type (integers, f32/f64, complex) plus f16/bf16, so *integer*
matmul (absent on CUDA today) works through naive, and `ext_matmul_uninit<T, T, T>` (same dtype
through the ext trait) routes to the **full chain** with the flag-derived mode instead of naive.

➡️ Yes.

---

❓ **Q24** - **Handle-state for the classic driver**: the FP64 emulation path needs
`cublasSetMathMode` + emulation knobs, which are **handle-global**. The cuBLAS driver sets them
inside the existing handle lock, launches, and **restores the defaults** before unlocking (so the
next caller on that handle never inherits the state); the typed f32 path does the same for
`CUBLAS_TF32_TENSOR_OP_MATH`. Ex and Lt carry their compute type per call and touch no handle
state.

➡️ Yes.

---

❓ **Q25** - **`#[non_exhaustive]` on `CudaRuntimeFlags`**: add it, and add a builder-style
`DeviceCuda::with_flags(self, flags) -> Self` (immutable twin of `set_flags_override`) so the
`..Default::default()` need of a non-exhaustive struct is paid once.

➡️ Yes.

---

❓ **Q26** - **Test matrix**: (1) host-side lib tests for the mode resolution, the static driver
table and the Lt argument/layout construction (no GPU); (2) a hardware test per **type triple**
that runs every enabled driver **individually** (enable set = one bit) and compares to the CPU
with a tolerance per `(triple, mode)` — TF32 ~1e-3, half storage ~1e-2, emulation per mantissa
bits, else 1e-9-class; (3) a layout matrix (as in Q20) for **naive** and **tiled** only, the
accelerated drivers being covered by contiguous/transposed/batched; (4) the naive kernel doubles
as the on-device oracle for a second comparison. Each hardware test skips without a device, as
today.

➡️ Yes.

---

❓ **Q27** - **Benchmarks**: a criterion `bench_matmul` (square, tall-skinny, batched; f32, f64,
c64, f16->f16, f16->f32) with the driver fixed per sample, run **before** (cuBLAS-typed baseline at
`9c7d49a`) and after; numbers in this directory's README with the usual environment table. No
performance gate in CI.

➡️ Yes.

---

❓ **Q28** - **Branch and batches**: new branch `261125/cuda-matmul` cut from the current
`261021/cuda` head (keeps the unpublished-crate freedom to rename `ComputeType32F`'s path and drop
the environment variables); batches: B1 tiled check → B2 `matmul/` skeleton, move, naive kernel,
flags refactor (behaviour-preserving for f32/f64) → B3 Ex + classic drivers with modes → B4 Lt
driver → B5 half/complex triples + tensor API + `ext_matmul` → B6 tests/bench/README. No commit or
push in the rstsr repo without your go per batch.

➡️ Yes.

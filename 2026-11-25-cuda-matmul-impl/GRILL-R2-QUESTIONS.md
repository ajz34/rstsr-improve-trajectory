# Round 2 questions — CUDA matmul implementation

## Background (agent-authored): round 1 answers as given on screen

- **Q1** generally accepted: (a) in principle. **Any** `(TA, TB) -> TC` combination should be
  accepted by the *naive* way; cuBLASLt/Ex/cuBLAS are only the accelerated drivers and are used
  "to try best". FP8 is a separate decision (rstsr has no fp8 dtype) — out of this grill.
- **Q2, Q3, Q5, Q6, Q9**: recommendations accepted.
- **Q4**: accepted with changes. The device-level function is *internal* and named
  `matmul_with_mode_inner`; `mod.rs` should stay a pure import/utility file, so the implementation
  goes elsewhere. Add a **tensor-level** `matmul_with_mode`, similar to `matmul_with_output` plus a
  mode.
- **Q7**: no environment variable for driver selection. Add an entry to `CudaRuntimeFlags` to
  enable/disable drivers; the other environment variables should be refactored out too. API shape
  is not fixed.
- **Q8**: `mod.rs` lightweight; the naive matmul gets its own file; the tiled kernel's correctness
  and efficiency are **not** fully established — a separate but related issue.
- **Q10**: like the CPU implementation, matmul should work for arbitrary layouts (strided, not
  c/f-prefer, negative and zero strides). Dtype promotion is desirable but not a must; casting to
  the output type first is acceptable (apparently what CuPy/PyTorch do). Check how the tiled SIMT
  kernel works.

### Facts checked for this round (agent)

**Tiled kernel (`src/kernels/gemm.rs`) — what layouts it really takes.**

- Matrix axes: strides enter the kernel as signed `rstsr_isz` and are used in plain address
  arithmetic, so **negative and zero strides on A/B are already fine**; C is written straight to
  global memory with signed strides, so the output takes any strides too.
- Refusal 1 (`lane_axis`): an operand with **no unit-stride axis** is refused, although that
  choice only decides which axis the thread id enumerates (coalescing), not correctness.
- Refusal 2 (`linear_step`): a **non-affine batch** (`[4, 3]` with strides `[6, 1]`) is refused;
  the kernel takes one scalar batch step per operand (`grid.z`).
- Single `@T@` for A, B, C, accumulators and `alpha/beta`; zero spelled `(@T@)0`, which is why
  complex is excluded. Shared tiles `64x64x16`, 256 threads, 4x4 per thread.
- cuBLAS side (`cublas_args::args`) refuses: negative stride with extent > 1, `scn != 1`, no
  unit-stride operand axis.
- The CPU serial implementation is the naive triple loop over the layouts with `TC` arithmetic
  (`TA: Mul<TB, Output = TC>`).

**Environment variables read by the crate today.**

| variable | where | effect |
|---|---|---|
| `RSTSR_CUDA_MATMUL=tiled` | `kernels/gemm.rs:365` | force tiled driver |
| `RSTSR_CUDA_DEBUG_GEMM` | `kernels/gemm.rs:376` | print geometry/args once |
| `RSTSR_CUDA_COMPUTE_TYPE_32F` | `flags.rs:47` | seed the global flags |
| `RSTSR_CUDA_STRICT_FP` | `kernel.rs:189` | `--fmad=false` NVRTC flag |
| `RSTSR_CUDA_ARCH` | `kernel.rs:219` | compile target arch |
| `RSTSR_CUDA_INCLUDE`, `RSTSR_CUDA_CACHE`, `RSTSR_CUDA_CACHE_DIR` | `kernel.rs` | NVRTC include paths / PTX cache location |

The first three change *op behaviour*; the last four change *what gets compiled*, so they are
fixed before the first kernel compile and are part of the kernel-cache key.

---

❓ **Q11** - **The naive driver is a new universal kernel, not the tiled one**: `drivers/naive.rs`
holds a new, deliberately simple kernel in the elementwise-kernel style: one thread per output
element, a loop over `k`, full layout addressing (signed strides, zero strides, and a **per-axis
batch stride list** so non-affine batches work), templated on `@TA@ @TB@ @TC@`. It accepts every
type triple with a CUDA arithmetic type and every layout, so nothing is ever refused with
`UnImplemented`. For `k`, batch and output it uses the existing plan tiers (T0/T1/T2) like the
elementwise kernels.

The cast rule is "cast each product to `TC`": `acc += (TC)a * (TC)b`, i.e. promotion by cast to the
output type, accumulating in `TC`. (Half output therefore accumulates in half on this driver,
unlike cuBLAS's 32F accumulate — documented as a fallback-accuracy note, and the reason the
accelerated drivers come first.)

➡️ Yes. Keeping it trivially correct is the point: it is the oracle and the universal fallback,
and it does not depend on the tiled kernel's review.

---

❓ **Q12** - **Accumulator type of the naive kernel**: give the kernel a fourth template
parameter `@TACC@` chosen by the output type (`f32` for `f16`/`bf16` output, else `TC`), so the
naive fallback matches the cuBLAS default accumulation for half at no API cost? Alternative: no
`TACC`, always accumulate in `TC` (one fewer moving part).

➡️ Yes to `TACC`: one extra type name in the template, and it removes the one place where the
fallback would be visibly less accurate than the driver it replaces.

---

❓ **Q13** - **Order of the chain and the tiled kernel's place**: full order is
**cuBLASLt > cuBLASEx > cuBLAS > tiled > naive**. The tiled kernel stays in the default chain for
`f32`/`f64` (it is today's fallback and `test_matmul_tiled` grades it against the CPU); each of its
two refusal cases (no unit-stride axis, non-affine batch) now lands in the naive kernel instead of
an error. Its correctness/efficiency review (and its promotion to `TA/TB/TC` template) is recorded
as a **separate follow-up task**, not done here.

➡️ Yes.

---

❓ **Q14** - **Dtype promotion (`DeviceExtMatMulAPI`) in scope, naive only**: implement
`ext_matmul_uninit` for CUDA through the naive kernel with in-kernel casts (no materialized cast
tensors), for every promoted pair (`f16 x f32 -> f32`, `i32 x f64 -> f64`, `f32 x f64 -> f64`, ...).
Accelerated drivers serve same-dtype operands only, so an `ext_matmul` call always runs naive
until a later wave adds a "promote-then-GEMM" path (cast to common type in a kernel, then Lt).

➡️ Yes: it is nearly free once the naive kernel is three-typed, and it gives `rt::ext_matmul` a
correct CUDA answer today. Mixed-operand GEMMs are a single-tenant workload for now.

---

❓ **Q15** - **Tensor-level entries**: an extension trait in `rstsr-cuda` (core cannot name CUDA
modes; same pattern as `TensorPinnedHostViewAPI`), `src/matmul/tensor.rs`:

```rust
pub trait TensorCudaMatMulAPI<R, TC, B, D> { /* ... */ }
// c.matmul_from_mode(&a, &b, alpha, beta, mode)   // GEMM form; beta = 0 never reads c
// a.matmul_with_mode::<TC, DB, DC>(&b, mode)      // allocating, output dtype by turbofish
// a.matmul_with_output_mode(&b, &mut c, mode)     // alpha = 1, beta = 0
```

`mode: <TC as CudaMatMulMode>::Mode` with a public, tiny marker trait `CudaMatMulMode`
(`type Mode`), implemented for `f32, f64, Complex<f32>, Complex<f64>, f16, bf16`.

➡️ Yes, with those three names (`*_mode` suffix, next to the existing `matmul_from` /
`matmul_with_output`). `matmul_with_mode` allocates like `rt::matmul`, but `TC` cannot be inferred
from `Mul` there, hence the turbofish.

---

❓ **Q16** - **Flags refactor — per-device runtime flags**: `CudaRuntimeFlags` becomes

```rust
pub struct CudaRuntimeFlags {
    pub compute_type_32f: ComputeType32F,          // existing
    pub compute_type_64f: ComputeType64F,          // Q3 round 1
    pub compute_type_16f: ComputeType16F,
    pub fixedpoint_mantissa_bits: u32,             // only read by EmulatedFixedPoint
    pub matmul_drivers: MatMulDrivers,             // enable set, see Q17
    pub trace_matmul: bool,                        // per-launch eprintln of driver + compute type
}
```

and the **three op-behaviour environment variables are removed**
(`RSTSR_CUDA_MATMUL`, `RSTSR_CUDA_DEBUG_GEMM`, `RSTSR_CUDA_COMPUTE_TYPE_32F` seeding); tests set
flags programmatically (`set_flags_override`). The **four kernel-build variables stay environment
variables** (`ARCH`, `STRICT_FP`, `INCLUDE`, `CACHE*`): they are process-global build inputs and
part of the PTX cache key, so a per-device value would silently poison the cache.

➡️ Yes.

---

❓ **Q17** - **Driver enable set**: a `#[non_exhaustive]` bitflag-style value

```rust
#[non_exhaustive]
pub struct MatMulDrivers(u8);
impl MatMulDrivers {
    pub const CUBLAS_LT: Self; pub const CUBLAS_EX: Self; pub const CUBLAS: Self;
    pub const TILED: Self; pub const NAIVE: Self;
    pub const fn all() -> Self; pub const fn contains(self, d: Self) -> bool;
    // union by BitOr, like std bitflags
}
```

Selection = the static ordered table filtered by `device.flags().matmul_drivers`. "Force one
driver" is "enable only that bit". If the chain ends with nothing able to serve (naive disabled, or
the forced one cannot), it is an **error naming the driver and the type triple** — not a silent
promotion. Default: `all()`.

➡️ Yes. A single concept (an enable set) covers both "disable Lt to compare" and "force tiled",
and the static table keeps the order.

---

❓ **Q18** - **The static driver table** (per output type and mode; same-dtype operands unless
stated). Confirm or correct:

| call | 32F/64F/16F mode | candidates in order |
|---|---|---|
| `f32 -> f32` | any 32F mode | Lt, Ex, cuBLAS (typed + math mode), tiled, naive |
| `f64 -> f64` | `Default`/`Pedantic` | Lt, Ex, cuBLAS, tiled, naive |
| `f64 -> f64` | `EmulatedFixedPoint` | **cuBLAS only**, then tiled, naive |
| `c32`/`c64` | `Default`/`Pedantic` | Lt, Ex, cuBLAS (Cgemm/Zgemm), naive |
| `f16 -> f16` | 16F `Default` (32F acc) / `Pedantic` / `Acc16F` | Lt, Ex, naive |
| `f16 -> f32`, `bf16 -> f32`, `bf16 -> bf16` | 32F `Default`/`Pedantic` | Lt, Ex, naive |
| mixed operands (`ext_*`) | — | naive only |

("tiled" absent for half/complex because it is single-`@T@`; a mode the library rejects, such as
`bf16 -> bf16` with 16F accumulate, falls through to the next candidate at launch time.)

➡️ Yes, this table.

---

❓ **Q19** - **File layout** (`src/matmul/`, `mod.rs` holds only module docs, `mod` lines and
re-exports):

```
matmul/
  mod.rs          docs + mod/re-export only
  mode.rs         ComputeType32F/64F/16F, CudaMatMulMode, mode -> cublas compute/scale type
  flags.rs        MatMulDrivers (bit set) + moved CudaRuntimeFlags fields  [see note]
  geom.rs         GemmGeom, resolve_geom, linear_step (moved)
  cublas_args.rs  CublasArgs, args() (moved)
  inner.rs        matmul_with_mode_inner + DeviceMatMulAPI / DeviceExtMatMulAPI impls
  tensor.rs       TensorCudaMatMulAPI
  drivers/
    mod.rs        GemmDriver enum, the static table, the ordered dispatch loop
    cublas_lt.rs  handle + workspace + heuristic cache, launch
    cublas_ex.rs  GemmEx / GemmStridedBatchedEx
    cublas.rs     typed gemm + math mode / emulation knobs
    tiled.rs      the moved tiled kernel
    naive.rs      the new universal kernel
```

Note: `CudaRuntimeFlags` itself stays in `src/flags.rs` (it is device-level, not matmul-only);
only `MatMulDrivers` lives in `matmul/flags.rs`. `src/kernels/{gemm,cublas_gemm}.rs` are deleted
after the move.

➡️ Yes.

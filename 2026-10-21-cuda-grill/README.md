# CUDA backend grill — comprehension docs and synthesis

Task directory for the 2026-10-21 CUDA backend proposal (`initial-proposal.md`).
This README indexes the four comprehension documents produced against the
sources below and synthesizes the answers to the proposal's questions. Formal
grill rounds (decision stress-testing) have **not** started; §4 proposes the
topics they should cover.

| Datum | Value |
|---|---|
| rstsr master analyzed | `1c3bc11` (2026-10-20, post-#163) |
| CuPy studied | 14.3.0.dev0 @ `8f7a35d16` (2026-10-19, main) |
| cudarc studied | 0.19.10 @ `ec1cac7` (2026-09-24); candle 0.11.0, dfdx 0.13.0, cubecl 0.11.1, oxicuda 0.1.3, CUTLASS 4.8.0 |
| Workload | `/home/a/rest_pack/rest/src/ri_pt2/py/dfmp2_addons.py` (PyTorch DF-MP2, single/multi-GPU, no autograd) |
| Local GPU | RTX 5070 Laptop, 8 GB, **sm_120** (Blackwell), driver 595.91.07, CUDA toolkit 13.2 |
| Maintainer constraint | **cudarc is the sole external CUDA dependency**; other crates are references only; gaps go through rstsr-ffi-style bindings of our own |

## 1. Document index

- **[01-rstsr-backend-architecture.md](01-rstsr-backend-architecture.md)** — the
  full backend contract: trait-surface census (~90 device trait names, 223 `rt::`
  functions), device object model, storage/`Raw` analysis, element-bound
  inventory, host-synchrony inventory, threading internals, OpenBLAS-crate
  integration template, and the **restrictive-assumption inventory R1–R26**.
- **[02-cupy-architecture.md](02-cupy-architecture.md)** — CuPy's architecture
  vs NumPy: layering, eager-host/async-device execution, memory pools (arena per
  stream), kernel-as-data machinery (NVRTC, specialization axes, cache keys),
  BLAS handle management, TF32 handling, dtype coverage; 18 mapped lessons.
- **[03-rust-gpu-ecosystem.md](03-rust-gpu-ecosystem.md)** — cudarc 0.19.10
  coverage and precise gaps (all fillable via `result`/`sys`), candle/dfdx/
  oxicuda as references, cubecl 0.11.1 status, NVRTC-vs-cubecl verdict, and the
  M0–M6 build-first sketch with validation experiments.
- **[04-dfmp2-workload-gap-analysis.md](04-dfmp2-workload-gap-analysis.md)** —
  capability table: every torch idiom in the workload vs current rstsr master;
  precision/TF32 design; async semantics; multi-GPU mapping; ordered blocking
  gaps and the estimated `rt::cuda::` surface.

## 2. Answers to the proposal's questions

**Is the framework too restrictive for a GPU backend?** No — the architecture is
genuinely extensible where it matters. `DeviceRawAPI::Raw` is an associated
type, the compute path never inspects it, the device lives per-tensor in
`Storage` (stateful, cloneable — `DeviceCpuRayon` already holds an
`Arc<ThreadPool>`), element-type bounds sit on device impls, and
TypeId/dtype-dispatch plus f16/bf16 in `rstsr-dtype-traits` already exist
(doc 01 §4, §9 positives). The real restrictions are four (doc 01 §9):

- **R1/R2 — host-slice element access**: `iter`/`Index`/`as_ptr` are gated on
  `Raw = Vec<T>`; `Index` returning `&T` is structurally impossible for device
  memory. Blocks *test parity* (8/85 shared core_func files use `a[[i,j]]`),
  not the compute path. Candidate fix: `TensorAny::get/set` + migration.
- **R10 — BLAS driver tier**: `BlasDriverBaseAPI` pins `Raw = Vec<T>` +
  rayon, so `rstsr-blas-traits`' 14 driver traits cannot be implemented for
  CUDA. cuBLAS/cuSOLVER need either a generalized driver layer or a separate
  surface (`DeviceMatMulAPI` route + new linalg traits).
- **R12/R13 — host-closure ops**: `map`/custom-reduce/sort-custom take host
  `Fn`; impossible as kernels. Omit-on-CUDA (compile error at call site) vs
  host round-trip is a policy decision.
- **R15 — fancy indexing perf**: index resolution is an O(n) D2H+H2D
  round-trip; the `ArrayAuxIndexer` design already anticipates device-side
  resolution.

Plus the API-level holes found from the workload side (doc 04 §5): no
tensor-level `astype` (G-007, hot-loop for the fold), `DeviceChangeAPI` is a
CPU↔CPU relabel (no `.cpu()`, and `Display` needs it), no stream/sync API, no
`memory_info`/`device_count`, no math-mode concept, `DeviceAPI: Default`
forces an ordinal-0 default.

**Do we need lazy/streaming evaluation?** No. CuPy — the reference GPU-NumPy —
is eager on the host and asynchronous on the device, with no lazy engine
anywhere; even its `fuse` is an opt-in per-signature trace, not a lazy mode
(doc 02 §2, §4). rstsr's eager API can sit unchanged on async kernel launches:
validate/broadcast → pool-allocate → enqueue → return, synchronizing only at
the enumerable host-read points (`to_vec`/`to_scalar`/scalar reductions/
`Display`/data-dependent shapes) (doc 01 §6, doc 04 §3). The one real hazard is
**Drop**: freed device memory must return to a **stream-ordered pool** (CuPy's
arena-per-stream invariant) or a drop can race a pending kernel. CUDA graphs
remain a later opt-in optimization, not a requirement.

**NVRTC (cupy way) vs cubecl — verdict on the initial lean: agree, with
nuance** (doc 03). NVRTC-compiled CUDA C++ is more flexible (arbitrary source,
intrinsics, CUTLASS headers includable, per-dtype/per-layout specialization),
debuggable (source-level Nsight), and its PTX is on par with nvcc; cubecl's
wins are autotuning and vendor portability, not codegen quality, and it is
self-described alpha whose CUDA backend itself sits on cudarc. GEMM's ceiling
is cuBLAS/Lt either way. The cost of NVRTC is engineering, not performance:
kernel-source generation, typed launchers, a source+options+arch-hashed disk
cache, and a caching allocator — all first-class work items. **cudarc-only is
viable**: every identified gap (Lt f64/compute-type control, module caching,
pool allocator, cuSOLVER wrappers) is fillable through its `result`/`sys`
layers without a second dependency.

**TF32 dynamically, not by env var — yes, and the mechanism is known**: prefer
per-call compute type (`cublasLtMatmul` computeType /
`CUBLAS_COMPUTE_32F_FAST_TF32`) over per-handle `cublasSetMathMode`; cupy's
process-global `CUPY_TF32`/`set_compute_type` is exactly the anti-pattern to
avoid (doc 02 §5). Recommended shape: `DeviceCuda` shares
`Arc<CudaDeviceState>` with `math_mode: AtomicU8`, set via method + RAII
`with_math_mode` guard; `same_device` compares ordinal/stream but **not**
math_mode (doc 04 §2). Caution from doc 03: cudarc's cuBLASLt *safe* layer
hard-wires f32→`FAST_TF32` — for strict-f32-by-default we must issue Lt calls
through the `result` layer ourselves (which also unlocks f64 and heuristic
caching).

**Can dfmp2_addons.py be ported?** Not today, but the *computational* surface
is nearly complete (doc 04): the single-large-GEMM reshape trick is zero-copy
expressible, `sum_with_dtype::<f64>(([1,2], false))` is exactly torch's
f32-elements/f64-accumulate fold (already merged, PR #110), broadcasting/`.mT`/
`to_contig`/`into_shape` match, and `slice_mut` + `+=` make the whole assembly
doable on device. Blocking API items in order: the backend itself; tensor
`astype`; real cross-device `change_device` (also unlocks `Display`);
stream/sync; introspection; math-mode. The multi-GPU driver maps cleanly:
one `DeviceCuda` per thread/ordinal, results as device tensors, single
assembly sync.

## 3. Cross-doc synthesis — the emerging shape

- **Crate**: `crates-device/rstsr-cuda` on the OpenBLAS template (build/link,
  symlinked shared bodies after R1 migration), member-crate wiring into the
  facade as `rt::cuda::` (the `rt::dlpack` precedent), `Raw = CudaBuffer<T>`
  as a (ptr, len, alloc/stream) struct (R18), own caching pool on cudarc
  (arena-per-stream, doc 02 §3), NVRTC kernel layer with disk cache.
- **Default order row-major**; strict FP32 by default; f32/f64 first,
  f16/bf16 next (already first-class in dtype traits; no BLAS matmul path for
  them yet — cuBLAS(Lt) covers f16/bf16 on CUDA where the CPU BLAS tier does
  not — an rstsr first).
- **Async-eager**: per-op launches on the device's stream; documented sync
  points; `rt::cuda::synchronize`; host-index APIs (`index_select`,
  `searchsorted` sorter) get device-index overloads later (R16).
- **Testing**: no GPU CI vendor → T3/T4 manual/local per `rules/testing.md`;
  shared `core_func` bodies need the R1 `get/set` migration first; a
  CPU↔CUDA copy-and-compare parity harness is the practical conformance shape.

## 4. Proposed grill topics (for the decision rounds)

1. Crate placement and facade wiring (`rstsr-cuda`; `rt::cuda::` member crate).
2. `Raw` buffer contract: length-carrying handle, `device_ptr()`, element
   `get`/`set` API and test-body migration (R1/R2).
3. Allocator: own arena-per-stream pool vs `cudaMallocAsync`; injectable?
   (doc 02 §3, doc 01 Q3).
4. Kernel strategy: NVRTC runtime codegen now vs precompiled v1 subset;
   typed-launcher layer; disk-cache layout (doc 03, doc 02 §4).
5. Math-mode/TF32: shared `AtomicU8` per device + RAII guard; cublasLt via
   `result`; strict-f32 default; interaction with `same_device` (doc 04 §2).
6. BLAS/linalg tier: generalize `rstsr-blas-traits` off `Raw = Vec<T>` vs a
   separate CUDA linalg surface; cuSOLVER scope (R10, doc 01 Q5).
7. Host-closure ops policy: omit vs host round-trip (R12/R13).
8. Dtype set and dispatch for v1; `CudaScalar` local bound vs dtype tag
   (doc 01 §5, Q7).
9. Streams in v1: implicit per-device stream vs user-visible `Stream`;
   `same_device` policy (R22, doc 01 Q4).
10. Element access / indexing: `get/set` migration; device-side fancy-index
    resolution (R15/R16).
11. Test/CI strategy and tier assignment (doc 01 Q9).
12. Milestone ordering M0–M6 and the local-RTX-5070 validation experiments
    (doc 03).

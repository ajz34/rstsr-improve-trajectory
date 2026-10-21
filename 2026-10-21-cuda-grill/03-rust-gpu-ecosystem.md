# Rust GPU/CUDA ecosystem survey and kernel-technology strategy for RSTSR-CUDA

| | |
|---|---|
| Purpose | Verify or refute the maintainer's lean: "runtime-compiled kernels (NVRTC, the CuPy way) rather than CubeCL" for the planned RSTSR CUDA backend |
| Method | Read-only survey of local checkouts under `/home/a/Git-Others/`; web research restricted to CubeCL; no builds, no clones. All API claims below were checked against source (`file:line`), not documentation alone |
| Constraint adopted mid-run | `cudarc` is to be the **only external CUDA dependency** for `rstsr-cuda` if possible. candle / dfdx / oxicuda / CubeCL / CUTLASS are precedents, not dependencies |
| Local GPU | NVIDIA RTX 5070 Laptop, 8 GB, compute capability **12.0** (Blackwell, `sm_120`), driver **595.91.07**, CUDA toolkit **13.2** |
| RSTSR context | Eager, statically typed (`TensorAny<R, T, B, D>`), device traits per backend; existing CPU/BLAS devices; CUDA planned as a device crate |

## Versions and commits studied

| Project | Version / commit | Date | License | Role here |
|---|---|---|---|---|
| cudarc | **0.19.10**, `ec1cac71dab29498240704a9e4a264fbbec19495` (`v0.19.10`) | 2026-09-24 | MIT OR Apache-2.0 | Proposed sole CUDA dependency |
| candle | **0.11.0**, `12bb5d1f6350f4ea9827d9086f213feec1f0af6a` | 2026-10-14 | MIT OR Apache-2.0 | Build-time-nvcc precedent |
| candle-kernels | 0.11.0, `cudaforge 0.1.2` build dep | 2026-10-14 | MIT OR Apache-2.0 | Kernel build pipeline precedent |
| dfdx | **0.13.0**, `4722a99d303f347d6088d95867d007c75ca6dd78` | 2024-01-25 | MIT OR Apache-2.0 | Stale nvcc + cudarc-0.9 precedent |
| oxicuda | **0.1.3**, `b9a18f44a70fe92eedd94d37165c5e11ccc954b7` | 2026-04-17 | Apache-2.0 | Pure-Rust PTX-DSL alternative |
| CUTLASS | **4.8.0** (`v4.8.0-10-g78e46228`), `78e46228a1bfc5c972e2f9a634223af784cd6949` | 2026-10-14 | BSD-3-Clause | Hand-GEMM source, note only |
| CuPy (model reference) | `8f7a35d1613d06cc1d885b92a80750d9bc817805` | 2026-10-19 | MIT | NVRTC kernel-cache model |
| CubeCL | **0.11.1** (crates.io, 2026-10-20); repo `main`; `cubecl-cuda` depends on `cudarc ^0.19.6` | 2026-10-20 | MIT OR Apache-2.0 | Kernel-language alternative |

---

## 1. Local crate surveys

### 1.1 cudarc 0.19.10 — the substrate

**Scope and layering.** Every FFI area is exposed at three levels: `sys` (bindgen), `result` (per-call `Result`), `safe` (RAII). `cudarc/src/lib.rs:1-100` lists driver, NVRTC, cuRAND, cuBLAS, cuBLASLt, NCCL, cuDNN, cuSPARSE, cuSOLVER, cuFILE, CUPTI, NVTX, cuFFT, cuTENSOR. The default feature set is `std, cublas, cublaslt, curand, driver, runtime, nvrtc, fallback-dynamic-loading` (`cudarc/Cargo.toml:32-43`). With that default, **no CUDA library or header is needed at build or link time**; symbols are `dlopen`-ed at runtime (`cudarc/src/driver/sys/mod.rs:8-12`, `cudarc/build.rs:96-104`). One sharp edge: `build.rs` still *requires* a CUDA-version feature (`cuda-13020`, `cuda-version-from-build-system`, …) and panics otherwise (`cudarc/build.rs:112-127`). Downstream must pin one; a conservative pin such as `cuda-12080` still runs on the 13.2 driver (driver API is backward compatible), while `cuda-13020` matches the local toolkit.

**Driver / context.** `CudaContext::new(ordinal)` retains the device primary context (`cudarc/src/driver/safe/core.rs:74`); `new_non_primary` and `new_cig` also exist. `CudaContext` is `Send + Sync` and binds itself to the calling thread before every call (`core.rs:29-70`), so it can be shared across threads but work submitted from different threads lands on the same context. Device enumeration is `CudaContext::device_count()` (`core.rs:279`), per-device info via `compute_capability()` (`core.rs:302`), `name()`, `total_mem()`, `uuid()`, `mem_get_info()`. Multi-GPU is plain ordinals; cross-device `memcpy_dtod` automatically takes the `cuMemcpyPeerAsync` path when the contexts differ (`core.rs:1750-1790`).

**Streams and events.** `default_stream()` is the legacy null stream (`core.rs:682`); `per_thread_stream()` wraps the `0x2` sentinel (`core.rs:691`) — note the CUDA semantics: the sentinel resolves *per calling thread*, so sharing one `Arc<CudaStream>` across threads silently gives each thread a different stream. `new_stream()` / `new_stream_with_priority()` create non-blocking streams (`core.rs:726`, `:748`). `CudaEvent` supports record/wait/synchronize/elapsed/query (`core.rs:532-645`), and `CudaStream::fork/join` are convenience wrappers. Event tracking is automatic: `CudaSlice` carries optional read/write events and `LaunchArgs` inserts cross-stream waits when the context is in multi-stream mode (`core.rs:457-500`, `launch.rs:100-160`); it can be disabled unsafely for latency.

**Memory.** `CudaSlice<T>` is the `Vec<T>` analogue (`core.rs:843`). Allocation uses `cuMemAllocAsync` when the device supports memory pools, else synchronous `cuMemAlloc` (`core.rs:1623-1650`); `CudaSlice::drop` frees on its stream (asynchronously when pool allocation is supported, otherwise with a stream synchronize; `core.rs:853-880`). There is **no pooling allocator**: every `CudaSlice` is one driver allocation. Copies are stream-ordered: `clone_htod`, `memcpy_htod`, `clone_dtoh`, `memcpy_dtoh`, `memcpy_dtod` (`core.rs:1685-1800`), with `CudaView`/`CudaViewMut` sub-slices. Pinned host memory: `CudaContext::alloc_pinned` / `alloc_pinned_with_flags` return `PinnedHostSlice<T>` with a blocking event (`core.rs:1483-1520`). Unified memory: `alloc_unified` / `UnifiedSlice` (`driver/safe/unified_memory.rs`). Raw pool management (`cuMemPoolCreate/TrimTo/SetAttribute/AllocFromPoolAsync`) is exposed at the `result` level only (`driver/result.rs:1547-1640`), not integrated with `CudaSlice`.

**CUDA graphs.** `CudaStream::begin_capture` / `end_capture` / `CudaGraph::{launch, upload}` are present (`driver/safe/graph.rs:22-140`), with the instantiate-flags bitmask. Graphs are not thread-safe, and host/device copies cannot be captured naively — rstsr would have to pre-stage parameters (the same problem candle solves with an HTOD cache, `candle-core/src/cuda_backend/device.rs:38-52`).

**NVRTC.** `compile_ptx(src)` / `compile_ptx_with_opts(src, CompileOptions)` compile a CUDA C++ string to PTX (`nvrtc/safe.rs:100-130`). `CompileOptions` covers `ftz`, `prec_sqrt`, `prec_div`, `fmad`, `use_fast_math`, `maxrregcount`, `include_paths`, `arch`, plus an arbitrary `options: Vec<String>` (`nvrtc/safe.rs:231-300`). `Ptx` can also wrap a PTX file, PTX source, or **cubin binary** (`nvrtc/safe.rs:16-40`), and `CudaContext::load_module` dispatches to `cuModuleLoadData` for images/source/cubin and `cuModuleLoad` for files (`core.rs:2266-2295`). Gaps: the safe API only ever calls `nvrtcGetPTX`, so it cannot produce a cubin; `nvrtcGetCUBIN` exists only at `sys` level (`nvrtc/sys/mod.rs:216`). `CompileOptions.arch` is `Option<&'static str>`, so a runtime-detected `sm_120` must go through the `options` vector as a formatted string. There is no header-injection/Jitify layer; `include_paths` plus `options` (`--pre-include`) is the available mechanism. NVRTC's default arch is `compute_75`, not the current device, so arch must always be set explicitly (NVIDIA NVRTC 13.4 docs, supported options).

**cuBLAS.** `CudaBlas::new(stream)` creates a handle bound to one stream (`cublas/safe/mod.rs:27-60`); `set_stream` is `unsafe` and mutates the handle, so concurrent use needs one handle per stream. `Gemm<T>` is implemented for **f16, bf16, f32, f64** (`cublas/safe/gemm.rs:63,143,226,288`), plus strided-batched variants; `Gemv` and `Asum` exist. f32 uses `cublasSgemm` (strict FP32, no TF32); f64 uses `cublasDgemm`; f16/bf16 use `gemm_ex` with `CUBLAS_COMPUTE_32F` (FP32 accumulate). There is no compute-type knob in the safe `GemmConfig`; TF32 for f32 must be requested by calling `cudarc::cublas::result::gemm_ex` directly with `CUBLAS_COMPUTE_32F_FAST_TF32` (signature at `cublas/result.rs:375-395`; `cublasSetMathMode` also exists in `sys`). A grouped-GEMM wrapper also exists (`cublas/safe/grouped_gemm.rs`).

**cuBLASLt.** `CudaBlasLT::new(stream)` allocates a per-handle workspace sized 32 MiB on Hopper+ and 4 MiB otherwise (`cublaslt/safe.rs:31-80`). `MatmulConfig` covers transposes, m/n/k, alpha/beta, leading dims, optional batch strides, bias stride, batch size (`cublaslt/safe.rs:289-310`), with `Activation::{Relu, Gelu}` epilogue fusion. But the high-level `Matmul<T>` trait is much narrower than the C API:

- f32 compute type is hard-wired to `CUBLAS_COMPUTE_32F_FAST_TF32` (`cublaslt/safe.rs:432-440`) — **using this impl silently enables TF32**, which is wrong as a NumPy-parity default.
- f16 and bf16 use `CUBLAS_COMPUTE_32F` (`:443-462`); **there is no f64 impl**.
- The matmul descriptor's scale type is hard-coded to `CUDA_R_32F` (`:362`), so f64 scales are impossible even at the result level without writing new descriptor code.
- The heuristic is queried **on every call** (`:370-385`); no algorithm is cached or selectable.
- `MatmulPreference` is private and only `MAX_WORKSPACE_BYTES` is set (`:246-265`). The `result` level does expose `create_matmul_pref` + `set_matmul_pref_attribute` and the `sys` enum includes `MATH_MODE_MASK`, `SEARCH_MODE`, `IMPL_MASK`, etc. (`cublaslt/result.rs:134-165`, `cublaslt/sys/mod.rs:733-746`), so a full wrapper is possible without leaving cudarc.

**cuSOLVER / others.** The `safe` cusolver layer contains only handle/params types (`cusolver/safe.rs:15-215`); `cusolver/result.rs` has five functions, all handle creation. The actual routines (`cusolverDnXgesvd`, `…potrf`, `…getrf`, `…gels`, and their `_bufferSize` companions) are fully bound in `cusolver/sys/mod.rs` (8,663 lines). cuSPARSE is `sys` + `result` only (`cusparse/mod.rs`). cuRAND has safe `fill_with_uniform/normal/log_normal` on a stream (`curand/safe.rs:30-100`). So a cudarc-only backend can reach every routine, but rstsr must write thin wrappers for cusolver/cusparse itself.

**Maintenance, license, maturity.** Authored by Chelsea Lowman; 0.19.10 tagged 2026-09-24 with an active log (CUDA 13.4 support, graph flags, pinned-allocation fix, raw `CUfunction` exposure in the last weeks). Dual MIT/Apache-2.0. Version cadence is roughly monthly with occasional breaking API changes across minors (the `CudaDevice` → `CudaContext`/`CudaStream` redesign is recent history), so rstsr should pin `=0.19.10` or a tight range.

### 1.2 oxicuda 0.1.3

A 28-crate, ~260k-line pure-Rust reimplementation of the CUDA stack (driver via `libloading`, memory, launch, runtime, PTX *text generation* DSL, autotuner, BLAS/DNN/FFT/sparse/solver/RNG, plus Metal/Vulkan/WebGPU/ROCm/LevelZero backends). It is not a binding layer: kernels are emitted as PTX text from Rust data structures (`crates/oxicuda-ptx/src/lib.rs:1-10`) and cached on disk by `(kernel_name, params_hash, sm_version)` (`crates/oxicuda-ptx/src/cache.rs:1-90`). It has no NVRTC and no cuBLAS; its GEMM is hand-written PTX (`crates/oxicuda-blas/src/level3/gemm_api.rs`).

Assessment: technically interesting as a *PTX assembler plus autotuner*, but wrong for rstsr as a dependency. It is at 0.1.x, explicitly awaiting "GPU hardware benchmark validation", its documented platform matrix is Linux/Windows with macOS compile-only, and it replaces rather than wraps the vendor libraries whose GEMM quality rstsr needs. The one idea worth borrowing is the disk-cached, parameter-keyed PTX cache (rstsr gets the same from NVRTC + a hash cache with less risk).

### 1.3 dfdx 0.13.0 (stale)

The CUDA backend is the classic build-time model: `build.rs` globs `src/**/*.cu`, queries `nvidia-smi` for the compute capability, clamps it to the newest `nvcc --list-gpu-code`, then runs `nvcc --ptx --default-stream per-thread -O3` per file (`dfdx-core/build.rs:77-200`). PTX is embedded with `include_str!(concat!(env!("OUT_DIR"), "/x.ptx"))` and loaded lazily per op through cudarc 0.9.15 (`dfdx-core/Cargo.toml:33`, `dfdx-core/src/tensor_ops/utilities/cuda_kernels.rs:63-80`). It has a small exact-size slice cache (`CachableCudaSlice`, `dfdx-core/src/tensor/cuda/device.rs:130-180`), a `par_stream` fork, and a cublas workspace mutex.

Its value is as a cautionary tale: the build hard-depends on `nvidia-smi` and `nvcc`, the target arch is frozen at build time (bad for a redistributed library), kernels are specialized by macro for a fixed dtype list, and the crate has been unmaintained since January 2024. The `--default-stream per-thread` flag is the one durable idea: device code should not assume the legacy default stream.

### 1.4 candle 0.11.0 — the main Rust precedent

**Authoring and build.** Kernels are `.cu`/`.cuh` files under `candle-kernels/src/` (11 modules: binary, unary, reduce, cast, fill, indexing, conv, sort, ternary, affine, quantized, plus MoE/GGUF variants). `candle-kernels/build.rs` uses `cudaforge 0.1.2` to compile them to PTX at build time (`-O3 -std=c++17 --expt-relaxed-constexpr`), writes a generated `ptx.rs` of `include_str!` constants, and additionally builds MoE/quantized kernels into a static `libmoe.a` linked with `cudart` — i.e. some kernels are AOT-compiled SASS, not PTX. The workspace enables cudarc with `cuda-version-from-build-system` and `dynamic-linking`, so a full CUDA toolkit is mandatory at build time (`Cargo.toml:48-60`). This is the largest build-friction difference from an NVRTC design.

**Loading and launching.** `CudaDevice` holds `Arc<CudaContext>`, one `Arc<CudaStream>`, a `ModuleStore` of lazily loaded `CudaModule`s, a custom-module map, `CudaBlas`, and `CudaRng` (`candle-core/src/cuda_backend/device.rs:59-70`). `get_or_load_func` lazily loads a module's PTX and caches the module (`device.rs:339-357`); `get_or_load_custom_func` does the same for user-supplied PTX strings (`device.rs:314-337`). Launches are hand-built argument lists via cudarc's `LaunchArgs` with a `SlicePtrOrNull` parameter cache for strided metadata (`cuda_backend/mod.rs:1150-1185`). `BackendDevice::new` uses `context.per_thread_stream()` (`device.rs:396-398`).

**Coverage and limits.** Dtypes: u8/u32/i16/i32/i64, f16/bf16/f32/f64, FP8 E4M3, MX FP6/FP4, FP8 E8M0 (`dtype.rs:9-38`). GEMM goes through cublas (`gemm_strided_batched_{f16,bf16,f32,f64}` with a process-global "reduced precision" switch, `cuda_backend/mod.rs:2552-2660`). Reduction covers sum/min/max/argmin/argmax (`op.rs:20-26`). Layout handling is explicit: contiguous fast paths, `copy_strided_src` materialization otherwise (`cuda_backend/mod.rs:2362-2510`).

What is missing for a general NumPy-like tensor library: no runtime kernel generation, hence no fusion and no user-defined kernels except precompiled PTX; every op/dtype combination is a build-time decision; no kernel autotuning (cuBLAS defaults only); one stream per device (no user-visible stream API); no general allocator or size-class pool (raw `CudaSlice` per storage, with a `ParamCache` only for kernel metadata); no complex dtypes; no general `einsum` (composed from matmul/reshape); many ops (sort, conv, indexing) are special-cased rather than generated. Its `CustomOp` trait can only supply CPU/Metal/CUDA implementations written in Rust, with the CUDA path still needing a precompiled PTX module. In short, candle is an excellent proof that cudarc + PTX modules + cuBLAS is sufficient for a real DL runtime, and equally good evidence of the ceiling a fixed build-time kernel set imposes.

### 1.5 CUTLASS 4.8.0 (note only)

NVIDIA's C++ header/template library, BSD-3-Clause, actively developed (SM110a changes in October 2026). Relevant only as a source of hand-tuned GEMM/conv kernels if cuBLAS/Lt ever proves insufficient, or as headers included by NVRTC-compiled source (`include_paths` supports this). It is not a Rust dependency and should not be one; the correct use is "compile a CUTLASS-based kernel through NVRTC when a specific shape/precision needs it", or offline nvcc + cubin.

### 1.6 CuPy's NVRTC model (the design being copied)

CuPy is the most complete existence proof of the NVRTC approach at NumPy scale. Its compiler layer chooses the target from the device's compute capability, capped by what the installed NVRTC supports (`cupy/cuda/compiler.py:174-226`), JIT-compiles through NVRTC, and caches the *binary* (cubin), not the PTX, in a pluggable cache backend (`compiler.py:718-781`). The cache key is a hash of source, options, headers, arch, and CuPy's own cache-key version; modules built with `name_expressions` store the mangled kernel names ahead of the cubin so repeated lookups do not recompile (`compiler.py:545-620`). Higher-level constructs (`ElementwiseKernel`, `ufunc`, `RawKernel` in `cupy/_core/_kernel.pyx:901,1282`) are source generators over that one compile-and-cache primitive.

Three lessons transfer directly: (a) cache the compiled artifact on disk, not just modules in memory; (b) put the arch and the full option set in the key; (c) make the generated source deterministic and hashable, so identical kernels from different processes share a cache entry. RSTSR does not need CuPy's `name_expressions` complexity initially, but the key design and the "one compile primitive, many generators" layering should be copied.

---

## 2. CubeCL 0.11.1

**What it is.** A Rust kernel language plus JIT and runtime collection: kernels are ordinary Rust functions annotated `#[cube]`, expanded by macro into CubeCL IR (now built on `pliron`), then lowered to LLVM/PTX (CUDA), LLVM/AMDGPU (HIP), MSL, SPIR-V, WGSL, or native CPU. `cubecl-cuda` depends on `cudarc ^0.19.6` and executes even externally supplied PTX through it. Version 0.11.1 was published 2026-10-20; 0.11.0 (2026-10-06) was a large refactor ("Make everything Pliron", new frontend, `Client`/`Server` rename) with a CUDA LLVM backend added in the pre-releases.

**Execution model.** CubeCL itself is a *kernel launch layer*: a `Client` launches a `CubeKernel`; `Handle` names device memory. It is not an eager tensor library and does not itself decide eager vs lazy. The "both models" impression comes from Burn: Burn's API is eager, while its fusion backend defers operations into segments, fuses elementwise ops, JIT-compiles and autotunes the fused kernel, then executes it — "eager API with lazy, dynamic fusion" (Tracel blog, *Fusion Tensor Operation Streams*). That fusion layer is Burn code (`VirtualTensor`), not part of CubeCL core, and as of the article it covered elementwise ops, not matmul/reduction/conv.

**JIT and caching.** Only launched variants are generated; compilation is cached in memory and in a named "environment", and `cubecl::environment::bundle()` ships a pre-warmed compilation/autotune cache. `dry_run` compiles and tunes without executing. This is functionally the same caching problem rstsr would solve with NVRTC, but CubeCL already ships the machinery.

**Autotuning.** A first-class framework: runtime micro-benchmarks select kernel variants and comptime specializations; 0.11 added an adaptive scheduler with early elimination, roofline bounds, and observability.

**Backends and portability.** CUDA, HIP/ROCm, Metal, Vulkan/WebGPU (via wgpu), and CPU. This is CubeCL's real differentiator: one kernel source across vendors, including a non-NVIDIA path that cudarc cannot offer.

**Users and maturity.** Burn is the primary production user; CubeK provides higher-level tiling. The project self-describes as **alpha**, with breaking changes expected between minor versions and advice to pin. This matters for a library like rstsr whose device layer is a public integration surface.

**Performance evidence.** The vendor post *State-of-the-Art Multiplatform Matrix Multiplication Kernels* (2025-07-18) benchmarks square matmuls 512³–8192³ on RTX 4080, RTX 3070 Laptop, Radeon 8060S/7600, and M2 Pro. On the RTX 4080 it reports CubeCL's Simple/Multi-Row tensor-core variants generally beating the cuBLAS/CUTLASS reference; above 4096³ the baseline is CUTLASS rather than cuBLAS. Caveats stated by the authors: tensor-core variants only, square shapes, heavy shape-specific tuning, and "state-of-the-art results are currently achieved only in certain contexts rather than universally". No independent reproduction was found. For memory-bound elementwise/reduction code there is no public head-to-head against handwritten CUDA.

**Debugging.** `cubecl-inspect` exists for IR inspection and `debug_print` for device-side printing, but the artifact under Nsight Compute/cuda-gdb is generated PTX from an IR, not CUDA C++ with a source mapping to Rust. Source-level profiling is therefore weaker than the NVRTC path. The `cuda-cpp` alternative compiler emits C++ (better tooling), but the default path is LLVM.

**Standalone use.** Yes: runtimes are independent of the frontend; a library can depend on CubeCL without a runtime; external PTX can be launched. But using it as an eager compute layer means adopting its client/server, handle, and pool model, or writing a tensor/dispatch layer on top — essentially what Burn did.

### 2.1 Why CubeCL's runtime is not a drop-in substrate

CubeCL's `Client`/`Server` architecture is asynchronous and queue-based by design: launches are submitted to a server, and memory is named by opaque `Handle`s whose pools belong to `cubecl-server`. RSTSR's device traits expect synchronous-looking, typed storage with `Result` returns and direct control over allocation, streams, and copies. Adopting CubeCL as the runtime would mean either (a) writing an adapter that maps every RSTSR storage operation onto handles and waits, which reintroduces the very launch/queue complexity the DSL was meant to hide, or (b) rewriting the device layer in Burn's model. Using CubeCL only as a *kernel compiler* (generate a function, launch it through cudarc) is possible but then it is a heavier, alpha-versioned replacement for NVRTC with a narrower language — not a win for the CUDA-only case.

---

## 3. Comparison

### 3.1 Options against the axes

| Axis | (1) NVRTC C++ strings on cudarc | (2) build-time nvcc + cudarc | (3) CubeCL | (4) hand PTX/cubin + CUTLASS |
|---|---|---|---|---|
| Runtime flexibility | **Highest**: generate per dtype/layout/op, fuse arbitrary expressions, embed user source, use any CUDA C++ (half/bf16 intrinsics, WMMA, inline PTX, CUTLASS headers) | Low: kernel set frozen at build; new dtype/op needs a rebuild; user kernels only if shipped as PTX | High within the DSL: comptime specialization, generics, fusion; no raw C++ escape hatch beyond `gpu_asm`/intrinsics | Highest for the specific GEMM, lowest overall |
| First-call latency / cache | NVRTC compile per variant (tens–hundreds of ms), driver PTX JIT on load; needs disk cache of PTX/cubin keyed by source+options+arch | Zero JIT; PTX embedded, module load only | JIT + autotune on first use, cached; bundle can pre-warm | Zero compile; module load |
| Authoring ergonomics (Rust) | Strings/templates; no Rust type checking of kernel bodies; generator code is Rust | C++ files, `include_str!`; ergonomic but static | **Best**: `#[cube]` Rust functions, traits, generics | Worst: assembly/C++ only |
| Type safety | Low at the Rust/kernel boundary (arg list is `unsafe`, see `cudarc/src/driver/safe/launch.rs:40-100`); recoverable with generated typed wrappers + arity/param checks | Medium: fixed kernel signatures, but still untyped cudarc launch | **Highest**: kernel body type-checked by rustc before codegen | Low |
| Performance ceiling | Same NVVM backend as nvcc when optimized; elementwise/reduction limited only by kernel quality; GEMM should still be cuBLAS | Same as (1) for the compiled set | Vendor claims cuBLAS-class matmul on some shapes; generic codegen; autotune closes gaps | CUTLASS/cuBLAS-class for GEMM |
| Tooling/debugging | **Best**: CUDA C++, `-lineinfo`, cuda-gdb, Nsight Compute source/SASS views; `printf`, compute-sanitizer | Same, but offline build | Generated PTX/IR; `cubecl-inspect`, `debug_print`; source mapping to Rust is indirect | PTX/SASS only |
| Portability | CUDA only | CUDA only | **Multi-vendor** (CUDA/HIP/Metal/Vulkan/WebGPU/CPU) | CUDA only |
| Downstream build friction | Driver + libnvrtc at runtime; no toolkit at build (cudarc dynamic loading); one `cuda-XXXX` feature must be pinned | Full CUDA toolkit + nvcc at build (candle's model) | Rust-only build, but LLVM/pliron toolchain and a large dependency tree; still needs CUDA driver at runtime | nvcc/CUTLASS at build |
| Maintenance risk | Low–medium: mature crate, but 0.x and API churn; NVRTC is a stable NVIDIA API | Low (no runtime compiler) | **High for rstsr**: alpha, breaking minors, large surface | Low (frozen artifact), high authoring cost |
| Licensing | MIT/Apache-2.0 | MIT/Apache-2.0 | MIT/Apache-2.0 | BSD-3 (CUTLASS) |
| Fusion / user kernels | Any fusion expressible in C++; user kernels are just more source | Fixed set; user kernels require shipping PTX | Fusion via Burn's layer, not core; DSL-constrained user kernels | None (fixed artifact) |
| Dtype coverage effort | One generator instantiation per dtype; new dtype is a template + test | New dtype means new .cu + rebuild | Generic over `CubeType`; some dtype work still needed | Per-GEMM instantiation |
| Eager fit (no graph) | **Direct**: one generated kernel per `(op, T, layout class)`; no graph | Direct but rigid | Requires adopting a kernel abstraction and possibly a fusion layer | Only for a GEMM escape hatch |

### 3.2 What cudarc 0.19.10 supplies vs. what rstsr must add

| Need | cudarc 0.19.10 status | rstsr-side work |
|---|---|---|
| Contexts, ordinals, multi-GPU | Ready: `CudaContext::new(ordinal)`, `device_count`, `compute_capability`, peer copies | Device registry keyed by ordinal/UUID; per-device pool and handles |
| Streams/events | Ready: default, per-thread, explicit non-blocking, priority, events, join/fork, auto event tracking | Stream policy (one explicit stream per device by default; never per-thread sentinel across threads); `rt::cuda::synchronize` |
| H2D/D2H/D2D | Ready: sync/async on stream, views, pinned staging | Async `outof_cpu_vec` variant; pinned-staging ring; layout-aware strided copies |
| Unified memory | Ready (`alloc_unified`) | Optional; not a substitute for a pool on an 8 GB laptop |
| Allocator / pool | **Missing at the tensor level**: raw `CudaSlice` allocations, raw `cuMemPool*` at `result` level | Build a size-class caching allocator (`PooledSlice` with custom `Drop`), per device, with a release threshold and a hard cap for 8 GB |
| CUDA graphs | Ready (capture/replay/upload) | Later: stable-pointer capture API, parameter pre-staging |
| NVRTC compile | Ready to PTX; cubin only via `sys` | Kernel-source generator, `sm_XX` selection, disk cache, module cache, error/log surfacing |
| Kernel launch | Ready but untyped (`unsafe`) | Generated typed launchers; param-size/ABI checks; `LaunchConfig` policy |
| GEMM f16/bf16/f32/f64 | Ready via cuBLAS (`gemm_ex`, `sgemm`, `dgemm`) | Layout mapping (row-major RSTSR ↔ column-major cuBLAS), batching, TF32 opt-in |
| GEMM via cuBLASLt | Partial: **f32 hard-wired TF32, no f64, scales fixed f32, heuristic per call, prefs private** | Use result/sys to set compute type (strict 32F vs FAST_TF32), cache algorithms, expose epilogue; fall back to cuBLAS for f64 |
| cuSOLVER | Bindings complete at `sys`; only handle wrappers | Write rstsr-owned thin wrappers (Cholesky/LU/SVD/eigh), workspace sizing |
| cuSPARSE | `sys`/`result` only | Wrappers if/when sparse ops are needed |
| cuRAND | Ready for uniform/normal f32/f64 | Seeding/offset policy for reproducibility |
| NCCL | Ready | Later, multi-GPU collectives |
| Module/PTX cache | **Missing** | In-memory + disk cache keyed by hash of source, options, arch, headers |

Nothing in this table requires a second crate: every missing high-level wrapper has a `sys` or `result` entry inside cudarc. If a symbol were genuinely absent, the fallback is an rstsr-owned FFI crate following the existing `rstsr-ffi` pattern (vendored header + bindgen + provenance), not candle or CubeCL.

### 3.3 Substrate decisions that hold regardless of kernel strategy

These are orthogonal to NVRTC vs CubeCL and should be settled before writing kernels:

- **Streams.** One explicit non-blocking stream per `DeviceCuda` is the simplest correct default for an eager library; expose `rt::cuda::stream`/`synchronize`. Avoid `default_stream()` (legacy null stream, implicit synchronization with other libraries) and avoid sharing the `per_thread_stream()` sentinel across threads (it resolves per host thread, so the "same" handle means different streams). If RSTSR later supports multiple streams, cudarc's automatic event tracking on `CudaSlice` inserts the necessary waits; disabling it is an unsafe performance lever, not a default. cuBLAS handles are bound to one stream, so a second stream needs a second handle (or an unsafe `set_stream`).
- **Multi-GPU.** Contexts are per ordinal and `Send + Sync`; `CudaContext::device_count()` + `new(ordinal)` is the whole model. Cross-device copies already use `cuMemcpyPeerAsync`. RSTSR's device value should carry the ordinal and a stable UUID (for cache/registry keys), and every allocation must be attributable to one device. Collectives wait for NCCL.
- **Host transfers.** Distinguish blocking APIs (`outof_cpu_vec`, `to_vec`) from an async API that takes a `PinnedHostSlice` and a `CudaEvent`. Without pinned staging, "async" H2D is synchronous in practice, which is exactly the DF-MP2 workload's observation about `torch.as_tensor`.
- **Unified memory.** `alloc_unified` exists but is not a fix for an 8 GB laptop: oversubscription hides OOM at the cost of unpredictable migration. Keep it out of the default allocator.
- **CUDA graphs.** Capture/replay is available and attractive for repeated eager sequences, but requires stable device pointers and pre-staged kernel parameters. Treat it as a later opt-in on top of the pool, never as a correctness dependency.
- **GEMM knobs.** The public device surface needs a TF32 policy (off/strict by default; opt-in fast), a reduced-precision-f16/bf16 policy, and a deterministic mode. cudarc's cublas handles the strict paths; cublasLt needs RSTSR's own descriptor code to avoid the hard-wired TF32 default. f64 always goes through `dgemm`, and cublasLt should not be used for it.
- **cuSOLVER later.** The bindings exist; the work is workspace sizing, row/column-major conversion, and error mapping. No new dependency is needed, but expect to write and test each routine against LAPACK reference results.

### 3.4 First-order cost model

| Cost item | NVRTC path | Build-time nvcc | CubeCL |
|---|---|---|---|
| Per-kernel engineering | Source template + cache key + typed launcher | C++ kernel + macro instantiation | Rust kernel + tuning hooks |
| New dtype | Template instantiation + tests | New .cu variants + rebuild | Mostly generic, less work |
| First call in a process | NVRTC compile + module load (cache miss) | Module load only | JIT + autotune (cache miss) |
| Warm call | Kernel launch + arg marshalling | Kernel launch | Launch through client/server |
| Distribution | Needs `libnvrtc` at runtime (ships with the CUDA toolkit; redistributable with an app) | Needs toolkit at build for consumers | Pure Rust build |
| Debug iteration | Edit generator, rerun; no C++ build step | Rebuild crate + kernels | Rebuild Rust; inspect IR |

The NVRTC path trades a one-time, cacheable compile cost and Rust-side type-safety work for runtime arch/dtype flexibility and the best debugging story. For a library that will be built once and run on unknown GPUs, that is the right trade; for a single-machine application with a fixed GPU, build-time nvcc would be simpler.

---

## 4. Recommendation

### 4.1 Primary path

**NVRTC-generated CUDA C++ on cudarc, with cuBLAS/cuBLASLt for GEMM and an rstsr-owned caching allocator.** This is option (1) plus the parts of (2) and (4) that are cheap to keep in reserve:

- `rstsr-cuda` depends on `cudarc` only, with `default-features = false`, features `["std","driver","nvrtc","cublas","cublaslt","curand","cuda-13020","dynamic-loading"]` (or a lower `cuda-12xxx` pin plus runtime NVRTC detection for wider distribution).
- Kernel families are emitted as CUDA C++ source from Rust templates: a common prelude (`rstsr_cuda_common.cuh` as a string constant) with strided-index helpers, dtype traits via preprocessor tokens, vectorized loads, and `extern "C"` entry points. One module per `(op, dtype, variant)` or one module containing all dtype variants of an op; both are cache-friendly, and a single module per op reduces `cuModuleLoad` count.
- Compile with explicit `--gpu-architecture=sm_<major><minor>` obtained from `CudaContext::compute_capability()`; `-lineinfo` in debug builds; `--use_fast_math` off by default and exposed as a runtime device flag; `-std=c++17`; optional `--extra-device-vectorization`.
- Cache: hash `(source, options, arch, cudarc version)` → PTX (or cubin via the NVRTC `sys` path) under `dirs::cache_dir()/rstsr/cuda/`; in-memory `HashMap<(kernel_id, dtype, arch), Arc<CudaModule>>`. This is the CuPy model (`cupy/cuda/compiler.py:718-781, 1061-1081`), including a disk backend and per-module multiple `name_expressions`.
- GEMM: cuBLASLt through `result` for f16/bf16 and for f32 with an explicit compute type (`CUBLAS_COMPUTE_32F` strict by default; `CUBLAS_COMPUTE_32F_FAST_TF32` only when the user enables TF32 on the device, matching the dynamic TF32 requirement in the DF-MP2 workload); cuBLAS `sgemm`/`dgemm` for f32/f64 fallback and small shapes; cache `(algo, workspace)` per `(dtype, m,n,k, layouts, arch)` instead of re-querying the heuristic every call.
- Memory: `PooledSlice` wrapping `CudaSlice<u8>`/typed slices with size-class bins, stream-ordered reuse, a configured release threshold, and a device-level cap. On an 8 GB laptop this is not optional: the target workload allocates multi-GB temporaries, and per-op `cuMemAlloc`/`Free` would dominate.

### 4.2 Fallback ladder

1. **cudarc `result`/`sys`**: use for cuBLASLt prefs/algo cache, cuSOLVER routines, cuSPARSE, and cubin extraction. Still one dependency.
2. **Build-time nvcc for selected kernels** (candle/dfdx model): if a kernel needs CUTLASS headers, a feature NVRTC lacks, or first-call latency is unacceptable, embed precompiled PTX/cubin via `include_str!` and load with `CudaSlice`/`load_module`. Keep it opt-in behind a feature so the default build needs no toolkit. This is a *technique*, not a dependency change.
3. **rstsr-ffi-style bindings** for any genuinely missing symbol, following the existing vendored-header/bindgen/provenance pattern.
4. **CubeCL** only if rstsr decides multi-vendor (AMD/WebGPU) is a product requirement, or if maintaining the kernel generator becomes a larger burden than adopting an alpha DSL. It would sit on the same cudarc runtime, so it is not an alternative substrate.

### 4.3 Verdict on "prefer NVRTC over CubeCL"

**Agree, with three corrections.**

1. *Flexibility*: correct, and stronger than stated. NVRTC accepts arbitrary CUDA C++ — `__half`/`__nv_bfloat16`, WMMA, inline PTX, CUTLASS headers via `include_paths`, and per-call generated source. CubeCL can express only what its DSL and intrinsics expose, and its escape hatches (`gpu_asm`) are narrower than "write CUDA".
2. *Performance*: only conditionally correct. For elementwise/reduction kernels the ceiling is set by kernel quality, and NVRTC emits the same NVVM/PTX backend as nvcc, so a well-written generated kernel is at parity with a hand-written one. For GEMM the ceiling is cuBLAS/cuBLASLt regardless of authoring model — CubeCL's published matmul wins come from autotuning and shape specialization, not from a faster compiler, and its own CUDA runtime uses cudarc. Do not expect NVRTC to beat cuBLAS on GEMM; do expect it to remove the fusion/launch overhead that dominates small eager ops.
3. *Cost*: the real trade is type safety and engineering. cudarc's launch API is untyped (`unsafe` arg list), so rstsr must generate typed launchers, validate argument counts/sizes, and own a kernel cache and error-log plumbing. CubeCL gives Rust-level kernel type checking and a ready autotuner, at the price of an alpha API, a large dependency tree, weaker source-level profiling, and a DSL that constrains codegen. For an eager, dtype-static, NumPy-parity library whose hard parts are layout/broadcast/dtype coverage, generated CUDA C++ is the better fit; the type-safety loss is containable because kernels are generated, not hand-written by users.

One further nuance: choosing NVRTC does **not** isolate rstsr from cudarc's gaps (allocator, cublasLt prefs, cusolver wrappers). Those must be built either way; CubeCL would hide some of them but would also impose its own client/server and pool model.

### 4.4 What to build first

Ordered so each step is independently testable on the local RTX 5070 (8 GB, `sm_120`, driver 595.91.07, CUDA 13.2):

1. **M0 — device, storage, memory (1–2 weeks).** `DeviceCuda` (`Arc<CudaContext>`, one explicit `CudaStream`, `CudaBlas`, `CudaBlasLT`, arch string, pool); `CudaStorage<T>` over `PooledSlice`; alloc/zeros/empty; `clone_htod`/`clone_dtoh` with a pinned staging path; `synchronize`; device query and `change_device` plumbing.
2. **M1 — NVRTC infrastructure.** Source builder + prelude, compile with explicit arch, in-memory and disk cache, module/function cache, typed launch wrappers, compile-error surfacing with the NVRTC log. Gate every later kernel on this.
3. **M2 — elementwise and copy.** Unary/binary/compare/where/cast, contiguous fast path plus a strided/broadcast variant; `assign`, `to_contig`, strided gather/scatter. This covers the bulk of RSTSR's op surface and validates the codegen/cache design.
4. **M3 — reductions.** sum/prod/min/max (axis and all), f64-accumulator variants, argmin/argmax; two-pass reduction for large axes; match RSTSR's `OpReduceAPI`/`OpReduceDtypeAPI` semantics.
5. **M4 — GEMM.** Row-major↔column-major mapping, batched/strided, cuBLASLt for f16/bf16/f32 with strict-vs-TF32 switch, cuBLAS sgemm/dgemm fallback, algorithm cache. Validate against faer/OpenBLAS on CPU and against PyTorch on the GPU (FP32 strict bitwise-tolerant, TF32 with a documented tolerance).
6. **M5 — creation/RNG and host scalars.** `arange`/`linspace`/`full`, cuRAND uniform/normal with offsets, `to_scalar` sync path.
7. **M6 — later.** cuSOLVER wrappers via cudarc `sys` (Cholesky/LU/SVD), CUDA graphs for repeated eager sequences, NCCL, unified memory for overcommit experiments.

Validation experiments to run before committing to details: (a) NVRTC compile latency and cached module-load latency per kernel on `sm_120`; (b) elementwise bandwidth vs PyTorch/CuPy on the same 8 GB laptop; (c) cuBLASLt strict-FP32 vs TF32 accuracy and throughput on the DF-MP2 GEMM shapes; (d) pool behavior under the workload's ~3.5 GB per-task allocations; (e) whether `cudarc 0.19.10` with `cuda-13020` builds and loads `sm_120` PTX under driver 595.91.07 (the version matrix in `cudarc/README.md` covers 13.0–13.4, and NVRTC 13.x supports `sm_120`; verify the exact `--gpu-architecture` spelling accepted by the installed NVRTC before coding the arch mapper).

### 4.5 Risks

- **cudarc 0.x churn**: pin exactly; budget for a migration at each minor.
- **cuBLASLt safe-layer narrowness**: the f32-TF32 default is a correctness trap; route all GEMM through rstsr's own wrapper from day one.
- **NVRTC first-call cost**: without a disk cache, every process pays compile latency; ship a cache from M1, not later.
- **8 GB memory wall**: pooling plus a hard cap and clear OOM errors; avoid unified memory as a silent fallback.
- **Arch coverage**: a redistributed binary must compile NVRTC output for the *runtime* GPU, not the build machine; this is exactly why the NVRTC path is preferable to candle/dfdx-style build-time arch freezing.

### 4.6 Evidence gaps

- No independent benchmark of CubeCL vs handwritten CUDA for elementwise/reduction kernels was found; the only public numbers are the vendor's matmul post, on square shapes, with tensor cores, and with a CUTLASS baseline above 4096³. Treat "CubeCL matches cuBLAS" as a vendor claim, not a settled fact.
- No measurement was possible here (no builds allowed). NVRTC compile latency, module-load latency, and the cost of a heuristic query per cublasLt call should be measured on the local RTX 5070 before sizing the cache and the GEMM wrapper.
- cudarc 0.19.10's `cuda-13020` feature is present in source, but the exact NVRTC arch string accepted by the installed 13.2 toolkit (`sm_120` vs `compute_120` vs `sm_120a`) should be verified with a one-line compile before the arch mapper is written. The NVRTC 13.4 documentation lists 120/120f/120a, and Blackwell consumer parts are `sm_120`; `sm_120a` is the architecture-specific variant and is not needed for portable kernels.
- Nobody has yet measured how much of RSTSR's op surface needs the strided/broadcast kernel variant versus a contiguous materialization. The DF-MP2 workload suggests broadcasting is common, so the strided path is worth building early, but the split should be measured.

## Sources

Local: `/home/a/Git-Others/cudarc` (0.19.10), `/home/a/Git-Others/candle` (0.11.0), `/home/a/Git-Others/dfdx` (0.13.0), `/home/a/Git-Others/oxicuda` (0.1.3), `/home/a/Git-Others/cutlass` (4.8.0), `/home/a/Git-Others/cupy` (`8f7a35d`).

Web: <https://github.com/tracel-ai/cubecl>; <https://docs.rs/cubecl/latest/cubecl/>; <https://docs.rs/cubecl-runtime/latest/cubecl_runtime/>; <https://docs.rs/cubecl-cuda/latest/cubecl_cuda/>; <https://crates.io/api/v1/crates/cubecl>; <https://github.com/tracel-ai/cubecl/releases>; <https://tracel.ai/blog/sota-multiplatform-matmul/>; <https://tracel.ai/blog/fusion-tensor-operation-streams/>; <https://docs.nvidia.com/cuda/nvrtc/index.html>.

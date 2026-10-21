# CuPy architecture: what changes when NumPy moves to the GPU

- **Subject**: CuPy `main`, commit `8f7a35d1613d06cc1d885b92a80750d9bc817805` (2026-10-19, merge of PR #10392).
- **Version**: `14.3.0.dev0` (`cupy/_version.py:3`); `git describe` = `v14.0.0rc1-1760-g8f7a35d16`.
- **Studied at**: `/home/a/Git-Others/cupy` (all paths below relative to this root).
- **v14 context**: NumPy-2/NEP-50 semantics; cuDNN support *removed* (`docs/source/upgrade.rst:15-25`);
  bfloat16 added as "minimal support" (`upgrade.rst:44-47`).

The centerpiece question: **which NumPy mechanisms cannot survive on a GPU, and what replaced them?** Short
answer: NumPy's synchronous, host-resident, precompiled C loops are replaced by (a) stream-ordered
asynchronous execution, (b) a pooling allocator, and (c) runtime-JIT CUDA C++ kernels specialized per
dtype/op/shape-class, fronted by a small Cython runtime. Everything below serves that claim.

---

## 1. Layering & package layout

```
 user code
  ├─ cupy/*.py, cupy/{linalg,fft,random}/, cupy/{_creation,_manipulation,_sorting}/   ← NumPy-shaped API
  │    thin Python: checks, dispatch, re-exports (cupy/__init__.py has ~430 re-export statements)
  ├─ cupy/_core/*.pyx                                        ← the engine (Cython/C++)
  │    core.pyx (ndarray, allocation, strides, get/set, DLPack/CAI, 3.4k lines)
  │    _kernel.pyx (ElementwiseKernel, ufunc, type resolution, source generation, 1.8k lines)
  │    _reduction.pyx; _routines_{math,logic,linalg,indexing,manipulation,sorting,...}.pyx
  │    _dtype/_scalar/_carray/_cub_reduction/_fusion_*
  ├─ cupy/_core/include/cupy/*.cuh                            ← device-side C++, injected into every
  │    carray.cuh (CArray/CIndexer), float16.cuh, bfloat16.cuh,   JIT-compiled kernel
  │    complex.cuh, atomics.cuh, _cccl/{cub,thrust,libcudacxx}, xsf
  ├─ cupy/cuda/*.pyx + cupy_backends/cuda/api/*                ← CUDA runtime layer
  │    device, stream, memory, pinned_memory, function, compiler, texture, graph, cufft, cub
  │    api/runtime.pyx, api/driver.pyx (soft-linked via ctypes function tables)
  └─ cupy_backends/cuda/libs/*.pyx + cupy_backends/cuda/*.h/.cu ← vendor libraries
       cublas, cusolver, cusparse, cusparselt, curand, nccl, cutensor, nvrtc, nvtx
       cupy_cub.cu / cupy_thrust.cu / cupy_cufftXt.cu (C++ bridges)
```

- **The public namespace is assembled from private modules.** `cupy/__init__.py` re-exports `cupy._core.core.ndarray`, `cupy._core._kernel.ufunc`,
  creation/manipulation/linalg routines. There is no single monolithic `multiarray`.
- **Ufunc/reduction objects are built at import time** from explicit dtype-loop tables (`_routines_math.pyx:919-946`): CuPy's
  replacement for NumPy's generated `loops.c`.
- **`cupy_backends` is the portability seam**: `cupy/cuda/__init__.py:41-104` lazily imports vendor wrappers; `api/runtime.pyx`/`driver.pyx`
  dispatch through `_softlink.pyx` (`ctypes.CDLL` + function-pointer tables), and HIP is a build-time variant.
- **`cupyx/`** is SciPy-shaped client code (`scipy.sparse/signal/ndimage`, `distributed` for NCCL, `jit`, `profiler`, `optimizing`,
  `cutensor`), not a second engine. cuDNN has no wrapper in v14.

---

## 2. Execution model: eager, asynchronous, stream-ordered

### 2.1 Streams

Each device's current stream lives in a *thread-local, per-device list* (`cupy/cuda/stream.pyx:14-33`). `get_current_stream()`
(`:87-97`) reads it, `Stream.use()` (`:253-262`) sets it, `with Stream():` push/pops it (`:217-228`). Defaults are
singletons `Stream.null` (legacy, ptr 0) and `Stream.ptds` (`:646-647`); `get_default_stream()` (`:73-74`) chooses from the
**import-time** flag `CUPY_CUDA_PER_THREAD_DEFAULT_STREAM` (`cupy_backends/cuda/stream.pyx:10-11`). Under PTDS the null pointer maps to `cudaStreamPerThread` (`:28-40`) and
the memory pool keys arenas per thread (`memory.pyx:877-889`). `Stream` also exposes priority, `record`/`wait_event`,
host callbacks, and capture (`:100-455`). A launch always targets one stream: explicit `stream=` > current
stream (`cupy/cuda/function.pyx:85-89`).

### 2.2 Eager but async

Every op is eager on the host, async on the device: validate/broadcast views → allocate
output from the pool → enqueue kernel/copy (`driver.launchKernel`, `cupy_backends/cuda/api/driver.pyx:273-286`) → return. `ufunc.__call__` (`_kernel.pyx:1376-1479`) and
`ElementwiseKernel.__call__` (`:990-1070`) do this inline. There is **no expression graph, no lazy node list**. CUDA's stream *is*
the queue; the only thing laziness could amortize is host launch overhead, which CuPy attacks with pooling,
kernel caching, and opt-in CUDA graphs.

### 2.3 Host synchronization

Host sync points are explicit and few: `ndarray.get()`/`asnumpy()` (default `blocking=True`) do async D2H + `stream.synchronize()`
(`core.pyx:2106-2216`, `cupy/__init__.py:825-864`); `item()/tolist()/tobytes()` call `get()` (`core.pyx:682-700`); `Device.synchronize()` (`device.pyx:199-207`); `Stream/Event.synchronize()`; `__getbuffer__`
syncs because host code will read (`core.pyx:455-479`); CAI-v3 consumption syncs the producer stream (`core.pyx:3357-3367`);
`MemoryAsyncPool.free_all_blocks` syncs before `cudaMemPoolTrimTo` (`memory.pyx:1903-1926`). `syncdetect` exists but is deprecated (unreliable detection,
`_core/syncdetect.py:11-70`). Consequence: **reductions return 0-d arrays, not scalars**, to avoid implicit D2H (`docs/source/user_guide/difference.rst:84-110`).

### 2.4 Events and graphs

`Event` is RAII (`stream.pyx:100-178`); CUDA graphs only via stream capture (`stream.pyx:349-424` → `cupy/cuda/graph.pyx:8-80`).
Capture forbids synchronous D2H and (in CuPy) cuBLAS calls (`cublas.pyx:642-656`). This is opt-in batching, not laziness.

### 2.5 Device selection

`cupy.cuda.Device(i)` is the context manager (`device.pyx:125-197`); there is no `using_device`. Arrays carry
`device_id` in `MemoryPointer`; kernels auto-enable peer access with a warning if operands live elsewhere (`_kernel.pyx:127-145`,
`device.pyx:90-107`). Per-device resources (handles, FFT plans, pools, compute-type table) are keyed by device id.

---

## 3. Memory management

### 3.1 Objects

`Memory` (`cudaMalloc`, `:105-124`), `MemoryAsync` (`cudaMallocAsync` on a stream, `:146-178`),
`UnownedMemory` (foreign ptr + owner, `:181-219`), `ManagedMemory` (unified, `:223-270`), `SystemMemory` (host `malloc` on
HMM/ATS, `CUPY_ENABLE_UMP=1`, `:274-354`), `PooledMemory` (a chunk; `free()` returns it, `:945-1038`). An `ndarray` holds
a `MemoryPointer`, never a bare pointer (`core.pyx:284-303`). Lifetime is still Python refcount, but deallocation is
**return-to-pool**: `PooledMemory.__dealloc__ → free() → pool.free(chunk)` (`:993-1038`); at exit `_exit_mode` suppresses it (`:45-52, 1035-1038`).

### 3.2 The pool

`cupy/__init__.py:904-908` installs `MemoryPool()` at import. `MemoryPool` is a façade over one `SingleDeviceMemoryPool` per device
(`:1566-1612`), which keeps **one `_Arena` per stream identifier** (`:1218-1292`; `_get_stream_identifier`, `:877-889`). Freed
blocks return to the arena of the stream they were freed on and are reused only when that stream is current —
this avoids cross-stream hazards without per-block events. Each `_Arena` is a size-ordered `std::set<(size, chunk*)>`:
best-fit via `lower_bound`, split, merge with free neighbours, and a "pending free" list so frees can proceed
without the mutex (`:1055-1199`). Sizes round to 512 B (`ALLOCATION_UNIT_SIZE`, `:40, 1042-1046`).

Pooling is essential because `cudaMalloc/Free` synchronize and cost microseconds–milliseconds; NumPy-style
per-temporary allocation would serialize the pipeline. CuPy's docs state both the speed benefit and the
"memory looks leaked" cost (`docs/source/user_guide/memory.rst:1-30`). Accounting (`:1464-1563`): a per-pool limit (`CUPY_GPU_MEMORY_LIMIT`, `set_limit(size|fraction)`) is
enforced by atomic CAS; on exhaustion it frees all blocks, retries, `gc.collect()`s twice, retries, then raises
`OutOfMemoryError(size, total, limit)`.

### 3.3 cudaMallocAsync

There is **no `CUPY_CUDA_ALLOC_CONF`**; the async allocator is opt-in via `set_allocator(MemoryAsyncPool().malloc)` (or `malloc_async`).
`MemoryAsyncPool` (`:1740-1926`) wraps one `cudaMemPool_t` per device (`'default'|'current'|int`), maps `set_limit` to `cudaMemPoolAttrReleaseThreshold` (`:1966-2040`), and
implements `free_all_blocks` as sync + `cudaMemPoolTrimTo(pool, 0)`. Requires CUDA ≥ 11.2 + `cudaDevAttrMemoryPoolsSupported` (`:127-142`); no memory hooks.
Reuse/ordering is the driver's job.

### 3.4 Pinned and managed memory

`PinnedMemoryPool` is a host-side per-size free-list pool, 512 B rounded (`cupy/cuda/pinned_memory.pyx:245-330`). H2D from a
fresh NumPy array is staged through pinned buffers so it is truly async; the buffer lives until the copy event
fires (`:159-190`), and `array`/`asarray` default `blocking=False` (`cupy/_creation/from_data.py:9-57`, `core.pyx:3100-3135`). Managed/system memory are
allocator choices with `prefetch`/`advise`; with HMM/ATS, `asnumpy` can return a zero-copy view (`core.pyx:2160-2170`).

### 3.5 Versus NumPy

NumPy: `malloc` per array, refcount free, host-only, no limit. CuPy: refcount still decides
*when*, but free returns to the stream's arena; memory is a scarce global resource with a limit, statistics
(`used_bytes/total_bytes/n_free_blocks`), hooks, and an OOM retry protocol.

---

## 4. Kernel machinery (the heart)

### 4.1 Elementwise kernels and ufuncs

An `ElementwiseKernel` is a **code template**, not a compiled kernel. Construction parses parameter declarations
(`ParameterInfo`, `_kernel.pyx:434-488`) like `'T in0, T in1'`; `T` is a placeholder C type resolved from actual dtypes at call
time (`_decide_params_type_core`, `:812-860`).

At each call (`:990-1070`): (1) `KernelArguments.create` normalizes args, converts CuPy-like objects (CAI), wraps scalars in `CScalar`,
handles `out=`/`where=` (`:493-560`); (2) broadcasting is applied as *views* (`:657-666`); (3) outputs are
allocated from the pool (`:668-695`); (4) `_reduce_dims` squashes contiguous/unit dims **before** codegen
(`:330-431`); (5) an `_ArgInfo` per argument records kind, Python type, dtype, **ndim, C-contiguity,
32-vs-64-bit indexing** (`:159-308`); (6) a memoized generator emits source and NVRTC compiles it
(`:868-898`).

Generated source (template `:78-107`), schematically:

```cuda
typedef float in0_type;                       // _TypeMap typedefs
extern "C" __global__ void cupy_add__float32_float32_float32(
    CArray<float, 2, true, true> _raw_in0,    // ndim, contiguity, 32-bit index
    CArray<float, 2, true, true> _raw_in1,
    CArray<float, 2, true, true> _raw_out0,
    CIndexer<2, true> _ind) {
  #pragma unroll 1
  CUPY_FOR(i, _ind.size()) {                  // grid-stride loop, carray.cuh:68-71
    _ind.set(i);                              // flat index -> multi-index
    const float &in0 = _raw_in0[_ind.get()];
    const float &in1 = _raw_in1[_ind.get()];
    float &out0 = _raw_out0[_ind.get()];
    out0 = in0 + in1;                         // user/ufunc expression
  }
}
```

`CArray<T, ndim, c_contiguous, use_32bit>` and `CIndexer<ndim, use_32bit>` are templates (`cupy/_core/include/cupy/carray.cuh:229-460, 519-625`). **ndim, contiguity, and index width are compile-time
constants** — hence dim squashing, and hence an array crossing 2 GiB changes the kernel binary (`core.pyx:280, 303`).

A **ufunc** is a family of such kernels plus a resolution table. `create_ufunc` registers an ordered loop list
(`'bb->b', 'BB->B', ... 'ff->f', 'dd->d', 'FF->F', 'DD->D'`, plus bool/fp16/bf16; `_routines_math.pyx:925-935`). `_Ops.guess_routine` (`_kernel.pyx:1734-1819`) walks it in order and takes the first loop
to which all inputs can safely cast (`numpy.can_cast`), with NEP-50 weak-scalar correction (`:1249-1279`). The loop
fixes compute/output dtypes; `fix_cast_expr` inserts casts (`:1140-1156`). `where=` becomes a guard + `_where`
parameter (`:1159-1235`). Kernel names embed arg dtypes, e.g. `cupy_add__float32_float32_float32` (+`_where`) (`:1481-1493`).

**Variant count**: per exercised loop, one binary per distinct `(arg kind, dtype, ndim, c_contiguous, index_32_bits)` tuple (× `where` presence) —
roughly `#loops × #shape-classes-per-arg × 2`. This is why `_reduce_dims` exists, why the in-process memo is global and keyed by `(device, arginfos, type_map)`
(`_util.memoize(for_each_device=True)`, `cupy/_util.pyx:39-73`; `_kernel.pyx:1083-1110`), and why custom kernels compile per dtype/ndim combination.

### 4.2 Reductions

`create_reduction_func`/`ReductionKernel` generate one kernel from `map_expr`, `reduce_expr`, `post_map_expr`, `identity` (`_reduction.pyx:40-122, 572-830`). The
template is a classic block reduction: each block owns `_block_stride` outputs; threads cooperatively walk the
reduce axis with a grid-stride loop; partials live in `__shared__` and combine in a power-of-two tree
(`:55-107`). **Axis handling is view permutation, not a general iterator**: reduce axes move to the front
(`_set_permuted_args`, `:175-199`), then `_reduce_dims` squashes (`:402-405`).

Launch geometry (`_get_block_specs`, `:235-249`): block size 512 CUDA / 256 HIP (`:228-232`); `block_stride` from contiguous
run length and reduce/output ratio; grid = `ceil(out_size/block_stride)`; 64-bit `IndexT` when the in/out ratio exceeds 2³¹
(`:354-358`). Accelerated paths tried first, silent fallback: CUB `DeviceSegmentedReduce` (`_cub_reduction.pyx:617+`; eligibility `:285-365`,
threshold 128 `:27-30`, block ≤512, 4 items/thread `:386-410`) and optional `cuda.compute` (`_reduction.pyx:372-396`).
Accelerator list: `CUPY_ACCELERATORS`, default `"cub"` (`_core/_accelerator.pyx:52-61`).

**No autotuning by default.** `cupyx.optimizing.optimize` is an opt-in Optuna context (≤100 trials, ≤1 s) over `(block_size, block_stride, out_block_num)`, cached
by shape/strides/dtypes (`cupyx/optimizing/_optimize.py:1-70`, `_reduction.pyx:443-480`).

### 4.3 Fancy indexing

Indexing splits into layout-only and kernel work (`_routines_indexing.pyx:33-57`): basic slicing is pure view algebra; a **single
integer-array index** computes strides + a linear `reduced_idx`, then runs one elementwise gather (`:560-585, 844-900`); a
**boolean mask** runs a prefix-sum over the mask, then `if (mask) out[mask_scanned-1] = a` (`:756-773, 775-843`; warp scan `:496-556`); **multiple
advanced indices** broadcast and fold into one flat index with `out += stride * (idx mod dim)`, then gather (`:1169-1219`);
**scatter/`ufunc.at`** writes `a[...]`, with `atomicAdd/atomicSub` variants (`:670-700`).

This explains the documented divergences: duplicate indices make the stored value undefined, and out-of-bounds
indices *wrap* instead of raising (`difference.rst:42-78`). NumPy's last-write-wins/bounds-error semantics are not
implementable without serialization.

### 4.4 Compilation, caching, first-call latency

**NVRTC is the default**, `nvcc` a fallback (`cupy/cuda/compiler.py:624-790`). No AOT kernel library: source is compiled at first
use. Options include `--std=c++17`, CuPy's include dir, the CUDA/ROCm include dir, bundled CCCL paths (`core.pyx:2589-2633`).
Arch is `-arch=sm_<cc>` (cubin) when possible, else `compute_<cc>` (PTX) (`compiler.py:208-236`). Env knobs: `CUPY_CACHE_DIR`, `CUPY_CACHE_SAVE_CUDA_SOURCE`,
`CUPY_CACHE_IN_MEMORY`, `CUPY_NVRTC_USE_PCH`, `CUPY_COMPILE_WITH_PTX`, `CUPY_CUDA_COMPILE_WITH_DEBUG`, `CUPY_DUMP_CUDA_SOURCE_ON_ERROR`, `NVCC` (`docs/source/reference/environment.rst`; `compiler.py:276-288, 637-641, 673-674`).

**Cache key** (`compiler.py:690-712`): SHA-1 over `(arch, options, nvrtc version, backend, preprocessed compiler banner, source, extra_source, CUPY_CACHE_KEY)`; `CUPY_CACHE_KEY` is a build-time checksum of bundled device headers
(`core.pyx:69-78`). `name_expressions` adds the sorted expressions plus a payload-format version. Disk filename = hash;
payload = cubin prefixed with its own SHA-1, and for `name_expressions` a header carrying NVRTC's lowered names
(`compiler.py:545-621`; `_compiler_cache.py:24-28, 89-98`). Default dir `~/.cupy/kernel_cache`.

**First-call latency is accepted, not hidden**: compilation is synchronous host work; docs advise warm-up,
persisting the cache across CI runs, and `cupyx.profiler.benchmark` with warm-up (`docs/source/user_guide/performance.rst:74-95`). Generated elementwise/ufunc
kernels are `extern "C"` with deterministic dtype-encoded names; `RawModule`/`cupyx.jit` use C++ template
instantiation and resolve names via `nvrtc.getLoweredName` (`compiler.py:855-876`). Jitify is deprecated, staged for removal in v15
(`compiler.py:328-343`).

### 4.5 Launch configuration

| Kernel kind | Block | Grid | Source |
|---|---|---|---|
| Elementwise/ufunc | 128 (default; `block_size=` override) | `ceil(N/128)`, grid-stride | `function.pyx:153-171`, `_kernel.pyx:1022, 1068` |
| Reduction generic | 512 CUDA / 256 HIP | `ceil(out_size/block_stride)` | `_reduction.pyx:228-249, 521-522` |
| Reduction CUB | ≤512, 4 items/thread | per segment | `_cub_reduction.pyx:386-410` |
| Integer matmul | 16×16 threads, 128×128 tiles | `ceil(m/128) × ceil(n/128)` | `_routines_linalg.pyx:384-433` |
| cuBLAS gemm | library-chosen | library-chosen (`DEFAULT`/`TENSOR_OP`) | `_routines_linalg.pyx:808-814` |

No occupancy query except cooperative launches (`function.pyx:110-124`).

Runtime env knobs touched above: `CUPY_CUDA_PER_THREAD_DEFAULT_STREAM` (§2.1), `CUPY_ACCELERATORS` (§4.2),
`CUPY_TF32` (§5.3), `CUPY_GPU_MEMORY_LIMIT` / `CUPY_ENABLE_UMP` (§3), `CUPY_CUDA_ARRAY_INTERFACE_SYNC` /
`CUPY_DLPACK_EXPORT_VERSION` (§2.3), plus the compile/cache set in §4.4 (`NVCC`, `CUDA_PATH`).

### 4.6 Sorting and searching

`sort`/`argsort`/`lexsort` are **Thrust**, not hand kernels: `cupy/cuda/thrust.pyx:86-160` forwards a runtime `dtype_id`
(`cupy/cuda/common.pyx:7-40`) to C++ bridges (`cupy/cuda/cupy_thrust.cu`). Multi-dimensional sorts stable-sort `(row_key, value)` tuples to emulate
per-axis segmented sort (`cupy_thrust.cu:108-210, 331-400`). `partition`/`argpartition` are hand kernels (`_routines_sorting.pyx:116-311`); `searchsorted` is an
`ElementwiseKernel` doing binary search per query (`cupy/_sorting/search.py:258-477`). CUB is used for reductions/scans, not sort.

### 4.7 `cupy.fuse`: an opt-in trace, not a lazy mode

`cupy.fuse` still exists, is exported (`_core/fusion.pyx:891-926`, `cupy/__init__.py:898`), and is maintained (commits through 2026-08): **not
deprecated**. It traces *one call*: a thread-local flag redirects ufunc/reduction calls to proxies
(`_fusion_thread_local.pyx:32-42`), the op list is compiled into one CUDA kernel (`new_fusion.pyx:25-35`, `_fusion_kernel.pyx:31-54`), cached per argument
signature (`fusion.pyx:806-889`). Limitations: simple elementwise/reduction only, no matmul/reshape (`docs/source/user_guide/kernel.rst:478-536`). Key
datapoint: **CuPy has no general lazy/streaming execution mode**; fusion is per-signature kernel
generation on top of eager execution.

---

## 5. BLAS & linear algebra

### 5.1 Handles

`Device._get_handle(name, create, destroy)` caches handles in **thread-local dicts keyed by device id**, created lazily with the
device made current (`cupy/cuda/device.pyx:233-310`); `get_cublas_handle()` (`:62-64`) is the only accessor. Handle count = threads × devices.
`cublas.create/destroy` are raw `cublasCreate/cublasDestroy` (`cupy_backends/cuda/libs/cublas.pyx:602-621`); `Handle.__dealloc__` destroys on thread teardown (`device.pyx:52-60`). **Stream binding
is per call**: every wrapper starts with `_setStream(handle)` → `cublasSetStream(handle, get_current_stream_ptr())` (`cublas.pyx:642-666`); cuSOLVER/cuSPARSE do the same
(`cusolver.pyx:1495-1530`). cuBLAS during capture is rejected (`cublas.pyx:642-656`); pointer mode is set explicitly where scalars are
on device (`cupy/cublas.py:397-416`).

### 5.2 matmul/dot dispatch

`matmul`/`dot` (`_routines_linalg.pyx:895+`) routes: **integer/bool** → tiled CUDA kernel (128×128 tiles, 16×16 threads), with
an int8 `gemmEx(CUBLAS_COMPUTE_32I)` fast path on sm_61+ when aligned (`:676-700, 384-433`); int8 output wraps mod 256. **fp16/bf16** →
`gemmEx` with `CUDA_R_16F/16BF` A/B/C and `CUDA_R_32F` compute (`:706-765, 780-854`). **f32/c64** → `COMPUTE_32F`; **f64/c128** →
`COMPUTE_64F` (older `sgemmEx/dgemm/cgemm/zgemm` fallbacks for CC<50/HIP). Algorithm: `CUBLAS_GEMM_DEFAULT_TENSOR_OP` on sm_80+ (sm_70+ for fp16/int8),
else `DEFAULT` (`:808-814`). cuBLAS is column-major, so operands are passed swapped (`tensordot_core_v11(transb, transa, ..., b, ldb, a, lda, c, m)`). Batched:
`gemmStridedBatchedEx`, or `gemmBatched` with a pointer array built by an elementwise kernel for broadcast batches
(`:1089-1140, 856-891`).

### 5.3 TF32

TF32 is a process-global, per-dtype, lazily initialized table. `compute_types = [TBD]*6` indexed by dtype class
(`:39-46`). `get_compute_type` (`:488-497`) reads `CUPY_TF32` **once per dtype** and stores the result permanently;
`set_compute_type` (`:80-99`) can also set it (TF32/BF16 require CC ≥ 8.0, else warn + DEFAULT). Modes are `_routines_linalg.pxd:21-27`.
The value is read at each gemm (`:785`) and maps to `CUBLAS_COMPUTE_32F_FAST_TF32` only for f32/c64 (`:791-795`). There is **no
per-call/stream/thread switch**; `CUPY_TF32` is documented as a global env var (`docs/source/reference/environment.rst:100-104`), and runtime
changes go through the global table. The legacy `CUBLAS_TF32_TENSOR_OP_MATH` math-mode enum is **not used**;
CuPy relies on the per-call compute type. fp16/bf16 tensor cores use a different mechanism: `setMathMode(CUBLAS_TENSOR_OP_MATH)`
toggled *around the call* on the legacy path (`:710-733`). `cupyx._cusolver` reuses the same table for `gesvdj`
(`cupyx/_cusolver.pyx:969, 1053`).

### 5.4 cuSOLVER / cuSPARSE

The style is hand-written `extern` declarations + `cpdef` wrappers with `check_status`.
Workspace follows *query-then-allocate* in Python: `*_bufferSize` → `workspace = cupy.empty(buffersize)` → call with `workspace.data.ptr` → read
`dev_info` (`cupy/linalg/_decomposition.py:77-100, 210-225, 340-390`). No workspace pool/cache. Since cuSOLVER lacks batched routines, `cupyx/_cusolver.pyx` loops over
matrices (`gesvd_loop_internal` etc., `:76-280`). cuSPARSE uses descriptor objects (`createMatDescr`, SpMat/SpVec; `cusparse.pyx:1272-1310`),
consumed by `cupyx.scipy.sparse`.

### 5.5 What matmul does not do

No cuBLASLt, no autotuning, no split-K, no epilogue fusion, no heuristic search
(only the fixed enum; `:384-393` even has a TODO about autotuning the integer kernel). No TF32 by default, no
fp32 down-cast, no workspace caching, no transposed-C (`beta=0`).

---

## 6. Dtype coverage

Core dtypes (`docs/source/overview.rst:13`; C-type map `_core/_scalar.pyx:16-30`):

| dtype | device C++ type | notes |
|---|---|---|
| `bool` | `bool` | bool loops use `|`/`&` for add/multiply |
| `int8..64`, `uint8..64` | `signed char` … `unsigned long long` | same-width loops; promotion by loop order |
| `float16` | `float16` over `__half` (`float16.cuh`) | arithmetic via `float`; bit-level `isnan/isinf`; `min/max` via float |
| `float32/64` | `float`, `double` | |
| `complex64/128` | `thrust::complex<float/double>` | bundled `complex.cuh` |
| `bfloat16` | `bfloat16` over `__nv_bfloat16` (`bfloat16.cuh`) | **v14**, via `ml_dtypes.bfloat16`, NumPy ≥ 2.1.2, "minimal" (`_scalar.pyx:278-292`, `upgrade.rst:44-47`) |
| structured/void | bytes via `CArray` | only field get/set; **any kernel launch fails** in v14 (`_dtype.pyx:52-74`, `upgrade.rst:49-58`) |
| `complex32` | recognized as `CUDA_C_16F` | not a NumPy dtype; cuFFT raises `NotImplementedError` (`cufft.pyx:1189-1191`) |
| object/string/datetime | — | rejected by `check_supported_dtype` (`_dtype.pyx:37-79`) |

Mechanics: `get_typename` maps dtype → C++ type + required header, and `format_type_decls` topologically sorts header deps
for deterministic source (`_scalar.pyx:16-30, 63-108`). bf16/fp16 headers are pulled in only when used. `cupy._util.bf16_loop(...)` expands to a
loop signature only when `ml_dtypes` imports (`_util.pyx:225-262`), so each ufunc explicitly inserts bf16 after fp16
(`_routines_math.pyx:925-935`). Promotion = ordered loop table + `numpy.can_cast` + weak scalars (`_kernel.pyx:1734-1819, 1249-1279`), with a `promote_types` callback
for e.g. mixed signed/unsigned comparisons (`_routines_logic.pyx:103-121`); output casting uses the ufunc `casting=` rule
(`same_kind` default) via `_raise_if_invalid_cast` (`_dtype.pyx:154+`). Library dispatch (Thrust/CUB/cuBLAS) uses runtime `CUPY_TYPE_*`
ids, not templates (`cupy/cuda/common.pyx:7-40`).

---

## 7. NumPy vs CuPy: the architecture comparison

| Concern | NumPy | CuPy | Why NumPy's mechanism cannot survive |
|---|---|---|---|
| Memory | `malloc/free` per array; refcount frees immediately | per-device pool, per-stream arenas, free returns chunk; optional `cudaMallocAsync` | `cudaMalloc/Free` synchronize and cost µs–ms; per-op allocation serializes the pipeline |
| Execution | synchronous; op completes before return | eager host enqueue on a stream; device async | the GPU runs independently; per-op blocking destroys throughput |
| Scalars | reductions return NumPy scalars | 0-d `cupy.ndarray`; scalar only via `.item()` | returning a host scalar forces D2H (`difference.rst:84-110`) |
| Elementwise/ufunc | precompiled C loops per dtype, selected by a type table | CUDA C++ generated at runtime per (loop, arg dtype/ndim/contiguity/index width), NVRTC | no CPU ISA to target; specialization must happen at runtime |
| Kernel organization | `PyUFuncGenericFunction` + `nditer` | `_Ops` loop table + `KernelArguments` + `_ArgInfo`; `CArray`/`CIndexer` encode shape class | `nditer`'s dynamic stride walk replaced by compile-time ndim/contiguity + grid-stride |
| Reduction | pairwise loops over an iterator; arbitrary axes | generated block-reduce; axes by view permutation + dim squashing; CUB fast path | shared-memory tree + launch geometry must be chosen per shape |
| Indexing | generic gather/scatter, bounds checks, sequential semantics | take/scatter kernels; masks via prefix-sum; multi-index folded to linear | parallel scatter has no last-write-wins; per-element bounds checks costly |
| Linalg | OpenBLAS/LAPACK, one synchronous call | cuBLAS/cuSOLVER, handles per thread×device, stream per call, workspace query+allocate | handle/stream/workspace state is a GPU-library artifact |
| Dtype machinery | scalar type objects + C loops; object dtype first-class | C++ type map; `object` impossible; structured only field access | compiled C++ cannot dereference PyObjects |
| Host↔device | arrays are host memory; no transfer concept | `asarray/asnumpy/get/set` first-class, pinned staging, blocking flag, DLPack/CAI | device memory is a separate address space |
| Streams | none | `Stream` user-visible, per-thread/per-device current stream | needed for overlap and interop |
| Device selection | none | `Device` context; per-device pools/handles/plans; peer access | multiple GPUs per process |
| Errors | Python exceptions from C | `CUDARuntimeError/CUBLASError/CUSOLVERError/CUSPARSEError`, status per call; async faults surface at next sync | GPU faults are asynchronous and sticky |
| Array-likes | lists and NumPy arrays accepted | CuPy arrays/scalars only (`difference.rst:126-140`) | implicit H2D would hide a transfer |
| Deliberate omissions | — | object/string dtypes, `numpy.matrix`, subclasses (`subok`), full structured dtypes, some `sort(kind/order)` options | no device representation or no demand |

**The three forcing functions.**

1. **The device is remote and asynchronous.** Any semantics assuming "the result exists when the call returns"
   must be re-expressed. CuPy's answer is not laziness but *stream ordering plus a few explicit sync points*
   (§2): hence 0-d arrays instead of scalars, `asnumpy(blocking=)`, and a synchronizing buffer protocol.
2. **Device memory is scarce and allocation is a driver call.** NumPy's
   allocate-per-temporary/free-at-refcount-zero would be catastrophic. CuPy keeps refcount lifetime but makes
   free a pool operation, with stream-local arenas for concurrency safety (§3); the pool limit and OOM retry
   are the explicit analogue of "malloc may fail".
3. **There is no fixed instruction set or preexisting compiled loop.** NumPy's loops are written once in C and
   selected; on GPU the loop must be *generated and compiled* per dtype and shape class. This is the largest
   architectural difference: a kernel is data (source + cache key), produced by `_kernel.pyx`/`_reduction.pyx`, compiled
   by NVRTC, cached in-process and on disk (§4). `_reduce_dims`, dtype loop tables, name mangling, and header
   checksums in the cache key all follow from it.

Equally instructive is what CuPy did **not** build: no lazy IR, no graph compiler, no hot-path autotuner, no
object dtype, no attempt to reproduce NumPy's sequential-indexing or NaN-order semantics. It documents
divergences (`difference.rst`) instead of hiding them.

---

## 8. Lessons for `rstsr-cuda`

Each item traces to a section above; "rstsr" assumes the current architecture (device value in every
`Storage`, trait-per-op with internal dtype dispatch, eager evaluation).

1. **Keep eager-async; do not build a lazy engine** (§2.2, §4.1). CuPy allocates and launches inline; streams
   provide all ordering. In rstsr, `DeviceStorageAPI` ops should enqueue on a stream held in `DeviceCuda` and return a
   tensor whose storage already exists. Laziness only amortizes host launch overhead; pooling, kernel caching,
   and optional CUDA graphs (`Stream.begin_capture`, §2.4) are cheaper.
2. **Pool per device, arenas per stream** (§3.2). This is the key correctness/perf invariant: never reuse a
   freed block on another stream without synchronization. Mirror `_get_stream_identifier`, including the PTDS per-thread case
   if rstsr exposes it.
3. **Make the allocator injectable** (§3.3). CuPy supports raw, async, managed, system, and pooled allocators
   behind one `alloc(size)` interface. In Rust, a `DeviceMemoryPool` trait in the device value; `cudaMallocAsync` should be a
   separate implementation, not a flag.
4. **Cache compiled kernels on disk with compiler + arch + headers + options in the key** (§4.4). CuPy's key
   is a proven checklist; add a payload-format version tag and optional source dump. Expect first-call latency
   and provide a warm-up/persistent-cache story from day one.
5. **Specialize on ndim/contiguity/index width, then squash dimensions** (§4.1). Compile-time `CArray<T, ndim, contiguous, 32bit>` is
   what makes inner loops fast but multiplies variants; canonicalize layouts (`_reduce_dims`-style) and key the
   cache on the canonical shape class, not raw `Layout`.
6. **Represent dtype dispatch as an ordered loop table + promotion predicate** (§4.1, §6). CuPy's
   `_Ops`/`can_cast` walk is the runtime analogue of rstsr's dtype-dispatched ops. Keep the table per op,
   ordered narrow→wide, and make NEP-50 weak-scalar behavior explicit (CuPy needed a dedicated `_check_should_use_weak_scalar`).
7. **Reduction: one generator, four expressions** (§4.2). `map/reduce/post_map/identity` + axis permutation + dim squashing covers
   sum/prod/min/max/arg*/nan*; add a CUB fast path with conservative eligibility checks and silent fallback.
   Copy the block-stride formula before inventing one; expose launch params.
8. **Fancy indexing cannot preserve NumPy semantics** (§4.3, §7). Plan for wrap-around bounds, undefined
   duplicate-index writes, and mask indexing via prefix-sum + scatter; document these as rstsr divergences.
9. **Do not autotune in the hot path** (§4.2, §5.5). CuPy ships fixed heuristics; its only autotuner is an
   opt-in Optuna context for reductions. Deterministic heuristics + optional offline tuning is the pragmatic
   baseline.
10. **Linalg needs per-(thread, device) handle maps and per-call stream binding** (§5.1). Put the map in
    `DeviceCuda` (Arc'd, thread-local or mutex-guarded), create lazily, set the stream before every call. This
    also keeps `change_device` cheap, consistent with rstsr's per-tensor device model.
11. **Use the column-major swap and gemmEx compute-type table** (§5.2): f16/bf16 → `CUDA_R_16F/16BF` + `COMPUTE_32F`;
    f32/c64 → `COMPUTE_32F`; f64/c128 → `COMPUTE_64F`; integers → own kernel or `COMPUTE_32I`. Keep the integer fallback.
12. **TF32: copy the capability, not the mechanism** (§5.3). `CUPY_TF32` + a process-global `compute_types` table is
    the pattern to avoid. Since `CUBLAS_COMPUTE_32F_FAST_TF32` is a *per-call* argument, carry math mode in the device object or an
    explicit per-op argument/scoped guard and pass it at each gemm: dynamic, thread-safe, no cached global, no
    env-var re-read semantics.
13. **Dtype support is a compile-time surface; decide it early** (§6). fp16/bf16 need device-side wrapper
    types with explicit conversions and float-promoted arithmetic; bf16 needs an external 16-bit type (CuPy
    used `ml_dtypes`, only in v14). Object dtype is impossible; structured dtypes are metadata-only. Close
    rstsr-cuda's numeric trait set over supported dtypes and dispatch internally.
14. **Host↔device transfer is first-class** (§2.3, §3.4): blocking and non-blocking forms, pinned staging for
    async H2D, DLPack/CAI-style zero-copy import. Any host-visible view of device memory must synchronize
    (CuPy's buffer-protocol precedent).
15. **Errors: one typed error per library, checked at every call; async faults surface later** (§7). Model
    `RSTSRError` variants for runtime/BLAS/solver/sparse and document launch-time vs execution-time; rstsr's
    `_f` convention maps well.
16. **Expose memory statistics and a limit from day one** (§3.2): `used_bytes/total_bytes/free_all_blocks/set_limit` plus the OOM retry protocol (free,
    GC, retry, typed OOM).
17. **Vendor libraries are optional at runtime.** CuPy soft-links everything and treats absence as a warning +
    on-use error (`cupy/cuda/__init__.py:41-104`, `_softlink.pyx`). A Rust backend should probe with `libloading` and expose capability
    flags rather than fail at startup.
18. **Document divergences as an artifact.** `docs/source/user_guide/difference.rst` is part of the architecture: out-of-bounds wrap,
    duplicate indices, NaN order, contiguity, 0-d reductions. rstsr-cuda will have its own list; writing it
    early prevents "bug" reports that are really design choices.

---

**One-line takeaway**: CuPy is not "NumPy with kernels swapped" — it is an **eager, stream-ordered,
pool-backed runtime that manufactures dtype/shape-specialized CUDA kernels on demand and caches them**, with
the NumPy API layered thinly on top. Streams (§2), pooling (§3), and runtime kernel generation + cache (§4)
are the minimum viable architecture for any NumPy-shaped Rust GPU tensor library; the global (not per-call)
TF32 toggle is the one design choice rstsr should deliberately invert.

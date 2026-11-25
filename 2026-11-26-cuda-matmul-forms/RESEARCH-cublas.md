# cuBLAS survey for the matmul forms (CUDA 13.2 headers, cudarc 0.19.10)

Sources on this machine: `/usr/local/cuda-13.2/include/cublas_api.h` (`CUBLAS_VER 13.4.0`),
`cublasLt.h`, and the bound functions of `cudarc-0.19.10/src/cublas/sys/mod.rs` and
`cublaslt/sys/mod.rs` (the version in rstsr's `Cargo.lock`). "bound" means callable without new
FFI. The first fact table in `GRILL-INIT-PROMPT.md` under-reported the bindings (a too narrow
search); this file supersedes it.

## dot (BLAS 1)

| routine | bound | note |
|---|---|---|
| `cublas{S,D}dot_v2`, `cublas{C,Z}dot{c,u}_v2` | yes | typed, real and complex (`c` = conjugate, `u` = unconjugated) |
| `cublasDotEx`, `cublasDotcEx` | yes | `xType`/`yType`/`resultType`/`executionType`; `f16`, `bf16`, `f32`; real only |
| `cublasSetPointerMode_v2`, `cublasGetPointerMode_v2` | yes | CUDA 13 name | cudarc binds the `_v2` names unconditionally and loads them at run time; a CUDA 12 runtime has only the un-suffixed symbol, so the setter is usable only where `_v2` exists (the CI pin 12060 must be probed before relying on it) |

`result` is a **host** pointer unless the handle is in `CUBLAS_POINTER_MODE_DEVICE`; in host mode
the call blocks until the GPU has written the result. Device mode makes the dot asynchronous (the
result is written to a one-element device buffer), at the price of a handle-global state: the
handle rstsr keeps per stream is shared by every call, so a dot would have to switch the mode
and switch it back inside the handle lock, and the `_v2` setter is not there on a CUDA 12
runtime. B0 therefore times the cuBLAS dot **both** ways (blocking host result, and device mode
on a private handle) and the choice is made on the numbers (D3). *Corrected 2026-11-26: the
first version of this file said the setter was not bound; the search used the pre-CUDA-13 name.*

## gemv (BLAS 2)

| routine | bound | since | note |
|---|---|---|---|
| `cublas{S,D,C,Z}gemv_v2` | yes | — | `y = alpha op(A) x + beta y`, free `incx`/`incy` |
| `cublas{S,D,C,Z}gemvStridedBatched` | yes | CUDA 11.6 (cuBLAS extension) | `strideA`, `stridex`, `stridey`; a stride of 0 broadcasts that operand |
| `cublas{S,D,C,Z}gemvBatched` | yes | — | device arrays of pointers (one `A`, `x`, `y` per batch item) |
| `cublas{HSH,HSS,TSS,TST}gemvStridedBatched`, `...gemvBatched` | **no** | CUDA 11.6 | `f16`/`bf16` in, `f16`/`f32` out |
| `cublasGemvBatchEx` (mixed real types) | **no** | CUDA 11.7+ | — |
| cuBLASLt gemv | none in 13.2 | — | `cublasLt.h` offers only `cublasLtMatmul` |

`gemvStridedBatched` takes exactly the per-operand batch steps the geometry already computes
(`GemmGeom::affine_batch`), so an affine batch (contiguous, or broadcast with step 0) costs no new
addressing. A gapped multi-axis batch is not affine and would need `gemvBatched` with a device
array of pointers.

## syrk / herk (BLAS 3, symmetric)

| routine | bound | note |
|---|---|---|
| `cublas{S,D,C,Z}syrk_v2` | yes | one triangle (`uplo`); `cublasCsyrk` is the plain-transpose form |
| `cublas{C,Z}syrkEx`, `syrk3mEx` | yes | complex with explicit `Atype`/`Ctype` |
| `cublas{S,D,C,Z}syrkx` | yes | `C = alpha A B^T + beta C`, two different operands |
| `cublas{C,Z}herk_v2`, `herkEx`, `herk3mEx`, `herkx` | yes | conjugate form |
| batched syrk / herk of any kind | none exists in 13.2 | no `syrkStridedBatched`, no `XsyrkBatched` |
| `f16`/`bf16` syrk | none exists in 13.2 | — |
| cuBLASLt syrk / herk | none in 13.2 | — |

## batched gemm

| routine | bound | since | note |
|---|---|---|---|
| `cublasGemmStridedBatchedEx` | yes | — | in use today: one step per operand, broadcast by step 0 |
| `cublas{S,D,C,Z,H}gemmBatched`, `cublasGemmBatchedEx` | yes | — | device arrays of pointers, mixed types for the `Ex` form |
| `cublasGemmGroupedBatchedEx` | yes | CUDA 12.5 | groups of `(m, n, k, trans, ld)` in one call |

The pointer-array and grouped entries are the cuBLAS route for a **non-affine** batch (a gapped
multi-axis batch part, which today falls through to the tiled kernel). They need a device array of
pointers per call: built on the host from the layouts (they are host values) and staged with an
async copy.

## Consequences for the forms

- Inner dot: own kernel (`vecdot`) unless the cuBLAS dot is significantly faster in device mode (D3).
- gemv: typed real and complex, unbatched **and** affine-batched, through the classic driver. Half
  gemv exists in cuBLAS but is not bound: out of scope under "no new FFI".
- syrk: typed real and complex, unbatched only (cuBLAS has no batched form).
- Non-affine batches (gemm, gemv): a separate item; internal to `prepare`, no interface impact.

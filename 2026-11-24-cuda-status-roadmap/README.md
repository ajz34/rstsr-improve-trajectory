# rstsr-cuda — status, coverage and roadmap (2026-11-24)

A consolidation of where the CUDA backend actually stands, what is missing for
ordinary `rstsr` use, what is missing specifically for streaming, and what to do
next. Written after reading the code at `261021/cuda` @ `9c7d49a`, PR
[ajz34/rstsr#4](https://github.com/ajz34/rstsr/pull/4) (body, commits, and the
owner's three review comments), and every CUDA directory in this repository.

No new measurements were taken. Performance claims are quoted with the date and
commit they were measured at; see [EVIDENCE.md](EVIDENCE.md) for how each
correctness/coverage claim was checked.

**Baseline**

| | |
|---|---|
| rstsr branch | `261021/cuda` @ `9c7d49a` (2026-11-24), 1 commit ahead of `ajz34/261021/cuda` (`f11eeda`) |
| PR #4 | merged as `6d17d04` (2026-11-23), fork only; `a4d717a` is the reviewed head on `261119/pr-4-cuda` |
| upstream | no PR to `RESTGroup/rstsr`; the work is not on `main` |
| machine | RTX 5070 Laptop 8 GiB, sm_120, driver 595.99.02, CUDA toolkit 13.2, Ryzen 9 9955HX |
| tests | 183 total (`--features faer --test-threads=1`): 72 host-side lib tests + 111 hardware integration tests; the lib half runs anywhere |

**Verdict in one paragraph.** For the slice it was built for — f32/f64
elementwise, reductions, `matmul`/`tensordot`/`outer`/`cross`/`vecdot`,
`index_select`/`index_put`, and a chunked pinned-host streaming pipeline — the
backend is real, tested on hardware, and fast (elementwise at CuPy parity,
streaming at 2.29× the naive flow). It is **not yet a general `rstsr` device**:
plain `-a`, `tril`, `sort`, `nonzero`, boolean masking, `array_index`,
`allclose`, complex `matmul`, and every cuSOLVER-shaped linalg call fail to
compile. The streaming *primitives* are complete, but the allocator framework,
DLPack, graphs and multi-GPU peer copy are not. The largest structural blockers
are outside `rstsr-cuda`: core gates `Raw = Vec<T>` (element `Index`/`iter`,
`take_along_axis`, `put_along_axis`) and a linalg tier hard-wired to the
`DeviceBLAS` type alias.

---

## 1. Master status

Labels: **DONE** = implemented and exercised on hardware at the baseline;
**PARTIAL** = works with a documented dtype/layout/scope limit; **MISSING** = no
implementation, a use site fails to compile or is rejected; **CORE** = blocked
by `rstsr-core`, not by this crate; **DEFERRED** = decided but not built;
**BY DESIGN** = intentionally not implementable/reachable.

| # | Area | State | What works / what is missing |
|---|---|---|---|
| 1 | Storage & transfers | **DONE** | `CudaBuffer<T>` (byte-erased, deep-copy `Clone`), H2D/D2H, D2D, CPU↔CUDA, layout/strided copies, D2H-free `Display` via visible-gather. Pageable transfers block by design; async edges exist only for pinned buffers. |
| 2 | Runtime, streams, events | **DONE** | per-ordinal registry, per-device-value stream (`new_stream`/`with_stream`/`set_stream`), per-stream cuBLAS handle, `CudaEvent`, `CudaSyncAPI`, cross-stream `order_input`/`order_output`. No stream priorities; one event allocated per cross-stream op. |
| 3 | Kernel engine | **DONE** | NVRTC → PTX, arch override, in-memory + disk cache (format v4), `FlatLaunch`, layout tiers T0/T1/T2, narrow-`u32` plan, ILP4 grid-stride, no per-op sync. |
| 4 | Elementwise | **PARTIAL** | arithmetic, comparison, transcendentals (real+complex), rounding, `abs`/`sign`/`conj`, `where`/`clip`, `isnan`/`isinf`/`signbit`, `ext_add/sub/mul/div`, compound assign, owned `a+b`. Missing: **`neg`**, `ext_bitand`/`bitor`/`bitxor`/`shl`/`shr`, half arithmetic; promoting binaries (`pow`, `maximum`, …) and `where`/`clip` are same-dtype only. |
| 5 | Reductions | **PARTIAL** | f32/f64: `sum`/`prod`/`mean`/`l2_norm`/`min`/`max`/`all`/`any`/`count_nonzero` (+ dtype), `var`/`std` (+ dtype), `argmin`/`argmax`/`nanarg*`, `cumsum`/`cumprod`. Integers: `min`/`max` and scans; `sum`/`prod` only through `*_dtype` with an f32/f64 accumulator. Complex: none. Missing: `cumsum_dtype`/`cumprod_dtype`, unraveled `arg*`, complex reductions/scans, custom reduce; layouts outside the single-run plan are refused (`UnImplemented`), with `to_contig` as the only workaround. |
| 6 | Linalg | **PARTIAL** | `outer`, `cross`, `matmul`/`matmul_uninit` (cuBLAS, tiled fallback), `tensordot` (view-only tier), `vecdot`. **f32/f64 only** for `matmul`/`tensordot` — no complex, no integer, no half; no `ext_*` mixed-dtype forms; no cuSOLVER family (`solve*`, `inv`, `det`, `cholesky`, `eig*`, `svd*`, `qr`, `pinv`, `matrix_rank`, `slogdet`, norms). |
| 7 | Indexing / masking | **PARTIAL** | `index_select`, `index_put` (host index slices). Missing: `take_along_axis`/`put_along_axis` (**CORE**: tensor layer demands `Raw = Vec<usize>`), `mask_select`/`mask_fill`, `array_index`/`array_index_assign`, `nonzero`. |
| 8 | Sorting / search / set | **MISSING** | `sort`, `argsort`, `sort_custom`, `unique*`, `searchsorted`, `isin` — all absent. |
| 9 | Comparison / closeness | **PARTIAL** | `equal`/`not_equal`/ordering present; `isclose`/`allclose` absent (`OpIsCloseAPI`, `OpAllCloseAPI`). |
| 10 | Dtypes | **PARTIAL** | f32/f64 full; `Complex<f32>`/`Complex<f64>` for elementwise (incl. transcendentals) but **not** reductions, scans, or GEMM; all integer widths for arithmetic/compare and `min`/`max`; bool; `f16`/`bf16` storage/fill/copy/compare only; no `i128`/`u128` (by design). |
| 11 | Streaming | **DONE (primitives)** | pinned `DeviceCudaPinned` + `PinnedBuffer`, non-blocking `h2d_pinned_async`/`d2h_pinned_async`, async `to_device` both ways, zero-copy `host_view`, drop-time read events, 3-stream per-chunk pipeline benchmarked at **2.29×** the naive flow (95.6 % of the `max(compute, h2d+d2h)` bound). Missing: allocator pool/config, DLPack `kDLCUDA`/`kDLCUDAHost`, CUDA graphs, multi-GPU peer copy, stream priorities, high-level pipeline helper. |
| 12 | Facade / packaging | **MISSING** | no `cuda` feature in `rstsr`, no `rt::cuda` namespace, no crate `readme.md`, no `publish = false`, no book page. Users must depend on `rstsr-cuda` directly (the generic `rt::*` functions then work). |
| 13 | Tests / CI | **PARTIAL** | 111 hardware integration tests + 72 host-side lib tests (183 total); tier-3 CI runs only `--lib` and `--no-run` of the hardware suite (no GPU runner). No perf gate for reductions/linalg/indexing; elementwise bench table is pre-PR and has not been re-run at `9c7d49a`. |
| 14 | Docs | **PARTIAL** | crate rustdoc is unusually good (numerics contract, divergences, env knobs), but the "absent, and why" section is stale in three places and it labels omissions "deliberate, not pending" that common use will need. |

### What changed since the last trajectory record

The trajectory has records up to the elementwise wave (2026-10-24) and the
pinned/streaming work (2026-11-24), but **nothing for PR #4's reductions,
linalg, indexing, numerics, arch-matrix and owned-op work, and nothing for the
owner's review directives**. That is the main documentation hole this directory
fills; the detail is in [PR-4.md](PR-4.md).

---

## 2. The ten things to do next

Ordered by "ordinary `rstsr` code starts working" first, then streaming, then
structural. Full batches, acceptance criteria and sizes are in
[ROADMAP.md](ROADMAP.md).

| P | Action | Why now | Size |
|---|---|---|---|
| P0 | Add `OpNegAPI` | `let b = -&a;` and `rt::neg` do not compile. One kernel row; the most embarrassing hole. | S |
| P0 | Fix the `lib.rs` "absent, and why" section | It claims `allclose_axes` blocks `allclose` (false: `allclose` works on CPU), that `take_along_axis` takes a *device* index buffer (the tensor layer demands a *host* `Vec`), and does not mention `neg`. Misleads every reader. | S |
| P0 | `tril`/`triu` via `DeviceCreationTriAPI` | Standard array-API creation; small kernel; needed by any triangular-mask code. | S |
| P0 | `isclose`/`allclose` | Two small elementwise/reduction traits; every numerical test wants them. | S/M |
| P0 | `nonzero` + `mask_select`/`mask_fill` | Boolean masking is ordinary use; needs one compaction primitive (`OpNonzeroAPI`) reused by `array_index`. | M |
| P0 | `array_index`/`array_index_assign` | The core layer already materializes resolved indices as device buffers, so only `DeviceArrayIndexAPI` is missing. | M |
| P1 | Sorting/search/set family | `sort`/`argsort`/`unique*`/`searchsorted`/`isin`; a bitonic/radix family, one wave. | L |
| P1 | cuSOLVER linalg tier | `solve`, `inv`, `det`, `cholesky`, `eig*`, `svd*`, `qr`, `pinv`, `rank`, `slogdet`, norms. Needs the `DeviceBLAS`-aliased `rstsr-linalg-traits::blas_impl` generalized, plus cuSOLVER wrappers. | L |
| P1 | Complex/integer/half GEMM, `ext_*` linalg | `Complex<f64>` matmul is the domain default outside this crate's tests; cuBLAS typed `gemm` already supports it, the bound is the only blocker. | M |
| P1 | TF32/cuBLASLt: make `compute_type_32f` real (or delete it) | Today the flag and env var are silently ignored; a user who sets `fasttf32` gets f32-pedantic results with no warning. | M |

Streaming-specific next actions (all P1): allocator framework D7
(`RSTSR_CUDA_ALLOC_CONF`, `empty_cache`, OOM protocol), multi-GPU
`cuMemcpyPeerAsync` + peer access, DLPack pinned/CUDA export, CUDA graphs, and a
GPU CI/perf gate. See [STATUS.md](STATUS.md) §7 and [ROADMAP.md](ROADMAP.md) W7.

---

## 3. Silent surprises (fix or document)

These are not "missing features"; they are places where the current behavior
differs from what a reasonable user would assume.

1. **`RSTSR_CUDA_COMPUTE_TYPE_32F` / `DeviceCuda::compute_type_32f()` is dead.**
   The value is parsed, stored, and compared by `same_device`, but no launch
   path reads it. `cublas_gemm` always calls typed `cublas<t>gemm`.
2. **Cross-ordinal copy is not a peer copy.** `into_device` allocates on the
   target and calls `memcpy_dtod_async` on the target stream, then synchronizes.
   No `cuMemcpyPeerAsync`, no peer-access check; on a multi-GPU host without
   peer access this is at best a blocking staged copy, at worst a driver error.
3. **`matmul`'s "tiled fallback" is not a dtype fallback.** The bound
   `T: CublasScalar` (f32/f64) is on the whole impl, so the tiled kernel is only
   reachable through `RSTSR_CUDA_MATMUL=tiled`; complex/integer/half matmul fail
   to compile rather than falling back.
4. **Reduction layout policy is refusal, not fallback.** A reduction over a
   layout the classifier does not accept returns `UnImplemented`; the only
   workaround is a `to_contig` copy. This is the owner-approved policy for now,
   but it is a common-use cliff.
5. **`same_device` ignores the stream.** Two device values on different streams
   are the same device; cross-stream ordering is the caller's job through the
   op-level events. Correct, but surprising if you expected a stream mismatch
   error.
6. **`to_vec()` returns the device buffer, not a `Vec<T>`.** The method is
   generic over `Raw`, so on CUDA it returns a `CudaBuffer<T>`; a host `Vec`
   needs the consuming `into_vec()` or `a.to_device(&cpu)` first. The doc
   example (`let v: Vec<i32> = a.to_vec()`) only holds for CPU devices.

---

## 4. Directory map

| File | Contents |
|---|---|
| [STATUS.md](STATUS.md) | Layer-by-layer status, missing device/op trait inventory, dtype matrix, streaming capability matrix, test/CI/docs status, known defects |
| [COVERAGE.md](COVERAGE.md) | Function-level "common `rstsr` usage" matrix: creation, manipulation, operators, reductions, indexing, sorting, linalg, interop, iteration |
| [ROADMAP.md](ROADMAP.md) | Prioritized checklist (P0/P1/P2) with acceptance criteria, sizes, dependencies, and suggested waves |
| [PR-4.md](PR-4.md) | What PR #4 contains that the trajectory never recorded, the owner's review directives, and corrections to the PR body |
| [EVIDENCE.md](EVIDENCE.md) | How claims were checked: commits, trait diffs, compile probes, file references, commands |

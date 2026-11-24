# rstsr-cuda — detailed status

Companion to [README.md](README.md). Everything here is at baseline
`261021/cuda` @ `9c7d49a` unless a date/commit says otherwise.

---

## 1. Provenance and environment

| Item | Value |
|---|---|
| Crate | `crates-device/rstsr-cuda`, package `rstsr-cuda` 0.10.3 |
| Branch | `261021/cuda`, HEAD `9c7d49a` (2026-11-24) |
| Fork remote | `ajz34/261021/cuda` @ `f11eeda`; local is 1 commit ahead |
| PR #4 | `ajz34/rstsr#4`, merged into `261021/cuda` at `6d17d04` (2026-11-23); reviewed head `a4d717a` on `261119/pr-4-cuda` |
| Upstream | none; `RESTGroup/rstsr` `main` does not contain the crate's work |
| Toolchain | workspace `rust-toolchain.toml` (nightly), edition 2021, rust-version 1.82 |
| CUDA | toolkit 13.2, driver 595.99.02, sm_120; `CUDARC_CUDA_VERSION=12060` when nvcc is absent |
| cudarc | 0.19, features `driver`, `nvrtc`, `cublas`, `cuda-version-from-build-system`, `fallback-dynamic-loading` |
| Crate features | `default = []`; `launch_trace`, `faer` (CPU-reference interop), `dynamic-linking` |
| CI | `.github/workflows/tier3-cuda.yml`: `--lib --release` + `--all-targets --no-run`; **no GPU runner** |

The crate is publishable today (no `publish = false`, no `readme.md`), which
is a release-plz hazard: the release workflow treats every workspace member as
publishable unless configured otherwise.

---

## 2. Layer status

| Layer | State | Notes / evidence |
|---|---|---|
| Storage (`buffer.rs`) | DONE | `CudaBuffer<T>` wraps a byte-erased `CudaSlice<u8>`; logical length for `unique*` truncation; deep-copy `Clone` via async D2D; manual `Send`/`Sync`; `MaybeUninit` twin. |
| Transfers (`transfer.rs`, `conversion.rs`) | DONE | H2D/D2H/D2D, pageable blocking paths, cross-ordinal D2D (see defect §9.5), same-ordinal relabel, `DeviceCpuSerial`/`DeviceFaer` hops. |
| Pinned host (`pinned/`) | DONE | `PinnedBuffer<T>` (`cuMemHostAlloc` + one `CU_EVENT_BLOCKING_SYNC` event), `DeviceCudaPinned` (`Raw = PinnedBuffer<T>`), zero-copy `host_view` into `DeviceCpuSerial`, async edges, `Drop` waits. |
| Runtime (`device.rs`, `registry.rs`, `stream.rs`, `flags.rs`) | DONE | one `CudaOrdinalObj` per ordinal (`Arc::ptr_eq` canonical), per-device-value `Arc<CudaStreamObj>`, per-stream cuBLAS handle behind `Mutex`, env-seeded `CudaRuntimeFlags`. |
| Events / sync (`event.rs`, `sync.rs`) | DONE | `CudaEvent` record/wait/synchronize/query; `CudaSyncAPI` for device, tensor, pinned tensor. Crate-local, not the core `DeviceSyncAPI` (which does not exist yet). |
| Kernel engine (`kernel.rs`, `kernels/`) | DONE | NVRTC→PTX, two-tier cache (memory + disk, `FORMAT_VERSION = 4`), arch override, `FlatLaunch` (arg marshalling, ept, explicit shape), T0/T1/T2 layout tiers, narrow-`u32` plan, read-lifetime guards. |
| Elementwise (`operators.rs`, `kernels/elementwise.rs`, `exprs.rs`, `ops_binary_extra.rs`, `ops_ternary.rs`) | PARTIAL | Missing `neg` and the mixed-integer `ext_*` family; same-dtype-only promotion; see §4 and §6. |
| Reductions (`reduce_ops.rs`, `arg_ops.rs`, `kernels/reduce_*`, `reduce_plan.rs`) | PARTIAL | Direct rows on f32/f64 only; ints get `min`/`max` plus `*_dtype` accumulators; complex has none. Plan is a pure function of shape/strides; `MIN_BLOCKS = 512` policy constant; lanes on unit-stride axis; deterministic, not CPU-bit-identical. |
| Scans (`scan_ops.rs`, `kernels/scan.rs`) | PARTIAL | `cumsum`/`cumprod` on `CudaReal` (ints, f32, f64), `include_initial`, axis split with chunk carry; complex refused by bound. |
| Linalg (`linalg.rs`, `kernels/gemm.rs`, `cublas_gemm.rs`) | PARTIAL | `matmul`/`tensordot` are f32/f64 only (typed cuBLAS); `outer`/`cross`/`vecdot` are generic. `resolve_geom` is shared by the cuBLAS and tiled paths; batch strides in elements (a real bug fixed in PR #4). |
| Indexing (`adv_index_ops.rs`, `kernels/adv_index.rs`) | PARTIAL | `index_select`, `index_put`; host index slices; no device index tensors. |
| Formatting (`format.rs`) | DONE | visible-gather D2H only, mirrors `rstsr-native-impl` traversal exactly. |
| Tests | PARTIAL | 183 green on hardware (`--features faer --test-threads=1`), 72 host-side; see §8. |
| Docs | PARTIAL | crate-level rustdoc is excellent; stale claims listed in §6. |

---

## 3. Missing device traits

Derived by diffing the rustdoc impl lists of `DeviceCuda` and
`DeviceCpuSerial` at `9c7d49a` and dropping blanket/sugar traits
(`ByRef`, `Imply`, `Pointable`, `DeviceNumAPI`, `DeviceComplexFloatAPI`,
`Display`, `ToString`). Method: [EVIDENCE.md](EVIDENCE.md) §2.

| Missing trait | User-facing consequence | Blocker / note |
|---|---|---|
| `DeviceCreationTriAPI` | `tril`, `triu` | Small triangular-mask kernel. |
| `DeviceArrayIndexAPI` | `array_index` (fancy indexing by integer arrays) | Core already hands device index buffers + layouts; one gather kernel. |
| `DeviceArrayIndexAssignAPI` | `array_index_assign` (fancy set) | Scatter; duplicate-index policy must be chosen. |
| `DeviceMaskIndexAPI` | `mask_select`, `mask_fill` | Needs `OpNonzeroAPI` for `mask_select`; `mask_fill` only this trait. |
| `DeviceTakeAlongAxisAPI` | `take_along_axis` | **CORE**: `take_along_axis_f` bound requires `DeviceAPI<usize, Raw = Vec<usize>>`, i.e. a host index tensor; a device impl cannot satisfy it. |
| `DevicePutAlongAxisAPI` | `put_along_axis` | Same core gate. |
| `DeviceExtMatMulAPI` | `ext_matmul` (mixed dtype) | Route-B `promote_pair`; CPU has it on `DeviceCpuSerial`/`DeviceFaer` only. |
| `DeviceExtVecdotAPI` | `ext_vecdot` | Same. |
| `DeviceExtTensordotAPI` | `ext_tensordot` | Same. |
| `DeviceExtOuterAPI` | `ext_outer` | Same. |
| `DeviceExtCrossAPI` | `ext_cross` | Same. |

---

## 4. Missing op traits

| Missing trait | User-facing function/operator | Note |
|---|---|---|
| `OpNegAPI` | `-x`, `rt::neg` | **Confirmed compile error** at `9c7d49a`; one kernel row. |
| `OpIsCloseAPI` | `rt::isclose` | Elementwise; straightforward. |
| `OpAllCloseAPI` | `rt::allclose` | Needs a bool reduction of the closeness predicate; `allclose_axes` is a separate, CPU-unimplemented entry. |
| `OpNonzeroAPI` | `rt::nonzero`; required by `mask_select`, `array_index` | Compaction: count (prefix sum) + scatter. |
| `OpSortAPI` | `rt::sort` | Bitonic/radix family. |
| `OpArgSortAPI` | `rt::argsort` | Same. |
| `OpSortCustomAPI` | `rt::sort_custom` | Host comparator — **BY DESIGN** impossible. |
| `OpUniqueAPI` | `rt::unique_all/counts/inverse/values` | Sort + adjacent-dedup + inverse scatter. |
| `OpSearchSortedAPI` | `rt::searchsorted` | Binary search over a sorted buffer. |
| `OpIsinAPI` | `rt::isin` | Sort/hash based. |
| `OpPackTriAPI` / `OpUnpackTriAPI` | `Tensor::pack_tri` / `unpack_tri` (+ `pack_tril`/`pack_triu`, `unpack_tril`/`unpack_triu`) | rstsr-specific triangular packing. |
| `OpCumSumDtypeAPI` / `OpCumProdDtypeAPI` | `rt::cumsum(..., dtype=…)` / `cumprod` | Scan with a different accumulator dtype. |
| `OpExtBitAndAPI` / `OpExtBitOrAPI` / `OpExtBitXorAPI` | `rt::ext_bitand` etc. | Mixed-integer bitwise. |
| `OpExtShlAPI` / `OpExtShrAPI` | `rt::ext_shl` / `ext_shr` | Mixed-integer shifts. |
| `OpLConsumeRemAPI` | `a % b` with owned `a` | Plain `&a % &b` works. |
| `OpRConsumeRemAPI` / `OpRConsumeShlAPI` / `OpRConsumeShrAPI` | owned RHS remainder/shift | Plain ref-ref forms work. |
| `OpUnraveledArgMinAPI` / `OpUnraveledArgMaxAPI` | `rt::unraveled_argmin/argmax` | Multi-axis index decomposition. |
| `OpReduceCustomAPI` | `rt::reduce_custom` | Host closure — **BY DESIGN** impossible. |
| `Op_Mut*_Ref*_API` (7 traits) | `rt::mapv` and composed host-closure helpers | Host closures — **BY DESIGN** impossible. |

The `Op*` closure family is not a gap to close: a CUDA kernel is a string
compiled at launch, and a Rust closure cannot cross into it. The consequence is
that any *composed core API* that reaches for a closure is silently host-only;
this is documented in `rstsr-core` after PR #4's `mapv` note.

---

## 5. Missing linalg (cuSOLVER-shaped)

`rstsr-linalg-traits` defines 19 tensor-tier entry points. All are implemented
for `DeviceFaer`; 12 also exist for the selected BLAS device via
`blas_impl/*`, but that module is written against the compile-time
`DeviceBLAS` alias, so CUDA gets none of them. 19 missing:

| Group | Entries |
|---|---|
| Decomposition | `cholesky`, `qr`, `svd`, `svdvals`, `eig`, `eigh`, `eigvals`, `eigvalsh` |
| Solve / inverse | `solve_general`, `solve_symmetric`, `solve_triangular`, `inv`, `pinv`, `matrix_power` |
| Scalar-valued | `det`, `slogdet`, `matrix_rank` |
| Norms | `matrix_norm`, `vector_norm` |

A CUDA implementation has two routes:

1. **Write `LapackDriverAPI`-style cuSOLVER wrappers** (`potrf`, `getrf`,
   `getri`, `gesv`, `gesvd`, `gesdd`, `syev`, `syevd`, `sygv`, `sygvd`, …) and
   generalize `rstsr-linalg-traits::blas_impl` off `DeviceBLAS` so the existing
   generic bodies apply. This is the owner's stated direction ("cuBLASLt /
   cuSOLVER wrappers to follow rstsr-blas/linalg-traits style").
2. Write CUDA-specific tensor-tier impls, duplicating the batching/order rules.
   Faster to land, worse to maintain.

`matrix_power` and the norm entries are compositions over `inv`/`svdvals` plus
core ops, so they follow once the driver layer exists.

---

## 6. Dtype matrix

`CudaDType` = bool, i8/i16/i32/i64/isize, u8/u16/u32/u64/usize, f32, f64,
`Complex<f32>`, `Complex<f64>`, `f16`, `bf16`. `i128`/`u128` deliberately have
no CUDA C type and keep host round-trips.

| Dtype | Elementwise | Reduce | Scan | MatMul/Tensordot | Index | Sort | Format |
|---|---|---|---|---|---|---|---|
| f64 / f32 | full | full (all rows) | yes | cuBLAS | `index_select`/`index_put` | no | yes |
| c64 / c32 | full (transcendentals too) | **no** — `ReduceAcc` is f32/f64 only | **no** | **no** | `index_select`/`index_put` | no | yes |
| ints | arithmetic/compare/bitwise | `min`/`max`/`arg*`/`count_nonzero`; `sum`/`prod` only via `*_dtype` (f32/f64 accumulator); no `mean`/`var`/`std`/`l2_norm` | yes | **no** | `index_select`/`index_put` | no | yes |
| bool | compare, `&`/`\|`/`^`, `!` | `all`/`any`/`sum` (→ `usize` count) | no | no | mask ops missing | no | yes |
| f16 / bf16 | compare/query only (no arithmetic) | **no** — `CudaArith`/`ReduceAcc` exclude half | no | no | `index_select`/`index_put` | no | yes |
| i128 / u128 | host round-trip (not a `CudaDType`) | host | host | no | no | no | yes |

Fill/`zeros`/`ones`/copy/H2D/D2H work for every `CudaDType` (the TypeId fill
list includes complex and half). Half arithmetic is "a later wave" per
`dtype.rs`; low-precision ML workloads are therefore storage-only today.

---

## 7. Streaming capability matrix

The pinned-host round (2026-11-23/24) closed the host-boundary gap. What exists
now, and what does not:

| Capability | State | Detail |
|---|---|---|
| Ops enqueue and return; no per-op host sync | DONE | only host reads (`to_vec`, `to_scalar`, `Display`) synchronize. |
| Per-device-value execution stream | DONE | `new_stream`, `new_stream_with_ordinal`, `with_stream`, `set_stream`; `CudaStreamObj` strictly compared for identity. |
| `same_device` ignores stream, flags/order compared | DONE | same ordinal + same flags + same order; cross-stream ops are legal and ordered by events. |
| Cross-stream input ordering | DONE | `order_input`/`order_input_raw`: event on the buffer's alloc stream, wait on the exec stream. |
| Cross-stream write ordering | DONE | `order_output`/`order_output_raw`: event on exec, wait on the alloc stream, so a later host read of the alloc stream is safe. |
| Kernel-read lifetime (foreign-stream temporary) | DONE | `9c7d49a`: `read_ptr_on` returns the cudarc guard; `FlatLaunch::arg_buf`/`arg_ptr` and the cuBLAS/D2D fast paths hold it past `launch()`. Regression test `foreign_stream_temporary_kernel_input_ordered`. |
| Stream-ordered alloc/free | DONE | cudarc `stream.alloc` → driver memory pool when available; `CudaSlice::drop` free is stream-ordered. |
| Allocator framework (D7) | **MISSING** | no `CudaAlloc` trait, no `RSTSR_CUDA_ALLOC_CONF`, no release threshold, no `empty_cache`, no OOM-retry protocol. |
| User events | DONE | `CudaEvent::new_f/record/wait/synchronize/query`. |
| Explicit readiness wait | DONE | `CudaSyncAPI` (device, CUDA tensor, pinned tensor); core `DeviceSyncAPI` does not exist. |
| Pinned host allocation | DONE | `PinnedBuffer::new_f/with_flags_f` (`PORTABLE`, `MAPPED`, `WRITE_COMBINED`). |
| One ordering event per pinned buffer | DONE | record on copy, wait before host read; `CU_EVENT_BLOCKING_SYNC`. |
| Non-blocking H2D/D2H | DONE | `h2d_pinned_async`, `d2h_pinned_async`, async `to_device` pinned↔CUDA. |
| Zero-copy host view of pinned | DONE | `TensorPinnedHostViewAPI::{host_view, host_view_mut, host_view_on, host_view_mut_on}` into `DeviceCpuSerial`/`DeviceFaer`. |
| D2H source drop-time read event | DONE | `07e3d2e`: `DevicePtr::device_ptr(slice, exec)` records a read event on the copy stream, honored at `CudaSlice::drop`. |
| 3-stream chunked pipeline | DONE | `benches/bench_streaming.rs`: 2.29× the naive flow, 95.6 % of the `max(compute, h2d+d2h)` bound, bitwise equal. |
| High-level pipeline helper | MISSING | users must copy the bench's recipe (streams + events + per-chunk issue). |
| Pinned allocator pool | **MISSING** | `malloc_host`/`free_host` per buffer; no caching pool, no async alloc. Large per-chunk allocations can stall the host. |
| Per-stream event sets | **MISSING** (deferred) | every cross-stream op creates/destroys one `CUevent`; no cached event pool. |
| Write-combined default | PARTIAL | flag exists; default is cached pages; the bench measured no D2H gain from WC. |
| Pageable async transfers | BY DESIGN blocking | `to_device` CPU↔CUDA is synchronous; overlap requires the pinned path. |
| DLPack `kDLCUDA` / `kDLCUDAHost` | **MISSING** | no dlpack module in `rstsr-cuda`; `rstsr-cpu-dlpack` is host-only. |
| CUDA graphs | **MISSING** | out of scope so far; launch overhead remains per op. |
| Stream priorities | **MISSING** | no priority constructor; all streams default priority. |
| Multi-GPU peer copy | **PARTIAL** | cross-ordinal `into_device` uses `memcpy_dtod_async` on the target stream + sync; no peer-access check, no `cuMemcpyPeerAsync`; blocking. Untested on a second GPU. |
| NCCL | BY DESIGN absent | not planned in the current scope. |

### Pipeline numbers (2026-11-24, `9c7d49a`, `bench_streaming`)

| Mode | Result |
|---|---|
| `serial_all` (pageable, blocking) | 105.4 ms |
| `pipeline` (pinned, 3 streams, per-chunk) | 44.5–46.1 ms |
| speed-up | **2.27–2.29×** |
| efficiency vs `max(compute, h2d+d2h)` | 91–95.6 % |
| transfer hidden | ~89–90 % |
| controls (`pipeline_blocking`, `pipeline_1stream`) | at the 79.7 ms phase sum (no overlap) |
| correctness | bitwise equal to `serial_all` |

The two laws the bench established: (a) issue per chunk, because an event covers
everything enqueued on its stream so far; (b) never make the producer stream
wait for its D2H consumer — use a read event at the consumer plus a drop-time
wait.

---

## 8. Tests, CI, benchmarks, docs

**Tests** (source counts at `9c7d49a`):

| Target | Tests | Runs without GPU? |
|---|---|---|
| `--lib` | 72 | yes |
| `tests/test_device_basic.rs` | 13 | no (`require_cuda!`) |
| `tests/test_kernels.rs` | 55 | no |
| `tests/test_matmul_tiled.rs` | 5 | no |
| `tests/test_ops_wave.rs` | 26 | no |
| `tests/test_pinned.rs` | 12 | no |
| total | 183 (last full hardware run) | — |

- Every hardware test starts with `require_cuda!()` and *passes* on a machine
  with no driver after printing `skipping: no CUDA device visible`. A green run
  on a GPU-less machine means nothing ran.
- Hardware runs use `--features faer -- --test-threads=1` (pinned↔Faer
  conversions; single device context).
- No property/fuzz tests; no cross-device randomized parity harness. The
  existing tests are hand-written per case.
- The tier-3 workflow comment still says "68 tests"; the count is 72.

**CI**: `.github/workflows/tier3-cuda.yml` runs on `ubuntu-latest`, no GPU:
`cargo test -p rstsr-cuda --lib --release` and
`cargo test -p rstsr-cuda --all-targets --no-run --release`. It catches host-logic
regressions and signature breakage, not kernel correctness. No arch-matrix job,
no perf gate, no rented GPU.

**Benchmarks**:

| Bench | Cells | Harness |
|---|---|---|
| `bench_elementwise` | 19 (fill/add/strided/broadcast/slice/step/flip/where/clip/rem/sin/pow/copy_T/to_contig) | custom median-of-R, `harness = false` |
| `bench_streaming` | h2d/compute/d2h/serial/pipeline/blocking/1-stream/pageable/WC | custom, assertions + ratios |

No benchmark exists for reductions, scans, matmul/tensordot, indexing, or
sorting. The PR #4 body explicitly said throughput was not measured; the
elementwise table in the trajectory is from 2026-10-24, before PR #4, and has
not been re-run at `9c7d49a`.

**Docs**: crate-level `lib.rs` documents the execution model, divergences,
numerics contract, env knobs, and the arch matrix. There is no
`crates-device/rstsr-cuda/readme.md`, no book page, no example beyond
`examples/arch.rs`, and no migration/porting guide for CPU→CUDA code.

---

## 9. Known defects and inconsistencies

1. **`compute_type_32f` is dead.** Parsed from `RSTSR_CUDA_COMPUTE_TYPE_32F`,
   stored in `CudaRuntimeFlags`, exposed by `DeviceCuda::compute_type_32f()`,
   compared by `same_device`, and never read by a launch path.
   `cublas_gemm::launch` always calls typed `cublas<t>gemm` (no
   `cublasGemmEx`, no cuBLASLt). A user selecting `fasttf32` silently gets
   f32-pedantic math.
2. **`neg` is missing.** `-&a` fails with
   `DeviceCuda does not implement OpNegAPI<f64, f64, Vec<usize>>` (compile
   probe in [EVIDENCE.md](EVIDENCE.md) §3).
3. **The `lib.rs` "Absent, and why" rationale is wrong in three places.**
   - `allclose`: it says `allclose_axes` is unimplemented on the CPU too, but
     user-facing `allclose` calls `allclose_all`, which needs `OpAllCloseAPI`;
     the CPU implements it, so `allclose` works on CPU and not on CUDA.
   - `take_along_axis`/`put_along_axis`: it says they take a *device* index
     buffer, but the tensor layer requires `Raw = Vec<usize>` (a *host* index
     tensor), so a CUDA impl is blocked at core level, not by read-back policy.
   - `neg` is not listed at all.
4. **The `lib.rs` "deliberate, not pending" framing is stale for common use.**
   `sort`/`unique*`/`searchsorted`/`isin`/`nonzero`/`tril` are called "no
   caller"; that is a statement about the first workload, not about `rstsr`
   users. The next wave should either implement them or keep the label
   explicitly workload-scoped.
5. **Cross-ordinal copy is not a peer copy.** `into_device` allocates on the
   target and calls `memcpy_dtod_async` + `synchronize`, with no
   `cuCtxEnablePeerAccess` or `cuMemcpyPeerAsync`. D8 of the device-basic
   decisions specified a peer copy; the code does not match it. On a multi-GPU
   host without peer access this is a driver error or a staged blocking copy.
6. **`matmul`'s tiled path is not a functional fallback.** The impl bound is
   `T: CublasScalar + num::Zero` (f32/f64) on the whole `DeviceMatMulAPI`
   impl, so unsupported dtypes cannot reach the tiled kernel; it is only
   selectable with `RSTSR_CUDA_MATMUL=tiled`.
7. **PR body vs code: cuBLAS handle scope.** The PR body says "one registry
   object per ordinal → one stream → one cuBLAS handle" and a later section
   says one handle per ordinal; the code has one handle **per stream**
   (`CudaStreamObj::cublas_slot`), created lazily and shared through a mutex.
   The code is the better design (a handle must be bound to a stream); the PR
   text should be corrected if it is reused.
8. **Stale comment in `creation.rs`.** The `full_impl` comment says half and
   complex fall back to the host doubling path, but `is_fill_supported`
   includes `Complex<f32>`, `Complex<f64>`, `f16`, and `bf16`. The host path is
   only for `i128`/`u128` (not `CudaDType`).
9. **Per-element `get_index`/`set_index` synchronize** (`transfer.rs`
   `DEFER` comments). Fine for `to_scalar`; a loop over a CUDA tensor would be
   pathologically slow. This matters for the R1/R2 shared-test migration.
10. **No `publish = false`.** `rstsr-cuda` is a workspace member with a
    version matching the released crates, no readme, and no release-plz
    exclusion; a release run could try to publish it.
11. **Reduction dtype coverage is narrower than the docs suggest.**
    `OpSumAPI`/`OpProdAPI` are bounded by `ReduceAcc`, implemented only for
    f32/f64, so `i32.sum()` and `Complex<f64>.sum()` fail to compile; an
    integer sum needs `sum_dtype::<f64>()`. The module table's `acc = T` rows
    do not state the restriction, and `lib.rs` lists "reductions" without a
    dtype qualifier. `mean`/`var`/`std`/`l2_norm` add `num::Float` and are
    f32/f64 too.

---

## 10. Scope assumptions to revisit

The PR-#4 scope was "the families the first workload needs". That produced
excellent depth on a narrow slice. The user-facing aim now is "common `rstsr`
usage plus streaming", which changes three assumptions:

| Assumption in the code/docs | Why it no longer holds |
|---|---|
| "No caller" for sort/search/set/nonzero/tril | Ordinary Array-API code uses them; a backend that cannot sort is not a general backend. |
| "Refuse unknown layouts" is enough for reductions | A user with a transposed/sliced view expects a slow answer, not `UnImplemented`; a generic fallback kernel is the safety net. |
| "Host closures are the only impossible thing" | True, but `take_along_axis`/`put_along_axis` and element `Index`/`iter` are blocked by `Raw = Vec<T>` gates, which look like implementation gaps to a user. These need core work, not device work. |

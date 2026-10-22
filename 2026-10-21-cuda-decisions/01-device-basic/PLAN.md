# PLAN — implement `01-device-basic` (device + data representation)

Implements `DECISIONS.md` D1–D8 of this directory. Record lives here;
**code lands in the rstsr repo** (branch `261022/cuda-device-basic`, no
auto-commit, manual review per main-repo policy).

Conventions adopted:
- `cudarc = "0.19"` (project convention — caret range, not `=0.19.10`;
  Cargo.lock pins the concrete patch anyway).
- cudarc role per F6/owner: substrate for driver/maintenance/cuda-version
  gates/`CudaSlice`; BLAS/Lt FFI is ours (not needed in this slice).

---

## P — Pre-flight probe (before any rstsr changes; lives in `pre-flight/` here)

Throwaway bin, cudarc 0.19 latest, `default-features = false`, features
`["std", "driver", "<cuda-ver>", "fallback-dynamic-loading"]`:

- **P1 substrate**: `device_count` → `CudaContext::new(0)` →
  `compute_capability` (expect `(12,0)`) → `mem_get_info` → `new_stream` →
  `alloc::<u8>`/`alloc_zeros` → `clone_dtoh` → drop. Run under both
  `cuda-12080` (conservative) and `cuda-13020` (local toolkit 13.2); record
  which pass on driver 595.91.07.
- **P2 mempool**: confirm `has_async_alloc == true` (FreeAsync path);
  alloc/free loop shows second-iteration caching;
  `mem_pool::set_attribute(RELEASE_THRESHOLD)` + `trim_to(0)` through the
  `result` layer work; `empty_cache` semantics observable via
  `mem_get_info`.
- Result: pick the default `<cuda-ver>` feature (expect `cuda-12080` default
  + passthrough features `cuda-13*` / `cuda-version-from-build-system`).

## B — Batches (rstsr repo, one branch; each batch compiles + tests green)

| # | Batch | Content (decision refs) |
|---|---|---|
| B1 | Skeleton + device value | Workspace member `crates-device/rstsr-cuda`; deps `rstsr-core`(+`std`), `rstsr-common`, `cudarc 0.19`; module tree; `DriverError → RSTSRError` mapping helper (existing `DeviceError`/`MemoryError` variants). `CudaStreamObj`, `CudaOrdinalObj`, registry `OnceLock<Mutex<HashMap<usize, Weak<_>>>>` (D2); `DeviceCuda` + `Clone/Debug/Display("cuda:0")/Default(new(0), panicking)`; `DeviceBaseAPI` (`same_device` = ordinal-ptr-eq ∧ stream ∧ flags-as-stored ∧ order), `DeviceRawAPI` (`Raw = CudaBuffer<T>`), `DeviceAPI` marker. |
| B2 | `CudaBuffer` + storage traits | `CudaBuffer<T>{CudaSlice<u8>, len, PhantomData}` (D1); alloc through `CudaAlloc`; deep-D2D `Clone` (panic via mapping); `truncate`/`device_ptr()`/`stream()`; manual `unsafe Send/Sync` + safety comments; `DeviceStorageAPI` (`len`, `get/set_index` = 1-element stream copy + sync, ptr accessors); hand-written `uninit_impl`/`assume_init_impl` (layout-identical rebuild, no transmute of internals). |
| B3 | Creation | `DeviceCreationAnyAPI`: `empty` (uninit alloc), `zeros` (`alloc_zeros`/memset — device-native), `full`/`ones`/`from_cpu_vec` (naive host-gen + `outof_cpu_vec`), `outof_cpu_vec` (blocking H2D on device stream, D5), `arange`/`linspace` naive. `tril/triu` **absent** (kernel round). Naive items tagged `// naive path` + registered below. |
| B4 | Transfers + `rt::cuda` helpers | `DeviceToCpuAPI` (D2H len-elements + sync); `DeviceChangeAPI` CUDA→CPU, CPU→CUDA (copies), CUDA→CUDA same-ordinal relabel (zero-copy, ptr-equal) + cross-ordinal peer copy (D8), lifetime-preserving Repr/ReprTo; free functions `device(ordinal)`, `device_count`, `is_available`, `memory_info`, `synchronize` (D4). |
| B5 | Flags | `CudaRuntimeFlags` (`Copy+PartialEq+Debug`, additive), `MathMode` `#[non_exhaustive]`; global `CUDA_FLAGS_DEFAULT: RwLock` env-seeded; typed accessors `flags()/flags_override()/math_mode()`, `set_flags_override(&mut self)`, `rt::cuda::set_default_flags` (D3). Env seed name proposed: `RSTSR_CUDA_MATH_MODE` — quick ruling wanted at impl time. |
| B6 | Allocator | Private `CudaAlloc` trait; driver-mempool impl; release threshold default unlimited; `empty_cache` (trim_to 0); OOM = threshold→0 → trim → retry once → typed `MemoryError` (D7); `RSTSR_CUDA_ALLOC_CONF` parser → global config → snapshot into `CudaOrdinalObj` → per-device setters. |
| B7 | Formatting | `DeviceFormatAPI` impl (D6). Interim gather = **per-visible-element `get_index`** (O(visible) single-element D2H; no kernel needed, no full copy ever); final = one gather kernel in the kernel round. Shared visible-index helper alongside `format_tensor_cpu_serial` (tiny `rstsr-native-impl` addition) so CPU/CUDA print identically. |
| B8 | Naive `OpAssignAPI` | Contiguous fast path = `memcpy_dtod`; strided = host round-trip. Unlocks `to_contig`/non-contig `into_vec`/`into_owned` for tests and sanity. Tagged naive. |
| B9 | Tests + docs | See below. |

Dependency order: P → B1 → B2 → B3 → B4; B5/B6 after B1 (parallel); B7 after
B4; B8 after B3; B9 continuous, final pass last.

## Tests (crate-local; no GPU CI → local/manual tier, recorded in `tests/README.md`)

Shared `core_func` bodies stay out (R1 bracket-index migration is deferred
ledger item 6; compile-blocked until then).

1. Registry identity: `new(0)` twice shares `CudaOrdinalObj`; distinct
   ordinals differ (skip without ≥2 GPUs).
2. `same_device` matrix: order / stream / flags-as-stored (`Some(x)` vs
   `None`-under-`x` → false; both `None` → true) / same-ordinal.
3. Flags: accessor resolution (`None` tracks global flip; `Some` pins),
   `set_flags_override` deep-clone semantics.
4. `CudaBuffer`: alloc/free; `truncate` shrinks logical len only; deep
   `Clone` (mutate copy, original unchanged); empty buffer ops; `device_ptr`
   non-null.
5. Creation round-trips vs CPU reference (`asarray` upload → `to_vec`
   compare; `zeros`/`ones`/`full`/`arange` values).
6. Transfers: CUDA→CPU→CUDA content equality; same-ordinal `change_device`
   relabel keeps `device_ptr` (zero copy); cross-ordinal peer copy (skip
   w/o 2 GPUs); view→view / owned→owned repr preservation.
7. Formatting: `Display`/`Debug`/`{:e}` strings identical to CPU device
   printing of the same values (incl. truncation at `MAX_PRINT`, strided
   views, 0-d, empty).
8. Allocator: second-iteration alloc_cached (loose timing); `empty_cache`
   increases `mem_get_info` free; absurd request → `MemoryError` (request
   ≈ `usize::MAX` bytes so it fails before reserving).
9. `get_index`/`set_index` round-trip incl. negative-stride view offsets.

## Naive-path register (replaced by the kernel round — do not fossilize)

| Path | Interim | Replacement |
|---|---|---|
| `full`/`ones`/`from_cpu_vec`/`arange`/`linspace` | host-generate + blocking H2D | device fill/sequence kernels |
| `OpAssignAPI` strided | D2H → CPU assign → H2D | strided copy kernel |
| `DeviceFormatAPI` gather | per-visible-element `get_index` | single gather kernel |
| `to_contig` non-contig | via naive assign | copy kernel |

## Out of scope (deferred-with-intent, per DECISIONS.md)

Op traits (unary/binary/reduction/indexing/sort/set), NVRTC + kernel cache,
cuBLAS/Lt handles + math-mode *consumption* (flags accessor only here),
events/pinned/`async_h2d`, guard sugar, TLS stream override, CUDA graphs,
DLPack bridge, NCCL, shared test-body migration, core `DeviceAllocatorAPI`,
upstream core proposals (`get_index_f`, R3/R5).

## Delivery

- **PR1**: crate + tests (this plan, B1–B9), review-gated, fork `ajz34` →
  `RESTGroup/rstsr` per `pr-writeup`.
- **PR2** (follow-up): facade wiring — `rstsr` feature `cuda =
  ["dep:rstsr-cuda"]`, `rt::cuda` namespace + flat `DeviceCuda` re-export
  (`prelude-conventions` skill at impl time).
- Acceptance demo (test or example): create `cuda:0`, upload + slice view,
   `to_device(CPU)` compare vs CPU compute, `Display` parity, flags
   accessor + mismatch error, `memory_info`/`synchronize`.

## Risks / checks

- cudarc 0.19.z patch-level churn → lockfile pins; probe re-run on bump.
- `cuda-12080` vs `cuda-13020` feature gates differ slightly in cudarc
  internals (some ctor cfg lists stop at 13010) — probe decides, primary-ctx
  path unaffected.
- Registry `Weak` cleanup: dropped device → recreated heavy obj must rebind
  pool/handles (test 1 covers indirectly).
- `get_index` sync cost on formatting interim (~µs × visible elements) —
  acceptable for printing; measured in B7 tests.

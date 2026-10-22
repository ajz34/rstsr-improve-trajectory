# DECISIONS — rstsr-cuda device and data representation (grill closed 2026-10-22)

Three grill rounds (R1 → R3, questions files in this directory; answers
received on screen and transcribed by the agent). Baseline facts were
extracted 2026-10-21 from rstsr @ `1c3bc11` + #164 (`000575d`), cudarc
0.19.10 @ `ec1cac7`, candle 0.11.0, dfdx 0.13.0, oxicuda 0.1.3, pytorch,
cupy @ `8f7a35d` (fact register F1–F10 in `GRILL-R1-QUESTIONS.md`).

**Final answers (R3, received on screen 2026-10-22 — agent transcription):**
(a) compare flags as stored — yes; (b) gate all ops — yes; (c) typed
accessors — yes; (d) `RwLock` global — yes; §3.1 Option A (strict stream
comparison) — yes; §3.2 defer shared override_stream (counter-proposal
accepted); §3.3 acknowledged. With these, every open item is closed.

---

## D1 — `Raw` buffer type: `CudaBuffer<T>`

```rust
pub struct CudaBuffer<T> {
    slice: CudaSlice<u8>,       // owning alloc: ptr + capacity + allocation stream
    len: usize,                 // logical element count (mutable: unique* truncation)
    _m: PhantomData<*const T>,  // !Send/!Sync poison; manual impls below
}
```

- Why not `CudaSlice<T>`: `CudaSlice<MaybeUninit<T>>` is unconstructible
  (`MaybeUninit: !DeviceRepr`) while `uninit_impl` requires
  `DeviceRawAPI<MaybeUninit<T>>::Raw`; the tensor layer transmutes
  `&mut Raw<T> ↔ &mut Raw<MaybeUninit<T>>` at ~10 sites; `assume_init_impl`
  is hand-written per device. A T-independent layout + `PhantomData`
  satisfies all of it (F2).
- Allocation: `stream.alloc::<u8>(len * size_of::<T>())` — `DeviceRepr`
  stops mattering at the Raw boundary (no dtype excluded from storage).
- `Clone` = **deep D2D copy** on the owning stream (`memcpy_dtod`), panicking
  through the error mapping (backtrace-preserving `rstsr_unwrap` style).
  Rationale: `Tensor::clone` → `into_owned` → `Raw::clone`; a shallow clone
  would alias two owned tensors (F1). #164 removed `Raw: Clone` from
  host-read paths; materializing ops still require it, so it is implemented.
  (Load-only future backends, e.g. hdf5, may omit `Raw: Clone` entirely.)
- Inherent API: `len`, `truncate(u)` (= `len = u.min(len)`; the `unique*`
  contract trusts the device to shrink its raw — F3), `device_ptr()` (what
  `get_index_ptr` returns; documented "never dereference on host" — answers
  R7 without a core change), `stream()`.
- Empty buffer: null-slice (`CudaStream::null::<u8>()`) behind a single
  constructor, so switching to a real 0-byte alloc is a one-site change.
- Manual `unsafe impl Send/Sync` with safety comments (data touched only via
  device ops; host aliasing impossible).
- **No pool-ready internals now** (no allocator backref field): `Raw` is
  opaque to core, so internals evolve non-breaking when a Rust pool lands.

## D2 — Device object model

```rust
static CUDA_FLAGS_DEFAULT: RwLock<CudaRuntimeFlags> = /* env-seeded */;

pub struct DeviceCuda {
    ordinal: Arc<CudaOrdinalObj>,      // heavy identity, registry per ordinal
    stream: Arc<CudaStreamObj>,        // execution stream (D4)
    flags: Option<CudaRuntimeFlags>,   // None = track global (D3)
    default_order: FlagOrder,          // light snapshot, compared (CPU precedent)
}
```

- **`CudaOrdinalObj` (heavy)**: `Arc<CudaContext>`, `default_stream`
  (`CU_STREAM_NON_BLOCKING`; never the legacy null stream), allocator config
  (D7), cuBLAS/Lt handle management (per-ordinal, behind a mutex; our own
  FFI over cudarc `sys`/`result` — cudarc is the substrate for maintenance,
  cuda-version gates and `CudaSlice`, *not* necessarily its safe wrappers),
  later the NVRTC kernel/module cache. **No `override_stream`** (D4).
- **Registry**: process-global
  `OnceLock<Mutex<HashMap<ordinal, Weak<CudaOrdinalObj>>>>`, fetch-or-create
  in `DeviceCuda::new(ordinal)`. The only process-global besides the flags
  default. Consequence: two `new(0)` share one pool/handles/flags-default —
  no state forking (F9).
- `Default` = `new(0)`, panics via error mapping on CUDA-less machines (no
  lazy init machinery).
- `same_device`: `Arc::ptr_eq(&ordinal) && stream == && flags == &&
  default_order ==` — ordinal identity by ptr-eq; **stream strictly
  compared** (D4); **flags compared as stored, not resolved** (D3);
  order compared (CPU precedent).
- `Display` = `cuda:0` (ordinal only); `Debug` = ordinal + stream id +
  order + flags detail.

## D3 — Flags

- `CudaRuntimeFlags`: small `Copy + PartialEq + Debug` struct, additive
  fields (`math_mode: MathMode` now; `fast_math: bool` when NVRTC lands;
  ...). `MathMode` is `#[non_exhaustive]` (Strict f32 / Tf32; later
  reduced-precision variants). Flags are identity, compared per D2.
- Global `CUDA_FLAGS_DEFAULT: RwLock<CudaRuntimeFlags>` (plain lock in a
  static — no `Arc`), seeded once from env at first device creation,
  safely mutable via `rt::cuda::set_default_flags` (+ getter).
- Per-device `flags: Option<CudaRuntimeFlags>`: `None` tracks the global,
  `Some(x)` pins. **Launch-time resolution**: a flag-consuming op (GEMM
  compute type first) reads `flags.or(global)` from the input tensor's
  device. This is what makes the torch-style toggle work without re-tagging
  (creation-time snapshots would strand existing tensors — rejected).
- Comparison semantics (both explicitly accepted): representation as stored
  (`Some(x)` ≠ `None` even when the global is `x` — accepted
  false-positive; fix is a one-line re-tag) **and** flags gate *all*
  multi-tensor ops, not only flag-consuming ones (`add` errors across
  differing flags — same price `default_order` already pays).
- Typed accessors only (no stringly keys): `flags() -> CudaRuntimeFlags`
  (resolved), `flags_override() -> Option<..>`, `math_mode() -> MathMode`,
  `set_flags_override(&mut self, Option<CudaRuntimeFlags>)`. Stringly keys
  exist only in the env parser.
- Guards deferred: `with_default_flags` RAII over the global is additive
  later; nothing depends on it now.

## D4 — Streams

- `CudaStreamObj`: wrapper over a cudarc non-blocking stream; `new(priority)`,
  `wait(&other)` / `join`, `synchronize`; identity equality (handle +
  context) used by `same_device`. `CudaEvent` exposure deferred to the
  transfer round (`wait`/`join` suffice for manual ordering).
- The device value carries the **execution stream**; `set_stream(&mut self,
  &Arc<CudaStreamObj>)` is the mutating setter (`set_default_order`
  precedent). `new(ordinal)` starts on the ordinal's default stream.
- **Strict cross-stream policy (Option A)**: `same_device` compares the
  stream; mixed-stream operands fail with `DeviceMismatch`. Cross-stream
  producer/consumer is explicit: clone the device, `set_stream`, re-tag
  entry tensors via same-ordinal `to_device` (view relabel — D8). Warn-only
  was rejected; LHS-implicit launch was rejected.
- **No shared `override_stream`** (priority-chain proposal withdrawn — the
  R24 `with_blas_num_threads` cross-thread hazard). A TLS-based override may
  be added later as purely additive sugar; `with_stream` stays reserved for
  that guard. Deferred, not designed.
- **Buffer vs device stream invariant**: `CudaBuffer`'s allocation stream
  (inside `CudaSlice`) owns free/reuse ordering (stream-ordered pool); the
  device value's stream owns kernel launch. Re-tagging never moves data.
- **Event tracking stays ON** (needed even under strict comparison: Drop of
  a buffer allocated on s1 while the device points at s2; launch-time waits
  for cross-stream operand slices). `unsafe disable_event_tracking()` is the
  documented single-stream opt-out lever.
- `rt::cuda::synchronize(&device)`. cuBLAS enqueue: `cublasSetStream(<op
  stream>)` before each call on the per-ordinal handle.

## D5 — Host reads / transfers

- `DeviceToCpuAPI` (post-#164 capability trait): stream-ordered D2H on the
  buffer's stream + sync; `to_vec`/`into_vec`/`Debug` route through it.
- `get_index`/`set_index` stay infallible in-trait; our impl syncs the
  one-element copy then panics via the error mapping on driver failure. A
  fallible core twin (`get_index_f`) is a separate upstream proposal, not a
  blocker.
- `outof_cpu_vec(Vec<T>)`: blocking by contract (consumes the host Vec);
  H2D on the device value's stream. This is the naive path.
- True async H2D/D2H on side streams: separate explicit API
  (`rt::cuda::async_h2d`-style) with pinned staging + events — designed in
  the transfer round; deferred with intent.

## D6 — Formatting (post-#164)

- CUDA implements `DeviceFormatAPI` directly; no `change_device` round trip.
  **Visible-gather**: enumerate the O(ndim·min_print) elements visible under
  `MIN_PRINT`/`MAX_PRINT` truncation, D2H that small gather, format from it
  — never a full-tensor copy. The visible-index enumeration helper is shared
  with `rstsr-native-impl` (next to `format_tensor_cpu_serial`) so CPU and
  CUDA print identically. Contract fixed now; the gather kernel lands with
  the kernel round (interim: full D2H only below a small-size cutoff).

## D7 — Allocator

- **Private `CudaAlloc` trait** inside rstsr-cuda
  (`alloc_async`/`free`/`set_release_threshold`/`trim`); default
  implementation = **driver mempool** (`cuMemAllocAsync`/`cuMemFreeAsync`
  via `CudaSlice`; stream-partitioned reuse by driver construction).
- Release threshold default: unlimited (torch/cupy-like caching). Manual
  valve `rt::cuda::empty_cache(&device)` → `cuMemPoolTrimTo(0)` (+ momentary
  threshold-0). No release-on-idle (impossible without thrashing).
- **OOM protocol**: on allocation failure → force threshold 0 (returns all
  cached blocks) → retry once → still failing → typed
  `RSTSRError::MemoryError` with requested/total/free.
- Config chain: `RSTSR_CUDA_ALLOC_CONF` (torch-style `key=value,key=value`,
  one env var, extensible) → parsed once into a global atomic config →
  snapshotted into `CudaOrdinalObj` at device creation → per-device setters
  override thereafter.
- A Rust-side pool may replace the driver pool later behind the same trait
  (driver-pool fragmentation/latency would be the trigger); non-breaking via
  the trait + opaque `Raw`.
- A core-level `DeviceAllocatorAPI` (CPU `Vec<T, A>` generalization) is a
  **separate future grill** — it touches every `Raw = Vec<T>` site (iter/
  Index family, `DeviceFormatAPI` blanket impl, host-view `asarray`,
  `rstsr-blas-traits` pin) and needs a concrete CPU use-case first.

## D8 — `change_device`

- Lifetime-preserving, `change_shape`-style: view→view, owned→owned without
  clone; cloned owned as fallback. Repr/ReprTo per direction: pass-through
  for same-ordinal relabel, `DataOwned` for copying directions, `DataRef`
  re-tag for same-ordinal borrows.
- Same ordinal: **relabel** — new device value (stream/flags/order), buffer
  untouched, zero data movement.
- Different ordinal (CUDA→CUDA): peer copy (`cuMemcpyPeerAsync` on the
  source stream + sync), allocation on the target stream.
- CUDA↔CPU: D2H/H2D copies, both directions implemented in rstsr-cuda
  (coherence fine; printing no longer depends on it post-#164).

## Names (public API)

`rstsr-cuda` crate · `DeviceCuda` · `CudaBuffer<T>` · `CudaOrdinalObj` ·
`CudaStreamObj` · `CudaRuntimeFlags` · `MathMode` · `CUDA_FLAGS_DEFAULT` ·
`rt::cuda::*` facade namespace · `RSTSR_CUDA_ALLOC_CONF` env var.

## Deferred with intent (later rounds, in rough order)

1. NVRTC kernel layer + disk cache + typed launchers (grill topic 4).
2. cuBLAS/cuBLASLt GEMM wiring + math-mode consumption; linalg tier / R10
   generalization (topics 5-6).
3. Dtype set + TypeId dispatch for kernels (topic 8).
4. Transfer round: `CudaEvent` exposure, pinned staging, `async_h2d/dtoh`
   (side-stream overlap).
5. Additive sugar: `with_default_flags` / TLS `with_stream` guards.
6. Test/CI strategy (CPU↔CUDA parity harness; R1 `get/set` migration for
   shared test bodies) (topic 11); milestone ordering (topic 12).
7. CUDA graphs; DLPack bridge; NCCL.
8. Core `DeviceAllocatorAPI` framework grill (needs CPU use-case).
9. Upstream proposals: fallible `get_index_f`; `axes_iter` bound relaxation
   (R3); generic `DeviceCpuSerial → BOut` change_device (R5).

## Changelog of the debate (for the record)

- R1 positions vs R2/R3 outcomes: registry kept; live-atomic `math_mode`
  **replaced** by the user's Option+global design (stronger: zero-retag
  global toggle + loud mixed-mode errors); single-hidden-stream **replaced**
  by user-visible streams with strict comparison; `with_stream` value
  transform **replaced** by `set_stream` mutation + explicit re-tag;
  driver-mempool allocator kept; guard machinery dissolved into deferred
  additive sugar.
- Corrections to the grill-doc baseline accepted along the way: doc 04 §3's
  "cheaply clonable handle Raw" is wrong for rstsr (deep D2D instead);
  #164's capability traits removed the Display→change_device and
  Debug→`Raw: Clone` couplings my R1 assumed.

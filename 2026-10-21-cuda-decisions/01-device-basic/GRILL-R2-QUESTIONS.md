# GRILL R2 — device and data representation for `rstsr-cuda`

This round absorbs the R1 answers (received on screen 2026-10-21; transcribed
as background in §0 — agent-authored summary, the answers file remains
user-owned), re-checks two of them against rstsr master `000575d` (#164, see
§1), and goes deep on the three decisions you flagged as hard to revert: the
device state model (§2), user-visible streams (§3), and the allocator trait
(§5). Per your heading note, "v1/v2" framing is dropped — each decision below
is presented as *the* design, with an explicit reversibility note.

---

## 0. Background — R1 answers transcribed (agent-authored)

| Item | Your answer |
|---|---|
| Heading | Step-by-step, but the first version must be well-polished; hard-to-revert ideas need deeper challenge. |
| #164 | `Raw: Clone` bound removed from host-copy paths (I verified; §1). CUDA data should still implement `Clone` = deep copy. Future hdf5-like load-only backends may omit `Raw: Clone`. |
| F6 | cudarc is used for maintenance + `cuda-*` feature gates + `CudaSlice`; **not** necessarily its safe wrappers — call cuBLAS/cuBLASLt through `result`/`sys` like the CPU BLAS backends do. |
| Q1 | Agreed (`CudaBuffer` wrapper). Buffer's stream: to be introduced or ignored — settle here. Clone = deep copy, keep. |
| Q1-p2 | Empty buffer as null-slice OK, but ensure easy revert to a real 0-byte alloc. |
| Q1-p3 | No pool-ready internals now; wary of Arc-everything ("we are not a DL framework"). |
| Q2 | Proposal: a `DeviceAllocatorAPI` for both CPU (`Vec<T, A: Allocator>`) and CUDA. Default no-op. |
| Q2-p1 | Cache like torch/cupy by default (threshold None = use all memory); wondered about release-on-idle. |
| Q2-p2 | Want OOM-retry: retry until everything is released; only then error. |
| Q2-p3 | Env var accepted; asks where env config lives (device object vs global mutable static). |
| Q3 | Wants heavy/light split explained properly; heavy = one per GPU (equal→true, clone→shallow), light = flags (equal→all flags same, clone→deep). |
| Q3-p1 | No lazy Default — panic on CUDA-less machines is fine. |
| Q3-p2 | math_mode = light-weight, compared; ordinal/heavy = not compared (same-GPU identity); stream = maybe a third category (heavy-ish but comparable?). |
| Q3-p3 | Device `Display` = `cuda:0` (ordinal only); `Debug` detailed. |
| Q4 | Prefers streams user-visible from day one; asked to be re-challenged. |
| Q5 | Heavy-global + light-per-device; TLS mode "straightforward" in that scheme. |
| Q5-p1 | One flag; make it comparable. |
| Q6 | RAII guard OK if the device split fits; if flags were snapshot state, closure-scoped would be the only way. |
| Q7 | #164 changed Display — check whether full D2H is avoidable; fewer copies matters more than speed for printing. |
| Q8 | same-ordinal `change_device` = no data copy (device/flag relabel); different ordinal = copy; prefer Cow-style output repr so caller chooses `.into_owned()`/view. |
| Q9 | Confirm `MathMode` vs a broader flag set. |

---

## 1. rstsr master moved: #164 (`000575d`) and what it changes

Verified against the tree (not just the commit message):

- **`DeviceToCpuAPI<T>`** is a new capability trait; `to_cpu_vec`/`into_cpu_vec`
  on `DeviceStorageAPI` forward to it; `DataCloneAPI` no longer requires
  `Self::Data: Clone`, and `DataOwned`'s `DataCloneAPI` impl is unconditional
  (moving an owned buffer is a move, not a clone). Host reads therefore
  constrain neither `Raw` nor dtype globally — the CUDA impl decides cost.
- **`DeviceFormatAPI<T, D>`** with one method per format mode; `Display`/
  `LowerExp`/`UpperExp`/`Debug` bound only on it. The `DeviceChangeAPI` bound
  on `Display` and the `Raw: Clone`/host-copy bounds on `Debug` are **gone**;
  the walker lives in `rstsr-native-impl` with a blanket impl for
  `Raw = Vec<T>`; non-Vec devices implement the four methods themselves.
- Remaining `Raw: Clone` bounds are exactly the materializing ops
  (`Clone for Tensor`, `into_owned`, `to_contig`/reshape, binary ops...) —
  so `CudaBuffer: Clone` (deep D2D) is still required and sufficient.

**Consequences for this grill:** my R1 Q7 is revised — CUDA printing is a
*device-side formatting capability*, not a `change_device` round trip (§6);
and nothing about `CudaBuffer`'s shape changes (F2/F3 reasoning unaffected).

---

## 2. R2-Q1 — Device state model: classify every field (heavy / live / light)

This merges your Q3 + Q5 answers into one coherent model, and is where I
push back on one specific placement (math_mode).

### 2.1 Why R1's option (c) was "worst of both" — the explanation you asked for

(c) was: registry dedups only heavyweight resources (pool, caches), flags stay
per-instance. The failure mode: any flag that must act like a *policy* (set
once on a handle, affect all subsequent launches, including ops on tensors
created earlier) silently stops doing so, because every tensor's `Storage`
holds its own device clone — a per-instance flag is a **snapshot frozen at
tensor creation**. Mutating your original handle never reaches existing
tensors. So for every field you must *individually* decide: is this a
snapshot (light) or live (shared)? (c) pretended that decision could be made
once for "resources" vs "flags" — it can't; it's per-field. That is exactly
the heavy/live/light classification below, so I'm adopting your split
framing, refined.

### 2.2 The classification, with the litmus tests

Three kinds of state (your "stream as third category" hunch is close — it
gets its own class, §3):

| Class | Storage | Clone | `same_device` | Examples |
|---|---|---|---|---|
| **Heavy** (identity) | inside `Arc<CudaHeavy>`, one per ordinal (registry) | shallow (Arc) | compared as identity (same Arc ⟺ same ordinal) | `CudaContext`, default stream, allocator config, BLAS/Lt handle mgmt, (later) kernel/module cache |
| **Live policy** (shared, mutable) | inside `CudaHeavy`, atomics/Mutex, mutated via `&self` | n/a (shared) | **never compared** | `math_mode: AtomicU8` |
| **Light snapshot** | by value in the device struct | deep (it's Copy-cheap) | **compared** | `default_order` (exactly the CPU precedent: `DeviceCpuSerial.same_device` compares only `default_order`) |
| **Selector** (§3) | by value in the device struct (`Arc<CudaStreamObj>`) | shallow (Arc) | **never compared** | `stream` |

```rust
pub struct DeviceCuda {
    heavy: Arc<CudaHeavy>,        // registry-canonical per ordinal
    stream: Arc<CudaStreamObj>,   // = heavy.default_stream or a user stream (§3)
    default_order: FlagOrder,     // light, compared
}
pub struct CudaHeavy {
    ctx: Arc<CudaContext>,
    default_stream: Arc<CudaStreamObj>,
    flags: CudaLiveFlags,         // math_mode: AtomicU8, +additive fields
    allocator: CudaAllocConfig,   // §5
    blas: BlasHandles,            // our own FFI handles over cudarc sys/result (F6)
    // later: kernel/module cache
}
```

**The litmus tests** (all three must pass; cite them when challenging any
placement):

- **S1 — policy toggle across reused tensors** (the DF-MP2 pattern: tensors
  uploaded once, then used in *both* strict-f32 and TF32 calls):
  `rt::matmul(&a, &b)` where `a` was created before the toggle and `b` after
  must still work. If `math_mode` is light+compared, `same_device` fails →
  `DeviceMismatch` → **workload breaks**. This is my direct challenge to your
  Q3-p2/Q5 answer ("math_mode light, compared"): light+compared fails S1.
  Light+uncompared is incoherent (a snapshot nobody reads at launch is dead
  state; reading the *input tensor's* snapshot at launch makes op results
  depend on creation order — worse). So math_mode must be **live** (shared
  atomic in heavy, read at launch from the input device) — which is doc 04
  §2's original conclusion, now with the framework change (#164) not
  affecting it. "Comparable" is then just `MathMode: PartialEq` for the
  getter/comparison API, not `same_device` participation.
- **S2 — stream overlap**: producer/consumer on two streams must be
  expressible without banning mixed-stream operands (§3).
- **S3 — purity**: `d.with_math_mode(Tf32)` is not a different device;
  `d.with_stream(s)` is not a different device either — they are the *same
  GPU*, and ops between their tensors must not fragment.

Note the `DeviceFaer` precedent *does* compare `num_threads` — but
`num_threads` is genuine light state (per-value, affects execution, never
toggled mid-workload on shared tensors). The distinction is not
"config-vs-resource" but "snapshot-tolerable vs toggle-live". Our inventory:
`default_order` snapshot-tolerable (CPU-identical semantics), `math_mode`
toggle-live. If you accept S1, this resolves Q3-p2/Q5-p2.

### 2.3 Registry = the mechanism for "heavy is global"

Your "let heavy part be global" = my registry: process-wide
`OnceLock<Mutex<HashMap<ordinal, Weak<CudaHeavy>>>>`, `DeviceCuda::new(ordinal)`
fetch-or-creates. Two `new(0)` calls return values sharing one heavy (same
pool, same handles, same live flags — no 8 GB cache fight). This is the only
process-global we add; it is a read-mostly identity cache, all *policy*
mutability stays inside `CudaHeavy`. `Default` = `new(0)`, panics through the
error mapping if CUDA init fails (your Q3-p1 — no lazy machinery).

`Display` = `cuda:0` (ordinal only); `Debug` = ordinal + stream id + order +
math_mode snapshot (your Q3-p3).

### 2.4 `same_device` — final proposal

```rust
fn same_device(&self, other: &Self) -> bool {
    Arc::ptr_eq(&self.heavy, &other.heavy) && self.default_order == other.default_order
}
```

Stream excluded (S2/S3, §3), math_mode excluded (S1), ordinal identity via
heavy ptr-eq. This is your heavy/light scheme with the two exclusions the
litmus tests force. **Reversibility:** the *struct shape* is hard to revert
once published; the `same_device` predicate is semantics (soft-revertable
with a release note, but churns user code) — hence deciding it now.

---

## 3. R2-Q2 — Streams visible from day one: yes; ambient "current stream": no

You asked to be re-challenged on preferring user-visible streams (R1 Q4b).
Challenge delivered — and I **agree on visibility**, but I argue against the
torch/cupy *ambient TLS* half, on a rstsr-specific ground:

**The killer argument: rstsr has no ambient dispatch point.** Every tensor
creation takes `&device` explicitly (`rt::asarray((vec, &d))`,
`rt::zeros((shape, &d))`), and every op derives its device from the *input
tensor's stored device*. There is no place where a hidden "current stream"
could be picked up that the explicit device value doesn't already provide. A
TLS current-stream would be a second source of truth for what the device
value already says — pure hazard, zero ergonomic gain (torch needs TLS
because `torch.zeros(..., device="cuda")` kernels dispatch through ambient
context; we don't).

**The design — streams as first-class values carried by the device:**

- `rtsr-cuda` exposes `CudaStream` (wrapper over cudarc's non-blocking
  stream: `new(priority)`, `wait(&other)`/join, `synchronize`; events
  deferred unless you want them now).
- The device value carries a **stream selector** (`stream` field, §2.2):
  defaults to `heavy.default_stream` (created once per ordinal,
  `CU_STREAM_NON_BLOCKING` — never the legacy null stream).
- **Stream selection = device-value transformation**:
  `let d2 = d.with_stream(&s);` — creates a new cheap `DeviceCuda` value
  (same heavy, different selector). All ops/creations through `d2` run on
  `s`. Scope end (shadowing) restores `d` — **this is the Rust form of
  Python's `with stream(s):`**: lexical instead of dynamic, no guard object,
  no `mem::forget` caveat, no TLS. dfdx's internal `par_stream` dance is this
  pattern done manually; we make it the public primitive.
- **Cross-stream operands are legal**: `same_device` ignores stream (§2.4);
  the op launches on the LHS/first input's stream. Safety is *by
  construction* through cudarc event tracking (F5): launch args insert
  `stream.wait(write_event)` for every operand slice from another stream —
  the consumer kernel waits for the producer's last write. So S2 passes
  without banning anything, with a documented per-launch event cost. This is
  why your "stream as comparable third category" should be **no**: comparing
  streams would make the producer/consumer pattern a type error while the
  event machinery already makes it correct.
- **Event tracking stays ON** (default). The `unsafe disable_event_tracking()`
  lever exists for single-stream hot paths, documented as "you promise one
  stream per ordinal". (Cost, from the extraction: ~2 event ops per kernel
  argument — measured, not guessed, before anyone disables it.)
- **Buffer-stream vs device-stream invariant**: `CudaBuffer` keeps its
  allocation stream (inside `CudaSlice`) — frees stay stream-ordered
  (F4/F8) and correct even after `with_stream` re-tagging or same-ordinal
  `change_device` (§7). Kernels launch on the *device value's* stream;
  buffers record their *allocation* stream. Two different things, both
  needed; document. This answers your Q1 stream remark: the buffer's stream
  is **not ignored** — it owns free/reuse; the device's stream owns
  execution.
- **cuBLAS handles** (F6 + your wrapper stance): our own FFI over
  `cudarc::cublas::sys/result`. One handle per heavy part behind a `Mutex`;
  `cublasSetStream(current op stream)` before each call (handle creation
  per-stream is the later optimization; the mutex is held only for enqueue,
  ~µs, and only serializes same-GPU threads).
- **Async H2D/D2H on side streams** needs pinned staging + events (CuPy §3.4)
  — `outof_cpu_vec` stays blocking (it consumes the host Vec); the pinned
  async API is its own later round. The overlap win you'd want (S2) is
  still available for D2D/compute via two device values.

**What this makes revertable vs not:** the `Stream` type, `with_stream`, and
stream-blind `same_device` are public surface (decide now — done above);
the event-tracking default, handle strategy, and pinned-staging are internal
(freely changeable). I believe this is the "polished first version" shape:
nothing ambient, everything explicit, Python-`with` expressible lexically.

**If you still want ambient TLS** (torch muscle-memory porting, or future
graph capture that wants a dynamic scope): it can be added *later* as a
sugar layer that `with_stream`-style values already subsume — the reverse
(remove ambient after publishing it) is the breaking direction. That
asymmetry is the whole argument.

---

## 4. R2-Q3 — math_mode placement + the guard (Q5/Q6 closed out)

Given §2 (math_mode = live atomic in heavy) and §3 (no ambient):

- `DeviceCuda::set_math_mode(&self, mode)` / `math_mode(&self) -> MathMode`
  — interior mutability on `&self` (a `&DeviceCuda` recovered from any
  tensor can flip it; the change is visible to all handles and all
  subsequent launches, existing tensors included — S1 passes).
- **RAII guard works** precisely *because* the state is shared (your Q6
  conditional, resolved in favor of (a)):
  `rt::cuda::with_math_mode(&d, Tf32) -> MathModeGuard` storing the previous
  value; `Drop` restores; nesting composes; `mem::forget` leaks the override
  (documented, same class as `MutexGuard`). No closure form needed; your
  "closure is the only way" case applied to *snapshot* flags, and we now
  have none (order is CPU-precedent `set_default_order(&mut self)`).
- **TLS per-thread mode: rejected** — it reintroduces ambient state against
  §3's argument, and a tensor's device has no thread. Per-GPU semantics
  stands (two threads sharing one GPU with overlapping guards interfere —
  recorded divergence; your workload is one worker per GPU).
- Q9 follow-up: the live-flags container is `CudaLiveFlags` (in heavy) —
  `math_mode: AtomicU8` now; **additive** fields later (a `fast_math`
  compile flag when NVRTC lands; a `reduced_precision` fp16-accumulate
  policy if needed). `MathMode` keeps its name (it selects the cublasLt
  compute-type axis); unrelated future knobs get their own fields, not
  enum variants. `#[non_exhaustive]` on `MathMode`.

---

## 5. R2-Q4 — Allocator: internal trait now, core trait separately

### 5.1 Your `DeviceAllocatorAPI` idea — challenge: scope

The CUDA need is real but small (§5.2); the CPU half (`Vec<T, A>`) is a
**framework change whose cost lands far from CUDA**:

- Every `Raw = Vec<T>`-gated site in core would need generalizing to
  `Vec<T, A>` or would simply not match a custom-allocator device: the new
  `DeviceFormatAPI` blanket impl (`Raw = Vec<T>` exactly), the
  iterators/`Index`/`as_ptr` family (R1/R2 blockers), host-view `asarray`
  (`Vec::from_raw_parts`), `pack_array`, and `rstsr-blas-traits`'
  `Raw = Vec<T>` pin (R10). A `Vec<T, MyAlloc>` device today would be a
  second-class citizen across the whole surface.
- Rust already allows a *process-global* allocator override
  (`#[global_allocator]`) for CPU — the novel bit your trait buys is
  *mixed allocators in one process*. That needs a concrete CPU use-case
  (an arena device? mimalloc-per-device?) before it earns the refactor.

**Proposal:** (i) now — a **private** `CudaAlloc` seam inside rstsr-cuda
(trait with `alloc_async/free/set_release_threshold/trim`; default impl =
driver mempool; a Rust pool can slot in later, and `CudaBuffer` internals
change then — non-breaking, `Raw` is opaque); (ii) separately — a
`DeviceAllocatorAPI` framework grill (candidate `02-allocator-framework`)
if/when you can name the CPU use-case. CUDA does not block on it, and it
does not block CUDA. Challenge back: what CPU allocator use-case do you
actually have in mind? If it's just "nice to have", it's R10-scale work for
zero current consumers.

### 5.2 Driver-mempool defaults (your p1–p3)

- **p1 — threshold**: default = unlimited (torch/cupy-like: freed blocks
  stay cached). **Release-on-idle: not possible and not wanted** — there is
  no driver idle callback, and an auto-trim heuristic would thrash
  (trim→realloc is a `cudaMalloc` round trip). Provide
  `rt::cuda::empty_cache(&device)` → `cuMemPoolTrimTo(0)` (+
  `set_release_threshold(0)` momentarily), documented as the manual valve.
- **p2 — OOM retry** (Rust translation of torch's protocol; no GC exists,
  drops are deterministic and owned by callers): on `cuMemAllocAsync`
  failure → force release threshold to 0 (returns all cached blocks) →
  retry once → still failing → typed `RSTSRError::MemoryError` with
  requested/total/free. "Retry until everything released" is exactly these
  two attempts; anything more would need to drop *user's* tensors, which we
  cannot.
- **p3 — env config home**: your "global static mutable" instinct, refined:
  a `static CUDA_CONF: CudaConf` of atomics (release threshold, later kernel
  cache dir...), parsed **once** from `RSTSR_CUDA_ALLOC_CONF`
  (torch-style `key=value,key=value`, extensible without new vars) at first
  device creation; device creation **snapshots** it into
  `CudaHeavy.allocator`; per-device setters override thereafter. So: env →
  global default → per-device override; programmatic writes to the static
  affect only devices created later. No per-device env re-reading, no TLS.

---

## 6. R2-Q5 — Formatting and host reads after #164 (Q7 revised)

- **Printing does NOT full-D2H** (your requirement, now structurally easy):
  CUDA implements `DeviceFormatAPI` itself — enumerate the *visible*
  elements under `MIN_PRINT`/`MAX_PRINT` truncation (O(ndim·min_print)
  elements, not n), D2H that small gather, format from it. The visible-index
  enumeration should be shared with `rstsr-native-impl` (a
  visible-slice helper next to `format_tensor_cpu_serial`) so CPU and CUDA
  print identically; the gather itself is one small kernel (lands with the
  kernel round; until then: full D2H only for tensors below a small-size
  cutoff, visible-gather after — the *contract* is fixed now, the fast path
  arrives with kernels).
- `DeviceToCpuAPI` implemented for CUDA: stream-ordered D2H on the buffer's
  stream + sync (§3 invariant); required by `to_vec`/`into_vec`/`Debug`.
- `get_index`/`set_index` remain infallible in-trait; our impl syncs the
  one-element copy then panics through the error mapping on driver failure.
  A fallible `*_f` core twin: propose upstream separately, not a blocker.
- `outof_cpu_vec`: blocking, v-now and by contract (consumes the Vec).

## 7. R2-Q6 — `change_device` semantics (Q8, near-settled)

Confirm my reading of your note:

- **Same ordinal**: relabel — replace the device value (flags/stream/order
  from the target), keep `CudaBuffer` untouched, **zero data movement**
  (CPU↔CPU relabel precedent). Works for `with_stream`-differing targets
  too (buffer keeps its allocation stream; §3 invariant).
- **Different ordinal (CUDA→CUDA)**: peer copy
  (`memcpy_peer`/`cuMemcpyPeerAsync` on the source stream, then sync), new
  allocation on the target device's stream.
- **CUDA↔CPU**: D2H/H2D copy, both directions implemented in the crate
  (coherence fine; #164 removed the `Display` dependency, so this is no
  longer forced by printing — it's for `.cpu()`-style workflow only).
- **Repr types**: consuming `change_device`/`into_device` → `Repr =
  DataOwned` when a copy happened, `Repr = R` (pass-through) for
  same-ordinal relabel — matching the CPU impl shape. Borrowing
  `to_device` → `ReprTo = DataRef` re-tag for same ordinal, `DataOwned` for
  copying directions (a CPU-side *view of CUDA memory* cannot exist, so no
  borrow-repr cross-space). Your "Cow output" wish: I read it as "don't
  force `.into_owned()` semantics on callers" — the `Repr`/`ReprTo`
  associated types already give per-direction output reprs, and `change`
  (non-consuming tensor→tensor) returns the same repr family the input had
  where legal. If you literally want `TensorCow` returns from `to_device`,
  say so and I'll spec `ReprTo = DataCow` (Owned arm for copies) — but note
  the Cow-Ref arm is only ever constructible same-ordinal.

## 8. Ledger

**Settled (this round, pending your R2 answers confirming):**
`CudaBuffer<T>{CudaSlice<u8>, len, PhantomData}` + deep D2D `Clone` +
`truncate` + null-empty behind a single constructor (easy revert to real
0-byte alloc); cudarc-substrate-with-own-BLAS-FFI (F6); heavy/live/light
classification with registry; `Default` = panicking `new(0)`; device
`Display cuda:0` / detailed `Debug`; streams as values + `with_stream`
lexical scoping + stream-blind `same_device` + event tracking ON;
`math_mode` live-atomic + RAII guard + `#[non_exhaustive]` +
`CudaLiveFlags` container; private `CudaAlloc` + driver mempool +
unlimited threshold + `empty_cache` + trim-retry OOM + `RSTSR_CUDA_ALLOC_CONF`
global-snapshot-override; formatting by visible-gather; `change_device`
relabel/copy split.

**Open (need your R2 answers):**
1. §2: math_mode live-not-light (the S1 challenge) — accept?
2. §3: no-ambient streams, `with_stream` as the `with` equivalent — accept?
3. §5.1: `CudaAlloc` private now + core `DeviceAllocatorAPI` as a separate
   grill — and what's your concrete CPU allocator use-case?
4. §7: the Cow-vs-ReprTo reading of your Q8 note.
5. §3: events deferred (Stream has `wait`/`synchronize` only) — enough?

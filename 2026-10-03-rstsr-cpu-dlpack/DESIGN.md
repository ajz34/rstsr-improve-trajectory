# `rstsr-cpu-dlpack` — design

- **Date**: 2026-10-03 · **Status**: design for the prototype (no code yet in this directory)
- **Inputs**: `../2026-10-03-rust-numpy-review/DECISIONS-R2.md` (scope: pure crate, no pyo3;
  assumptions A1–A4) and `../2026-10-03-rust-numpy-review/RESPONSE-discussion-R1.md` (verifications
  V1–V5, corrections C1–C5)
- **Pinned sources**: rstsr `f179c46` (v0.9.0); `dlpack-ffi` 1.3.0; numpy `v2.5.2`
  (`~/Git-Others/numpy`); DLPack v1.3 header. Line anchors below are against these.

The crate is the first consumer of `dlpack-ffi` and the first out-of-tree rstsr storage repr; every
design choice below was checked against the sources, and anchors are given so the claims can be
re-verified cheaply.

## 1. Scope

**In scope (v1).**

- `rstsr` → DLPack export: owned (move), shared (zero-copy, read-only share), and copy fallback.
- DLPack → `rstsr` import: `DLManagedTensorVersioned` (v1.x) and legacy `DLManagedTensor`, as a
  zero-copy **read-only** tensor that owns the producer's lifetime (deleter travels with it).
- CPU devices only (every rstsr CPU device has `Raw = Vec<T>`; see `REVIEW.md` and V2).
- Rust-side API, plus a reference Python-side capsule holder (~30–80 lines, `examples/` + the
  prototype's `python/`) — `PyCapsule` is a CPython type and cannot be produced in pure Rust
  (`DECISIONS-R2.md` §2).

**Out of scope (v1), with the reason.**

| Not in v1 | Why |
|---|---|
| Mutable zero-copy import/export | `DataMutAPI::raw_mut` hands out `&mut Vec<T>`; safe code could `push` into a span that must never be resized (foreign allocation) → unsound surface. A bounded `&mut [T]` accessor would be needed first (see §9 Q3) |
| Adopting a foreign allocation into a `Vec` | Allocator ownership: the producer allocated it (NumPy: `PyDataMem`), Rust cannot free it with `Global`; the owner must stay in the repr (`DECISIONS-R2.md` §2, V2) |
| Legacy **export** | Legacy structs have no `version`/`flags`; NumPy itself exports versioned only (`dlpack.c:412-413`) |
| `DLPackExchangeAPI` (v1.3 C table) | Runtime/streaming concern for accelerator frameworks; not needed for CPU pull-style exchange |
| GPU device types | Import rejects `device_type != kDLCPU` with `UnImplemented`; a future `rstsr-cuda-dlpack` would add the mapping |
| pyo3 / packaging | `DECISIONS-R2.md` D2 |

## 2. Crate identity

```toml
[package]
name = "rstsr-cpu-dlpack"
version = "0.1.0"          # placeholder; rstsr workspace version at port time
edition = "2021"
rust-version = "1.84.0"    # honest MSRV: faer 0.22.6 needs 1.84 (V3); workspace's 1.82 is stale

[dependencies]
dlpack-ffi = { version = "1.3", default-features = false }
rstsr-core = { version = "0.9", default-features = false }
rstsr-common = { version = "0.9", default-features = false }   # error type, via rstsr-core re-export

[features]
default = ["row_major", "std"]
row_major = ["rstsr-core/row_major"]
col_major = ["rstsr-core/col_major"]
std = ["rstsr-core/std"]
```

- Order features are pass-through: the crate is order-agnostic (layout accessors only), and exactly
  one of `row_major`/`col_major` must be on, same rule as `rstsr-core` itself.
- No pyo3, no `cdylib`, no `include/` headers in the crate. The prototype's `demo-ffi` (a separate
  crate in the task dir) is the thing that is `cdylib`; the deliverable crate stays an `rlib`.
- Workspace placement at port time: the crate is neither a device nor a plugin; proposal is a new
  `crates-interop/` group (or top level next to `rstsr-core`). Decided at port time (§9 Q4).

## 3. Module layout

```
src/lib.rs      crate docs, re-exports, the `with_dlpack_dtype!` dispatch macro
src/dtype.rs    `DlpackDtype` trait + impls + DLDataType ↔ T mapping
src/device.rs   `DeviceDlpackAPI<T>` seam (kDLCPU) + blanket impl
src/repr.rs     `DataDlpack<C, O>` storage repr, owners, tensor aliases
src/export.rs   `DlpackExport` guard + `into_dlpack_*` / `to_dlpack_*` entry points
src/import.rs   `unsafe fn from_dlpack_versioned*` / `from_dlpack_legacy*` + validation
```

Everything uses `rstsr_core::prelude::*`; fallible functions follow the rstsr convention
(`_f` suffix → `rstsr_common::error::Result`, panic twin without suffix).

## 4. Public API surface

### 4.1 dtype (`dtype.rs`)

```rust
pub trait DlpackDtype: Copy + 'static {
    const DLTYPE: DLDataType;                  // code/bits/lanes, lanes == 1
    fn from_dltype(dtype: DLDataType) -> Result<()>;   // exact-match check, else error
}
```

| Rust `T` | `code` | `bits` | NumPy 2.5 consumable |
|---|---|---|---|
| `f64`, `f32`, `half::f16` | `kDLFloat` (2) | 64/32/16 | yes |
| `half::bf16` | `kDLBfloat` (4) | 16 | **no** (stock NumPy cannot consume) |
| `i8/i16/i32/i64`, `u8/u16/u32/u64` | `kDLInt` (0) / `kDLUInt` (1) | 8…64 | yes |
| `i128`, `u128` | 0 / 1 | 128 | **no** |
| `bool` | `kDLBool` (6) | 8 | yes |
| `num::Complex<f64>` / `num::Complex<f32>` | `kDLComplex` (5) | 128 / 64 | yes |

`lanes == 1` only; vector lanes and sub-byte float codes (`kDLFloat8_*`) are rejected with
`UnImplemented`. Exporting a `T` that is not `DlpackDtype` is a compile-time error; importing a
dtype the crate cannot represent is a runtime error.

Dynamic dispatch for hosts (the Python adapter, demo-ffi) — the crate exports a macro instead of a
`DlpackTensorDyn` enum so hosts pick their own enum:

```rust
with_dlpack_dtype!(dtype, |T| { /* body generic over T */ });
```

### 4.2 device seam (`device.rs`)

```rust
pub trait DeviceDlpackAPI<T> {
    fn to_dlpack_device(&self) -> DLDevice;    // v1: always { kDLCPU, 0 }
}
impl<T, B> DeviceDlpackAPI<T> for B where B: DeviceAPI<T, Raw = Vec<T>> { ... }
```

The blanket impl covers `DeviceCpuSerial`, `DeviceFaer` and every `crates-device/*` backend (all
`Raw = Vec<T>`) — this is the answer to Q12 ("trait implement all that `DeviceAPI<Raw = Vec<T>>`").
A future accelerator crate defines its own impls for its device types (their `Raw` is not
`Vec<T>`), and import routes `device_type` through the same seam in reverse.

### 4.3 export (`export.rs`)

```rust
pub struct DlpackExport { /* Box<inner>, owned */ }
impl DlpackExport {
    pub fn managed(&self) -> &DLManagedTensorVersioned;   // inspect before handing off
    pub fn data_ptr(&self) -> *const c_void;              // zero-copy checks
    pub fn into_raw(self) -> *mut DLManagedTensorVersioned; // ownership leaves Rust
}

// entry points (panic twins: into_dlpack / to_dlpack_shared / to_dlpack_copy)
pub fn into_dlpack_f<T, B, D>(tensor: Tensor<T, B, D>) -> Result<DlpackExport>
pub fn to_dlpack_shared_f<T, B, D>(tensor: &TensorDlpackShared<T, B, D>) -> Result<DlpackExport>
pub fn to_dlpack_copy_f<R, T, B, D>(tensor: &TensorAny<R, T, B, D>) -> Result<DlpackExport>
where R: DataAPI<Data = Vec<T>> + DataCloneAPI<Data = Vec<T>>
```

- `into_dlpack_f` moves the tensor into the export context — no copy, Rust gives up access.
- `to_dlpack_shared_f` is zero-copy and repeatable: each call bumps the repr's `Arc` and builds a
  fresh managed tensor, so a Python holder may call `__dlpack__` more than once (NumPy may retry –
  `notes/numpy-dlpack.md`).
- `to_dlpack_copy_f` works for **any** tensor or view, at the cost of one deep copy. This is the
  safe answer for "export this slice of a bigger tensor"; a zero-copy borrowed-view export is
  intentionally not offered in v1 (§9 Q2).
- `DlpackExport` is a guard: if it is dropped without `into_raw()`, everything is freed; `into_raw()`
  hands the pointer to the consumer with the deleter as the only way back. `as_ptr()` therefore does
  not exist — the pointer can only leave Rust together with the responsibility.

Inner layout (one allocation, `#[repr(C)]`, `managed` first so the deleter can cast the pointer):

```rust
#[repr(C)]
struct DlpackExportInner<K> {
    managed: DLManagedTensorVersioned,  // version {1,0}; shape/strides/data filled below
    shape: Box<[i64]>,
    strides: Box<[i64]>,                // always emitted (DLPack ≥ 1.2 forbids NULL)
    keepalive: K,                       // the tensor (move) or the shared repr
}
unsafe extern "C" fn deleter<K>(ptr: *mut DLManagedTensorVersioned) { drop(Box::from_raw(ptr as *mut DlpackExportInner<K>)) }
```

Field policy, all matching what NumPy's consumer reads (`dlpack.c:630-660, 745-770`):

| field | value | note |
|---|---|---|
| `version` | `{major: 1, minor: 0}` | the v1.0 interchange subset; NumPy rejects `major > 1` and itself exports `{1,0}` (`dlpack.c:412-413, 639`) |
| `device` | `{kDLCPU, 0}` | §4.2 |
| `dtype` | `T::DLTYPE` | §4.1 |
| `shape` | layout shape as `i64` | error if a dim > `i64::MAX`; NumPy caps `ndim ≤ NPY_MAXDIMS` on import (`dlpack.c:662`) |
| `strides` | layout strides in **elements** (may be negative) | same unit as rstsr; NumPy multiplies by itemsize on import (`dlpack.c:757`) |
| `data` | base + `bounds_index().0` elements; `NULL` if size 0 | byte offset is baked into the pointer, `byte_offset = 0`, exactly like NumPy's producer (`dlpack.c:359-371`) |
| `flags` | move/copy export: `IS_COPIED`; shared export: `READ_ONLY` | table below |

Flag policy (the one place where this design deviates from a letter of A1 — flagged for veto):

| export entry | flags | what the consumer sees in NumPy | safety argument |
|---|---|---|---|
| `into_dlpack_f`, `to_dlpack_copy_f` | `IS_COPIED` | writeable array | the buffer is *solely* owned by the consumer for its whole lifetime (DLPack's own wording for `IS_COPIED`); no aliasing exists |
| `to_dlpack_shared_f` | `READ_ONLY` | read-only array | Rust may read concurrently; the consumer must not write |

A1 said "read-only zero-copy import/export"; writeability here is not a zero-copy *view* but the
transfer case, which is what `copy=True` is on the NumPy side (V4). If the maintainer prefers the
strict reading, setting `READ_ONLY` on move-exports too is a one-line change (§9 Q1).

### 4.4 import (`import.rs`)

```rust
pub type TensorDlpack<T, B = DeviceCpu, D = IxD> =
    TensorBase<Storage<DataDlpack<Vec<T>, DlpackForeignOwner>, T, B>, D>;

pub unsafe fn from_dlpack_versioned_f<T, B, D>(ptr: *mut DLManagedTensorVersioned) -> Result<TensorDlpack<T, B, D>>
where T: DlpackDtype, B: DeviceDlpackAPI<T>;
pub unsafe fn from_dlpack_legacy_f<T, B, D>(ptr: *mut DLManagedTensor) -> Result<TensorDlpack<T, B, D>>;
```

- `unsafe` because the pointer itself cannot be validated; everything else is (§7).
- On **error** the foreign tensor is left untouched — the caller (capsule holder) still owns it.
  On **success** the deleter obligation moves into the returned tensor's owner and the caller must
  not free it (the capsule protocol renames the capsule to `used_*` for exactly this reason).
- Legacy semantics: read-only (`dlpack.c:651-656`) and `strides == NULL` allowed (pre-1.2
  producers), meaning compact row-major.

## 5. `DataDlpack<C, O>` — the storage repr (`repr.rs`)

One repr serves both directions; `C = Vec<T>` and `O` is the keep-alive owner:

```rust
pub struct DataDlpack<C, O> { span: ManuallyDrop<C>, owner: O }

pub struct DlpackForeignOwner { /* ptr + deleter (versioned or legacy) */ }   // import
pub struct OwnedArc<C> { raw: Arc<C> }                                       // export share

pub type TensorDlpack<T, B, D>        = ...DataDlpack<Vec<T>, DlpackForeignOwner>...;  // import
pub type TensorDlpackShared<T, B, D>  = ...DataDlpack<Vec<T>, OwnedArc<Vec<T>>>...;    // shareable
```

`OwnedArc` exists because **rstsr's `DataArc` has no `Clone` and no accessor to its inner `Arc`**
(`storage/data.rs:34, 257-268` — only `strong_count`/`weak_count`) — a `TensorArc` cannot be
duplicated from `&TensorArc`, so sharing must go through an owner the bridge crate can clone.
`OwnedArc` is where a `SimpleArcTensor`-style shareable repr would also live if rstsr ever grows one
(§9 Q3).

Trait surface:

- `DataAPI<Data = C>` — always. Read-only by construction (`raw()` only).
- `DataCloneAPI` (`C: Clone`) — `into_owned`/`into_shared` **deep-copy** the span (never re-home
  it), so `.to_owned()`, `to_cpu_vec()` etc. work and are always sound.
- `Clone for DataDlpack<C, O> where O: Clone` — clones the *owner* (Arc bump / strong count) and
  rebuilds the span handle over the same memory. For `DlpackForeignOwner` this impl does not exist,
  so a double-free cannot be spelled.
- `unsafe impl Send/Sync where C: Send + Sync, O: Send/Sync` — imported tensors are `!Send`/`!Sync`
  (raw pointer owner), which is deliberate: dropping a NumPy-produced tensor off the Python thread
  would call `Py_DECREF` without the GIL. Shared tensors are `Send + Sync` when `C` is.

Span construction contract (the safety core; shared by import and by the shared export):

1. `base' = base + byte_offset + min_index * size_of::<T>()` where `min_index` is the minimum of
   `Σ i_k·stride_k` over the index space; `span_len = max_index - min_index + 1`.
2. `span = ManuallyDrop::new(unsafe { Vec::from_raw_parts(base' as *mut T, span_len, span_len) })`.
   The fabricated `Vec` must never be dropped, resized, or grown — guaranteed by `ManuallyDrop` +
   never handing out `&mut` to it + never implementing `DataMutAPI`/`DataForceMutAPI`.
3. layout `offset' = -min_index` (as `usize`), strides unchanged. Then
   `bounds_index() == (0, span_len)` and `TensorAny::new_f` accepts it
   (`tensorbase.rs:195-208`: `check_strides` + `idx_max <= storage.len()`).
4. All of `[base', base' + span_len)` lies inside the producer's allocation: the max-address element
   is a valid element, so both endpoints are inside the same allocation. Zero-size tensors use
   `NonNull::dangling()` with `span_len == 0`.
5. Alignment: `(base + byte_offset) % align_of::<T>() == 0` is required for constructing `&[T]`;
   otherwise `InvalidValue` error (§7 step 8; the *copy* reader could handle unaligned data, the
   zero-copy view cannot).

`check_strides(true)` (rstsr `layoutbase.rs:283`) is the arbiter of layout validity: overlapping
(non-broadcast, non-stride-0) layouts that NumPy can express through `as_strided` are rejected by
rstsr and therefore by import — an accepted limitation, surfaced as rstsr's own `InvalidLayout`.

## 6. Ownership and deleter discipline

- Export: the deleter is `extern "C"` and lives in the struct; the consumer calls it exactly once.
  Rust-side misuse is prevented by not exposing the pointer without `into_raw()` (§4.3).
- Import: the owner stores `(ptr, deleter)` and invokes it in `Drop` — exactly once, at the moment
  the last Rust handle dies. `deleter == NULL` (allowed by the spec) means "never free".
- `TensorDlpack` is read-only and cloneable only via deep copy, so "who owns what" has exactly one
  answer at all times; that is the reason the repr does not implement `DataMutAPI` (§9 Q3).

## 7. Import validation checklist

Order matters; all failures return without touching the foreign tensor.

1. pointer non-null; versioned: `version.major == 1` else `UnImplemented`.
2. `device.device_type == kDLCPU` else `UnImplemented` (device_id accepted as-is).
3. `dtype.lanes == 1`; `(code, bits)` must be one of §4.1, else `UnImplemented`; typed entry also
   requires `T::DLTYPE == dtype` else `InvalidValue`.
4. `ndim >= 0`; `ndim == 0` → scalar (`shape`/`strides` may be NULL, `numel = 1`).
5. `shape[i] >= 0`; each dim ≤ `i64::MAX` (trivially) — build `Vec<usize>`.
6. `strides`: NULL → compact row-major from shape (legacy producers); else read `ndim` entries.
7. `numel == 0` → data may be NULL, empty tensor (owner still retained).
8. alignment check (step 5 of §5), else `InvalidValue`.
9. `min/max_index` with checked arithmetic (`checked_mul`/`checked_add`, i128 → error on overflow).
10. build span + layout; `Layout::new(shape, stride, offset')?` then `TensorAny::new_f(...)?`
    (rstsr validates `check_strides` and bounds; its errors pass through unchanged).

Error taxonomy — the crate reuses `rstsr_common::error::RSTSRError` (so `?` composes with rstsr):

| condition | variant |
|---|---|
| null / malformed struct, unaligned data, dtype mismatch, bad version | `InvalidValue(String)` |
| valid DLPack not representable here (non-CPU device, exotic/vector dtype) | `UnImplemented(String)` |
| shape/stride arithmetic out of range | `ValueOutOfRange(String)` |
| layout rejected by rstsr (`check_strides`, bounds) | rstsr's own error |

## 8. The Python boundary (what stays outside the crate)

The crate speaks pointers; the capsule is boxed on the Python side. The holder shim must:

- **export**: create a fresh `DlpackExport` per `__dlpack__(stream, max_version, dl_device, copy)`
  call — `copy=True` maps to `to_dlpack_copy_f`, else `to_dlpack_shared_f` — wrap the raw pointer in
  a `PyCapsule` named `dltensor_versioned` whose destructor calls `managed.deleter` **iff** the
  capsule was never consumed (name still `dltensor_versioned`), and honour `max_version` (we only
  produce version 1.0);
- **import**: call `arr.__dlpack__(max_version=(1, 0))`, read the pointer, rename the capsule to
  `used_dltensor_versioned` *before* handing the pointer to the crate, then hand it over — the
  rename is what transfers the free obligation from the capsule to the imported tensor.

The capsule mechanics are already exercised end-to-end in
`../2026-10-03-rust-numpy-review/experiments/probe_numpy_dlpack.py`; the prototype's `python/`
directory reuses that shape. The prototype's `demo-ffi` crate provides the host-side entry points
(`extern "C"`, error string + null-on-error) that the shim calls.

## 9. Open questions for the maintainer

| # | Question | Recommendation |
|---|---|---|
| Q1 | Move-export flags: `IS_COPIED` (writeable in NumPy, sole transfer) vs strict `READ_ONLY` (A1's letter) | `IS_COPIED`; both are one line |
| Q2 | Zero-copy export of a *borrowed* view (without a copy, without ownership) | defer; needs an `unsafe` entry point with a caller-managed liveness contract |
| Q3 | Writable import (rung 2, `IS_COPIED` buffers) | defer: needs a rstsr-side bounded `&mut [T]`/span accessor (or force-mut gate) that does not expose `&mut Vec<T>`; allocator ownership already forbids re-homing (V4) |
| Q4 | Workspace placement and crate defaults at port time (`crates-interop/`? default features?) | proposal in §2; decide at port |
| Q5 | Upstream the capsule-name constants (`dltensor_versioned`, `used_*`) to `dlpack-ffi` | suggest yes (spec constants, currently only in the shim) |

## 10. Test plan

**Rust (`cargo test -p rstsr-cpu-dlpack`)**

1. dtype round-trip table (all of §4.1, plus rejects: lanes > 1, `kDLFloat8_*`, bits mismatch).
2. export field checks: version/dtype/shape/strides/byte_offset/flags/data-pointer for
   contiguous, transposed, sliced, negative-stride, zero-size, 0-dim, broadcast (stride-0) tensors.
3. ownership: `into_raw` + consumer-simulated deleter runs exactly once (counted by a test deleter
   on the import side; on the export side, drop-vs-deleter exclusivity asserted by pointer reuse /
   miri); `DlpackExport` dropped without `into_raw` frees.
4. import: fabricated producers (a Rust-side struct + deleter that flips an `AtomicUsize`):
   versioned/legacy, NULL strides, negative strides, byte_offset ≠ 0, 0-dim, zero-size, deleter =
   NULL; deleter fires exactly once when the last tensor handle dies (including through
   `to_owned()`/clone paths).
5. validation matrix = §7, every row asserted, and *the foreign tensor is untouched on error*.
6. results equality on shared memory: write through the imported view's `raw()`/device accessors
   from the producer side and observe through rstsr ops (both directions, ptr equality).
7. `miri` for the repr/owner paths if the toolchain has it (`cargo +nightly miri test`).

**Python end-to-end (`python/`, conda env `torch`)**

1. export/share: Rust builds `arange` → `rstsr_demo_export` → `np.from_dlpack(holder)`; assert
   values, `arr.ctypes.data == demo_data_ptr(handle)`, `arr.flags.writeable is False`; Rust keeps
   reading (sum) while the array lives.
2. export/move + round trip: `arr.flags.writeable is True`, mutate in NumPy, hand back via
   `arr.__dlpack__()` → `rstsr_demo_import` → Rust sees the mutated values.
3. import: NumPy array → capsule → Rust read-only tensor; zero-copy pointer equality; negative
   strides (`arr[::-1]`) and sliced (`arr[::2]`) layouts; `weakref` proof that the producer's
   lifetime is governed by the capsule/imported tensor (drop order exercises the deleter).
4. protocol: `__dlpack__` called twice (NumPy's TypeError retry path), capsule reuse, unconsumed
   capsule → destructor frees, consumed capsule → destructor does nothing (no double free).
5. failure injection from Python (ctypes-built malignant managed tensors): null data, bad version,
   CUDA device type, `lanes=2`, unaligned pointer, absurd strides — error, no crash, capsule still
   freeable by its owner.

## 11. Prototype layout (this task dir)

```
prototype/
  Cargo.toml            # workspace: rstsr-cpu-dlpack + demo-ffi, path deps to ../../../../rstsr, ../../../../dlpack-ffi
  rust-toolchain.toml   # nightly, mirroring rstsr
  rstsr-cpu-dlpack/     # the deliverable crate
  demo-ffi/             # cdylib host: handle table + extern "C" entry points for the shim
  python/               # shim + end-to-end suite (reuses the probe's capsule machinery)
```

Path dependency note: `rstsr-core` is consumed as a plain path dep (workspace-inherited fields
resolve through the rstsr workspace root); the prototype's own `Cargo.lock` is independent of
rstsr's. The port into the rstsr workspace (a patch, per this repo's charter) swaps the paths for
`workspace = true` entries — that is also where the stale `rstsr-*-ffi` version specs in the
workspace manifest can be refreshed (the `rstsr-ffi` repo is ahead of `rstsr/Cargo.toml`:
openblas 0.6.1 vs `"0.5"`, mkl 0.3.0 vs `"0.2"`, blis 0.2.1 vs `"0.2"`, aocl 0.3.0 vs `"0.2"`,
kml 0.2.1 vs `"0.2"`, cblas-base 0.1.2 vs `"0.1"`).

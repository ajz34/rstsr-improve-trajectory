# rust-numpy: a code literature review — and what rstsr can learn from it

- **Date**: 2026-10-03
- **Task dir**: `2026-10-03-rust-numpy-review/`
- **Status**: draft for discussion (no implementation intended; see `QUESTIONS-discussion-R1.md`)
- **Discussion trail**: `QUESTIONS-discussion-R1.md` (Q1–Q14) → `ANSWERS-discussion-R1.md` (maintainer, 2026-10-03) → `RESPONSE-discussion-R1.md` (verification, corrections, Q4 still open) → `DECISIONS-R2.md` (scope closed: pure `rstsr-cpu-dlpack`, heavy layer deferred). Claims corrected by R1 are marked “(corrected in `RESPONSE-discussion-R1.md` V#/C#)”.
- **Pinned revisions (everything below is relative to these)**:
  - rust-numpy `da6bf5be05d4053cf0c51afa12efc018a984ca7f` = v0.29.0 + 8 commits (2026-08-28), `numpy` crate 0.29.0
  - numpy reference checkout `~/Git-Others/numpy` at tag **v2.5.2** (`48fecee545`); installed numpy for probes **2.5.1** (conda env `torch`)
  - DLPack upstream `~/Git-Others/dlpack` at **v1.3**; `dlpack-ffi` crate 1.3.0 (`/home/a/rstsr_pack/dlpack-ffi`)
  - rstsr workspace master `f179c46` (v0.9.0)
- **Evidence layout**:
  - `notes/numpy-dlpack.md` — NumPy's DLPack producer/consumer, source-read + empirical probes
  - `notes/dlpack-v13-spec.md` — DLPack v1.3 contract and `dlpack-ffi` surface
  - `notes/rstsr-state.md` — rstsr bridge-relevant inventory (storage/layout/dtype/device/feature gaps)
  - `experiments/probe_numpy_dlpack.py`, `experiments/probe-output.txt` — raw probe evidence
  - this file = synthesis

---

## 0. TL;DR

1. **rust-numpy is the mature reference implementation of a NumPy C-API bridge in Rust.** It binds 235 array-API slots and 38 ufunc-API slots at runtime through the `_ARRAY_API` capsule — no NumPy headers at build time, no NumPy version pinned at compile time (`src/npyffi/array.rs`).
2. **Its boundary semantics are asymmetric**: Rust→NumPy is *ownership transfer* (a Rust allocation becomes the NumPy buffer, kept alive by a Python owner object, so it is zero-copy), while NumPy→Rust is *borrow-only* (views under an aliasing guard; never taking ownership of NumPy's buffer). See §3.3–3.5.
3. **Zero-copy for owned Rust data is solved, including the allocator problem**: `PySliceContainer` (a `#[pyclass]` holding raw `ptr/len/cap` + a monomorphized drop function) is attached via `PyArray_SetBaseObject`, so the Python GC frees Rust memory with the correct Rust allocator (`src/slice_container.rs`, `src/array.rs:253-297`).
4. **The borrow checker is the crown jewel and directly reusable prior art**: `PyReadonlyArray`/`PyReadwriteArray` maintain a *cross-extension* dynamic aliasing registry published as a capsule inside NumPy itself (`numpy.core.multiarray._RUST_NUMPY_BORROW_CHECKING_API`), with a Diophantine over-approximation of overlap on strides (`src/borrow/shared.rs`, `src/borrow/mod.rs`). Borrows survive GIL release, which is what makes long parallel compute with NumPy-backed data sound.
5. **Byte-order is never silently reinterpreted and never silently converted**: dtype extraction requires "no-casting" equivalence, so a `>f8` array fails `PyArray<f64>` extraction instead of being misinterpreted (verified in numpy 2.5.1: `np.can_cast('>f8','<f8','no') == False`). The cost is that the caller must byteswap explicitly.
6. **rust-numpy has no DLPack support at all** (grep over `src/`, `tests/`, `examples/`, `benches/` is empty; §5). NumPy 2.x, by contrast, both produces and consumes DLPack capsules, and this is the natural third-party-facing interface — a much smaller surface than the C-API (one versioned struct + capsule rules vs. hundreds of slots + struct layouts + per-version branches).
7. **DLPack is not a superset of the C-API bridge.** It has no byteorder field, no dtype for datetime64/timedelta64/strings/structured/object, no writeability out-of-band semantics before v1.0 flags, and no import-time casting story. Anything nontrivial beyond "strided numeric/boolean buffers, zero-copy, native-endian, CPU/GPU device" falls back to copy-with-conversion paths.
8. **rstsr is closer than it looks, but the owning path and the Python layer are absent.** Zero-copy *views* of foreign memory are already expressible with the existing `DataRef` + non-owning-`Vec` pattern that `asarray` itself uses (§8.2); *owning* a NumPy buffer needs a storage repr that carries the owner, definable in the bridge crate without a core change — the in-repo UB note is faer-specific (corrected in `RESPONSE-discussion-R1.md` V1/V2); there is no runtime dtype tag (G2), no device id (G3), and no pyo3/capsule/DLPack code anywhere (G4/G7). Notably, rstsr's array-API docs currently record `from_dlpack` as impossible for other languages — a decision this review asks to revisit (§8.7). The declared MSRV 1.82 is stale under default features (faer 0.22.6 requires 1.84), so pyo3 0.29's 1.83 raises nothing (Q10; `RESPONSE-discussion-R1.md` V3).
9. The strategic question is therefore not "rust-numpy or DLPack" but **which boundary each direction of travel uses**: NumPy→rstsr can be a DLPack import (zero-copy view or owned adoption) plus an `np.asarray`-style copy path for exotic dtypes; rstsr→NumPy can be a DLPack export (zero-copy, needs a Python object to hang `__dlpack__` on) or a NumPy-C-API write (via rust-numpy as a dependency, giving `.npy`-in-memory + exotic dtype support).
10. **Decision points are collected in `QUESTIONS-discussion-R1.md`** (maintainer answers in `ANSWERS-discussion-R1.md`, R1 response in `RESPONSE-discussion-R1.md`); none of them require implementing anything yet.

---

## 1. Scope and method

**Question this review answers**: what does the existing art (rust-numpy, NumPy's own DLPack support, the DLPack spec, dlpack-ffi) imply about how rstsr should inter-operate with NumPy — before any design or implementation is attempted?

**What was read**

| Source | Revision | Depth |
|---|---|---|
| rust-numpy | `da6bf5be` (v0.29.0+8) | full `src/` (~8.8k lines), `Cargo.toml`, README, CHANGELOG, tests/examples/benches/CI survey |
| NumPy | v2.5.2 checkout | DLPack implementation `numpy/_core/src/multiarray/dlpack.c` + tests + Python entry points (delegated; see `notes/numpy-dlpack.md`) |
| DLPack | v1.3 (`84d107b`) | header-in-comments spec + prose spec + NEWS (delegated; see `notes/dlpack-v13-spec.md`) |
| dlpack-ffi | 1.3.0 | generated surface + tests (`notes/dlpack-v13-spec.md`) |
| rstsr | `f179c46` | storage/layout/dtype/device inventory (delegated; see `notes/rstsr-state.md`) |

**Empirical probes**: run against numpy 2.5.1 in conda env `torch`; scripts and raw output under `experiments/`. Probe results are quoted inline where they decide a claim.

**Conventions**: claims cite `file:line`; anything not verified is marked *(uncertain)*; "rust-numpy" = the crate, "NumPy" = the C library.

---

## 2. rust-numpy at a glance

| Property | Value | Source |
|---|---|---|
| crate/version | `numpy` 0.29.0 (workspace of one crate) | `Cargo.toml` |
| role | PyO3-based Rust bindings of the NumPy C-API, with `ndarray` as the Rust-side array library | `src/lib.rs:1-13` |
| hard deps | `pyo3` 0.29 (macros), `ndarray` `>=0.15,<=0.17`, `num-complex`, `num-traits`, `num-integer`, `libc`, `rustc-hash` | `Cargo.toml` |
| optional deps (implicit features) | `half` (f16/bf16), `nalgebra` (0.30–0.35), `rand_core` | `Cargo.toml` |
| MSRV | 1.83 (follows PyO3) | README:5,16 |
| Python | 3.9–3.15 incl. free-threaded `3.14t/3.15t`, PyPy 3.11; Python 3.8 dropped in 0.29/unreleased | README:18-20, CI `ci.yml` |
| NumPy target | ABI v2 constant, runtime API target v1.15 (2.5 on abi3t); "numpy >= 1.16 recommended" | `src/npyffi/numpyconfig.rs`, README:26 |
| size | 8,759 lines Rust across 24 files | `wc -l src/**/*.rs` |
| dev tooling | `x.py` (fmt/clippy/msrv/examples/test driver), criterion benches, nox + pytest for examples | `x.py`, `benches/`, `examples/*/noxfile.py` |
| project history | in the PyO3 org; release cadence tracks PyO3 majors; CHANGELOG preserved since v0.1 | `CHANGELOG.md` |

The crate is *the* de-facto standard for Rust↔Python array interop: every Rust scientific extension that touches NumPy in PyO3 style either uses it or reimplements a subset.

---

## 3. Architecture

### 3.1 The C-API bridge (`npyffi/`) — runtime capsule, slot-index dispatch

No NumPy headers are used at build time; `build.rs` only configures PyO3. At runtime, on first use:

1. Import `numpy` and resolve the core module name for NumPy 1.x vs 2.x (`numpy.core` vs `numpy._core`), mirroring pybind11's strategy (`src/npyffi/array.rs:18-43`).
2. Fetch the capsule `_ARRAY_API` from `numpy[._]core.multiarray`, cache the pointer in a `PyOnceLock`, and deliberately leak the capsule reference so the pointer stays valid (`src/npyffi/mod.rs:32-47`, `array.rs:85-144`).
3. Read the API table as an array of `*const c_void` and dispatch every function through a *slot offset* — the same numbering NumPy's own `numpy_api.py` code generator uses. The macro `impl_api![offset; sig]` generates typed wrappers (`src/npyffi/mod.rs:58-70`; `src/npyffi/array.rs:149-433`; 235 active array-API slots — 258 `impl_api!` lines including commented-out re-additions; ufunc API: 38 active in `src/npyffi/ufunc.rs`).
4. Type objects are looked up by slot too (`PyArray_Type`, `PyArrayDescr_Type`, scalar types, `NpyIter_Type`, …) (`src/npyffi/mod.rs:73-151`).

Guard rails at first use (`src/npyffi/array.rs:92-137`):

- **ABI check**: `PyArray_GetNDArrayCVersion()` (slot 0) vs `NPY_VERSION = 0x02000000` — a module built against a newer ABI than the running NumPy is an error.
- **Feature/API check**: `PyArray_GetNDArrayCFeatureVersion()` (slot 211) vs target `0x0c` (v1.15), or `0x16` (v2.5) in the `abi3t` free-threaded configuration (`src/npyffi/numpyconfig.rs`).
- **Endianness check**: runtime CPU endianness must match the compile target.

Version-dependent behavior is localized rather than spread: `is_numpy_2(py)` flips descr-struct layout handling (`src/npyffi/objects.rs`) and the free-threaded "item data" accessors are only bound as API slots on NumPy ≥2.5 abi3t builds (`src/npyffi/array.rs:435-448`).

**Consequence**: the crate's FFI maintenance surface = (a) slot numbers/types it binds, (b) C struct layouts it reads directly (`PyArrayObject_fields`, `PyArray_Descr*` variants, iterator objects — `src/npyffi/objects.rs`), (c) per-version branches. That is the price of the C-API route; rust-numpy pays it and keeps it in one module.

### 3.2 Object model and type system

- `PyArray<T, D>` is `#[repr(transparent)]` over `PyAny` with `PhantomData<T>`, `PhantomData<D>` — a pointer-sized handle; the element type `T: Element` and rank `D: ndarray::Dimension` exist only at the type level (`src/array.rs:99-117`). Aliases `PyArray0..6`, `PyArrayDyn`.
- `PyUntypedArray` is the dynamic base (dtype/ndim/shape/strides/flags introspection) (`src/untyped_array.rs`). Checks live in `PyArray::extract`: object must be an ndarray (subclass allowed; `PyArray_CheckExact` used for exact-type paths), rank must match `D::NDIM` if fixed, dtype must be *equivalent* to `T::get_dtype()` (`src/array.rs:144-177`).
- `PyArrayDescr` wraps `numpy.dtype` with a full introspection API (`num`, `kind`, `char`, `byteorder`, `is_native_byteorder`, `itemsize`, `alignment`, `flags`, `names`, `get_field`, subarray accessors) (`src/dtype.rs:132-311`).
- `Element` is an `unsafe trait` with `const IS_COPY: bool` (trivially copyable), `get_dtype(py)`, `clone_ref(py)` (GIL-held clone; refcount bump for object dtype) plus bulk-copy hooks (`src/dtype.rs:469-508`).

**Element dtype coverage** (the honest table):

| Category | Types | Notes |
|---|---|---|
| booleans | `bool` → `NPY_BOOL` | |
| signed/unsigned ints | `i8..i64`, `u8..u64`, `isize/usize` | C-data-model-aware alias resolution (`src/dtype.rs:510-553`) |
| floats | `f32`, `f64` | |
| half | `f16` (feature `half`) | |
| bfloat16 | `half::bf16` (feature `half`) | **requires a third-party NumPy bfloat16 dtype provider**; the crate constructs the dtype by name and panics with a message if absent (`src/dtype.rs:617-632`) |
| complex | `Complex32`, `Complex64` (`num-complex`) | |
| object | `Py<PyAny>` | `IS_COPY = false`; refcounting through `clone_ref` |
| fixed strings | `PyFixedString<N>` (S), `PyFixedUnicode<N>` (U) | dtype built by mutating a fresh descr + cached per size (`src/strings.rs:161-201`) |
| datetime | `Datetime<U>`, `Timedelta<U>`, units Y..as | stores raw i64 + unit metadata (`src/datetime.rs`) |
| **absent** | `i128/u128`, `longdouble`/`complex256`, `float16`-outside-half, structured/void as `Element`, NumPy 2 `StringDType`, subarray dtypes | structured dtypes are reachable only through raw descriptor APIs |

There is **no runtime dtype enum** (a `DataType` enum was removed in 0.16, `CHANGELOG.md`); dynamic dispatch is done by user code via a cast cascade on `PyUntypedArray` (doctest in `src/untyped_array.rs:32-57`).

**Rank cap**: the conversion/views machinery assumes ≤32 dimensions (`[npy_intp; 32]` buffers, `assert!(strides.len() <= 32)`, `MAX_DIMENSIONALITY_ERR`) (`src/convert.rs:219-241`, `src/array.rs:1376`, `src/error.rs:13-16`). NumPy 2 raised `NPY_MAXDIMS` to 64, so 33–64-dim arrays are outside rust-numpy's contract even though NumPy itself accepts them. *(Stale assumption; likely a latent panic, not memory unsafety.)*

### 3.3 Memory and ownership — the asymmetry

**Rust → NumPy (ownership transfer, zero-copy).** `IntoPyArray` takes `self` by value:

- `Vec<T>` / `Box<[T]>` / `ndarray::Array<T, D>` (owned) are moved into a `PySliceContainer`, a `#[pyclass(frozen)]` storing `ptr/len/cap` plus a monomorphized `drop` fn-pointer that reconstructs the `Vec`/`Box` on drop (`src/slice_container.rs:9-91`).
- The array is created with `PyArray_NewFromDescr(..., data = rust_ptr, flags = NPY_ARRAY_WRITEABLE, ...)`, then `PyArray_SetBaseObject(array, container)` links lifetimes (`src/array.rs:253-297`; `from_raw_parts` at 285-297; `from_owned_array` at 412-424).
- Consequently the NumPy array is writeable, zero-copy, and **cannot be resized** (NumPy refuses because the buffer is not its own; doc + test `src/convert.rs:14-36`, `tests/array.rs`).
- The `Box<[T]>` path deliberately derives the data pointer *after* dissolving the Box to avoid `noalias` provenance issues (`src/convert.rs:52-59`, referencing rust-lang/unsafe-code-guidelines#326 — a nice piece of unsafe-Rust folklore the crate carries).

**Rust → NumPy (copy).** `ToPyArray` borrows `&self`, allocates in NumPy's heap, and copies: fast path `ptr::copy_nonoverlapping` when contiguous and `IS_COPY`, element-wise `clone_ref` otherwise (works for object dtype; non-contiguous input is materialized in C order) (`src/convert.rs:96-184`).

**NumPy → Rust (borrow-only).** There is *no* ownership-taking path. The only ways in are:

- `as_array()` / `as_array_mut()`: `ndarray::ArrayView(Mut)` over NumPy's buffer, non-contiguous allowed. Negative strides are handled by flipping the data pointer to the memory start and *inverting axes* — a documented hack with a FIXME against ndarray#842 (`src/array.rs:1361-1416`).
- `as_slice()`: only when `is_aligned() && is_contiguous()`; empty arrays special-cased (`src/array.rs:747-789`).
- `to_vec()` / `to_owned_array()`: copies (with `Element::clone_ref` for object dtype) (`src/array.rs:1521-1605`).
- `borrow_from_array` (unsafe): the reverse pattern — a Python object owns the ndarray, NumPy views it, caller upholds the invariant (`src/array.rs:299-351`).

**Why borrow-only?** Because NumPy owns the buffer and frees it with its own allocator; Rust cannot adopt it without a foreign-buffer storage type and a matching deleter. rust-numpy simply declines to model that (it has no such storage). This is precisely the gap DLPack-filled designs close: DLPack makes ownership transferable in *both* directions (a deleter callback travels with the tensor).

### 3.4 Conversion API surface

| Trait/type | Direction | Semantics |
|---|---|---|
| `IntoPyArray` | Rust → Py (move) | zero-copy for `Vec`, `Box<[T]>`, owned `ndarray::Array`; result not resizable |
| `ToPyArray` | Rust → Py (borrow) | copy into NumPy heap; C-order for non-contiguous input |
| `PyArray::from_slice/from_vec/from_vec2/from_vec3/from_iter/from_owned_array/from_owned_object_array` | constructors | mix of copy and move paths |
| `PyArrayLike<T, D, C>` | Python → Rust argument | `C = TypeMustMatch` (exact dtype) or `AllowTypeChange` (`numpy.asarray` semantics via `PyArray_FromAny` + `NPY_ARRAY_FORCECAST`); 1-D sequences go through `Vec<T>` extraction (`src/array_like.rs:135-192`) |
| `FromPyObject for PyReadonlyArray/PyReadwriteArray` | Python → Rust argument | extraction + borrow guard in one step (`src/borrow/mod.rs:241-250, 484-493`) |
| `PyArrayMethods::{get,uget,get_mut,to_vec,to_owned_array,copy_to,cast_array,permute,reshape,resize,...}` | mixed | thin wrappers over C-API slots |

Strides are converted to **bytes** at the boundary (`npy_strides`), C/F order detection included (`src/convert.rs:228-251`). Indexing (`NpyIndex`) converts element indices to byte offsets (`src/convert.rs:281-330`).

### 3.5 Borrow checking — the crown jewel

Problem: safe Rust references into a buffer that Python and other extensions can freely alias. rust-numpy's answer (`src/borrow/`):

- Guard types `PyReadonlyArray<'py, T, D>` / `PyReadwriteArray<'py, T, D>` acquire on construction (`try_readonly`/`try_readwrite`) and release on drop; conflicts return `BorrowError::{AlreadyBorrowed, NotWriteable}` (`src/error.rs:154-173`).
- The registry is keyed by **base-object identity** (the base-object chain is walked to the original allocation, `shared.rs:370-382`) with a per-array key `(data_range, data_ptr, gcd_strides)`.
- Overlap is decided by an over-approximation: ranges must intersect and the GCD of strides must divide the pointer difference — a Diophantine condition; false conflicts are documented and accepted (`shared.rs:218-255`, rationale in `borrow/mod.rs:151-166`).
- **Cross-extension by construction**: the registry lives in a `#[repr(C)] struct Shared { version, flags, fn acquire/acquire_mut/release/release_mut }` published as the capsule `numpy.core.multiarray._RUST_NUMPY_BORROW_CHECKING_API`; independently built extensions coordinate through the same table, versioned additively (`shared.rs:21-173`). NumPy itself doesn't understand it — it is a Rust-extension convention planted inside the NumPy module namespace.
- **GIL relationship**: acquire/release require the GIL, but borrows *stay valid after `Python::detach`*; they do not block, they fail loudly (`borrow/mod.rs:142-150`). This is the enabler for "release the GIL and compute on borrowed NumPy memory" patterns.
- Safety philosophy: unchecked code (Python, C, unsafe Rust) is its author's responsibility; safe Rust cannot *introduce* UB given that contract (`borrow/mod.rs:119-150`).

### 3.6 GIL, threading, free-threading

- Everything Python-facing is `Bound<'py, _>`/`Python<'py>`-bound; PyO3 0.29 API (`Python::attach`, `detach`).
- `Element: Send + Sync` is required (0.23, free-threading support); `PyReadonlyArray`/`PyReadwriteArray` guards are not `Send` (they hold GIL-bound handles) but the *borrow* outlives GIL release on the same thread.
- Free-threaded Python is supported through 0.14t→3.15t and abi3t (`CHANGELOG.md` unreleased, `#556`); the borrow registry mutex and the capsule table are the corresponding machinery.
- rust-numpy itself has **no rayon/parallel feature** (removed in 0.16; parallelization is delegated to `ndarray`'s features). The canonical pattern (see `examples/parallel`): take a `PyReadonlyArray`, get a view, compute with ndarray+rayon, `into_pyarray` the result — zero-copy out.

### 3.7 The rest of the surface (survey)

- `sum_products`: `dot`, `inner`, `einsum` via `PyArray_EinsteinSum` / matrix-product slots (`src/sum_products.rs`).
- `random`: `BitGenerator`/`PyBitGenerator` wrappers around NumPy's `bitgen_t` (post-0.29 unreleased work, #499) with `rand_core` integration (`src/random.rs`, `src/npyffi/random.rs`).
- `datetime`, `strings`, `nalgebra` integrations as covered above.
- Tests: integration tests in `tests/` (array, array_like, borrow, sum_products, to_py) plus in-file unit tests; benches via criterion (`array`, `array_like`, `borrow`); CI matrix 3.9–3.15t + PyPy on Linux/macOS/Windows, lint+clippy+MSRV+examples gates (`x.py check`).

---

## 4. Design lessons for rstsr

1. **Runtime FFI discovery beats build-time headers** for Python interop: no NumPy in the build graph, works in any environment where `import numpy` succeeds, version mismatch becomes a clear runtime error. rust-numpy's slot table is the mechanism; a DLPack bridge gets the same property for free but with a far smaller contract.
2. **Ownership transfer across the boundary needs a Python-side owner object with a Rust deleter.** `PySliceContainer` is the minimal design: one `#[pyclass]` + monomorphized free function + `PyArray_SetBaseObject`. Any rstsr→NumPy zero-copy export needs the same shape (a Python object owning rstsr storage), or DLPack's `manager_ctx`+`deleter` which is the library-neutral generalization.
3. **Borrow-based zero-copy into Rust requires an aliasing story.** rust-numpy's answer is a *global, cross-extension, base-object-keyed* registry with an over-approximate overlap test, planted as a NumPy capsule. If rstsr wants safe views of NumPy memory, this design is directly transplantable (it is versioned and additive). Note the acceptance criterion: false conflicts are tolerated; unsoundness is not.
4. **Compile-time dtype typing is ergonomic only for monomorphic kernels.** rust-numpy pays for it with type-level `Element`+`phantom` design and no runtime dtype; dynamic entry points require manual cast cascades. rstsr's `TensorAny` is the opposite philosophy (runtime dtype + device polymorphism) — which is *better* for an interchange layer, because an interchange layer must accept "whatever dtype this Python object has" without N monomorphizations. The gap: rstsr needs a *runtime dtype tag* to build the mapping table (DLPack `DLDataType <-> dtype`, numpy dtype num <-> dtype).
5. **Strides in bytes + offset, with negative strides and zero strides explicitly enumerated** are the boundary realities. rust-numpy's negative-stride inversion is a hack it would like to remove; a bridge that can express (ptr, byte_offset, byte_strides) natively (DLPack) is cleaner than one that must reconstruct ndarray views.
6. **Byteorder must be a first-class decision.** rust-numpy chose "reject rather than reinterpret, never auto-convert". DLPack has *no* byteorder field at all, so the same choice recurs: DLPack interchange is native-endian-only in practice.
7. **Borrows that survive GIL release are the key to performance**: acquire under the GIL, `detach`, compute for seconds, re-attach, drop. rstsr's rayon-first execution model maps onto this directly.
8. **The C-API route is expensive to maintain per NumPy release; the DLPack route is cheap and library-neutral**, at the cost of coverage (dtype/layout/writeability coverage tables) and of requiring a Python-visible object to carry the protocol methods.
9. **The 32-dim cap is a warning about hard-coded limits**: NumPy 2 changed an invariant under rust-numpy's feet. A bridge should express "whatever ndim the producer says" or fail with a clear error, not assert.
10. **Separate "view" from "ownership" in the design from day one.** A view is cheap (unsafe glue over existing `DataRef` machinery, no core changes); ownership is the real work (a storage variant carrying an owner+deleter, with the deleter running exactly once, possibly from a foreign thread or after Python finalization). rust-numpy only ever solved the view half for imports; DLPack is what makes the ownership half expressible.

---

## 5. Sharp edges and limitations of rust-numpy (checklist)

| # | Limitation | Evidence |
|---|---|---|
| L1 | **No DLPack support whatsoever** (no producer or consumer; `__dlpack__` never mentioned) | grep over `src/ tests/ examples/ benches/` empty |
| L2 | ≤32 dimensions assumed in conversion/view paths; NumPy 2 allows 64 | `src/convert.rs:232`, `src/array.rs:1376`, `src/error.rs:13-16` |
| L3 | NumPy buffers can only be borrowed, never adopted (no foreign-buffer storage, no deleter model) | §3.3; no such API in `src/` |
| L4 | Byte-swapped dtypes cannot cross (extraction fails; no auto-swap) | `src/dtype.rs:282-289`; numpy `multiarraymodule.c:1480-1499` (`PyArray_EquivTypes` = "no-casting" only, via `MinCastSafety`, `convert_datatype.c:274-287` "larger casting values are less safe"); probe: `can_cast '>f8'→'<f8' 'no'` = False |
| L5 | `bf16` needs a third-party NumPy dtype package; panics otherwise | `src/dtype.rs:617-632` |
| L6 | Missing scalar dtypes: `i128/u128`, long double/complex256 | `src/dtype.rs` impl list |
| L7 | Structured/void, NumPy-2 `StringDType`, subarray dtypes not surfaced as `Element` types | `src/dtype.rs:296-311` (raw descr access only) |
| L8 | Borrow-conflict over-approximation rejects some genuinely disjoint views | `src/borrow/mod.rs:151-166` |
| L9 | Non-contiguous `ToPyArray` silently materializes C-order copy (asymmetry with the zero-copy move path) | `src/convert.rs:113-183` |
| L10 | Negative strides handled by axis-inversion hack (open FIXME) | `src/array.rs:1382-1394` |
| L11 | Python-object dtype (`Py<PyAny>`) is GIL-bound and `IS_COPY = false`: every copy refcounts | `src/dtype.rs:642-653` |
| L12 | Writeability: extraction only; no in-band way to express "I will not write" via DLPack-style flags (NumPy's own `make_nonwriteable` is a side effect) | `src/borrow/mod.rs:529-544` |

## 6. NumPy's DLPack implementation (v2.5.2 source + v2.5.1 probes)

Full evidence: `notes/numpy-dlpack.md`; raw runs: `experiments/probe-output.txt`. Provenance: numpy vendor a DLPack **v1.0** header; DLPack-v1 support (versioned struct, flags, `max_version`) landed in **NumPy 2.1.0**; `kDLBool` in 1.25.0; unsupported-dtype/device errors became `BufferError` in 2.5.0; a user-dtype registry (`np.dtypes.register_dlpack_dtype`) appeared in 2.5.0.

### 6.1 `np.from_dlpack(x)` — the consumer

- **Accepts only objects with `__dlpack__`.** A bare `PyCapsule` is rejected (`AttributeError: 'PyCapsule' object has no attribute '__dlpack__'`); `__dlpack_device__` is *never* called despite the docstring (`dlpack.c:604-621`).
- **Exact call**: `x.__dlpack__(dl_device=None, copy=None, max_version=(1, 0))`; if that raises `TypeError` **and** the user passed neither `device` nor `copy`, numpy silently retries with a bare `x.__dlpack__()` — so **a producer can be invoked twice** (`dlpack.c:607-625`; probe lines 138-140). Producers must be idempotent.
- `device=` accepts only `"cpu"`; `copy=` is forwarded to the producer and **not enforced** by numpy (numpy never copies on import; its own producer ignores `copy=False`).
- Validation order (all `BufferError` unless noted): capsule name (`dltensor_versioned` then `dltensor`; anything else `ValueError`) → `version.major > 1` reject (minor ignored) → `ndim <= 64` (`NPY_MAXDIMS`) → device whitelist `kDLCPU`/`kDLCUDAHost`/`kDLROCMHost`/`kDLCUDAManaged` (**`device_id` is not validated**, round-trips verbatim) → `lanes == 1` → dtype mapping (below) → strides (`NULL` = C-contiguous; element strides multiplied by itemsize; negative/zero strides accepted) and `byte_offset` honored.
- **No alignment check at all**; a +1-byte-misaligned pointer imports and reports `flags.aligned == False`.
- Ownership: on success numpy renames the producer's capsule to `used_dltensor[_versioned]` and attaches an internal base capsule (`numpy_dltensor[_versioned]`) that calls the deleter exactly once when the array dies; on failure the capsule is released *un-renamed* (producer's destructor runs, per DLPack convention); a consumed capsule reused → `ValueError`.
- **Legacy (v0) imports are always non-writeable** (`dlpack.c:658`) since the legacy struct has no flags.

### 6.2 `ndarray.__dlpack__()` — the producer

- Signature `(*, stream=None, max_version=None, dl_device=None, copy=None)`; `stream` must be `None`; `max_version` is `None` or a 2-tuple (major only is read; bools count as ints).
- **Capsule flavor**: `max_version` major ≥ 1 → `DLManagedTensorVersioned` (named `dltensor_versioned`) with `version = (1, 0)`; **`max_version=None` (the default!) → legacy `DLManagedTensor`**. Requests above numpy's max are **clamped, never rejected**.
- Read-only arrays **cannot** be exported as legacy (`BufferError: Cannot export readonly array…`); versioned export sets `READ_ONLY` (bit 0) and, when `copy=True` was honored, `IS_COPIED` (bit 1).
- Layout: strides are **always** emitted (element units, never NULL for ndim > 0; `shape`/`strides` live in one 8-byte-aligned allocation); `byte_offset` is **always 0** (deliberate, to match PyTorch practice, `dlpack.c:346-358`); `data` is the array's raw pointer; 0-d arrays have NULL shape/strides; size-0 arrays export with **non-NULL** data (contrary to the header's advice).
- Stride restriction: a non-C-contiguous export requires every non-size-1 dimension's byte stride to be a multiple of itemsize, else `BufferError`; size-1 dims with non-multiple strides get **truncated** element strides (a `(10,5,1)` array with byte strides `(8,80,4)` exports element strides `[1,10,0]`).
- **Byte order**: non-native byte order → `BufferError: DLPack only supports native byte order.` (so byte-swapped arrays never leave NumPy via DLPack).
- Dtype coverage on export: bool→`kDLBool`/8; ints `kDLInt`, uints `kDLUInt` (8/16/32/64); floats ≤ 8 bytes → `kDLFloat` (16/32/64); complex ≤ 16 bytes → `kDLComplex` (64/128); **everything else → `BufferError`** (datetime64/timedelta64/strings/void/structured/object/float128/complex256; bfloat16 is not a NumPy dtype at all). Escape hatch: `np.dtypes.register_dlpack_dtype((code,bits), dtype)` (2.5+) makes user dtypes (e.g. `ml_dtypes` bfloat16) interoperable both ways.
- Device reported: `(kDLCPU, 0)` for ordinary arrays; an array that *arrived* via DLPack re-reports its origin device (base-chain walk).

### 6.3 What this means for a full-coverage matrix (NumPy side)

| rstsr dtype | DLPack code | numpy import | numpy export |
|---|---|---|---|
| bool | `kDLBool`/8 | ✅ | ✅ (≥1.25) |
| i8/i16/i32/i64, u8/u16/u32/u64 | `kDLInt`/`kDLUInt` | ✅ | ✅ |
| **i128 / u128** | `kDLInt`/`kDLUInt` bits=128 | ❌ `Unsupported dtype` (numpy has no int128) | n/a |
| f16/f32/f64 | `kDLFloat` 16/32/64 | ✅ | ✅ |
| **bf16** | `kDLBfloat`/16 | ❌ unless registered (numpy has no bfloat16) | ❌ (dtype doesn't exist) |
| Complex32/64 | `kDLComplex` 64/128 | ✅ | ✅ |
| (any byte-swapped dtype) | — | — | ❌ `native byte order` BufferError |
| datetime64/timedelta64, strings, object, structured | not expressible | ❌ | ❌ |

So: **every dtype NumPy has that rstsr can hold is interoperable, except `i128/u128` and `bf16` (and byte-swapped input, which NumPy refuses to export)**; conversely rstsr's `i128/u128/bf16` cannot be handed to stock NumPy via DLPack.

### 6.4 The constraints this puts on rstsr's bridge (both directions)

**Export (rstsr → NumPy).**
1. NumPy will only consume an **object with `__dlpack__`** — a pyclass (or a user-side Python shim holding a capsule). The "just hand over a capsule" idea does not work with `np.from_dlpack`.
2. Our `__dlpack__` must accept `dl_device=None`/`copy=None`/`max_version=(1,0)` kwargs, be **idempotent under a double call**, and clamp rather than reject high `max_version`s.
3. Produce a `dltensor_versioned` capsule with `version=(1,0)`; emit `READ_ONLY` when exporting a view whose data must not be written; honor `copy=True` (materialize + `IS_COPIED`) or raise `BufferError` if impossible.
4. Emit element strides (identical to rstsr's layout units), `byte_offset = offset × itemsize` (or shift `data`), `data` pointing at the logical origin; there is **no alignment obligation in practice** but a 256-byte-aligned `data` is still the polite choice.
5. rstsr tensors with ndim > 64, or dtypes `i128/u128/bf16`, must fail with a clear error toward NumPy (or go through an explicit cast/copy path).

**Import (NumPy → rstsr).**
6. Always request `__dlpack__(max_version=(1, 0))`; handle both capsule flavors; treat a legacy capsule as **read-only** (it cannot signal writability); consume/rename exactly once and call the deleter once.
7. Map element strides 1:1 (watch the size-1 truncation artifact: numpy may hand `stride[i] = 0` for size-1 dims — harmless but must not break layout equality checks), `offset = byte_offset / itemsize` (verify divisibility), accept negative and zero strides, accept `ndim == 0` and size-0, and do **not** require alignment.
8. Dispatch dtype by `(code, bits)` per the table; reject everything else with a precise error (or offer an explicit opt-in cast path). Read-only capsules map naturally onto rstsr's `TensorView`; writable ones can map to `TensorMut`/owned only after the G1 storage question is settled.
## 7. DLPack v1.3 — the contract a bridge must satisfy

Full evidence and citations in `notes/dlpack-v13-spec.md`. Digest (pinned to DLPack v1.3 = `84d107b`; the local checkout is actually `v1.3-4-g94485e2`, see drift note below):

**Structs and ownership**

- `DLManagedTensorVersioned` (80 B: `version, manager_ctx, deleter, flags, dl_tensor`) is the standard since 1.0; `DLManagedTensor` (64 B) is deprecated legacy. Both are *borrowing* wrappers: the consumer calls `deleter` exactly once when done; the deleter must free `manager_ctx` **and the struct itself**.
- Capsules: `"dltensor_versioned"` (legacy: `"dltensor"`); the consumer takes ownership by renaming to `"used_dltensor_versioned"` (legacy: `"used_dltensor"`); the producer's capsule destructor calls the deleter only if the name is still the producing one. Names must be statically allocated.
- Version rules: major mismatch ⇒ consumer calls the deleter and reads **nothing else**; minor mismatch ⇒ usable (minor bumps only add enum values). Python `max_version` is a 2-tuple; the producer may return a different version and the consumer must verify. Legacy producers are detected by catching `TypeError` on the `max_version` kwarg; version-negotiation failure recommends `BufferError`. **NumPy 2.5.2 emits and requests `(1, 0)`** — not 1.3.

**Layout**

- **Strides are in elements, not bytes** (the opposite convention from NumPy's C-API and from `dlpack-ffi`'s field naming); `byte_offset` is in bytes and is `uint64_t` (non-negative). Element `(0,…,0)` is at `(char*)data + byte_offset`.
- v1.2+ requires `strides != NULL` when `ndim != 0` (the prose spec's "NULL means row-major" is stale, though NumPy still tolerates NULL on import). `ndim == 0` may have NULL shape/strides; zero-size tensors should set `data = NULL`.
- Negative and zero strides: representable, undocumented, not validated by NumPy — pass-through territory.
- Alignment: the header claims 256-byte alignment but explicitly says nobody adheres and advises not to rely on it; NumPy deliberately sets `byte_offset = 0` and points `data` at element 0.
- `lanes > 1` = vector element; sub-byte types are packed (default) with `bits` required to be exact.
- Native endianness is assumed; **exporting non-native-endian data must raise an explicit error** (header).

**Flags** (versioned struct only): `READ_ONLY = 1`, `IS_COPIED = 2`, `IS_SUBBYTE_TYPE_PADDED = 4`. NumPy honors READ_ONLY (and raises `BufferError` rather than export a read-only array to a pre-1.0 consumer, which cannot express it). Recommended consumer policy: ignore READ_ONLY if unsupported rather than raise.

**Devices**: `{kDLCPU=1, kDLCUDA=2, … kDLTrn=18}` in v1.3; CPU is `{kDLCPU, 0}`. Cross-device exchange is consumer's-risk; NumPy (CPU-only) accepts `kDLCPU`/`kDLCUDAHost`/`kDLROCMHost`/`kDLCUDAManaged` and rejects the rest with `BufferError`.

**`copy` semantics**: `copy=True` ⇒ producer copies and **must** set `IS_COPIED`; `copy=False` ⇒ never copy, `BufferError` if impossible; `copy=None` ⇒ copy if needed.

**`dlpack-ffi` (1.3.0) gives exactly the raw types** — 7 structs, newtype enums (`.0` + associated consts), 3 flag constants, nullable deleters — and deliberately **no helpers** ("We do not add any safe-wrappers or abstractions"). A consumer crate must supply capsule logic, version checks, dtype/device tables, validation, and safe wrappers. Two wrinkles: the flag constants are typed `u32` while `flags` is `u64` (cast needed), and the crate's own ownership test does not demonstrate a spec-conforming deleter (it frees only `manager_ctx` because its structs are stack locals).

**Version drift**: upstream HEAD adds `kDLTPU/kDLTPUHost/kDLAscend` device types and `kDLBcomplex`; neither the v1.3 header nor `dlpack-ffi` has them. Pin to v1.3 unless there is a reason to track main.

**Implication for rstsr**: the DLPack contract is small enough to implement correctly in a few hundred lines, but it has real footguns — element-strides, the capsule rename protocol, deleter-called-exactly-once, major-version bail-out, and the "who keeps the allocation alive" question on negative-stride views. None of it belongs in `rstsr-core`; all of it maps cleanly onto a bridge crate over `dlpack-ffi`.
## 8. rstsr today — the bridge-relevant inventory

Full evidence with citations: `notes/rstsr-state.md` (1216 lines, sections A–H). Digest (rstsr `f179c46`, v0.9.0; spot-verified by reading `rstsr-common`/`rstsr-core` during synthesis):

**8.1 Storage and ownership.** Storage payload types are `DataOwned`, `DataRef<'a>`, `DataMut<'a>`, `DataCow<'a>`, `DataArc`, `DataReference<'a>` (`rstsr-core/src/storage/data.rs:11-39`); a tensor is `Storage::new(data, device)` (`storage/device.rs:81`) plus a `Layout`. There is **no foreign-buffer storage**: nothing that owns a pointer it did not allocate, and nothing that carries a deleter.

**8.2 Zero-copy is closer than expected — for views.** The *view* direction needs no new core types, only unsafe glue:

- *NumPy → rstsr view*: fabricate a non-owning `Vec` over the foreign pointer (`Vec::from_raw_parts(ptr, len, len)`), wrap it in `DataRef::from_manually_drop(ManuallyDrop::new(raw))` (`storage/data.rs:98`), `Storage::new` it, and build a `TensorView` via `TensorView::new_f` (`tensor/tensorbase.rs:195`) or `TensorBase::new_unchecked` (`:115`). This is exactly the pattern rstsr already uses for foreign slices: the `asarray` slice overloads (`tensor/asarray.rs:519-543`, mutable at `:693-718`, gated `B: DeviceAPI<T, Raw = Vec<T>>`) and `IntoRSTSR for MatRef/ColRef/MatMut` (`device_faer/conversion.rs:53-75`).
- *rstsr → foreign view*: `TensorAny::as_ptr()/as_mut_ptr()` already return offset-adjusted pointers (`tensor/ownership_conversion.rs:629/636`) and `layout()` exposes shape/stride/offset in **elements**.
- **The documented limit is narrower than it reads** *(corrected in `RESPONSE-discussion-R1.md` V1)*: an *owned* **faer** allocation must not be re-homed into a `Vec` — faer over-aligns (64 B) and pads its row capacity, so a `Vec`'s dealloc layout would differ (`device_faer/conversion.rs:85-94`; verified against faer 0.22.6 `src/mat/matown.rs:9-13, 83-85`). The transferable rule is: **never let a *deallocating* `Vec<T>` own foreign memory** — the repo's `ManuallyDrop` + external-owner pattern is the sound alternative. Owning a NumPy buffer therefore needs a storage repr that carries the owner (pointer + fabricated `Vec` + deleter/Python reference), and that repr can be defined **in the bridge crate with public API — no core change** (`RESPONSE-discussion-R1.md` V2: `Storage::new`, `DataAPI`/`DataMutAPI`, `TensorAny::new_f` are all public; core impls are generic over `R`).

**8.3 Layout units.** `Layout<D>` = `{shape, stride, offset}` with `stride: D::Stride` where `type Stride: AsMut<[isize]>` (`rstsr-common/src/layout/dim.rs:37`) — **signed, element-unit strides** — and `offset: usize` (element units) (`rstsr-common/src/layout/layoutbase.rs:15-23`). So: strides map to DLPack's element strides **without conversion**; `offset` maps to DLPack's **byte** `byte_offset` (× itemsize); a negative *offset* is not representable (negative *strides* are).

**8.4 Dtype system.** Compile-time traits only — `ExtNum`/`ExtReal`/`ExtFloat`/`DTypePromoteAPI` with macro enumerations (`rstsr-dtype-traits`, `ext_num.rs:65,103,149,200,253`). Supported element types: `u8..u128`, `usize`, `i8..i128`, `isize`, `f32`, `f64`, `Complex<f32/f64>`, `bool`, and feature-gated `half::f16`/`bf16`. **There is no runtime dtype tag/enum** — importing a DLPack tensor therefore needs a `(code, bits) → monomorphized constructor` dispatch table; exporting does not (a trait constant per element type suffices).

**8.5 Device model.** `DeviceBaseAPI` / `DeviceRawAPI<T>` / `DeviceStorageAPI<T>` / `DeviceAPI<T>` (`storage/device.rs:3/9/23/159`). Devices are *not* ZSTs (`DeviceCpuSerial { default_order }`, `DeviceFaer { base: DeviceCpuRayon }`) but they carry **no device kind or device id** — the mapping to `DLDevice` must be invented. Default device = `DeviceFaer` under `faer_as_default`, else `DeviceCpuSerial` (`rstsr-core/src/lib.rs:45-49`). No GPU crate exists anywhere in the workspace.

**8.6 Python/DLPack presence: zero.** No `pyo3`, no `PyCapsule`, no `dlpack`/`dltensor` occurrence in any `Cargo.toml` or `.rs` of the workspace. `dlpack-ffi` is deliberately pure (the `update-ffi-dlpack` skill forbids rstsr references), so bridge glue cannot live there.

**8.7 A recorded adverse decision to revisit — with a nuance.** rstsr's docs currently strike out `from_dlpack` (`rstsr-core/src/tensor/creation.rs:11`: `- [ ] ~`from_dlpack`~`), mark it status **D** (`rstsr-core/src/docs/array_api_standard.md:130`), and state under "Other Dropped Specifications" (`:378-380`): *"Functions related to Python Array API namespace and dlpack. … This is not possible for another language currently."*

The nuance: that rationale is **correct for `rstsr-core` as a pure-Rust array-API namespace** — the standard's `from_dlpack(x)` takes a *Python object* with `__dlpack__`, which a Rust crate cannot accept without becoming a Python extension. It is **not** a statement that rstsr cannot interchange with NumPy: with a pyo3 bridge crate (the thing this review exists to scope), both directions are implementable, and the Rust-side analogue is an ordinary Rust trait (`FromDLPack`/`AsDLPack`-style) rather than a Python-namespace function. The decision to keep the Python-array-API surface out of `rstsr-core` therefore does *not* preclude a bridge crate; this review recommends keeping that separation explicit. **Revisiting this recorded decision is the single most important consensus item.**

**8.8 Test/bench infrastructure.** Entry-binary tests (`entry_row_cpu`, col-major twins), no criterion harness yet in the workspace; a cross-language test would be a new category (needs a Python interpreter + NumPy in CI).

### Gap list (what does not exist today)

| # | Gap | Needed for | Effort/notes |
|---|---|---|---|
| G1 | foreign-owner storage (own pointer + deleter) | owned adoption of NumPy/DLPack buffers; safe "storable" imports | new storage **repr** (fabricated `Vec` + owner) — definable in the bridge crate; **no core change** (corrected in `RESPONSE-discussion-R1.md` V2) |
| G2 | runtime dtype dispatch for import | NumPy → rstsr for arbitrary dtype | macro-generated `(code,bits) → T` table; export needs only trait consts |
| G3 | device kind/id mapping | `DLDevice` production/validation | one trait method + CPU value now; extensible later |
| G4 | capsule + protocol layer (`__dlpack__`, rename rules, versioning) | both directions | pyo3-based; must be a new crate (G7) |
| G5 | unsafe glue for non-owning views (+ documentation of the invariants) | zero-copy import | pattern exists (`asarray`/`DataRef`); needs audited wrapper |
| G6 | byte_offset ↔ element-offset conversion + negative-stride policy | layout mapping | `offset × itemsize`; decide policy for `usize` offset vs DLPack's unsigned byte offset |
| G7 | a home for the bridge (new crate; not `rstsr-core`, not `dlpack-ffi`) | everything | pyo3 + `dlpack-ffi` deps; separate MSRV possible |
| G8 | Python/NumPy in the test loop | round-trip and lifetime tests | CI category that does not exist yet |

---

## 9. Bridge options — analysis

### 9.1 Design axes (what any option must pin down)

| Axis | Choices | Constrained by |
|---|---|---|
| D1 direction | import / export / both | Q1 |
| D2 mechanism | DLPack / NumPy C-API / byte copy / hybrid | §6–7 |
| D3 Python object | pyclass with protocol dunders / user-side shim / none | `np.from_dlpack` requires `__dlpack__` on the *object* (§6.1) |
| D4 ownership on import | view (lifetime-bound) / owned (deleter) / copy | rstsr has no foreign-owner storage (§8, G1) |
| D5 dtype policy | exact-and-error / opt-in cast / implicit cast | §6.3 |
| D6 device policy | CPU-only now, trait-tagged later | §7, §8.5 |
| D7 code home | new crate / core feature / separate repo | `dlpack-ffi` purity, MSRV (Q3, Q10) |
| D8 fallback | none / bytes / C-API | §6.3 gaps |

### 9.2 Option A — DLPack bridge crate (pyo3 + `dlpack-ffi` + rstsr)

*Shape*: new optional crate; a `#[pyclass]` wrapper carrying an rstsr tensor (or a holder)
with `__dlpack__`/`__dlpack_device__`; Rust-side functions `from_dlpack`/`asarray`-style for
the import direction; a foreign-buffer view (and optionally an owner+deleter storage, G1).

- **Pros**: zero-copy both directions; ownership transfer in both directions is the model's
  native semantics; library-neutral (works with PyTorch/CuPy/JAX too, all of which implement
  `__dlpack__`); tiny, versioned contract (vs. hundreds of C-API slots); device-neutral for
  future GPU; all of rstsr's *present* dtypes are expressible (§6.3).
- **Cons**: dtype gaps are structural (datetime64/strings/object/structured are not
  representable; `i128/u128`/`bf16` are not consumable by stock NumPy); needs pyo3 (MSRV
  1.83+); needs an audited unsafe layer; the export side *must* ship a Python-visible object.
- **Effort**: medium; risk concentrated in lifetimes/ownership and nondeterministic
  double-call semantics (§6.1) — all testable.

### 9.3 Option B — NumPy C-API bridge (depend on rust-numpy)

*Shape*: new crate depending on the `numpy` crate; conversions rstsr ↔ `PyArray`/borrow guards.

- **Pros**: NumPy's full dtype surface (datetime64/timedelta64/strings/object via raw descr
  APIs) and its casting machinery; mature, battle-tested FFI; convenient argument extraction
  (`PyArrayLike`, borrow guards) for the Python-facing API.
- **Cons**: NumPy-specific; brings `ndarray` + rust-numpy + pyo3 into the picture; its
  **import side is borrow-only** (no ownership adoption), so a storable rstsr tensor is a
  copy; its ≤32-dim assumption is already stale vs NumPy 2; zero-copy export still needs a
  Python-side owner object (same pyclass work as Option A) plus ndarray-view machinery.
  In short: no better than DLPack where DLPack works, thinner coverage than bytes where it
  doesn't, and more coupling.
- **Effort**: medium-high for a worse coverage/zero-copy trade.

### 9.4 Option C — hybrid (A + explicit fallback)

Zero-copy DLPack as the primary path; an explicitly-named copy fallback for the
non-expressible cases. The fallback can be:
- **bytes/`.npy`-style copy** (simplest; covers *every* dtype via an explicit
  `np.asarray(x).astype(...)` on the Python side); or
- **rust-numpy-based** (if NumPy casting semantics are desired in Rust).

### 9.5 Option D — copy-only interchange

Python-side helpers plus raw byte transfer (`tobytes`/`frombuffer`, or `.npy` files written
from Rust). No unsafe, no new Rust deps beyond an entry point (or none at all if file-based).
Cost: no zero-copy — which is the entire point of the exercise — so it is a fallback, not a
primary. (Note: as a *file* path it also works with zero Python glue at test time.)

### 9.6 Option E — C ABI + ctypes (no pyo3), spike only

`extern "C"` functions + a Python shim (ctypes can call `PyCapsule_New`). Avoids the pyo3
dependency/MSRV question entirely, but forces hand-rolled capsule and Python-object plumbing
and gives no safe ownership story; the export side still needs a Python object with
`__dlpack__`. Only worth a throwaway spike if Q10 blocks pyo3.

### 9.7 Comparison

| Criterion | A (DLPack) | B (rust-numpy) | C (A+fallback) | D (copy) |
|---|---|---|---|---|
| zero-copy NumPy → rstsr | ✅ (view; owned with G1) | ⚠️ borrow-only (copy to store) | ✅ | ❌ |
| zero-copy rstsr → NumPy | ✅ | ⚠️ needs ndarray view + pyclass owner | ✅ | ❌ |
| dtype coverage (read) | numeric/bool only | full NumPy surface | full (fallback copies) | full (copy) |
| dtype coverage (write) | all rstsr dtypes except i128/u128/bf16 → NumPy | full | full (fallback copies) | full (copy) |
| ownership transfer both ways | ✅ native (deleter) | ❌ import side | ✅ | ❌ |
| dependency burden | pyo3 (MSRV 1.83) | pyo3 + rust-numpy + ndarray | +fallback deps | minimal/none |
| NumPy-specific code | no (any DLPack peer) | yes | mostly no | no |
| future GPU | ✅ device field | ❌ CPU-only | ✅ | ❌ |
| maintenance surface | small, versioned | large (C-API slots, NumPy releases) | small | tiny |
| effort / risk | medium / medium | medium-high / medium | medium / low | low / low |

### 9.8 Recommendation

1. **Adopt Option A as the core mechanism** — it is the only mechanism that makes ownership
   transferable in both directions, it is the array-API ecosystem's standard, and it is the
   cheapest long-term maintenance surface.
2. **Keep an explicit copy fallback (Option C) for the structural dtype gaps** — the honest
   answer for datetime64/strings/object/structured is a copy; do not contort DLPack into
   covering them.
3. **Do not lead with Option B.** rust-numpy is invaluable as *prior art and reference*
   (this review), and possibly as an optional extra later for NumPy-specific semantics; it is
   the wrong primary boundary for rstsr.
4. **Option E only as a spike** if the pyo3/MSRV question (Q10) blocks Option A.

---

## 10. What rstsr would need to build (work items, not a plan)

Derived from the §8 gap list and the §6.4 constraints. Each item lists its acceptance
evidence; per `QUESTIONS-discussion-R1.md` Q13, none of this starts before the direction questions
are settled (R1 answered Q1–Q3/Q7/Q10/Q12; Q4 remains open — see `RESPONSE-discussion-R1.md` §5).

| # | Work item | Depends on | Acceptance evidence |
|---|---|---|---|
| W1 | bridge crate skeleton (new optional crate; pyo3 + `dlpack-ffi` + rstsr path dep; own MSRV policy) | Q3, Q10 | crate builds; `rstsr-core` unchanged |
| W2 | dtype mapping: export trait constants (`(code,bits)`) and import dispatch table `(code,bits) → T` | §6.3 | matrix test over every dtype × both directions |
| W3 | layout mapping: element strides pass-through, `offset × itemsize → byte_offset`, negative/zero strides, size-1 stride artifacts, ndim ≤ 64 guard | §6.4 | strided/reversed/broadcast/0-d/size-0 round-trips |
| W4 | device mapping trait (`DLDevice` + `__dlpack_device__`) with `kDLCPU/0` only | Q12 | device tuple round-trip |
| W5 | export protocol: pyclass with `__dlpack__` (kwargs, idempotence, clamping, `READ_ONLY`, `IS_COPIED`, `copy=True`), capsule creation, deleter keeping the tensor alive | §6.4 (1–5) | `np.from_dlpack(obj)` shares memory and survives Rust scope; double-call test |
| W6 | import protocol: consumer logic (call with `max_version=(1,0)`, capsule-name/version/device/dtype validation, rename-once, deleter-once) + foreign view via the existing `DataRef` pattern | §6.4 (6–8) | view aliases writes; GC of the NumPy array with the view alive behaves per policy |
| W7 | ownership repr for adopted buffers (only if Q6 says "owned"; **lives in the bridge crate, not core** — `RESPONSE-discussion-R1.md` V2) | G1, Q6 | deleter runs exactly once on last drop; no UB under borrow-checker tests |
| W8 | cross-language test harness (Python + NumPy in CI; round-trip, refcount/lifetime, error paths, idempotence) | Q13 | test list from §6.1–6.4 executed green |
| W9 | fallback copy path for non-expressible dtypes (bytes or `.npy`) | Q2, Q7 | explicit per-dtype error vs opt-in cast tests |

As verified in R1, the bridge design has **no core-touching items**: even W7's foreign-owner repr
can live in the bridge crate (`RESPONSE-discussion-R1.md` V2). Any rstsr-core change would be
surface-only (a public `TensorForeign` alias, a `from_dlpack` entry point) if it is ever wanted.

---

## 11. Open questions

The decision points this review feeds are collected in **[QUESTIONS-discussion-R1.md](./QUESTIONS-discussion-R1.md)**
(Q1–Q14, with recommendations), answered in **[ANSWERS-discussion-R1.md](./ANSWERS-discussion-R1.md)** and
processed in **[RESPONSE-discussion-R1.md](./RESPONSE-discussion-R1.md)** (verifications V1–V5, corrections
C1–C5, Q4 redux). R1 settled all questions except **Q4 (the Python-facing deliverable)**, which stays open
pending the five points in `RESPONSE-discussion-R1.md` §5. **No implementation or detailed design should
start before those are answered.**

---

## Appendix A — rust-numpy source inventory

| File | Lines | Purpose |
|---|---|---|
| `src/lib.rs` | 187 | crate docs, exports, prelude, `pyarray!` macro, free-threading guard |
| `src/array.rs` | 1763 | `PyArray<T,D>` + `PyArrayMethods` (constructors, views, casts, reshape/transpose) |
| `src/untyped_array.rs` | 320 | `PyUntypedArray` + shape/strides/flags/alignment accessors |
| `src/dtype.rs` | 831 | `Element` trait + scalar dtype impls, `PyArrayDescr` and methods |
| `src/convert.rs` | 330 | `IntoPyArray`, `ToPyArray`, `NpyIndex`, `ToNpyDims`, strides conversion |
| `src/array_like.rs` | 216 | `PyArrayLike` + `TypeMustMatch`/`AllowTypeChange` |
| `src/borrow/mod.rs` | 781 | borrow guards, rationale, `FromPyObject` impls |
| `src/borrow/shared.rs` | 1012 | cross-extension borrow registry (capsule API, conflict math), tests |
| `src/slice_container.rs` | 91 | `PySliceContainer` — Rust-owned memory freed by Python GC |
| `src/error.rs` | 188 | error types (BorrowError, AsSliceError, dtype/rank mismatch reporting) |
| `src/strings.rs` | 233 | `PyFixedString<N>` / `PyFixedUnicode<N>` element types |
| `src/datetime.rs` | 349 | `Datetime<U>` / `Timedelta<U>` element types + units |
| `src/random.rs` | 419 | BitGenerator/Generator wrappers |
| `src/sum_products.rs` | 200 | `dot` / `inner` / `einsum` |
| `src/npyffi/mod.rs` | 169 | API-table loading, type-object registry, `is_numpy_2` |
| `src/npyffi/array.rs` | 475 | 235 active array-API slots + ABI/feature/endianness checks |
| `src/npyffi/objects.rs` | 665 | C struct layouts (`PyArrayObject`, `PyArray_Descr*`, iterators, ufunc, strings) |
| `src/npyffi/types.rs` | 287 | C typedefs and enums (`NPY_TYPES`, `NPY_ORDER`, …) |
| `src/npyffi/flags.rs` | 83 | array/iter/dtype flag constants |
| `src/npyffi/ufunc.rs` | 113 | ufunc API table (38 active slots) |
| `src/npyffi/random.rs` | 11 | `bitgen_t` |
| `src/npyffi/numpyconfig.rs` / `npy_common.rs` | 28 / 8 | ABI/API version constants; endianness constants |

---

## Appendix B — how to re-verify the fact base

All commands are read-only (the probes only read NumPy memory).

```bash
# rust-numpy revision; and the negative claim "no DLPack support"
git -C ~/Git-Others/rust-numpy rev-parse HEAD          # da6bf5be05d4053cf0c51afa12efc018a984ca7f
grep -rin "dlpack\|dltensor\|__dlpack__" ~/Git-Others/rust-numpy/src   # (no output)

# NumPy revision and DLPack implementation location
git -C ~/Git-Others/numpy describe --tags              # v2.5.2
sed -n '1,60p' ~/Git-Others/numpy/numpy/_core/src/multiarray/dlpack.c

# empirical probes (installed NumPy 2.5.1, conda env torch)
conda run -n torch python experiments/probe_numpy_dlpack.py | tee experiments/probe-output.txt

# DLPack revision (+ the drift: the checkout is 4 commits past the v1.3 tag)
git -C ~/Git-Others/dlpack describe --tags             # v1.3-4-g94485e2
git -C ~/Git-Others/dlpack diff v1.3..HEAD -- include/dlpack/dlpack.h

# dlpack-ffi ABI/ownership tests
cargo test --manifest-path /home/a/rstsr_pack/dlpack-ffi/Cargo.toml
```

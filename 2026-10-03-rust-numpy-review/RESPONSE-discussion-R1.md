# Response to `ANSWERS-discussion-R1` — verification, corrections, and the Q4 redux

- **Date**: 2026-10-03 · **Companions**: `REVIEW.md` (fact base), `QUESTIONS-discussion-R1.md`,
  `ANSWERS-discussion-R1.md` (maintainer's answers), `notes/` (raw evidence)
- **Scope**: point-by-point disposition of the R1 answers; source-level verifications of the claims
  the answers rest on (V1–V5); corrections where either the review or the answers were
  imprecise (C1–C5); a concrete minimal shape for the re-opened Q4. **No implementation work**;
  nothing outside this directory was modified. All citations are at the pinned revisions (rstsr
  `f179c46`, numpy `v2.5.2`, faer 0.22.6, DLPack v1.3).

## 1. Disposition of the R1 answers

| Q | R1 answer | Disposition |
|---|---|---|
| Q1 | (c) both directions, two separable deliverables | accepted |
| Q2 | (c) DLPack primary + copy fallback; two layers (DLPack = light/C-interface; rust-numpy traits = heavy) | accepted; layer-2 shape refined in C5 |
| Q3 | (a) new workspace crate, `rstsr-cpu-*` naming | accepted; naming nit in C3; "no core change needed" verified in V2 |
| Q4 | "may need rediscussion"; no Rust-driven Python package, just make transfer possible | **open** — minimal shape proposed in §4 |
| Q5 | immutable view fine; mutable view still under consideration | constraint accepted; V4 adds a third rung that removes most of the difficulty |
| Q6 | `device_faer` UB comment may be wrong — check it | checked (V1): the comment is **correct** and *faer-specific*; review wording corrected (C2) |
| Q7 | personal scope f64/f32/i64/i32/complex; non-common types follow the recommendation | accepted (policy (a) + opt-in (b) stands) |
| Q8, Q9, Q11, Q13, Q14 | okay | accepted |
| Q10 | (a) bridge crate carries its own MSRV | accepted; the premise verifies and is even more permissive than assumed (V3) |
| Q12 | blanket trait impl over `DeviceAPI<Raw = Vec<T>>` | accepted; this is already the status quo of every CPU device (V2) |

## 2. Verifications

### V1 — the `device_faer` UB note is correct, and it is faer-specific (Q6)

The comment (`rstsr-core/src/device_faer/conversion.rs:85-94`, on `impl IntoRSTSR for Mat<T>`) says the
faer buffer cannot be re-homed into a `Vec` because:

> for the numeric element types (`f32`, `f64`, complex, ...) faer allocates with 64-byte alignment and
> pads its row capacity, while a `Vec<T>` always deallocates with `align_of::<T>()` and no padding.
> Deallocating with a different layout than the original allocation is undefined behavior by the
> allocator contract [...]

**Checked against faer 0.22.6 source — the comment is factually right** (in its stated scope):

- `faer-0.22.6/src/mat/matown.rs:9-13` — `align_for(size, align, needs_drop)` returns
  `max(align, 64)` when `!needs_drop && size.is_power_of_two()`; `f32`/`f64`/`Complex<f64>` are
  power-of-two-sized and `!needs_drop`, so their allocations are 64-byte aligned.
- `faer-0.22.6/src/mat/matown.rs:83-85` — when `align > size`, `row_capacity` is rounded up to a
  multiple of `align / size`; the row capacity (faer's innermost stride) is therefore *padded*.
- Consequence: handing faer's allocation to a `Vec<T>` that later deallocates it is UB (different
  size, different alignment, and a padded stride the `Vec` cannot represent).

**What this does *not* say** (the user's reading was right; the review's was too broad): it is not a
statement about rstsr in general, and it does not apply to a NumPy buffer. NumPy's memory is not
faer's; for the import direction the rule that transfers is narrower:

> never let a *deallocating* `Vec<T>` own foreign memory. The sound pattern is the one the repo
> already uses — a fabricated `Vec` wrapped in `ManuallyDrop`, plus an external owner whose `Drop`
> releases the real resource (`DataRef::from_manually_drop`'s documented contract,
> `storage/data.rs:87-100`; used by `asarray` and by `IntoRSTSR for MatRef`,
> `device_faer/conversion.rs:53-75`).

For NumPy specifically the "real resource release" is even less optional than for faer: the buffer
may belong to a *base object* (a view's memory cannot be freed at the pointer at all), so the owner
must be a Python reference or the DLPack deleter — never a `Vec` dealloc. See V4 for the cleanest
form of that ownership.

### V2 — every CPU device is `Raw = Vec<T>`, and a bridge-defined storage repr needs **no** rstsr-core change (Q3, Q12)

- All devices in the tree: `DeviceCpuSerial` (`rstsr-core/src/device_cpu_serial/device.rs:25`),
  `DeviceFaer` (`device_faer/device.rs:63`, and `DeviceCpu = DeviceFaer` by default,
  `lib.rs:46`), and the five BLAS backends
  (`crates-device/rstsr-{openblas,mkl,blis,aocl,kml}/src/device.rs:56`) — **all** `type Raw = Vec<T>`.
  So `B: DeviceAPI<T, Raw = Vec<T>>` is a complete bound today, exactly as Q12 assumes.
- A foreign-owner repr can live **outside** `rstsr-core`, using only public API:
  - `Storage<R, T, B>` + `pub fn new` (`storage/device.rs:14-21, 81-83`);
  - `DataAPI` / `DataCloneAPI` / `DataMutAPI` are public, **not sealed** (`storage/data.rs:285-307`);
  - `TensorAny<R, T, B, D>` is a public alias (`tensorbase.rs:98`), with a validated safe constructor
    `TensorAny::new_f` (`check_strides` + `bounds_index ≤ storage.len()`, `tensorbase.rs:195-208`);
  - the core API is *generic over `R`* throughout: e.g.
    `impl<R, T, B, D> TensorViewAPI for TensorAny<R, T, B, D> where R: DataAPI<Data = B::Raw>`
    (`ownership_conversion.rs:662-680`, mutable twin at `:725-744`), with the same pattern recurring
    in assignment, operators, reductions, manipulation and formatting (a scan of `for TensorAny<R`
    / `impl<R, T, B, D>` matches ~80 sites); op dispatch goes through those traits
    (`tensor/operators/op_with_func.rs:9-34`).
- Consequence: `struct DataForeign<T, O> { vec: ManuallyDrop<Vec<T>>, owner: O }` implementing
  `DataAPI<Data = Vec<T>>` (+ `DataMutAPI`, optionally `DataCloneAPI` as a deep copy) is enough to get
  `TensorAny<DataForeign<T, O>, T, DeviceCpuSerial, IxD>` through views, manipulation, assignment and
  the arithmetic/reduction operators. `W7` (G1) changes from "touch rstsr-core" to "a type in the
  bridge crate"; core is involved only if we *want* a public alias (e.g. `TensorForeign`) or a
  `from_dlpack` entry point in core — both are surface, not machinery.
- Caveats to carry into the design (all verified points, not open questions):
  1. the fabricated `Vec` must have `len ≥ bounds_index().1` of the exposed layout (`new_f` checks
     exactly that; the `MatRef` precedent sets it exactly) and must never be dropped/resized; its base
     pointer may be an interior pointer after the negative-stride normalisation of V5;
  2. `DataAPI::Data = Vec<T>` means any repr must present `&Vec<T>`; that is what keeps the pattern
     honest — the "owner stays in the struct, the `Vec` is a view" split;
  3. implementing `DataForceMutAPI<Vec<T>>` (the `unsafe fn force_mut`, `storage/data.rs:311-320`) is
     exactly the "mutable view" gate of Q5 — it is opt-in machinery, and implementing it for the
     foreign repr is a *decision*, not a necessity;
  4. `DeviceCpuSerial::to_cpu_vec`/`into_cpu_vec` clone the whole raw buffer
     (`device_cpu_serial/device.rs:33-38`), i.e. the *span*, not the logical tensor; the layout-driven
     extraction path `to_raw_f` (`ownership_conversion.rs:412-431`) is generic over `R` and returns
     exactly `size()` elements — that is the path to document.

### V3 — the workspace is de facto ≥ Rust 1.84 under default features (Q10)

- `faer = "0.22"` resolves to faer **0.22.6**, whose `Cargo.toml` declares `rust-version = "1.84.0"`
  (`~/.cargo/registry/.../faer-0.22.6/Cargo.toml:14`).
- `rstsr-core` has `default = ["row_major", "aligned_alloc", "faer", "faer_as_default"]`
  (`rstsr-core/Cargo.toml:37`), so the default build already requires ≥ 1.84, while the workspace
  declares `rust-version = "1.82.0"` (`rstsr/Cargo.toml:23`).
- Therefore pyo3 0.29's 1.83 does not raise the floor at all, and "the bridge crate carries its own
  MSRV" (Q10a) is safe and nearly free. Side note for the maintainer: the declared `1.82.0` is stale
  for default-feature builds (only the `--no-default-features` configuration is a true 1.82 build);
  worth a look independently of this task.

### V4 — `copy=True` gives a third rung: a **consumer-owned** buffer (affects Q5/Q6)

New fact, found while re-reading numpy's `__dlpack__` producer (`dlpack.c`):

- `arr.__dlpack__(copy=True, max_version=(1, 0))` makes NumPy **copy before exporting**
  (`dlpack.c:525-532`: `if (copy_mode == NPY_COPY_ALWAYS) { self = PyArray_NewCopy(self, NPY_KEEPORDER); }`)
  and records it in the versioned capsule (`dlpack.c:415-421`: `flags |= DLPACK_FLAG_BITMASK_IS_COPIED`).
- DLPack v1.3 defines that flag as: *"the tensor is considered solely owned throughout its lifetime by
  the consumer, until the producer-provided deleter is invoked"* (`dlpack-ffi/header/dlpack.h:315-321`).
- Generally, the versioned capsule carries `READ_ONLY` iff the **exported** array is not writeable
  (`dlpack.c:415-418`; with `copy=True` the exported array is the fresh copy, hence writeable), and a
  read-only array cannot be exported at all as a legacy capsule (`dlpack.c:537-543`).

So the import design has three rungs, not two:

| rung | call | ownership/aliasing | mutability |
|---|---|---|---|
| view (zero-copy) | `__dlpack__(max_version=(1,0))` | aliases the producer; contract "producer must not mutate while borrowed" | read-only if `READ_ONLY` set; writeable aliasing only under the Q5 contract |
| **copy (consumer-owned)** | `__dlpack__(copy=True, max_version=(1,0))` | **solely owned by rstsr**; no aliasing with Python | freely writeable, no contract |
| rstsr-side copy | import anything + copy yourself | rstsr-owned | writeable |

This is the clean answer to "owned adoption without the aliasing problem": ask the producer for a copy
(one memcpy, producer-side), and the foreign-owner repr then needs no `unsafe` aliasing contract at
all. A capsule from another producer carrying both flags simultaneously (`IS_COPIED | READ_ONLY`) is
not a contradiction — "copied, but the producer considers the content const" — and should be adopted
as owned-but-read-only unless rstsr makes an explicit copy of its own.

### V5 — strides, offset and alignment: the mapping is direct, with one normalisation and two checks

- NumPy's export writes **element strides** for every dimension (`dlpack.c:365-368`, “Strides in DLPack
  are items; in NumPy are bytes”), never `NULL`, and always `byte_offset = 0`
  (`dlpack.c:359-360, 371`, with an explicit comment that it ignores the 256-byte alignment suggestion
  and assumes exporting unaligned data is acceptable).
- Negative strides are **not** rejected on either side: the export check is only
  `shape[i] != 1 && strides[i] % itemsize != 0` (`dlpack.c:277-284`; `-8 % 8 == 0` passes), and the
  import simply multiplies back and builds the array (`dlpack.c:748-762`).
- rstsr `Layout` supports negative element strides, but `bounds_index` requires the address bound to be
  non-negative (`rstsr-common/src/layout/layoutbase.rs:235-260`, `rstsr_pattern!(min, 0..)`).
- Mapping recipe for import (unit-checked against both sides):
  `base' = data + byte_offset + min_index*itemsize`, `offset = -min_index` (elements), `strides` as
  given (may be negative), fabricated-`Vec` len `= max_index - min_index + 1`; `strides == NULL`
  (other producers, not NumPy) means C-contiguous. Export is the inverse: `byte_offset = offset*itemsize`,
  `strides = layout.stride()`, `data = raw.as_ptr()`.
- **Alignment must be checked on import** (rstsr-side, before fabricating any typed view):
  DLPack gives no alignment guarantee, and a producer can legally hand over data misaligned for its
  dtype (e.g. a NumPy array built over a buffer at a non-multiple offset — NumPy flags such arrays
  unaligned but still exports them). rstsr should reject (or copy) rather than fabricate a `Vec<T>`
  over a misaligned pointer. NumPy itself checks alignment in neither direction.

## 3. Corrections and refinements (C1–C5)

**C1 — Q4 as answered leaves one hard hole.** "Just make data transfer possible" still requires a
Python object with `__dlpack__` on the export side (bare capsules are rejected by `np.from_dlpack`;
REVIEW §6.1) and *some* Rust-callable entry point on the import side. The hole is not a problem with
the answer's spirit — it is that the answer does not yet say who writes those ~1 object + ~2 functions.
§4 turns this into three concrete shapes.

**C2 — where the error actually was (Q6).** The comment is right (V1); the *review* was the imprecise
one: it generalized a faer-specific dealloc-layout UB into "owning a foreign allocation is documented
UB" in rstsr (REVIEW §0.8, §8.2, G1). Corrected in place, with a pointer here. Nothing about the
import design depends on the wrong version — it depends on the correct one, which is *less* restrictive:
a bridge-crate repr may own a foreign buffer as long as the buffer is never freed by a `Vec`.

**C3 — crate naming (Q3).** `rstsr-cpu-pyo3` mixes the device axis (`cpu`) with the binding axis
(`pyo3`); pyo3 is not CPU-specific and would be shared by a future GPU bridge. Proposal:
`rstsr-cpu-dlpack` (device-axis name is justified: it will encode CPU storage assumptions and the
`kDLCPU` device mapping) for the pyo3-free core, and `rstsr-cpu-pyo3` or plain `rstsr-pyo3` for the
adapter. Either way `dlpack-ffi` stays pure types (no rstsr references), as its charter requires.

**C4 — Q12 needs no trait work in rstsr-core.** The blanket bound is already satisfied by every device
(V2). The "trait implementation" work lands in the bridge crate, where a *local* trait has no orphan
problems in either direction: it can be implemented for rstsr tensor types (foreign types, local
trait) and blanket-implemented over `B: DeviceAPI<T, Raw = Vec<T>>`. What rstsr-core *cannot* receive
from outside is a new `impl` of a core op-trait for a foreign type — and it does not need to, because
those impls are already generic over `R`.

**C5 — layer 2 ("implementing rust-numpy traits") is shaped worse than it looks.** rust-numpy's
`IntoPyArray`/`ToPyArray` are defined *in terms of ndarray* (`IntoPyArray::Dim: ndarray::Dimension`,
`rust-numpy/src/convert.rs:37-45, 129-137`), so implementing them for rstsr types drags `ndarray` +
`rust-numpy` + `pyo3` into the bridge crate and still routes the data through the NumPy C-API. Cheaper
alternatives with the same coverage: (i) provide `rstsr ↔ ndarray` conversions and let rust-numpy's own
ndarray impls do the work — no rust-numpy dependency; (ii) cover the expressiveness gap with the copy
fallback instead (`.npy` bytes — `npyz` is already a dev-dependency in the workspace — or
dtype-converting on the Python side). The heavy layer stays on the table, but I would not build it
first, and it does not need to be rust-numpy-shaped to be useful.

## 4. Q4 redux — the minimal Python-facing surface

Constraint that cannot be designed away: `np.from_dlpack(x)` needs `x` to be a Python object with
`__dlpack__`. So *someone* must own one pyclass and two thin functions. Three shapes:

| shape | who writes the Python-visible object | what rstsr ships | cost |
|---|---|---|---|
| **A. pure layer only** | the user (their own pyo3 module, or a ctypes shim over an exported C symbol) | `rstsr-cpu-dlpack` only | rstsr stays pyo3-free, but every user re-implements the fragile part: `max_version` negotiation, the double-call retry, capsule rename, deleter-exactly-once, refcount for repeated exports |
| **B. pure layer + thin adapter (recommended)** | rstsr, as a ~1 pyclass + 2 fn crate | `rstsr-cpu-dlpack` + `rstsr-cpu-pyo3` | one extra crate and a pyo3 dependency (in that crate only); no package, no wheels, no PyPI — built locally like any other crate, or vendorable into a user's own module |
| **C. A + reference example** | the user | `rstsr-cpu-dlpack` + an `examples/` ctypes + pyo3 adapter (not a crate) | keeps the workspace pyo3-free while still recording a working adapter to copy |

Recommendation: **B**, with the protocol logic (not the capsule plumbing) in the pure crate: the pure
layer produces/consumes `DLManagedTensorVersioned`/legacy pointers, owns the deleter/refcount
discipline, and exposes a small seam; the pyo3 crate does capsule creation, name checks, renames,
destructors and the holder class. If the maintainer prefers no pyo3 crate in the workspace, C delivers
the same capability with the adapter kept as executable documentation.

Also needed regardless of shape: a decision on the **hosting model**, because it decides what "the
Python side" is — Python importing an extension module, Rust embedding Python, or Python loading a Rust
`cdylib` through ctypes. Shape A is only ergonomic in the third case.

## 5. What R2 needs before any design is written

1. **Hosting model**: Python hosts Rust (extension/cdylib) / Rust hosts Python (pyo3 embed) / both.
2. **Q4 shape**: A, B or C (§4); if B, are `rstsr-cpu-dlpack` + `rstsr-cpu-pyo3` acceptable crate
   names and both in the rstsr workspace?
3. **Mutability policy for v1**: read-only zero-copy imports + `copy=True` for owned/writable ones
   (recommended, V4), and *no* mutable zero-copy import in v1 — or is a `force_mut`-style `unsafe`
   mutable path required from the start (Q5)?
4. **Milestone order**: export first (`np.from_dlpack(holder)` works), import first, or both in
   lockstep? (The two directions have disjoint difficulty: export needs the holder + protocol,
   import needs the foreign repr + dtype dispatch.)
5. **Core-surface question**: is `TensorAny<DataForeign<...>, ...>` (alias defined in the bridge
   crate) acceptable as the user-facing form, or should rstsr-core eventually carry a `TensorForeign`
   alias/`from_dlpack` entry point (surface only, no machinery, V2)?

## 6. Work items, updated after R1

| # | item | notes |
|---|---|---|
| W1 | `rstsr-cpu-dlpack` crate skeleton in the rstsr workspace | depends on `dlpack-ffi` (types only) + `rstsr-core`; MSRV 1.84 (V3) |
| W2 | foreign-owner repr + import (view rung) | V2 pattern; alignment check (V5); negative-stride normalisation (V5) |
| W3 | import, owned rung via `copy=True` / `IS_COPIED` | V4; retires the aliasing contract for owned imports |
| W4 | export path: transfer + refcount + versioned capsule, `READ_ONLY` policy | legacy only if a consumer cannot negotiate; never relax correctness |
| W5 | copy fallback for the expressiveness gap | dtype conversion helpers / `.npy` bytes; never silent |
| W6 | tests: deleter-exactly-once, capsule reuse, double-call, read-only/view/owned/aliasing matrix | error paths are the spec |
| W7 | pyo3 adapter crate (shape B) or `examples/` adapter (shape C) | Q4-dependent |
| — | *dropped from the earlier list*: "new storage variant in rstsr-core (G1)" | becomes W2 in the bridge crate (V2); core change only if R2.5 says yes |

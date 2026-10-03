# DLPack v1.3 contract — primary-source notes

Evidence gathered 2026-10-03 for the rstsr/NumPy interchange review.
Audience: rstsr maintainers writing a DLPack producer/consumer on top of the
`dlpack-ffi` crate.

**Citation legend** (absolute paths):

- `dlpack.h:NN` — `/home/a/Git-Others/dlpack/include/dlpack/dlpack.h` at tag
  **v1.3**, commit **84d107b** (the spec lives in the header comments). The
  vendored copy `/home/a/rstsr_pack/dlpack-ffi/header/dlpack.h` is byte-identical
  to that tag (verified by `diff` against `git show v1.3:include/dlpack/dlpack.h`),
  so its line numbers are the same.
- `python_spec.rst:NN` — `/home/a/Git-Others/dlpack/docs/source/python_spec.rst`
  (same at v1.3 and at current HEAD; the 4 commits after v1.3 touched only the header).
- `numpy/...` — NumPy v2.5.2 checkout `/home/a/Git-Others/numpy`.
- "array API std" — the published Python array API standard 2025.12 pages
  (`__dlpack__`, `from_dlpack`), which is where the normative *Python* rules live;
  `python_spec.rst` is DLPack's own copy/summary of that material.

> ## Version drift — read first
>
> The upstream checkout `/home/a/Git-Others/dlpack` is **not** clean at v1.3:
> `git describe` → `v1.3-4-g94485e2` (HEAD = 94485e2, 4 commits past the tag).
> The extra commits add: `kDLTPU = 19`, `kDLTPUHost = 20`, `kDLAscend = 21` to
> `DLDeviceType`, `kDLBcomplex = 18U` to `DLDataTypeCode`, and a doc-only change
> to the `data` comment (Metal `id<MTLBuffer>`). **None of these are in v1.3 or in
> the `dlpack-ffi` crate.** If rstsr ever targets upstream `main`, the vendored
> header/crate is stale by four commits. Nothing else in `docs/` changed.
> Command used: `git -C /home/a/Git-Others/dlpack describe --tags`,
> `git diff v1.3..HEAD -- include/dlpack/dlpack.h`.

---

## 1. Structs and ABI layout

Sizes/offsets below were independently re-derived by compiling the vendored
header with gcc (`_Alignof`/`offsetof`) on x86_64 — matching the crate's
hand-written tests, which pin the sizes `48/64/80` (`dlpack-ffi/tests/abi.rs:19-21`).

### `DLPackVersion` (`dlpack.h:42-66`) — 8 bytes, align 4

```c
typedef struct { uint32_t major; uint32_t minor; } DLPackVersion;
```

Normative semantics (`dlpack.h:45-59`):
- "A change in major version indicates that we have changed the data layout of
  the ABI - DLManagedTensorVersioned."
- "A change in minor version indicates that we have added new code, such as a
  new device type, but the ABI is kept the same."
- **MUST**: "If an obtained DLPack tensor has a major version that disagrees with
  the version number specified in this header file (i.e. `major != DLPACK_MAJOR_VERSION`),
  the consumer must call the deleter (and it is safe to do so). It is not safe to
  access any other fields as the memory layout will have changed."
- Minor mismatch: "the tensor can be safely used as long as the consumer knows
  how to interpret all fields."

`DLPACK_MAJOR_VERSION = 1`, `DLPACK_MINOR_VERSION = 3` (`dlpack.h:19,22`).
There is **no `DLPACK_VERSION` / `DLPACK_ABI_VERSION` macro in v1.3** — NEWS.md
mentions them for v0.7 (`NEWS.md` "## v0.7"), but since v1.0 the header defines
the two separate macros (verified for v1.0–v1.3 via `git show <tag>:...`).
When asked "what is `DLPACK_VERSION` in v1.3": it does not exist.

### `DLDevice` (`dlpack.h:125-136`) — 8 bytes, align 4

```c
typedef struct { DLDeviceType device_type; int32_t device_id; } DLDevice;
```

- `device_id`: "For vanilla CPU memory, pinned memory, or managed memory, this is
  set to 0." (`dlpack.h:131-135`)
- In C++ the enum is `enum : int32_t` (`dlpack.h:71-72`); in C it is an ordinary
  enum (int, 4 bytes). The Rust binding uses `c_uint` (u32); representation-wise
  identical for the defined non-negative values.

### `DLDataType` (`dlpack.h:184-215`) — 4 bytes, **align 2**

```c
typedef struct { uint8_t code; uint8_t bits; uint16_t lanes; } DLDataType;
```

- `code` is deliberately `uint8_t`, not the enum ("We keep it uint8_t instead of
  DLDataTypeCode for minimal memory footprint, but the value should be one of
  DLDataTypeCode enum values", `dlpack.h:203-208`). Only the low 8 bits of a
  `DLDataTypeCode` value are representable.
- "native endian-ness" is assumed; "An explicit error message should be raised
  when attempting to export an array with non-native endianness" (`dlpack.h:185-187`).
- `lanes`: "Number of lanes in the type, used for vector types" (`dlpack.h:213`);
  examples: `float4` = code 2, bits 32, lanes 4; `std::complex<float>` = code 5,
  bits 64, lanes 1; `bool` = code 6, bits 8, lanes 1 (`dlpack.h:189-197`).
- Sub-byte types are stored packed: "DLPack requires the data to be in little
  bit-endian, i.e., for a packed data set D, `((D >> (i * bits)) && bit_mask)`
  stores the i-th element" (`dlpack.h:199-201`).

### `DLTensor` (`dlpack.h:217-278`) — 48 bytes, align 8

| field | offset | type | meaning |
|---|---|---|---|
| `data` | 0 | `void*` | pointer to allocated data; may be opaque; 256-byte alignment claim (see below) |
| `device` | 8 | `DLDevice` | device of the tensor |
| `ndim` | 16 | `int32_t` | number of dimensions |
| `dtype` | 20 | `DLDataType` | element type |
| `shape` | 24 | `int64_t*` | shape; may be `NULL` iff `ndim == 0` |
| `strides` | 32 | `int64_t*` | strides **in number of elements, not bytes**; must not be `NULL` when `ndim != 0` (v1.2+) |
| `byte_offset` | 40 | `uint64_t` | "The offset in bytes to the beginning pointer to data" |

Semantics:
- Start of data: `(char*)data + byte_offset` is the address of element
  `(0,)*ndim` — stated in the header (`byte_offset` "should be used to point to
  the beginning of the data", `dlpack.h:224-225`) and in the repo's own app
  (`apps/numpy_dlpack/dlpack/dlpack.py:116-117`: "data + byte_offset gives the
  address of tensor element with index (0,) * ndim").
- Buffer size formula (`dlpack.h:236-244`): `prod(shape) * ((dtype.bits * dtype.lanes + 7) / 8)`.
- "Note that if the tensor is of size zero, then the data pointer should be set
  to `NULL`." (`dlpack.h:247-248`); the Python spec relaxes to "either `NULL` or
  `0`" (`python_spec.rst:161-163`).
- 256-byte alignment: "This pointer is always aligned to 256 bytes as in CUDA."
  (`dlpack.h:224`) — immediately followed by the caveat: "Note that as of Nov
  2021, multiple libraries (CuPy, PyTorch, TensorFlow, TVM, perhaps others) do
  not adhere to this 256 byte alignment requirement on CPU/CUDA/ROCm, and always
  use `byte_offset=0`. This must be fixed (after which this note will be updated);
  at the moment it is recommended to not rely on the data pointer being correctly
  aligned." (`dlpack.h:227-231`). NumPy states the same from the other side:
  "the `dlpack.h` header suggests/standardizes that `data` must be 256-byte
  aligned. We ignore this intentionally, because `__dlpack__` standardizes that
  `byte_offset` must be 0 (for now) to not break pytorch" (`numpy/.../dlpack.c:337-341`).

### `DLManagedTensor` (legacy) (`dlpack.h:280-308`) — 64 bytes, align 8

```c
typedef struct DLManagedTensor {
  DLTensor dl_tensor;                                        // offset 0
  void *manager_ctx;                                         // offset 48
  void (*deleter)(struct DLManagedTensor *self);             // offset 56
} DLManagedTensor;
```

- Header note: "This data structure is used as Legacy DLManagedTensor in DLPack
  exchange and is deprecated after DLPack v0.8. Use DLManagedTensorVersioned
  instead. This data structure may get renamed or deleted in future versions."
  (`dlpack.h:287-290`)
- "It is not meant to transfer the tensor. When the borrowing framework doesn't
  need the tensor, it should call the deleter to notify the host that the
  resource is no longer needed." (`dlpack.h:281-285`)
- `manager_ctx`: "the context of the original host framework ... It can also be
  NULL." (`dlpack.h:297-300`)
- `deleter`: "this should be called to destruct the `manager_ctx` which backs the
  DLManagedTensor. It can be NULL if there is no way for the caller to provide a
  reasonable destructor. The destructor deletes the argument `self` as well."
  (`dlpack.h:301-306`) — i.e. the deleter frees both `manager_ctx` and the
  `DLManagedTensor` allocation itself.

### `DLManagedTensorVersioned` (`dlpack.h:331-375`) — 80 bytes, align 8

Field order is **not** the intuitive one; note `flags` sits between `deleter` and
`dl_tensor`:

```c
typedef struct DLManagedTensorVersioned {
  DLPackVersion version;                             // offset 0
  void *manager_ctx;                                 // offset 8
  void (*deleter)(struct DLManagedTensorVersioned *self); // offset 16
  uint64_t flags;                                    // offset 24
  DLTensor dl_tensor;                                // offset 32 .. 80
} DLManagedTensorVersioned;
```

- "This is the current standard DLPack exchange data structure." (`dlpack.h:339`)
- Same borrowing/deleter wording as the legacy struct (`dlpack.h:331-337`, `:353-359`).
- `flags`: "By default the flags should be set to 0." (`dlpack.h:362-364`)
- Stability guarantee: "Future ABI changes should keep everything until this field
  stable, to ensure that deleter can be correctly called." (`dlpack.h:366-368`)
  — this is what makes the "call the deleter on major mismatch" rule safe.

### v1.3-only: `DLPackExchangeAPI` family (`dlpack.h:377-647`)

New in v1.3: C-level fast exchange. `DLPackExchangeAPIHeader { DLPackVersion
version; DLPackExchangeAPIHeader* prev_api; }` (16 bytes) and `DLPackExchangeAPI`
(56 bytes) holding `managed_tensor_allocator`,
`managed_tensor_from_py_object_no_sync`, `managed_tensor_to_py_object_no_sync`
(all required non-NULL), `dltensor_from_py_object_no_sync` (optional), and
`current_work_stream` (required). Exposed on the *type* as a PyCapsule named
`"dlpack_exchange_api"` (`dlpack.h:544-563`; `python_spec.rst:182-192`).
Consumers may ignore it entirely: "consumer frameworks can always start
implementing by only using the Python `__dlpack__` API, and then upgrade to the
C Exchange API later" (`python_spec.rst:190-192`). The API pointer "must stay
alive throughout the lifetime of the process" (`dlpack.h:600`).

---

## 2. Enums

### `DLDeviceType` (`dlpack.h:68-123`) — values present in v1.3

| name | value | note |
|---|---|---|
| `kDLCPU` | 1 | |
| `kDLCUDA` | 2 | |
| `kDLCUDAHost` | 3 | pinned by cudaMallocHost |
| `kDLOpenCL` | 4 | |
| *(5, 6 unused)* | — | renames from kDLGPU/kDLCPUPinned era; numbers retired |
| `kDLVulkan` | 7 | |
| `kDLMetal` | 8 | Apple GPU |
| `kDLVPI` | 9 | |
| `kDLROCM` | 10 | |
| `kDLROCMHost` | 11 | |
| `kDLExtDev` | 12 | reserved; "semantics can differ depending on the implementation" |
| `kDLCUDAManaged` | 13 | |
| `kDLOneAPI` | 14 | |
| `kDLWebGPU` | 15 | |
| `kDLHexagon` | 16 | |
| `kDLMAIA` | 17 | Microsoft MAIA |
| `kDLTrn` | 18 | AWS Trainium |

(Values verified by compiling the header; also `NEWS.md` for the v0.2–v0.8 history.)
Not in v1.3 but in upstream HEAD: `kDLTPU=19`, `kDLTPUHost=20`, `kDLAscend=21`.

### `DLDataTypeCode` (`dlpack.h:138-182`) — values in v1.3

`kDLInt=0`, `kDLUInt=1`, `kDLFloat=2`, `kDLOpaqueHandle=3`, `kDLBfloat=4`,
`kDLComplex=5`, `kDLBool=6`, `kDLFloat8_e3m4=7`, `kDLFloat8_e4m3=8`,
`kDLFloat8_e4m3b11fnuz=9`, `kDLFloat8_e4m3fn=10`, `kDLFloat8_e4m3fnuz=11`,
`kDLFloat8_e5m2=12`, `kDLFloat8_e5m2fnuz=13`, `kDLFloat8_e8m0fnu=14`,
`kDLFloat6_e2m3fn=15`, `kDLFloat6_e3m2fn=16`, `kDLFloat4_e2m1fn=17`.
No FP6/FP4 value may be assumed to be 6/4 bits: "Setting bits != 6 is currently
unspecified, and the producer must ensure it is set while the consumer must stop
importing if the value is unexpected" (`dlpack.h:171-180`). `kDLBcomplex=18U`
is upstream-HEAD-only (not v1.3).

Bit-packing of `DLDataType`: `code: u8` (widened from the enum), `bits: u8`,
`lanes: u16`; total 32 bits, struct alignment 2 (`dlpack.h:202-215`).

---

## 3. Flags (v1.3)

Defined only for `DLManagedTensorVersioned.flags` ("bit masks used in the
DLManagedTensorVersioned", `dlpack.h:310`):

| macro | value | semantics |
|---|---|---|
| `DLPACK_FLAG_BITMASK_READ_ONLY` | `1UL << 0` = 1 | "bit mask to indicate that the tensor is read only" (`dlpack.h:312-313`) |
| `DLPACK_FLAG_BITMASK_IS_COPIED` | `1UL << 1` = 2 | "bit mask to indicate that the tensor is a copy made by the producer. If set, the tensor is considered solely owned throughout its lifetime by the consumer, until the producer-provided deleter is invoked." (`dlpack.h:315-321`) |
| `DLPACK_FLAG_BITMASK_IS_SUBBYTE_TYPE_PADDED` | `1UL << 2` = 4 | "The default for sub-byte types (ex: fp4/fp6) is assumed packed. This flag can be set by the producer to signal that a tensor of sub-byte type is padded." (`dlpack.h:323-329`) |

- Default is `flags = 0` (`dlpack.h:364`).
- READ_ONLY is a producer declaration about the *data*; the header does not say
  what the consumer must do. The array API standard adds the de-facto rule:
  "A consumer that does not support read-only arrays should ignore this flag
  (this is preferred over raising an exception; the user is then responsible for
  ensuring the memory isn't modified)" (array API std, `__dlpack__` Notes).
  NumPy implements exactly that: `readonly = (managed->flags & DLPACK_FLAG_BITMASK_READ_ONLY) != 0`
  (`numpy/.../dlpack.c:648`).
- "Padded" is not further defined in v1.3 (uncertainty; see §10).

---

## 4. Ownership, deleter and capsule protocol

### Who calls the deleter, when

- The managed structs are for *borrowing*: "It is not meant to transfer the
  tensor. When the borrowing framework doesn't need the tensor, it should call
  the deleter to notify the host that the resource is no longer needed."
  (`dlpack.h:281-285`, repeated `:331-337`).
- The deleter "deletes the argument `self` as well" — it must free
  `manager_ctx` *and* the `DLManagedTensor(Verseioned)` allocation itself
  (`dlpack.h:301-306`, `:353-359`).
- If the deleter is `NULL` the caller cannot free the resource; the field is
  explicitly nullable ("It can be NULL if there is no way for the caller to
  provide a reasonable destructor", same lines). Consumers must therefore always
  null-check, as the spec's own example does: `if (managed->deleter) { managed->deleter(managed); }`
  (`python_spec.rst:120-123`).
- On major-version mismatch the consumer **must** call the deleter and **must
  not** read any other field (`dlpack.h:51-55`); the position of `deleter` is
  covered by the forward-compatibility note at `dlpack.h:366-368`.
- Call it **exactly once**. The Python capsule protocol exists precisely to
  arrange that (below). No spec text says "exactly once" in these words, but the
  capsule rename is designed so the destructor does not also call it
  (`python_spec.rst:97-106`).

### `manager_ctx` rules

- Opaque `void*`; may be `NULL` (`dlpack.h:297-300`, `:346-352`).
- Only the producer's deleter interprets it. It is the producer's hook for
  keeping the backing owner alive (e.g. NumPy stores the `PyObject*` array there;
  `python_spec.rst:151-157`).
- The deleter is allowed to do nothing if it cannot safely run (NumPy's example
  leaks when Python is finalized, `python_spec.rst:139-149`).

### Capsule protocol (Python binding of the struct)

Names (`python_spec.rst:80-104,126-127`; confirmed by NumPy's macros
`numpy/_core/src/common/npy_dlpack.h:8-11`):

| struct | produced name | renamed-to name |
|---|---|---|
| `DLManagedTensor` | `"dltensor"` | `"used_dltensor"` |
| `DLManagedTensorVersioned` | `"dltensor_versioned"` | `"used_dltensor_versioned"` |

The header itself never names the capsules; the rule is: "when
`DLManagedTensorVersioned` is in use the capsule names `dltensor` and
`used_dltensor` will need a `_versioned` suffix" (`python_spec.rst:83-85`).
"Capsule names ... must be statically allocated" (`python_spec.rst:126-127`)
and must not be copied — NumPy uses `PyCapsule_SetName` with a static string
(`numpy/.../dlpack.c:797-799`).

Rules:
- **Producer MUST** name the capsule `"dltensor"` (or `_versioned`) and install a
  `PyCapsule_Destructor` that calls the struct's deleter when the capsule is
  freed while still carrying the producing name (`python_spec.rst:92-95`).
- **Producer's destructor MUST** check the name first and do nothing if it was
  consumed: the reference implementation returns early on
  `PyCapsule_IsValid(self, "used_dltensor")` (`python_spec.rst:110-124`).
  So the producer checks *for the consumed name*, not for its own.
- **Consumer MUST** take ownership by renaming to `"used_dltensor"` (same
  versioned suffix), after which the capsule destructor will not call the
  deleter; the consumer's own array object then owns the struct and must call
  the deleter from *its* destructor (`python_spec.rst:97-106`).
- **Never consumed**: the capsule still holds the original name, so when its
  refcount reaches zero the destructor calls the deleter. "called either when
  the refcount on the capsule named `dltensor` reaches zero or the consumer
  decides to deallocate its array" (`python_spec.rst:104-106`). Ownership never
  gets lost, but the data stays alive until the capsule dies.
- The capsule is "consumed immediately within `from_dlpack` - therefore it is
  consumed exactly once, and it will not be visible to users of the Python API"
  (`python_spec.rst:87-90`).
- Threading/GIL: "The `DLManagedTensor` deleter must ensure that sharing beyond
  Python boundaries is possible, this means that the GIL must be acquired
  explicitly if it uses Python objects or API." (`python_spec.rst:129-133`).
  This is the only explicit thread-related rule in the corpus.

---

## 5. Version negotiation

Header-level (C, MUST):
- Consumer on major mismatch: call deleter, touch nothing else (`dlpack.h:51-55`).
- Consumer on minor mismatch: usable if it understands all fields; minor bumps
  only add enum values (`dlpack.h:57-59`).
- `DLManagedTensorVersioned.version` is "The API and ABI version of the current
  managed Tensor" (`dlpack.h:342-345`); the struct is the standard from DLPack
  1.0 on, `DLManagedTensor` is deprecated (`dlpack.h:287-290`).

Python-level (`max_version`; normative text lives in the array API standard,
summarized at `python_spec.rst:73-78`):
- Signature: `__dlpack__(*, stream=None, max_version: tuple[int, int] | None = None,
  dl_device=None, copy=None)` (array API std 2025.12). `max_version` is "the
  maximum DLPack version that the consumer ... supports, in the form of a 2-tuple
  `(major, minor)`". **Declared type is a 2-tuple** — a list is not specified;
  NumPy enforces `PyTuple_Check(...) && PyTuple_GET_SIZE(...) == 2` and raises
  `TypeError` otherwise (`numpy/.../dlpack.c:507-511`).
- The producer "may return a capsule of version `max_version` (recommended if it
  does support that), or of a different version. This means the consumer **must
  verify the version** even when `max_version` is passed." (array API std).
- Producer's recommended algorithm (array API std, `__dlpack__` Notes):
  - `max_version is None` → keep the DLPack 0.x implementation (legacy struct);
    "Note: from March 2025 onwards (but ideally as late as possible), it's okay
    to raise BufferError here".
  - `max_version >= our_own_dlpack_version` → return our max version.
  - else same major → "we should still be fine here - return our own max version".
  - else (consumer major older than producer's) → producer may fall back to the
    legacy `DLManagedTensor` if it still has that implementation; **otherwise it
    should raise `BufferError`** "to tell users that the consumer's max_version
    is too old to allow the data exchange to happen".
- Consumer's recommended algorithm (array API std): `try:
  x.__dlpack__(max_version=(1, 0), ...)`; `except TypeError: x.__dlpack__(...)`.
  So **legacy support is detected via `TypeError`** (unknown kwarg), not via a
  protocol flag. NumPy implements exactly this fallback and only when
  `device is None and copy is None` (`numpy/.../dlpack.c:608-621`); NumPy's own
  `max_version` is `(1, 0)` (`numpy/.../npy_static_data.c:212`) and it emits
  `version.major = 1, version.minor = 0` in the struct it produces
  (`numpy/.../dlpack.c:412-413`).
- Error type for an unsupported version is **`BufferError`** (recommended in the
  producer pseudo-code); the standard does not mandate a specific exception for
  version mismatch beyond this.
- Legacy/unversioned capsule seen by a modern consumer: the consumer looks up the
  capsule name; if it is `"dltensor"` (not `_versioned`) it must interpret it as
  a legacy `DLManagedTensor` (no `version`, no `flags`; no read-only signal).
  NumPy branches on `PyCapsule_IsValid(capsule, "dltensor_versioned")`
  (`numpy/.../dlpack.c:630-651`) and in the legacy branch treats the array as
  writable (no read-only flag exists).
- DLPack 1.0 note: the READ_ONLY flag is v1.0+ only, so a consumer asking for
  major 0 cannot be told about read-only data. NumPy refuses to export a
  read-only array to a pre-1.0 consumer: "Cannot export readonly array since
  signalling readonly is unsupported by DLPack (supported by newer DLPack
  version)." — `BufferError` (`numpy/.../dlpack.c:538-545`).

---

## 6. Data layout rules

- **Strides are in elements, not bytes** (`dlpack.h:263-267`); NumPy converts
  byte strides by dividing by itemsize on export and multiplying on import
  (`numpy/.../dlpack.c:366-367`, `:755-757`).
- `strides == NULL`:
  - **v1.3 rule (MUST):** "can not be NULL if ndim != 0, must points to an array
    of ndim elements that specifies the strides, so consumer can always rely on
    strides[dim] being valid for 0 <= dim < ndim." (`dlpack.h:263-267`)
  - "When ndim == 0, strides can be set to NULL." (`dlpack.h:269`)
  - v1.2 changed this: "Before DLPack v1.2, strides can be NULL to indicate
    contiguous data. This is not allowed in DLPack v1.2 and later. The rationale
    is to simplify the consumer handling." (`dlpack.h:271-274`)
  - **Inconsistency:** `python_spec.rst:161-163` still says "When the `strides`
    field ... is `NULL`, it indicates a row-major compact array" — stale prose
    from pre-1.2, contradicted by the header. De-facto, consumers remain
    tolerant: NumPy accepts `strides == NULL` and builds a C-contiguous array
    (`numpy/.../dlpack.c:736-755`), and the repo's own example app does the same
    (`apps/numpy_dlpack/dlpack/to_numpy.py:17-29`).
- **`byte_offset` is in bytes** (`dlpack.h:276-277`) and is `uint64_t`, so it
  **cannot be negative**. The element `(0,)*ndim` lives at
  `(char*)data + byte_offset`. De-facto most producers (NumPy, PyTorch et al.
  per the header note) always set it to 0 and put the first element directly in
  `data`, even for non-256-aligned data. A consumer must therefore compute
  `data + byte_offset` and must not assume `byte_offset == 0`.
- **Zero-dimensional tensors**: `ndim == 0`, `shape` may be `NULL`, `strides`
  may be `NULL`; the size formula yields 1 element, so `data` must be a valid
  pointer (the "size zero → data NULL" rule does not apply).
- **Zero-size dimensions**: any `shape[i] == 0` makes `prod(shape) == 0`; the
  header says set `data` to `NULL` (`dlpack.h:247-248`), the Python spec allows
  `NULL` or `0` (`python_spec.rst:161-163`). Shape/strides arrays are still
  required to be well-formed for `ndim != 0`.
- **Stride signs**: the header says nothing about negative strides, and the type
  is signed `int64_t`, so they are representable. De-facto they occur (NumPy
  exports views with negative strides unchanged; it does not normalize them, and
  its importer does not validate them: `numpy/.../dlpack.c:736-741`). A consumer
  must do pointer arithmetic that may step backwards from `data + byte_offset`.
- **0-stride (broadcast) dimensions**: not mentioned anywhere in v1.3. Signed
  `int64_t` permits 0; NumPy neither rejects nor special-cases it. Treat as
  de-facto legal (uncertainty, §10).
- **Alignment**: see §1 — "always aligned to 256 bytes as in CUDA" with the
  explicit advisory that in practice it often is not and consumers should not
  rely on it.
- **Lanes**: a `DLDataType` with `lanes > 1` describes a vector element of
  `bits * lanes` bits; consumers that cannot represent vector types must reject
  (NumPy: "Unsupported lanes in DLTensor dtype", `BufferError`,
  `numpy/.../dlpack.c:682-687`).
- **Sub-byte dtypes**: packed little-bit-endian by default (`dlpack.h:199-201`);
  "Setting bits != 6/4 is currently unspecified", producer must set bits
  correctly, "the consumer must stop importing if the value is unexpected"
  (`dlpack.h:171-180`); a producer may instead set
  `DLPACK_FLAG_BITMASK_IS_SUBBYTE_TYPE_PADDED` to declare padding.

---

## 7. Device rules

- A tensor is on exactly one device, described by `DLDevice` (`dlpack.h:125-136`;
  design note "everything assumes to be row major ... on a single device",
  array API std "Non-supported use cases").
- `device_id` is the producer's own numbering and need not match the consumer's:
  "In practice this will likely be the same numbering as that of the consumer,
  however that is not guaranteed." (`python_spec.rst:168-175`)
- CPU: `device_type = kDLCPU` (1), `device_id = 0` for vanilla/pinned/managed
  memory (`dlpack.h:131-135`).
- Cross-device exchange: the header is silent; the Python spec's rule is a
  recommendation, not a hard MUST: "If an array that is accessed via the
  interchange protocol lives on a device that the requesting (consumer) library
  does not support, it is recommended to raise a `BufferError`, unless an
  explicit copy is requested (see below) and the producer can support the
  request." (`python_spec.rst:40-43`). The array API std hardens this for
  `from_dlpack`: it "must raise `BufferError`" when the device is unsupported
  (from_dlpack Raises section), and the consumer may instead request
  `dl_device=(kDLCPU, 0)`; only `kDLCPU` support is mandated by v2023.12.
- De-facto NumPy as consumer only accepts CPU-like devices — `kDLCPU`,
  `kDLCUDAHost`, `kDLROCMHost`, `kDLCUDAManaged` — everything else is
  `BufferError: Unsupported device in DLTensor.` (`numpy/.../dlpack.c:671-678`).

---

## 8. Streams, threads, and `copy=True`

- Streams are a Python-level `stream` kwarg for CUDA/ROCm: "The consumer must
  pass the stream it will use to the producer; the producer must synchronize or
  wait on the stream when necessary." (`python_spec.rst:45-50`). Values: CUDA
  `None|1|2|>2` (0 disallowed), ROCm `None|0|>2`; `-1` means "producer must not
  perform any synchronization"; CPU must accept only `None` (array API std).
  NumPy only supports `stream=None` (`numpy/.../dlpack.c:519-523`).
- v1.3's C exchange API replaces stream passing with
  `current_work_stream(device_type, device_id, out)`; the export/import hooks
  "do not perform any stream synchronization. The consumer should query
  DLPackCurrentWorkStream to get the current work stream and launch kernels on
  it." (`dlpack.h:423-424`, `:454-455`, `:488-493`). For `kDLCPU` the consumer
  need not query, and "a CPU only framework can just provide a dummy
  implementation that always set out_current_stream[0] to NULL"
  (`dlpack.h:478-482`).
- Thread-safety: nothing in the header or prose spec states which threads may
  call the deleter or whether `DLTensor` may be read concurrently. The only
  threading rule is the GIL requirement for Python-side deleters
  (`python_spec.rst:129-133`). Treat "deleter may be invoked from any thread /
  late after Python finalization" as a de-facto producer obligation (NumPy's
  example handles both, `python_spec.rst:139-158`).
- `copy` semantics (from_dlpack / `__dlpack__`):
  - `copy=True` → "the function must always copy (performed by the producer)";
    **"When a copy happens, the `DLPACK_FLAG_BITMASK_IS_COPIED` flag must be
    set."** (`python_spec.rst:52-56`; array API std `__dlpack__` parameters).
  - `copy=False` → "must never copy, and raise a `BufferError` in case a copy is
    deemed necessary" (`__dlpack__`; `from_dlpack` also says `BufferError` for
    cross-device, while the `device` paragraph of `from_dlpack` oddly lists
    `ValueError` for the same condition — a spec inconsistency in 2025.12).
  - `copy=None` → "reuse the existing memory buffer if possible and copy
    otherwise".
  - The producer must set IS_COPIED when it copies; a consumer may require a copy
    when the requested device differs, `copy=True`, or the exported layout/dtype
    cannot be consumed as-is.

---

## 9. `dlpack-ffi` crate surface (v1.3.0, `dltensor`-free by design)

Files: `/home/a/rstsr_pack/dlpack-ffi/src/lib.rs` (generated by
`scripts/bindgen.py`, bindgen 0.73.2), `tests/abi.rs`, `tests/ownership.rs`,
`readme.md`, `header/dlpack.h` (vendored, identical to upstream v1.3 tag).

- **Names**: exactly the C names, `#[repr(C)]`, `pub` fields —
  `DLPackVersion`, `DLDevice`, `DLDataType`, `DLTensor`, `DLManagedTensor`,
  `DLManagedTensorVersioned`, `DLPackExchangeAPIHeader`, `DLPackExchangeAPI`
  (`src/lib.rs:16,62,113,124,143,154,206,215`).
- **Enums are bindgen "newtype" style** (`--default-enum-style newtype`,
  `scripts/bindgen.py`): `pub struct DLDeviceType(pub c_uint);` with associated
  constants `DLDeviceType::kDLCPU` etc. (`src/lib.rs:22-58`);
  `pub struct DLDataTypeCode(pub c_uint);` with `DLDataTypeCode::kDLInt` ...
  (`src/lib.rs:68-109`). `#[repr(transparent)]`, `Copy + Clone + Debug + Hash +
  Eq`. Values from newer minors are representable without UB; convert with `.0`.
- **Flags are module constants typed `u32`**: `DLPACK_FLAG_BITMASK_READ_ONLY: u32
  = 1`, `..._IS_COPIED: u32 = 2`, `..._IS_SUBBYTE_TYPE_PADDED: u32 = 4`
  (`src/lib.rs:10-12`). Note the type mismatch with the C header, where the
  macros are `1UL << n` (8 bytes on LP64) and `flags` is `uint64_t`: in Rust a
  consumer must cast `as u64` to OR them into `flags` (the crate's own test does
  this at `tests/ownership.rs:76,91-92`). A `u32`-typed flag set also cannot
  express future bits ≥ 32 without a crate update.
- **Deleters are nullable function pointers**:
  `pub deleter: Option<unsafe extern "C" fn(self_: *mut DLManagedTensor)>`
  (`src/lib.rs:149`) and the versioned analogue (`src/lib.rs:160`). Same style
  for the five `DLPack*` callback type aliases (`src/lib.rs:167-202`).
- `DLManagedTensorVersioned` **exists** in the crate (`src/lib.rs:154`), with the
  field order `version, manager_ctx, deleter, flags, dl_tensor`.
- `DLPackExchangeAPI` / `DLPackExchangeAPIHeader` also exist (`src/lib.rs:206-228`).
- **No helpers, no functions, no trait impls** — only types, type aliases and
  consts (39 `pub const` lines; no `pub fn`). Producing/consuming tensors,
  capsule handling, and version checks are entirely the consumer crate's job.
  `readme.md:5` states the policy: "We do not add any safe-wrappers or
  abstractions on top of the FFI layer."
- **Tests pin**:
  - `tests/abi.rs`: sizes on 64-bit (`DLPackVersion` 8, `DLDevice` 8,
    `DLDataType` 4, `DLTensor` 48, `DLManagedTensor` 64,
    `DLManagedTensorVersioned` 80), newtype enums 4 bytes and align 4
    (bindgen-generated layout tests are disabled via `--no-layout-tests`), a few
    constant values incl. `kDLTrn == 18`, `kDLFloat4_e2m1fn == 17`, and that
    unknown values are representable.
  - `tests/ownership.rs`: full round-trips for both structs — build a
    `DLManagedTensor`/`DLManagedTensorVersioned` over a Rust `Vec` stored in
    `manager_ctx`, read through the pointer, call the deleter exactly once.
  - **Not pinned**: field offsets (only total sizes), `align_of` for
    `DLDataType` (2) and the structs, the flag constant values, and the
    "deleter frees `self`" rule — the test deleters deliberately free only
    `manager_ctx` because the managed structs are stack locals
    (`tests/ownership.rs:15-26`), so the crate's own round-trip does **not**
    demonstrate a spec-conforming deleter.
- Consumer crate must supply: allocation/lifetime management, capsule
  name/rename logic, version comparison, dtype/device mapping tables, layout
  validation, and any safe wrapper.

---

## 10. Classification

### (a) MUST-level rules (normative, explicit)

1. Major-version mismatch → consumer calls the deleter and reads no other field
   (`dlpack.h:51-55`).
2. Minor-version mismatch → usable only if the consumer understands all fields
   (`dlpack.h:57-59`).
3. `strides` must not be `NULL` when `ndim != 0` (v1.2+); may be `NULL` when
   `ndim == 0` (`dlpack.h:263-274`).
4. Strides are in **elements**; `byte_offset` is in **bytes** (`dlpack.h:263-277`).
5. `dtype.bits` must be the true bit width for fp6/fp4; a consumer must stop
   importing on an unexpected value (`dlpack.h:171-180`).
6. Sub-byte packed data is little-bit-endian (`dlpack.h:199-201`).
7. Non-native-endian data must be rejected with an explicit error on export
   (`dlpack.h:185-187`).
8. `dtype.code` holds a `DLDataTypeCode` value (stored in a `u8`) (`dlpack.h:203-208`).
9. Data pointer must be 256-byte aligned (`dlpack.h:224`) — but see the
   advisory at `:227-231`; the alignment obligation is normative-on-producers,
   non-relied-upon by consumers.
10. Zero-size tensor → `data = NULL` (`dlpack.h:247-248`).
11. The deleter deletes `self` as well as `manager_ctx` (`dlpack.h:301-306`,
    `:353-359`).
12. Producer sets the PyCapsule name `"dltensor"`/`"dltensor_versioned"`; capsule
    names must be statically allocated; producer's destructor calls the deleter
    only for the unconsumed name (`python_spec.rst:92-95,126-127`).
13. Consumer renames to `"used_dltensor"`/`"used_dltensor_versioned"` to take
    ownership (`python_spec.rst:97-106`).
14. `copy=True` ⇒ producer copies and **must** set `IS_COPIED`
    (`python_spec.rst:52-56`).
15. Python-side deleters must acquire the GIL if they touch Python objects
    (`python_spec.rst:129-133`).
16. `from_dlpack` propagates the producer's `BufferError` and raises
    `AttributeError` if the methods are missing (array API std, from_dlpack).

### (b) De-facto conventions not spelled out in the v1.3 corpus

1. `max_version` is a Python 2-tuple; a list is not specified (array API std
   types it `tuple[int,int]`; NumPy rejects non-tuples). Older numpy/pytorch may
   accept lists — do not rely on it.
2. Legacy producers are detected by catching `TypeError` from
   `__dlpack__(max_version=...)` (array API std pseudo-code; NumPy
   `numpy/.../dlpack.c:608-621`).
3. NumPy as consumer *still* accepts `strides == NULL` on import (row-major),
   despite the v1.2 prohibition — tolerant-consumer behavior is widespread.
4. NumPy and PyTorch always export `byte_offset = 0` and point `data` at the
   first element, ignoring the 256-byte rule (header note; NumPy comment).
5. Negative and 0 strides are neither rejected nor documented; NumPy passes them
   through unvalidated.
6. NumPy rejects `lanes != 1`, fp8/fp6/fp4/bfloat16 (unless registered), and
   non-CPU-like devices with `BufferError`; producers should expect such
   rejections from CPU-only consumers.
7. `version.minor` emitted by real producers can lag the header: NumPy v2.5.2
   emits `(1, 0)` and asks for `max_version=(1, 0)`.
8. Error type for version negotiation failure: `BufferError` (recommended, not
   mandated); `TypeError` is only for unsupported kwargs/arguments.
9. The deleter may run on a foreign thread and even after Python finalization;
   NumPy's deleter guards with `Py_IsInitialized()` and leaks rather than crash.
10. Read-only: consumers that cannot honor it should ignore the flag rather than
    raise.

### (c) Uncertainties / gaps

1. **Upstream drift**: HEAD (94485e2) adds `kDLTPU=19`, `kDLTPUHost=20`,
   `kDLAscend=21`, `kDLBcomplex=18U` — not in v1.3 and not in `dlpack-ffi`.
   Decide whether rstsr pins to the tag or tracks main.
2. No definition of what "padded" means for
   `DLPACK_FLAG_BITMASK_IS_SUBBYTE_TYPE_PADDED` (padding unit/stride unspecified).
3. Negative strides: legal by type but undocumented; the base address of the
   data allocation is then *after* the first logical element, which interacts
   badly with a naive "own the data pointer" consumer. Producers without an
   owner object cannot express a negative-stride view safely.
4. 0-stride broadcast dims: undocumented; whether consumers must accept them is
   unclear.
5. Whether `data` must be the allocation base (256-aligned) with `byte_offset`
   as displacement, or the first element (byte_offset 0), is contradictory in
   practice; consumers should only ever use `data + byte_offset`.
6. `DLDataType.alignof == 2` (not 4) — relevant if a consumer re-declares the
   struct by hand instead of using this crate.
7. Legacy capsule + modern consumer: no `flags`, so read-only cannot be
   expressed; NumPy treats it as writable.
8. The published DLPack python_spec page still renders as
   "DLPack 0.6.0 documentation" (stale Sphinx version label), though its content
   matches v1.3.
9. `from_dlpack`'s `device` paragraph says `ValueError` where the `copy`
   paragraph says `BufferError` for the same "copy needed but copy=False"
   condition (array API std 2025.12) — pick `BufferError`, which matches
   `__dlpack__`.
10. No statement anywhere about concurrent access to a live `DLTensor` or about
    immutability of the metadata after publication.

---

## 11. Conformance checklists

### What a conforming DLPack **producer** must do

1. Allocate a `DLManagedTensorVersioned` (DLPack 1.x) and fill
   `version = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION}` (or the producer's
   own `(1, m)`), `flags = 0` unless a flag applies.
2. Provide a `deleter` that frees `manager_ctx` **and** the managed struct
   itself; it must be safe to call from a foreign thread and must null-check
   nothing (it is the producer's own function).
3. Fill `dl_tensor`: `device` (CPU ⇒ `{kDLCPU, 0}`), `ndim`, `dtype`
   (`code`/`bits`/`lanes` per the tables in §2), `shape` (`int64_t[ndim]`,
   `NULL` allowed only for `ndim == 0`).
4. Always fill `strides` with `int64_t[ndim]` **in elements** when `ndim != 0`;
   do not use `NULL` to mean contiguous.
5. Set `data` and `byte_offset` such that `(char*)data + byte_offset` is the
   first logical element; byte_offset is unsigned bytes. Prefer exporting
   `byte_offset = 0` and pointing `data` at element 0 (matches NumPy/PyTorch).
   If claiming 256-byte alignment, `data` must really be 256-byte aligned.
6. Reject non-native-endian data with an explicit error before exporting.
7. Set `data = NULL` when the tensor has a zero-size dimension.
8. Keep the memory alive for as long as the consumer may hold the tensor; the
   producer must not mutate/free it except through the deleter.
9. Expose data read-only by setting `DLPACK_FLAG_BITMASK_READ_ONLY` rather than
   relying on the consumer; set `IS_COPIED` if the exported buffer is a copy.
10. For sub-byte dtypes set `bits` exactly (4/6) and pack little-bit-endian, or
    set `IS_SUBBYTE_TYPE_PADDED` if padded.
11. Python: create the capsule with name `"dltensor_versioned"` (or `"dltensor"`
    for a legacy export) and a destructor that calls the deleter **only** when
    the capsule still carries that name; use a statically allocated name.
12. Python: accept the `max_version` kwarg when targeting modern consumers; on
    `max_version is None` export the legacy struct; if the requested major is
    older than what you can emit, raise `BufferError` (or fall back to the
    legacy struct if still implemented).
13. Python: honor `copy=True` by copying and setting `IS_COPIED`; raise
    `BufferError` when `copy=False` cannot be honored; honor `dl_device` for at
    least `(kDLCPU, 0)`.
14. Python: acquire the GIL in the deleter, and degrade gracefully if the
    interpreter is finalizing.
15. Do not set fields of a struct you did not allocate; the consumer owns the
    struct after the capsule rename.

### What a conforming DLPack **consumer** must do

1. Check the capsule name first: `"dltensor_versioned"` → read
   `DLManagedTensorVersioned`; `"dltensor"` → legacy `DLManagedTensor`.
   Anything else is an error.
2. On a versioned struct with `version.major != 1` (i.e. != the major you
   understand): call `deleter` if non-NULL and read no other field.
3. On `version.minor` newer than yours: proceed only if you understand every
   field you use (minor bumps add enum values only).
4. Rename the capsule to `"used_dltensor_versioned"`/`"used_dltensor"` **before**
   reading the payload, thereby taking ownership exactly once.
5. Never call `deleter` more than once; call it from your array object's
   destructor (or immediately on error after you have renamed).
6. Never assume `deleter` is non-NULL.
7. Read the element base address as `(char*)data + byte_offset`; never assume
   `byte_offset == 0`, and never rely on `data` being 256-byte aligned.
8. Validate `ndim` (e.g. against your max dims), `dtype.code`/`bits`/`lanes`
   (reject unknown codes/bits/lane counts rather than guessing), and the device;
   raise `BufferError` for anything you cannot represent.
9. Interpret `strides` in elements; handle negative and zero strides if you
   accept arbitrary views; handle `strides == NULL` on `ndim != 0` only as
   legacy tolerance (treat as row-major contiguous), never emit it.
10. Handle `ndim == 0` (scalar) with possibly `NULL` shape/strides, and any
    zero-size dimension with `data == NULL`.
11. Respect `READ_ONLY`: if you cannot represent read-only, either ignore it
    (documented preference) or copy; never hand out a writable alias of a
    read-only tensor without documenting it.
12. When you copy, you are the copy's owner; the producer's deleter still owns
    the original.
13. Python: pass `max_version=(1, 0)` (or higher) first; on `TypeError` retry
    without the newer kwargs to support legacy producers; verify the returned
    struct's version regardless.
14. Python: honor `copy`/`dl_device`; raise `BufferError` (not `ValueError`)
    when a needed copy is disallowed; propagate producer `BufferError`s;
    `AttributeError` if `__dlpack__` is absent.
15. Do not free `shape`/`strides`/`manager_ctx` yourself unless you are the
    producer that allocated them; the deleter is the only release path.

# NumPy 2.x DLPack interop: primary-source notes

Provenance

- Source checkout: `/home/a/Git-Others/numpy`, `git describe --tags` = **v2.5.2**,
  HEAD = `48fecee5453aa1d31e6b79dcb3969dc1a6d1a891` (`git status` clean — no local
  modifications). All `file:line` citations below are relative to that checkout root.
- Installed interpreter: conda env `torch`, Python 3.13.14, **numpy 2.5.1**
  (`/home/a/miniconda3/envs/torch/lib/python3.13/site-packages/numpy/__init__.py`),
  torch 2.14.0+rocm7.2 (used only for the cross-library sanity probe).
  The installed 2.5.1 differs from the 2.5.2 checkout; no DLPack-relevant
  difference was observed, but all source citations are for v2.5.2.
- Empirical probes: `experiments/probe_numpy_dlpack.py` (ctypes-built DLPack
  capsules + Python-level probes), raw output in `experiments/probe-output.txt`.
  Run with `conda run -n torch python probe_numpy_dlpack.py`.
- Vendored DLPack header: `numpy/_core/src/common/dlpack/dlpack.h`. **There is no
  `DLPACK_VERSION` macro in this checkout** (repo-wide `grep "define DLPACK_VERSION"`
  returns nothing). The header defines `DLPACK_MAJOR_VERSION 1` (line 22) and
  `DLPACK_MINOR_VERSION 0` (line 25), i.e. DLPack v1.0 ABI. Provenance comment at
  lines 1-3: taken from dmlc/dlpack commit `bbd2f4d32427e548797929af08cfe2a9cbb3cf12`
  "(but added typedef to DLManagedTensorVersioned)".

---

## 1. `np.from_dlpack` — accepted inputs, signature, dispatch

- Signature: `from_dlpack(x, /, *, device=None, copy=None)`.
  Parsed at `numpy/_core/src/multiarray/dlpack.c:577-582`; user-facing docstring at
  `numpy/_core/_add_newdocs.py:1798-1850`.
- C function name: `from_dlpack` in `numpy/_core/src/multiarray/dlpack.c:571-807`,
  registered in the `_multiarray_umath` module table at
  `numpy/_core/src/multiarray/multiarraymodule.c:4829-4830`, re-exported by
  `numpy/_core/multiarray.py:27,35` and `numpy/__init__.py:255`.
- **It accepts any object with `__dlpack__`** — it does not require
  `__dlpack_device__` (the C code never calls it; `dlpack.c:604-621`). Docstring says
  both are needed (`_add_newdocs.py:1812-1813`), but empirically an object with only
  `__dlpack__` imports fine (`probe-output.txt` "object without __dlpack_device__").
- **A bare `PyCapsule` is NOT accepted**: `np.from_dlpack(arr.__dlpack__())` raises
  `AttributeError: 'PyCapsule' object has no attribute '__dlpack__'` (empirical).
- Exact call numpy makes (`dlpack.c:589-606`): with `kwnames = (dl_device, copy,
  max_version)` (`npy_static_data.c:199-202`) and `call_args = {obj, <device|None>,
  copy, dl_max_version}`:
  - plain call → `obj.__dlpack__(dl_device=None, copy=None, max_version=(1, 0))`
  - `device="cpu"` → `obj.__dlpack__(dl_device=(1, 0), copy=..., max_version=(1, 0))`
    (`dlpack.c:592-601`; `dl_cpu_device_tuple` = `(1, 0)` at `npy_static_data.c:207`)
  - `dl_max_version` = `(1, 0)` at `npy_static_data.c:212`.
  - Empirical confirmation: `probe-output.txt:138-139`.
- Retry/fallback (`dlpack.c:607-625`): if the first call raises **TypeError** and both
  `device` and `copy` are `None`, numpy clears the error and calls `obj.__dlpack__()`
  with **no arguments at all**. Empirical: a strict-kwargs exporter records
  `[(None, {'dl_device': None, 'copy': None, 'max_version': (1, 0)}), (None, {})]`
  (`probe-output.txt:140`). The retry also fires for an *internal* TypeError; a
  probe exporter that always raises TypeError was called **twice** before the error
  surfaced (`probe-output.txt:194-196`).
- `device=` accepts only the string `"cpu"` (and `None`); anything else raises
  `ValueError: Device not understood. Only "cpu" is allowed, but received: <obj>`
  (`conversion_utils.c:1403-1419`; empirical `probe-output.txt:145-147`).
- `copy=`: accepted via `PyArray_CopyConverter` semantics (None/True/False/bool-like;
  strings rejected with `ValueError: strings are not allowed for 'copy' keyword. Use
  True/False/None instead.`). It is **forwarded to the exporter**; numpy does no copy
  of its own in `from_dlpack`. Empirical: `copy=True` → no shared memory; `copy=None`
  and `copy=False` → shared memory (`probe-output.txt:149-154`).

## 2. `ndarray.__dlpack__` — signature and semantics

- Registered as a method: `numpy/_core/src/multiarray/methods.c:3067-3073`
  (`METH_FASTCALL | METH_KEYWORDS`), implementation `array_dlpack`
  (`dlpack.c:484-559`), docstring `_add_newdocs.py:3195-3205`.
  `ndarray.__dlpack_device__` is `METH_NOARGS` (`methods.c:3071-3073`), impl
  `dlpack.c:561-569`, docstring `_add_newdocs.py:3207-3217`.
- Full accepted signature: `a.__dlpack__(*, stream=None, max_version=None,
  dl_device=None, copy=None)` (`dlpack.c:498-505`). Defaults:
  `stream=None`, `max_version=None`, `copy=NPY_COPY_IF_NEEDED`
  (`dlpack.c:488-490`).
- `stream`: only `None` is accepted; anything else raises
  `ValueError: NumPy only supports stream=None.` (`dlpack.c:519-523`; empirical
  with `stream=1` and `stream=(1,2)`).
- `max_version`: `None` or a 2-tuple (`dlpack.c:507-517`). Non-tuple/wrong length →
  `TypeError: max_version must be None or a tuple with two elements.`; element 0 is
  converted with `PyLong_AsLong`, so `(1.0, 0)` → `TypeError: 'float' object cannot
  be interpreted as an integer`, `(None, 0)` → `TypeError: 'NoneType' object cannot be
  interpreted as an integer`. Only **element 0 (major) is read**; minor is ignored.
  `(True, False)` counts as `(1, 0)` (bools are ints) → versioned capsule.
- **Which capsule flavor**: versioned (`DLManagedTensorVersioned`, capsule name
  `dltensor_versioned`) iff major ≥ 1, else legacy (`DLManagedTensor`, name
  `dltensor`) — `dlpack.c:553-555` (`create_dlpack_capsule(..., major_version >= 1,
  ...)`), names `npy_dlpack.h:8-11`. Consequently:
  - `max_version` **omitted (None)** → legacy/unversioned capsule
    (empirical `probe-output.txt:20`). The default is *not* versioned.
  - `(0, 0)`, `(0, 8)`, `(-1, 0)` → unversioned; `(1, 0)`, `(2, 0)`, `(100, 3)` →
    **versioned with `version = (1, 0)`** (numpy clamps to its own max; it never
    errors on a too-high request) — `dlpack.c:411-413`; empirical lines 23-25.
  - Legacy capsule + non-writeable array → `BufferError: Cannot export readonly
    array since signalling readonly is unsupported by DLPack (supported by newer
    DLPack version).` (`dlpack.c:537-543`). Hence plain `a.__dlpack__()` on a
    read-only array fails, but `np.from_dlpack(a)` works because numpy requests
    `max_version=(1,0)` (numpy's own `test_dlpack.py:135-144`).
- `dl_device` (Array API extension): `None`, or a 2-tuple of ints. Non-tuple →
  `TypeError: dl_device must be a tuple`; wrong arity → `TypeError: function takes
  exactly 2 arguments (1 given)` (from `PyArg_ParseTuple("ii")`). If it equals the
  array's current device → OK; if it is `(1, 0)` (kDLCPU) → the capsule's device is
  *relabelled* to CPU (no data movement); otherwise `BufferError: unsupported device
  requested` (`dlpack.c:454-481`; empirical `probe-output.txt:42-48`).
- `copy`: `True` → `PyArray_NewCopy(self, NPY_KEEPORDER)` (`dlpack.c:526-532`) and
  sets `DLPACK_FLAG_BITMASK_IS_COPIED`; `False`/`None` are no-ops for numpy as
  producer (it never *needs* a copy). Empirical: `copy=True` flags=2 (bit 1),
  `copy=False`/`None` flags=0 (`probe-output.txt:50-54`). `copy=np.array([1,2,3])` →
  `ValueError: The truth value of an array with more than one element is ambiguous`
  (converter quirk, noted in `test_dlpack.py:44-46`).
- Refcount/lifetime: the capsule holds a strong reference to the ndarray
  (`Py_INCREF(self)` at `dlpack.c:447-448`; `manager_ctx = self` at 407/428; the
  deleter decrefs at 108-111 / 126-128). Empirical: `sys.getrefcount(x)` is +1 while
  the capsule lives, back to baseline after `del capsule` (`probe-output.txt:119-121`).
- Device reported: always `(1, 0)` for ordinary CPU arrays; `array_get_dl_device`
  walks the `.base` chain (`dlpack.c:230-261`) and *re-reports whatever device the
  array was imported from* (see §5), so a CPU-relabelled or CUDAHost-imported array
  reports its original device tuple.

## 3. Version negotiation

- Producer max supported: DLPack **1.0** (`dlpack.c:411-413` sets `version.major=1,
  minor=0` unconditionally for versioned output; header `DLPACK_MAJOR_VERSION 1`,
  `dlpack.h:22`). Requests above it are clamped, never rejected.
- Producer min supported: legacy `DLManagedTensor` (v0.x) still produced on demand
  (major < 1). Source TODO says v0 support "should be deprecated in NumPy 2.1"
  (`dlpack.c:545-552`) but it is still present in 2.5.2.
- Consumer: accepts both capsule names — `dltensor_versioned` and `dltensor`
  (`dlpack.c:630-659`). Versioned: rejects only `version.major > 1` with
  `BufferError: from_dlpack(): the exported DLPack major version is too high to be
  imported by this version of NumPy.` (`dlpack.c:639-645`); **minor is ignored**
  (empirical: version (1, 99) imports; a capsule *named* `dltensor_versioned` with
  version (0, 9) also imports — the capsule name alone selects the struct layout,
  the version field is only checked for `major > 1`). Empirical: (2,0) and (3,5)
  rejected (`probe-output.txt:176-177`).
- Legacy import is permitted but always marks the array read-only (`readonly = 1` at
  `dlpack.c:658`), because the legacy struct has no flags field. Empirical:
  `writeable=False` and `y[0] = ...` → `ValueError: assignment destination is
  read-only` (`probe-output.txt:164-166`).
- Changelog: DLPack v1 support landed in **NumPy 2.1.0**
  (`doc/source/release/2.1.0-notes.rst:152`, gh-26501; commit
  `fb60521895 "ENH: Support dlpack version 1 ABI"`, first tag containing it: v2.1.0).
  Read-only export / `IS_COPIED` / `max_version` are part of that same commit
  (`git show fb60521895`). `kDLBool` export/import came in 1.25.0 (commit
  `89f61ff188`). `np.from_dlpack` now raises `BufferError` (not `RuntimeError`) for
  unsupported device/dtype/ndim — NumPy 2.5.0 (`doc/source/release/2.5.0-notes.rst:283-288`).
  User-dtype registry `np.dtypes.register_dlpack_dtype` added in 2.5.0
  (`doc/source/release/2.5.0-notes.rst:362-364`, gh-31256;
  `numpy/dtypes.py:26,29-72`).

## 4. Export side (ndarray → capsule)

Implementation: `fill_dl_tensor_information` (`dlpack.c:268-374`) +
`create_dlpack_capsule` (`dlpack.c:377-451`).

- **Strides are always exported** (never NULL for ndim > 0); `shape`/`strides` are one
  allocation inside the managed struct, aligned to 8 bytes (`dlpack.c:384-434`).
  For ndim == 0 both pointers are NULL and `ndim = 0` (`dlpack.c:433-434`; empirical
  `probe-output.txt:67`).
- **`byte_offset` is always 0** on export (set twice, `dlpack.c:360,371`); the comment
  at `dlpack.c:346-358` explains this is deliberate to match the common practice
  (pytorch) despite the header's 256-byte-alignment/`byte_offset` guidance.
- `data` is `PyArray_DATA(self)` — a raw view of the array; no copy unless `copy=True`.
- Dtype coverage (`dlpack.c:291-344`), emitted as `(code, bits=8*itemsize, lanes=1)`:
  - non-native byte order → `BufferError: DLPack only supports native byte order.`
    (checked before the dtype switch; header asks for exactly this, `dlpack.h:167`).
  - `bool` → `kDLBool` (code 6), bits 8.
  - signed/unsigned ints (any itemsize numpy has) → `kDLInt`(0)/`kDLUInt`(1).
  - floats with itemsize ≤ 8 → `kDLFloat`(2) (float16/32/64). `float128/longdouble`
    → `BufferError: DLPack only supports IEEE floating point types without padding
    (longdouble typically is not IEEE).` (same message for `complex256`).
  - complex with itemsize ≤ 16 → `kDLComplex`(5) (complex64/128).
  - everything else (`datetime64`, `timedelta64`, `str_`/`U`, `bytes_`/`S`, `void`/`V`,
    `object`, structured) → registry lookup, else
    `BufferError: DLPack only supports signed/unsigned integers, float and complex
    dtypes (or dtypes registered by third-party packages).`
    (empirical per-dtype lines `probe-output.txt:88-103`).
  - **bfloat16 is not supported** (not a numpy dtype); `float8` dtypes neither.
- Strides rule (`dlpack.c:277-286`): if the array is not C-contiguous **and**
  `size != 1`, every dimension with `shape[i] != 1` must have
  `strides[i] % itemsize == 0`, else
  `BufferError: DLPack only supports strides which are a multiple of itemsize.`
  Size-1 dims are exempt, and their byte strides are integer-divided by itemsize —
  so a size-1 dim with a non-multiple stride is exported with a **truncated** (often
  0) element stride (empirical: numpy `(10,5,1)` strides `(8,80,4)` →
  DLPack strides `[1, 10, 0]`, `probe-output.txt:63`). Strides are in *elements*
  (`strides[i] / itemsize`, `dlpack.c:367`).
- Non-contiguous, negative strides, 0-strides: all exported as-is
  (`a[::-1]` → `[-1]`; broadcast → `[0, 1]`; empirical lines 57-60). 0-d arrays,
  size-0 arrays, Fortran-order arrays export fine.
- Size-0: exported with `shape=[0, 3]`, `strides=[0, 0]` and a **non-NULL** `data`
  pointer (empirical line 68) — the vendored header suggests NULL data for size 0
  (`dlpack.h:222`), numpy does not follow that.
- Device: `(kDLCPU=1, 0)` by default; inherited from the base chain if the array was
  imported from a capsule (`dlpack.c:230-261`, see §5).
- Versioned-capsule flags (`dlpack.c:415-421`): `flags = 0`; bit 0
  (`DLPACK_FLAG_BITMASK_READ_ONLY`, `dlpack.h:276`) set iff the array is not
  `NPY_ARRAY_WRITEABLE`; bit 1 (`IS_COPIED`, `dlpack.h:284`) set iff `copy=True`.
  Empirical: readonly → 1; `copy=True` → 2; readonly + `copy=True` → 2 (copy is
  writeable, so no READ_ONLY bit) — `probe-output.txt:111-117`.
- Legacy capsules carry **no flags at all** (`DLManagedTensor` has none,
  `dlpack.h:257-274`), which is why read-only arrays cannot be exported as legacy.
- No alignment enforcement on export: an unaligned numpy array (e.g.
  `np.frombuffer(bytearray(64), dtype='f8', offset=1)`) exports successfully
  (empirical `probe-output.txt:245-248`).

## 5. Import side (capsule → ndarray)

Validation order in `from_dlpack` (`dlpack.c:630-805`), empirically confirmed:

1. Capsule name: `dltensor_versioned` is tried first, then `dltensor`
   (`dlpack.c:630-655`). Any other name (e.g. `used_dltensor`,
   `used_dltensor_versioned`, `torch`, `numpy_dltensor`) → `ValueError:
   PyCapsule_GetPointer called with incorrect name`. A non-capsule return value →
   `ValueError: PyCapsule_GetPointer called with invalid PyCapsule object`
   (empirical `probe-output.txt:134-135,250-256`).
2. `version.major > 1` → BufferError (see §3).
3. `ndim > NPY_MAXDIMS` → `BufferError: maxdims of DLPack tensor is higher than the
   supported maxdims.` (`dlpack.c:661-668`). `NPY_MAXDIMS == 64`
   (`numpy/_core/include/numpy/ndarraytypes.h:41`); empirically ndim 0/1/32/64 import,
   65 fails (`probe-output.txt:226-231`). (numpy's own test only exercises 0..32,
   `test_dlpack.py:110-115`.)
4. Device whitelist (`dlpack.c:670-679`): accepted are `kDLCPU=1`, `kDLCUDAHost=3`,
   `kDLROCMHost=11`, `kDLCUDAManaged=13` (host-accessible memory). Everything else
   (`kDLCUDA=2`, `kDLROCM=10`, `kDLOneAPI=14`, 0, …) →
   `BufferError: Unsupported device in DLTensor.` **`device_id` is not validated**:
   `(1,5)`, `(13,7)`, `(3,2)` all import and are reported back verbatim by
   `__dlpack_device__` (empirical `probe-output.txt:189-192`).
5. `dtype.lanes != 1` → `BufferError: Unsupported lanes in DLTensor dtype.`
   (`dlpack.c:681-686`; empirical lanes 2/4 fail).
6. Dtype mapping (`dlpack.c:688-746`), built-ins first, then user registry, else
   `BufferError: Unsupported dtype in DLTensor.` (`dlpack.c:74-78`):
   - `kDLBool`(6) bits 8 → `bool`; any other bits → unsupported.
   - `kDLInt`(0) 8/16/32/64 → int8/16/32/64; `kDLUInt`(1) same → uint8/16/32/64.
   - `kDLFloat`(2) 16/32/64 → float16/32/64; bits 8/128 → unsupported.
   - `kDLComplex`(5) 64/128 → complex64/128; bits 32 → unsupported.
   - `kDLBfloat`(4) bits 16 → **unsupported** (no bfloat16 dtype in numpy);
     `kDLOpaqueHandle`(3) and unknown codes → unsupported.
   - Full empirical matrix: `probe-output.txt:198-220`.
   - Escape hatch (2.5+): `np.dtypes.register_dlpack_dtype((code, bits), dtype)`
     registers a user dtype for both directions
     (`dlpack.c:810-886`, `numpy/dtypes.py:29-72`); built-in mappings win on import;
     conflicts raise `ValueError`. Empirically registering `(7,8)` for `S1` makes
     `S1` round-trip (`probe-output.txt:267-269`).
7. Shape/strides: `strides == NULL` means **C-contiguous row-major** (`dlpack.h:236`);
   numpy passes NULL through to `PyArray_NewFromDescr`, giving a C-contiguous view
   (`dlpack.c:753-765`). Otherwise DLPack element strides are multiplied by itemsize
   (`dlpack.c:756-758`). Negative strides are accepted (empirical `[2.,1.,0.]` for
   stride -1), zero strides are accepted (broadcast views; writes alias),
   `byte_offset` is honored: `data = dl_tensor.data + byte_offset`
   (`dlpack.c:761`; empirical offset 16 → elements 2..4).
8. **No alignment check on import**: a deliberately +1-byte misaligned `float64`
   pointer imports fine and yields `arr.flags.aligned == False`
   (empirical `probe-output.txt:245-248`). (The 256-byte rule in `dlpack.h:199-206`
   is advisory; the header itself notes most libraries ignore it.)
9. Writeability: versioned capsule → writeable iff `READ_ONLY` flag clear
   (`dlpack.c:648,763-765`); legacy capsule → always read-only (`dlpack.c:658`).
   Empirical: versioned writable import can be written through
   (`probe-output.txt:168-170`).
10. Ownership/deleter:
    - On **success** numpy creates a new internal capsule
      `numpy_dltensor[_versioned]` that owns the producer's managed pointer and
      becomes the array's `base` (`dlpack.c:772-794`; names `npy_dlpack.h:16-17`),
      then **renames the original capsule** to `used_dltensor[_versioned]`
      (`dlpack.c:796-803`) so its destructor skips the deleter
      (`dlpack.c:139-178`). The producer's deleter is therefore called exactly once,
      when the array (and its internal base capsule) dies
      (`array_dlpack_internal_capsule_deleter*`, `dlpack.c:187-224`).
      Empirical: after a successful import the consumed capsule is named
      `used_dltensor_versioned`, and `arr.base` is a `PyCapsule` named
      `numpy_dltensor_versioned` (`probe-output.txt:262-264`).
    - On **failure** numpy `Py_DECREF`s the original capsule on every error path
      without renaming it (e.g. `dlpack.c:635,643,655,666,677,684,735,743,768,785,791`),
      so the producer's capsule destructor runs and (per the DLPack convention in
      `dlpack.c:139-178`) calls the deleter. Empirical: after a failed import the
      capsule name is unchanged and numpy's reference is gone (`probe-output.txt:262`).
    - **Capsule reuse**: the second `np.from_dlpack` on the same capsule raises
      `ValueError: PyCapsule_GetPointer called with incorrect name` (both flavors;
      `probe-output.txt:156-162`).
11. 0-size / 0-d: both round-trip; imported 0-d arrays have `shape ()` and are
    ordinary ndarrays (`test_dlpack.py:157-160`; empirical ndim=0 line 227).

## 6. Docs / tests / changelog notes on the user-facing contract

- `np.from_dlpack` docstring: `_add_newdocs.py:1798-1850` (device must be `"cpu"`;
  `copy=False` "will raise `BufferError` in case a copy is deemed necessary").
- `ndarray.__dlpack__` / `__dlpack_device__` docstrings: `_add_newdocs.py:3195-3217`
  (deliberately minimal in 2.5.2 — no mention of `max_version`, versions, or flags).
- DLPack is *not* implicit conversion: "NumPy doesn't implicitly convert objects to
  ndarrays using DLPack" — `doc/source/user/basics.interoperability.rst:60-66`.
- Behavioral contract tests: `numpy/_core/tests/test_dlpack.py` — refcounts
  (20-26, 56-62), `stream` (28-34), `copy` arg parsing (36-46), stride multiple of
  itemsize (48-54), dtype passthrough incl. bool (64-77), unsupported datetime64
  (79-83), byte-swapped (85-90), non-contiguous (92-108), ndim 0..32 (110-115),
  `__dlpack_device__` (117-123), readonly export/import (135-144), writeability of
  old vs new (146-155), 0-d and size-1-dims (157-166), copy semantics (168-176),
  `dl_device`/`device` (178-188), dtype registry (191-262).
- Changelog touchpoints: `doc/source/release/2.1.0-notes.rst:152` (DLPack v1);
  `2.2.5-notes.rst:39` (writeable flag fix, #28632); `2.2.6-notes.rst:39`
  (from_dlpack thread-safety, #28889); `2.5.0-notes.rst:283-288` (BufferError) and
  `362-364` (dtype registry).

## 7. What NumPy does NOT support via DLPack (user-visible failure mode)

| Not supported | Failure mode |
|---|---|
| `datetime64`, `timedelta64` | `BufferError: DLPack only supports signed/unsigned integers, float and complex dtypes …` (both directions) |
| `U`/`S` strings, `V`/void, structured, `object` | same generic BufferError, unless registered via `np.dtypes.register_dlpack_dtype` |
| `float128`/`longdouble`, `complex256`/`clongdouble` | `BufferError: DLPack only supports IEEE floating point types without padding …` |
| byte-swapped dtypes | `BufferError: DLPack only supports native byte order.` |
| bfloat16 (import) | `BufferError: Unsupported dtype in DLTensor.` |
| float8/int128/complex32/any other (code,bits) | `BufferError: Unsupported dtype in DLTensor.` |
| lanes > 1 (vector dtypes) | `BufferError: Unsupported lanes in DLTensor dtype.` |
| ndim > 64 | `BufferError: maxdims of DLPack tensor is higher than the supported maxdims.` |
| GPU-resident transfer: `kDLCUDA`, `kDLROCM`, `kDLOneAPI`, `kDLMetal`, `kDLWebGPU`, … | `BufferError: Unsupported device in DLTensor.` (only CPU/host/pinned/managed accepted) |
| actual movement to/from a GPU device via `dl_device`/`device=` | numpy accepts only `"cpu"`; `__dlpack__(dl_device=(k>1, id))` → `BufferError: unsupported device requested`; `np.from_dlpack(..., device="gpu")` → `ValueError: Device not understood. Only "cpu" is allowed …` |
| readonly arrays over legacy (v0) capsules | `BufferError: Cannot export readonly array since signalling readonly is unsupported by DLPack …` |
| exporting a strided array whose element strides are not a multiple of itemsize | `BufferError: DLPack only supports strides which are a multiple of itemsize.` |
| data not in host-visible memory (managed/device pointers other than the host types) | not representable (no device support) |
| bare capsule input to `np.from_dlpack` | `AttributeError: 'PyCapsule' object has no attribute '__dlpack__'` |
| reuse of a consumed capsule | `ValueError: PyCapsule_GetPointer called with incorrect name` |

## 8. Surprising / decision-relevant

1. **`arr.__dlpack__()` with no arguments produces a legacy (v0) capsule**, not a
   versioned one; only `max_version` major ≥ 1 switches to
   `DLManagedTensorVersioned` (`dlpack.c:490-517,553-555`). Legacy capsules cannot
   signal read-only, so `ro_arr.__dlpack__()` raises even though
   `np.from_dlpack(ro_arr)` works. Third-party consumers should always pass
   `max_version=(1, 0)` unless they need backwards compatibility.
2. **numpy clamps, it never negotiates**: `max_version=(100, 3)` is accepted and yields
   `(1, 0)`; the producer never errors on a too-high request (the consumer does reject
   `major > 1` on import).
3. **`np.from_dlpack` may call `__dlpack__` twice** (silent TypeError retry,
   `dlpack.c:615-621`). Producers must be idempotent/side-effect free on repeated
   calls (and should be prepared for a first call with
   `dl_device`/`copy`/`max_version` kwargs).
4. **The consumer never calls `__dlpack_device__`** despite the docstring; it passes
   `dl_device=None` and relies on the exporter.
5. **`device_id` is ignored on import but preserved on re-export** (`dlpack.c:670-679`;
   `array_get_dl_device` 230-261). Non-zero ids are legal input and round-trip
   verbatim.
6. Legacy (v0) import **always yields a non-writeable array** (`dlpack.c:658`).
7. Versioned import ignores the minor version and fully trusts the ABI once
   major ≤ 1; the flag word is read from the same struct position.
8. kDLBool is the exported dtype for numpy `bool` (code 6, bits 8) — a consumer must
   know DLPack ≥ 0.8 (`dlpack.h:162`); torch and cupy do.
9. **`copy=False` is not enforced by numpy as consumer**: the docstring's promise
   ("will raise BufferError if a copy is needed") rests entirely on the exporter;
   numpy never copies in `from_dlpack` and its own `__dlpack__` ignores `copy=False`.
   (`_add_newdocs.py:1820-1824`; `dlpack.c:526-532`.)
10. Size-0 export has a non-NULL data pointer and strides `(0,0)`, contradicting the
    header's advice to use NULL (`dlpack.h:222`; empirical `probe-output.txt:68`).
11. numpy does **no alignment enforcement either way** (export comment
    `dlpack.c:346-358`; import has no check at all — misaligned pointers import with
    `flags.aligned == False`).
12. NumPy's `__dlpack__` accepts a `dl_device` kwarg whose only meaningful value
    besides the current device is `(1, 0)`; asking for CPU on a non-CPU-origin array
    *relabels* rather than copies (`dlpack.c:454-481`).
13. NumPy 2.5 grew a user-dtype registry, `np.dtypes.register_dlpack_dtype`, intended
    for `ml_dtypes`-style dtypes — the first extension point that makes future
    bfloat16/float8 (and even `S`/`U` strings) DLPack-interoperable with numpy
    (`numpy/dtypes.py:29-72`, `dlpack.c:21-87,810-886`). Built-in mappings take
    priority on import and conflicts raise `ValueError`.

## 9. Probe output (selected; full log in `experiments/probe-output.txt`)

Key excerpts (line numbers in `probe-output.txt`):

```
16  OK: bare capsule name (no max_version) -> 'dltensor'
23  OK: max_version=(1, 0) -> dltensor_versioned version=1.0 flags=0
24  OK: max_version=(2, 0) -> dltensor_versioned version=1.0 flags=0
63  OK: shape=(10,5,1) strides=(8,80,4) -> dl strides=[1, 10, 0]
68  OK: size-0 (0,3): ndim=2 shape=[0, 3] strides=[0, 0] data ptr null? False
112 ERR: BufferError: Cannot export readonly array since signalling readonly is unsupported by DLPack (supported by newer DLPack version).
138 OK: plain np.from_dlpack(obj) -> args=() kwargs={'dl_device': None, 'copy': None, 'max_version': (1, 0)}
140 OK: old-style exporter: attempts recorded by exporter = [(None, {'dl_device': None, 'copy': None, 'max_version': (1, 0)}), (None, {})]
176 ERR: BufferError: from_dlpack(): the exported DLPack major version is too high to be imported by this version of NumPy.
196 OK: exporter __dlpack__ called 2 time(s) before the error surfaced
262 OK: FAILED import: refcount delta=0, capsule name still dltensor_versioned ...
263 OK: SUCCESSFUL import: refcount delta=0, capsule renamed to used_dltensor_versioned -> numpy owns it
264 OK: array base type=PyCapsule, base capsule name=numpy_dltensor_versioned
274 OK: np.from_dlpack(torch.arange(6)) -> float32 [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
277 OK: np.from_dlpack(torch strided view) -> ok
```

The probes cover: round-trip and raw capsule, every `max_version` variant incl.
malformed, `stream`, `dl_device`, `copy` flags, strides/byte_offset/0-d/size-0/F-order
export, the full dtype matrix both directions, byteswapped, read-only (all four
paths), refcounts, `__dlpack_device__`, kwargs passed to the exporter, device/`copy`
kwargs on import, capsule reuse, versioned/legacy writeability, version and device
whitelists, dtype code/bits matrix, lanes, ndim 0..65, NULL/negative/zero strides,
byte_offset, misalignment, capsule names, validation order, ownership/rename,
the dtype registry, and torch ↔ numpy interchange.

## 10. Implications for a third-party producer/consumer

To interoperate with NumPy through DLPack, a non-numpy library must:

As a **producer** (relevant for rstsr exporting to numpy):
1. Implement `__dlpack__(*, stream=None, max_version=None, dl_device=None, copy=None)`
   — numpy calls it with `dl_device`, `copy`, `max_version` kwargs and falls back to a
   **bare no-arg call** only if the first attempt raises `TypeError` *and* the user's
   device/copy were both None. Handle `dl_device=None`; accept
   `dl_device=(1, 0)`/raise otherwise. Accept `max_version` whose major may be
   larger than yours; return your highest supported version. If you only produce the
   legacy `DLManagedTensor`, hold it read-only semantics for the consumer.
2. Emit capsule name `dltensor_versioned` for `DLManagedTensorVersioned` and
   `dltensor` for legacy; implement the deleter and keep the memory alive until it is
   called. numpy renames the capsule on success and calls the deleter once, via the
   array's base; on failure it releases the capsule un-renamed.
3. Produce device `(1, 0)` for CPU; do not emit GPU device types unless the consumer
   can handle them (numpy cannot). Set `byte_offset` only if non-zero; strides are in
   elements and may be negative/zero; NULL strides means C-contiguous.
4. Only emit dtypes numpy knows (bool8, int/uint 8/16/32/64, float 16/32/64,
   complex 64/128) unless the user enrolled a dtype via
   `np.dtypes.register_dlpack_dtype`. Use native byte order and lanes == 1.
   Strides must be multiples of itemsize except for size-1 dims (else numpy's
   `BufferError`).
5. Set `DLPACK_FLAG_BITMASK_READ_ONLY` (versioned only) if you keep ownership or the
   buffer is read-only; numpy will then return a non-writeable array. Legacy capsules
   cannot express this, so numpy protects itself by importing legacy as read-only.
6. Be prepared to be called twice (TypeError retry), and be idempotent.

As a **consumer** (relevant for rstsr importing from numpy):
7. Call `arr.__dlpack__(max_version=(1, 0))` (or at least be ready for both struct
   layouts) — numpy's *default* is the legacy struct, which loses the READ_ONLY and
   IS_COPIED flags. Handle both capsule names, reject `used_dltensor*` reuse, and
   rename the capsule you consumed to `used_dltensor[_versioned]` after taking
   ownership so a second import fails loudly instead of double-freeing.
8. Call the producer's deleter exactly once: on failure drop the capsule
   (its destructor calls the deleter), on success keep it alive until your array
   dies.
9. Accept `strides == NULL` as C-contiguous, element-unit strides (can be negative or
   zero), and `byte_offset`; numpy always emits a non-NULL data pointer even for 0-d
   and size-0 tensors, and never emits `byte_offset != 0`, so do not rely on those
   being absent.
10. Expect numpy `bool` to arrive as `kDLBool` (code 6, 8 bits) and read-only arrays
    to arrive as versioned capsules with the READ_ONLY flag; expect `np.from_dlpack`
    on your capsules to request `max_version=(1, 0)` and `dl_device=None`.
11. Do not expect numpy to move data between devices or to copy: `device=` only
    accepts `"cpu"`, and `copy=False` is advisory only.

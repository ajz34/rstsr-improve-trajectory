# R1 design questions — `rstsr-cpu-pyo3`

Date: 2026-10-04. Round 1 of the grilling session opened by `initial-prompt.md`.
Each question carries a recommendation (`➡️`); the maintainer's answers reshape the
tree and open R2. Facts below were verified against the repos on this date — they are
not questions.

## 1. Inherited constraints (settled upstream — not reopened here)

| # | Constraint | Source |
|---|---|---|
| C1 | Protocol logic stays in `rstsr-cpu-dlpack`; the pyo3 crate is boxing/packaging on top ("shape B ... a packaging step on top, not a redesign") | `DECISIONS-R2.md` §2 |
| C2 | Mutability rung: zero-copy is read-only in both directions; writeable only via `copy=True`; zero-copy adoption into an owned `Tensor` is rejected ("too unsafe") | A1; `FOLLOWUPS.md` §1 |
| C3 | No rust-numpy reimplementation (rust-numpy traits / `ndarray` / pyo3 *inside* rstsr deferred); rust-numpy is prior art only | D1 |
| C4 | Capsule discipline: fresh `DlpackExport` per `__dlpack__` call; `copy=True` → `to_dlpack_copy_f`, else shared flavor; capsule named `dltensor_versioned` with a destructor that calls the deleter **iff** unconsumed; import renames the capsule to `used_dltensor_versioned` *before* handoff; we produce DLPack version (1,0) and honour `max_version` | `DESIGN.md` §8 |
| C5 | Imported tensors are read-only `TensorDlpack` (`!Send`/`!Sync`, producer deleter exactly once); views of foreign buffers are copy-or-nothing; rstsr views export only via `to_dlpack_shared_view(base, view)` with a retained base | `FOLLOWUPS.md` §2–§3 |

## 2. Fact base (rstsr side, branch `261003/rstsr-cpu-dlpack` @ `d84f235`)

API the adapter consumes (all verified on the branch):

- **Export**: `into_dlpack[_f]` (consumes, `IS_COPIED`), `to_dlpack_shared[_f]`
  (repeatable, `READ_ONLY`), `to_dlpack_copy[_f]` (deep copy, `IS_COPIED`),
  `to_dlpack_shared_view[_f](base, view)`; guard type `DlpackExport`
  (`managed()`, `into_raw()`; Drop = deleter path).
- **Import**: `unsafe fn from_dlpack_versioned_f / from_dlpack_legacy_f(ptr) ->
  Result<TensorDlpack<T, B, D>>` — validates device/dtype/layout *before* taking
  ownership; a rejected import leaves the foreign tensor under the caller's control.
- **Representations**: `TensorDlpack` (foreign owner, read-only),
  `TensorDlpackShared` (shareable), `into_shared_dlpack_f(Tensor) ->
  TensorDlpackShared` (no copy); `DlpackSharedBaseAPI` implemented for
  `TensorDlpackShared` and core `TensorArc`; core gained zero-copy `Clone for
  TensorArc` (CoW on `raw_mut`).
- **Dtype dispatch**: `with_dlpack_dtype!` macro — i8..i128, u8..u128, f32/f64,
  Complex32/64, bool; f16/bf16 behind the default `half` feature. i128/u128 are not
  consumable by stock NumPy.
- **Devices**: blanket over `DeviceAPI<T, Raw = Vec<T>>` — `DeviceCpuSerial`
  (always available, no feature), `DeviceFaer` (= facade default `DeviceCpu`,
  feature `faer`), `DeviceCpuRayon`, five BLAS backends; all map to `(kDLCPU, 0)`.
- **State**: the branch is **unmerged** (fork `ajz34` only); rstsr `main` @ `f179c46`
  is clean and has **no pyo3 anywhere** (deps, code, CI).
- **Python-side prior art**: `../2026-10-03-rstsr-cpu-dlpack/prototype/` — ctypes shim
  (`rstsr_dlpack.py`), `demo-ffi` cdylib host with handle table, e2e suite
  (8 cases / 64 checks green, Python 3.13 + NumPy 2.5.1, conda env `torch`).

## 3. Open questions (R1 frontier)

### Q1 — Python surface scope: what is the Python-visible object?

- (a) **Protocol holder only**: a `Tensor`-like pyclass with `__dlpack__`,
  `__dlpack_device__`, properties (shape/strides/dtype/ndim/flags), and a module-level
  `from_dlpack(obj)`. Feature-equal to the ctypes shim, natively.
- (b) **(a) + creation and convenience**: `asarray` (from lists/buffers/numpy),
  `zeros`, `arange`, and `to_numpy(copy=False)` — pure-Python round trips become
  possible without a Rust host program.
- (c) **(b) + compute surface** (elementwise/reductions/linalg) — the rust-numpy-like
  library.

Protocol details settled either way (per C4): `stream` must be None/default (CPU),
`dl_device` must be `(kDLCPU, 0)` or None, `max_version` honoured at (1,0), NumPy's
TypeError-retry (double `__dlpack__` call) must be safe.

➡️ **(b) for v0.1.** "Approachable" fails if a Python user cannot *create* a rstsr
tensor without a Rust host; (a) only re-implements the demo. (c) is barred by C3/D1 in
spirit — the moment ops enter, scope creeps toward rust-numpy. The ctypes shim in
`rstsr-cpu-dlpack/examples/` stays as documentation of the pure-Rust host path.

### Q2 — Placement and branch strategy

- (a) **Prototype in this task dir first** (path dep into the rstsr checkout with
  `261003/rstsr-cpu-dlpack` checked out, exactly how `demo-ffi` did it); port to
  `crates-interop/rstsr-cpu-pyo3` once the dlpack branch merges.
- (b) **Directly in the rstsr workspace now**: branch `261004/…` stacked on
  `261003/rstsr-cpu-dlpack` (both live on fork `ajz34`), mirroring the maintainer's
  2026-10-03 redirect ("why not directly implement in rstsr repo?").
- (c) **Separate sibling repo** at pack root.

➡️ **(a).** The hard dependency is unmerged; stacking another branch on it in the
product repo compounds review debt, while this repo's charter (self-contained task
dirs, any linkage acceptable) exists precisely for this. Option (b) remains open at
port time — it is the same code motion the dlpack crate went through.

### Q3 — What the pyclass wraps; threading model

Export side: the holder needs a repeatable `__dlpack__`, so the natural canonical
repr is `TensorDlpackShared` (via `into_shared_dlpack_f`) or `TensorArc`, using
`to_dlpack_shared` per call; `copy=True` → `to_dlpack_copy_f`. The consuming
`into_dlpack` (move export) is unusable behind a Python object (protocol requires
repeatable calls).

Import side: `from_dlpack_versioned_f` yields `TensorDlpack`, which is
`!Send`/`!Sync` (C5) — pyo3 `#[pyclass]` demands `Send + Sync` unless declared
`unsendable` (thread-pinned object; not usable under free-threaded CPython).

- (a) **One `Tensor` pyclass**, interior enum `{ Shared(TensorDlpackShared),
  Foreign(TensorDlpack) }`, class marked `unsendable`.
- (b) **Two classes** (e.g. owned/shared `Tensor` sendable + imported `TensorView`
  unsendable), making the split visible in Python.
- (c) **Import always copies** into shared/owned repr (writeable, `Send`+`Sync`) —
  abandons zero-copy import, contradicting the task's stated aim.

➡️ **(a).** One class keeps the Python story simple ("a rstsr tensor is a rstsr
tensor"); the thread-pinning caveat is invisible under GIL Python and documented.
(c) defeats the purpose; (b) leaks Rust repr distinctions Python users should not
care about.

### Q4 — Device default for the extension

- (a) `DeviceCpuSerial` baked in for v0.1 (no features, matches the dlpack crate's
  examples; creation functions are the only device consumers since there are no ops).
- (b) Facade default (`DeviceCpu = DeviceFaer`, pulls `rayon`) for parity with the
  `rstsr` crate experience.
- (c) Cargo-feature seam now (`serial` default; `rayon`/`faer`/BLAS selectable).

➡️ **(a) now, (c) at port time.** v0.1 has no compute surface, so feature weight and
build time buy nothing yet; the seam is one `type` alias + feature away when ops or
the facade enter.

### Q5 — Packaging and naming

- Crate: `rstsr-cpu-pyo3` (given). Python distribution/module name: `rstsr_cpu`?
  `rstsr`? Build: maturin; abi3 (which minimum — py310?); PyPI: now / after merge /
  never (in-pack dev only)?

➡️ **Distribution/module `rstsr_cpu`, maturin, `abi3-py310`, no PyPI until the crate
ports into the rstsr workspace and the API stabilises.** Keeps the `rstsr-cpu-*`
family naming symmetric (`dlpack` crate ↔ protocol, `pyo3` crate ↔ bindings) without
claiming the bare `rstsr` name on PyPI before there is anything more than a bridge.

### Q6 — Test and CI plan

- Rust: unit tests for capsule destructor semantics (deleter iff unconsumed, rename
  before handoff, double-call safety) against `DlpackExport` directly.
- Python: port the prototype's 8-case/64-check e2e suite to pytest against the real
  extension (zero-copy pointer equality, negative/strided/transpose layouts, lifetime
  via weakref, failure injection with malignant managed tensors, torch cross-check).
- CI: none in the task dir (local, conda env `torch`); a GitHub Actions workflow with
  a Python matrix is added only when the crate ports to the rstsr workspace
  (mirroring `rstsr-cpu-dlpack-test.yml`).

➡️ **As described.** The e2e suite is the acceptance gate; CI can wait for the port
(Q2) so the unmerged-dependency problem does not have to be solved twice.

## 4. Explicitly out of scope (pending maintainer override)

- Any op/compute surface on the Python object (Q1c).
- Mutable zero-copy in either direction, adoption of foreign memory (C2).
- rust-numpy-style compile-time dtype typing in Python (C3).
- GPU/accelerator devices (the dlpack crate rejects them; a future bridge crate's
  problem).

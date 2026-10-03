# Decision points: rstsr ↔ NumPy interop (for discussion)

- **Date**: 2026-10-03 · **Companion**: `REVIEW.md` (fact base) · `notes/` (raw evidence)
- **Purpose**: reach consensus on *direction and shape* before any design/implementation work.
- **Status**: none of these are decided; each carries a recommendation for discussion, not a plan.

Each question is phrased so that a one-line answer unblocks the next layer. Q1–Q4 are
load-bearing; the rest refine.

---

## Q1. What does "bi-directional" mean for v1?

| Option | Meaning |
|---|---|
| (a) | NumPy → rstsr only (read NumPy arrays into rstsr) |
| (b) | rstsr → NumPy only (expose rstsr tensors to NumPy) |
| (c) | both, as two separable deliverables |

**Recommendation**: (c), but keep them independently shippable — they have different
difficulty profiles (export needs a Python-visible object carrying `__dlpack__`; import needs
a foreign-buffer storage + runtime dtype dispatch).

## Q2. Which interchange mechanism?

| Option | What it buys | What it costs |
|---|---|---|
| (a) DLPack only | zero-copy both ways; standard, versioned, library-neutral contract; device-neutral (future GPU); tiny spec vs. C-API; matches the array-API ecosystem | cannot express datetime64/timedelta64/strings/structured/object/non-native-endian; no casting; needs a Python object (pyo3) to carry the protocol |
| (b) NumPy C-API (depend on rust-numpy or reimplement) | full NumPy dtype coverage; `.npy`-style semantics; numpy's own casting rules (`np.asarray`) | NumPy-specific; heavyweight; couples to rust-numpy+ndarray+pyo3; rust-numpy itself has no ownership-adoption for NumPy buffers (borrow-only) and a ≤32-dim cap |
| (c) hybrid: DLPack primary + copy fallback (bytes/.npy or numpy C-API) for the expressiveness gap | zero-copy for the cases that matter; correctness for the rest | two mechanisms to maintain |
| (d) copy-only (bytes / `.npy` files / `tobytes`) | trivial to reason about; no unsafe; no deps beyond a Python shim | no zero-copy; perf cliff at the boundary |

**Recommendation**: (c). Rationale: rstsr's dtype set is *nearly* fully covered by DLPack —
the exceptions are `i128/u128` and `bf16` (expressible in DLPack but not consumable by stock
NumPy; §6.3), and the gap is otherwise only NumPy→rstsr for exotic dtypes, where a copy
fallback is the honest answer anyway. Details in `REVIEW.md` §9.

## Q3. Where does the bridge code live?

| Option | Notes |
|---|---|
| (a) new optional crate in the rstsr workspace, e.g. `rstsr-dlpack` / `rstsr-python` | keeps `rstsr-core` free of pyo3; separate MSRV possible; can depend on `dlpack-ffi` (pure types) |
| (b) rstsr-core optional feature | pulls pyo3 into core's dependency graph; rejected by packaging policy? |
| (c) separate repository | cleanest isolation, but splits the ecosystem map |

**Recommendation**: (a). Precedent: the ecosystem already separates concerns by crate
(`rstsr-blas-traits`, device crates); `dlpack-ffi` itself is deliberately pure (no rstsr
references), so the glue must live somewhere else.

## Q4. What is the Python-facing deliverable?

Constraint discovered in §6.1: **`np.from_dlpack` only accepts an object with a `__dlpack__`
method — a bare `PyCapsule` is rejected** (`AttributeError`). So "just hand NumPy a capsule"
is not an option; the export side must present a Python object (or the user wraps the
capsule in a shim class themselves).

| Option | Notes |
|---|---|
| (a) user-side shim: Rust returns a `PyCapsule`; the user writes a small Python holder class calling through | no pyclass in rstsr; ugly, ignores the protocol kwargs, easy to get lifetimes wrong |
| (b) pyclass wrapper exposing `__dlpack__`/`__dlpack_device__` (and maybe `__array__`, `shape`, `dtype`) | `np.from_dlpack(tensor)` "just works"; correct kwargs/idempotence handling; natural path to a real Python API |
| (c) pip-installable extension (`maturin`) | product-level commitment (wheels, CI, versioning); premature |

**Recommendation**: (b). Design the Rust-side traits so a future (c) is packaging only.
Defer (c).

## Q5. NumPy → rstsr: view or copy first?

- **view** (zero-copy) requires a foreign-buffer storage in rstsr: (ptr, layout, dtype,
  owner) + a drop hook that releases the NumPy reference (or calls the DLPack deleter).
- **copy** requires only a dtype-dispatched `from_raw_parts`/`asarray` path.

**Recommendation**: view first for DLPack-expressible dtypes (it is the whole point);
copy as the fallback (Q2c). Note both need the same runtime dtype dispatch table, and the
*view* path is unusually cheap in rstsr: the existing `DataRef` + non-owning-`Vec` pattern
(`asarray`/`IntoRSTSR` precedents, REVIEW §8.2) needs no new core types — only an audited
unsafe wrapper. The *owned* path is where the real design work is (G1).

## Q6. Ownership semantics of an imported NumPy buffer?

| Option | Notes |
|---|---|
| (a) borrow with a lifetime: `TensorView<'a>` tied to the Python object | simplest; matches rust-numpy's borrow philosophy; not storable past the scope |
| (b) owned adoption: `TensorArc`-style storage whose last drop calls the DLPack deleter | needed for pipelines that outlive the Python call; DLPack supports it (`manager_ctx` + `deleter`) |

**Recommendation**: (b) is the strategically right model (it is what DLPack is *for*), with
(a) as a trivially-derived special case. It requires a new owner+deleter storage variant
(G1) — and it is the main reason a bridge would touch `rstsr-core` at all; note rstsr
currently *documents* that owning a foreign allocation is UB (`device_faer/conversion.rs:85-94`).

## Q7. Policy for NumPy dtypes rstsr cannot represent

(datetime64/timedelta64, str/bytes/StringDType, object, structured/void, longdouble,
non-native byteorder)

| Option | Notes |
|---|---|
| (a) hard error with a precise message | safe, explicit |
| (b) opt-in `np.asarray`-like cast/copy (e.g. datetime64→i64 raw? object→?; endianness→swap) | matches NumPy culture; only sensible per-dtype |
| (c) implicit cast | rejected: silent precision/blowup surprises |

**Recommendation**: (a) by default in the zero-copy path; (b) via explicitly-named helpers
where a meaningful mapping exists (e.g. byte-swap copy; datetime64→i64 is a *lossy
reinterpretation* and should NOT be automatic).

Concrete gap (from §6.3): NumPy→rstsr fails for datetime64/timedelta64, `U`/`S` strings,
object, structured/void, float128/complex256, byte-swapped (NumPy refuses to export those at
all), and `kDLBfloat` (NumPy has no bfloat16 dtype). rstsr→NumPy fails for `i128/u128` and
`bf16` (NumPy has no such dtypes). Everything else in rstsr's dtype set round-trips
zero-copy: bool, i8–i64, u8–u64, f16/f32/f64, complex64/128.

## Q8. Byteorder

Facts (§6.2): DLPack carries no byteorder field, and NumPy **refuses to export** byte-swapped
arrays (`BufferError: DLPack only supports native byte order.`). A third-party producer could
still hand over non-native data silently, so the policy is about trust in producers, not
about numpy.

**Recommendation**: mirror rust-numpy's decision — never silently reinterpret, never silently
convert; document that a capsule's data is assumed native-endian; offer an explicit
copy-and-swap helper for the by-*value* (copy) path.

## Q9. GIL and threading contract

- Exports must keep the storage alive across arbitrary Python lifetimes → reference-counted
  ownership; who may mutate while exported?
- rstsr's rayon-first compute wants `detach`-style GIL release around kernels.

**Recommendation**: document the same stance as rust-numpy ("unchecked code is its author's
responsibility"; safe APIs cannot add UB), and provide an explicit "exported / on-loan" state
or guard if the cost is acceptable. Do not attempt a cross-extension borrow registry in v1
(unlike rust-numpy's, ours would have no second implementation to coordinate with).

## Q10. Dependency and MSRV policy

pyo3 0.29 requires Rust 1.83; the rstsr workspace is at 1.82.0.

| Option | Notes |
|---|---|
| (a) bridge crate carries its own MSRV (1.83+) while the workspace stays 1.82 | only works if the crate is not a workspace member with inherited rust-version, or the workspace bumps |
| (b) bump workspace MSRV to 1.83 | simple, mild cost |
| (c) avoid pyo3 (C-ABI + ctypes shim on the Python side) | zero Rust deps, but hand-rolled lifetimes and a Python shim to maintain |

**Recommendation**: (a) if the packaging works cleanly, else (b). Avoid (c) unless there is a
hard constraint.

## Q11. DLPack version policy

Facts (REVIEW §7, `notes/dlpack-v13-spec.md`): NumPy 2.5.2 emits and requests version
`(1, 0)` (not 1.3); it accepts both `dltensor_versioned` and legacy `dltensor` capsules;
legacy-only producers are detected by catching `TypeError` from the `max_version` kwarg;
the `READ_ONLY` flag exists only from v1.0 (NumPy raises `BufferError` rather than export a
read-only array to a pre-1.0 consumer).

**Recommendation**: *consume* both capsule kinds; *produce* `DLManagedTensorVersioned` with
proper `max_version` handling (version 1.x); set `READ_ONLY` for view exports; fall back to
a legacy capsule only if a consumer cannot negotiate (or raise `BufferError`), never silently
relaxing correctness.

## Q12. Devices

rstsr is CPU-only today; NumPy is CPU-only. But rstsr's device abstraction is the reason to
prefer DLPack (device field + `__dlpack_device__`).

**Recommendation**: design the device mapping as a trait method (device kind + id) with only
`kDLCPU` implemented now; do not bake CPU assumptions into the protocol layer.

## Q13. What is the first acceptance test?

Candidate definitions of "done" for a first milestone:

1. `np.from_dlpack(rstsr_tensor)` returns an array that shares memory (mutating one shows in
   the other), is writeable iff the export was writeable, and the export survives the Rust
   call's scope;
2. `rstsr` can wrap `numpy_array` zero-copy **as a view whose lifetime is tied to the NumPy
   array** (the cheap version), and — separately — decide whether "remains valid after the
   NumPy array is garbage-collected" (owned adoption, G1/W7) is in scope;
3. round-trip dtype/shape/strides equality for the expressible dtype set;
4. explicit failure (clear error, no UB) for every non-expressible dtype and for legacy
   capsules (read-only semantics).

**Recommendation**: (1) and (3) for the export milestone, (2-view) for the import milestone,
(4) throughout (error-path tests are the spec).

## Q14. Process: is the fact base of `REVIEW.md` accepted?

The recommendations above depend on facts that should be validated by the maintainer before
any design work: the DLPack coverage table (§6/§7), the claimed absence of a foreign-buffer
storage in rstsr (§8), and the option analysis (§9). **No implementation should start before
Q1–Q4 are answered.**

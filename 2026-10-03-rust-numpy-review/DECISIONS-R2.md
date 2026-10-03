# R2 decisions — scope closed on the pure `rstsr-cpu-dlpack`

- **Date**: 2026-10-03 · **Companions**: `REVIEW.md`, `QUESTIONS-discussion-R1.md`,
  `ANSWERS-discussion-R1.md`, `RESPONSE-discussion-R1.md`
- **Input**: maintainer, after reading `RESPONSE-discussion-R1.md` —
  *"I'm not sure if we need to fully implement the rust-numpy, but anyway, pure rstsr-cpu-dlpack is
  capable."*

## 1. Decisions

| # | Decision | Effect |
|---|---|---|
| D1 | The heavy interchange layer (rust-numpy traits / `ndarray` / pyo3 *inside* rstsr) is **deferred, not built now** | leaves the plan (C5's layer 2); the dtype expressiveness gap remains with the copy fallback (W5). Revisit only if zero-copy for exotic dtypes (datetime64/strings/object/structured) ever becomes a requirement |
| D2 | The deliverable is the **pure `rstsr-cpu-dlpack`** crate: `dlpack-ffi` types + `rstsr-core`, no pyo3, no packaging | Q3(a) confirmed with the `rstsr-cpu-*` name; Q4 resolves to shape **A/C** — rstsr ships the tensor/protocol side, the Python-side object stays outside the crate (a reference shim is documented) |

## 2. What "capable" covers, and the one boundary that cannot move

**Covered by the pure crate (all of the hard parts):** given any
`TensorAny<R, T, B, D>` with `B: DeviceAPI<T, Raw = Vec<T>>` and a DLPack-expressible `T`,

- `to_dlpack` produces a `DLManagedTensorVersioned` whose `manager_ctx` owns the tensor (moved or
  refcounted), with correct deleter/`READ_ONLY`/`IS_COPIED` behavior and repeat-export safety;
- `from_dlpack` consumes a versioned or legacy pointer into a tensor over a foreign-owner repr
  (view rung), or a consumer-owned buffer (owned rung, V4), with dtype dispatch, alignment check and
  stride normalisation (V5);
- any DLPack consumer that can receive a raw pointer — another Rust crate, C, Julia, a torch bridge —
  can use the result directly, which is the "DLPack as the C interface to rstsr" role from Q2.

**The boundary that cannot move:** `PyCapsule` is a CPython type, and `np.from_dlpack(x)` requires
`x.__dlpack__()` to *return a capsule*. So the last mile is Python-side:

- export: wrap rstsr's pointer in a capsule (`PyCapsule_New` with the DLPack name, a destructor that
  calls the crate's exported `extern "C"` deleter if the capsule was never consumed) and expose it as
  `__dlpack__(...)` on a small holder object;
- import: call `arr.__dlpack__(max_version=(1, 0))`, read the pointer out, rename the capsule to
  `used_*`, hand the pointer to the crate.

That is ~30 lines of Python and it is where the protocol footguns live (double call, rename, deleter
exactly once); the crate owns the *logic* and the Python shim only the capsule boxing. The capsule
mechanics via ctypes — creation with a kept-alive destructor, plus the full consumer-side validation
matrix — are already demonstrated in `experiments/probe_numpy_dlpack.py`; a shim over a Rust pointer
reuses that shape. If a pyo3 adapter is ever wanted, it is a packaging step on top (shape B), not a
redesign.

## 3. Working assumptions for the design (confirm or correct)

| # | assumption |
|---|---|
| A1 | **Mutability**: read-only zero-copy import/export; owned-and-writeable only via `copy=True` (V4); no mutable zero-copy path in v1 |
| A2 | **Milestone order**: export first (`np.from_dlpack(holder)` sharing memory), import second |
| A3 | The reference Python shim ships as `examples/` (ctypes) inside the crate — not as a crate, not as a package |
| A4 | The prototype is built and tested in a new task directory of this repository first (charter: task dirs carry complete code and a patch); the crate moves to the rstsr workspace under the normal review policy afterwards |

## 4. Next step

A design document for `rstsr-cpu-dlpack` — module layout, public API surface (traits vs free
functions), the foreign-owner repr and its safety contract, dtype dispatch table, error taxonomy,
device mapping seam, and the test plan (deleter-exactly-once, capsule reuse, double call, the
view/owned/read-only matrix, the failure-injection producer built on the ctypes harness) — followed by
a prototype in a new task directory. No code before the design is on paper.

## 5. Addendum (later on 2026-10-03)

The maintainer redirected implementation into the rstsr repo (*"why not directly implement in rstsr
repo? Just create a new branch on that"*), superseding A4's "prototype in this repo first":

- the crate now lives at `crates-interop/rstsr-cpu-dlpack` of the **rstsr workspace**, on branch
  `261003/rstsr-cpu-dlpack`, with `rstsr/Cargo.toml` gaining the member, the workspace dependency
  and `dlpack-ffi = "1.3"` (1.3.0 was published to crates.io the same day);
- the design and the Python host harness remain in the task directory
  `../2026-10-03-rstsr-cpu-dlpack/`; nothing is committed in the rstsr repo (no-auto-commit policy).

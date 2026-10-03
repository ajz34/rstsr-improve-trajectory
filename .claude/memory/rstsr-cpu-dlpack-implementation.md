---
name: rstsr-cpu-dlpack-implementation
description: rstsr-cpu-dlpack implemented in the rstsr repo on branch 261003/rstsr-cpu-dlpack (2026-10-03) — repr design, test/miri/Python status, and the gotchas hit while building it.
metadata:
  type: project
---

The `rstsr-cpu-dlpack` crate was implemented on 2026-10-03 (same day as its design) **inside the
rstsr workspace** — `crates-interop/rstsr-cpu-dlpack` on branch `261003/rstsr-cpu-dlpack`, with
`rstsr/Cargo.toml` gaining the member, the workspace dep and `dlpack-ffi = "1.3"` (published that
day). Left **uncommitted** per the no-auto-commit policy. Design + the Python host harness stay in
`rstsr-improve-trajectory/2026-10-03-rstsr-cpu-dlpack/` (see [[rstsr-numpy-interop-review]]).

Shape of the implementation (it passed 18 Rust tests, miri, clippy, and a Python 3.13 + NumPy 2.5.1
end-to-end suite of 8 cases / 64 checks):

- one repr `DataDlpack<C, O>` = fabricated `ManuallyDrop<Vec<T>>` span + owner `O`; import owner
  `DlpackForeignOwner` (raw DLManagedTensor[Versioned] pointer, calls the producer deleter exactly
  once on drop, `!Send`/`!Sync`), shared-export owner `Arc<Vec<T>>` (Clone → repeat export).
- import normalisation: `base' = data + byte_offset + min_index·itemsize`, layout `offset' =
  -min_index`, span length `max-min+1`; zero-size → `NonNull::dangling()` + offset 0.
- export: `#[repr(C)]` box with `DLManagedTensorVersioned` first (deleter casts the pointer back),
  `version {1,0}`, always-emitted element strides, `byte_offset = 0`, data = base + `layout.offset()`;
  flags: IS_COPIED for move/copy exports, READ_ONLY for shared exports.
- `DlpackExport::into_raw()` is the only way the pointer leaves Rust; dropping the guard runs the
  same deleter a consumer would.

Gotchas that cost time (worth remembering for bridge crates):

- **`DataArc` was not `Clone`** at implementation time (no inner-Arc accessor either) → the bridge
  used its own `Arc` owner; see [[rstsr-external-storage-repr]]. (Later on 2026-10-03 core gained
  `Clone for DataArc`/`TensorArc`, which enabled view export of `TensorArc` bases —
  [[rstsr-tensorarc-clone]].)
- `tensor.raw()` is *address order*, not logical order — read logical values through
  `storage().get_index(layout().index(&idx))` (bit me in tests: a reversed view's raw buffer looks
  ascending).
- rstsr-core's prelude does **not** export `Storage`/`DataOwned`/`DataArc` (they live in
  `rstsr_core::storage::exports::*`), nor `OpAssignAPI` (`rstsr_core::operators::exports::*`) and
  `DeviceCreationAnyAPI` (`storage::exports`).
- Mixing a *path* `dlpack-ffi` dep with the registry version makes two distinct crate instances
  ("multiple different versions of crate dlpack_ffi") — depend on the registry version everywhere.
- `Tensor::to_owned`/`to_dlpack_copy_f` need the heavy device bounds (`DeviceCreationAnyAPI`,
  `OpAssignAPI`, `DeviceRawAPI<MaybeUninit<T>>`); `TensorBase` needs `D: DimAPI` (not just
  `DimBaseAPI`).
- On nightly, bare `f16` resolves to the unstable primitive type and shadows nothing — name
  `half::f16` via the crate's re-export in tests.
- Adding a member that nobody depends on triggers cargo's `unused workspace dependency` warning for
  its `[workspace.dependencies]` entry (same as the `rstsr` facade entry).
- The `PyCapsule` destructor can be written in **Rust** (`PyCapsule_GetName`/`GetPointer` resolve
  from the CPython process at dlopen); the name check (`dltensor_versioned` vs `used_*`) is what
  prevents a double free.

Follow-ups from the post-implementation review (2026-10-03): the unsafe zero-copy *adoption*
draft (`from_dlpack_*_adopt_f`) was **rejected** by the maintainer (too unsafe), and the
*shared-view export* (`to_dlpack_shared_view[_f]`, with the core `Clone for TensorArc` it needed)
was **implemented** — see [[rstsr-cpu-dlpack-followups]], [[rstsr-tensorarc-clone]] and
`2026-10-03-rstsr-cpu-dlpack/FOLLOWUPS.md`. The crate itself (with the review-comment fixes:
optional `half`, `CODE_*` constants in the dispatch macro, `rstsr_raise!`, anchor docs on the
panic versions) is committed in the rstsr repo as `49a9c45`.

The crate is now also surfaced through the main `rstsr` prelude: cargo feature `rstsr/dlpack`,
items under `rt::dlpack::*` (rstsr commit `6eb3d90`). It required a bridge `prelude` module with
the `rstsr_traits`/`rstsr_structs`/`rstsr_funcs` groups (the member-crate pattern) that the facade
forwards behind `#[cfg(feature = "dlpack")]`. The pattern + the prelude conventions (including the
deviations that exist in the tree) are codified in the `rstsr-agents` skill `prelude-conventions`
(commit `b8828bd`).

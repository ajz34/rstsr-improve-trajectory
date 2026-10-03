# Open threads after the post-implementation review (2026-10-03)

> Status (end of 2026-10-03): **both threads are closed.** §1 (zero-copy adoption) was
> **decided against** by the maintainer — "decided not implement, at least currently; this is
> too unsafe" — and stays unimplemented; §2 (view export) was **implemented** in the same
> session. The crate lives in the rstsr repo on branch `261003/rstsr-cpu-dlpack`
> (committed as `49a9c45`; the view-export change is the current working tree).

Cross-framework DLPack usage evidence (numpy 2.5.1 / torch 2.14 probes) lives in
`../2026-10-03-rust-numpy-review/experiments/probe_dlpack_python_usage.py` +
`probe-output-python-usage.txt` (commit `21e1b61` of this repo).

## 1. Zero-copy adoption: `TensorDlpack` → owned `Tensor` (unsafe) — **REJECTED, not implemented**

**Decision (maintainer, 2026-10-03): do not implement — the unsafe conversion is too unsafe to
ship.** The analysis below is kept as the record of why; no code was added.

Motivation (for the record): imported tensors are read-only and `!Send`/`!Sync`; `to_owned()` is
a gather copy. A `Vec::from_raw_parts` adoption would drop the copy and give write access +
`Send`/`Sync`. The ecosystem offers no prior art: numpy/torch/cupy all keep the producer's
deleter — even `np.from_dlpack(t, copy=True)` yields `owndata=False` (probe), i.e. the copy stays
in the producer's allocator/deleter domain.

Why it cannot be a safe generic conversion — four facts the DLPack struct does not carry:

1. **base vs interior pointer** — DLPack permits views; NumPy bakes slice offsets into `data`
   (`np.arange(100)[10:20]` exports `data = base + 80`), and our own move export emits
   `data = base + layout.offset()` (`export.rs:171`). Re-homing an interior pointer is heap
   corruption; no portable runtime check exists (glibc `malloc_usable_size` is UB on interior
   pointers).
2. **capacity** — `Vec::from_raw_parts` needs the allocation's element count; DLPack carries the
   span only (a slice of a bigger buffer has span < capacity). glibc `free` ignores size/align,
   so a wrong capacity "works" until a stricter allocator or ASAN.
3. **allocator identity** — Rust's default `System` allocator is libc `malloc`, NumPy's default
   data allocator is malloc-backed, so the common case matches; a `#[global_allocator]` swap or
   NumPy's replaceable `PyDataMem` handler silently breaks it.
4. **deleter semantics** — a standard DLPack deleter frees the data; NumPy's deleter drops a
   Python reference (the buffer is owned by the array graph, possibly a base object such as a
   view's parent or `frombuffer`'s `bytes`). After adoption you must either call it (double free)
   or not (metadata leak), and cannot know which is right from the struct.

The only runtime gate is `IS_COPIED` ("solely owned throughout its lifetime by the consumer") —
it covers the aliasing half, nothing else.

Options as presented (all now moot):

- **(a)** the unsafe entry only (usable where the producer is a Rust component you control);
- **(b)** (a) + an `into_dlpack_*_adoptable` export flavor (`data` = allocation base,
  `byte_offset = offset·itemsize`, `cap`/`len` in `manager_ctx`, a private flag bit,
  metadata-only deleter) → self-verifying `export → adopt` round trips within controlled
  pipelines; caveat: a non-adopting consumer (NumPy) leaks the buffer;
- **(c)** neither; document the analysis only.

**Outcome: (c).**

## 2. Zero-copy export of basic-indexed views — **IMPLEMENTED (2026-10-03)**

Gap (design rationale): `into_dlpack_f` requires an owned `Tensor`; `to_dlpack_shared_f` requires
the whole `TensorDlpackShared` layout; views went out only through `to_dlpack_copy_f` (a copy).
NumPy's `view.__dlpack__()` works because view objects retain their base (refcount); rstsr views
are pure borrows (`Storage<DataRef<'a, Vec<T>>, T, B>` with `TrueRef(&root)`) carrying no owner —
the exporter must supply one.

What shipped (branch `261003/rstsr-cpu-dlpack`, working tree):

- **`to_dlpack_shared_view` / `to_dlpack_shared_view_f(base, view)`** in `src/export.rs`. Checks
  that the view's root container *is* the base's buffer
  (`ptr::eq(base.buffer_base_ptr(), view.storage().raw().as_ptr())`), then builds a fabricated
  span over the base buffer with a cloned owner, validates the view layout through the existing
  `new_f` path (`check_strides` + `idx_max <= buffer len`), and exports with `READ_ONLY`.
  `data = buffer base + view offset`, so a `(.., 1..3)` slice of a `[3, 4]` tensor exports
  `strides = [4, 1]`, `data = base + 1`. Both the base and the view may be dropped after
  `into_raw()`; repeatable; no `unsafe` in the entry points (the fabricated span is the crate's
  existing repr idiom).
- **`DlpackSharedBaseAPI<T>`** (4 members: `buffer_base_ptr`, `buffer_len`, `clone_buffer_owner`,
  `type Owner`) — the seam implemented by the bases:
  - `TensorDlpackShared` (bridge repr): owner = the cloned representation (Arcs bumped).
  - `TensorArc` (core): owner = `DataArc<Vec<T>>` clone. **Newly enabled by core changes below.**
  - Deliberately **not** implemented for `TensorDlpack` (foreign import): its owner cannot be
    cloned without double-freeing the producer deleter — views of foreign buffers stay
    copy-or-nothing (`to_dlpack_copy`).
- **Core additions** (`rstsr-core`, minor semver per maintainer):
  - `impl<C> Clone for DataArc<C>` — `Arc::clone`, zero-copy (`storage/data.rs`).
  - `impl Clone for TensorArc<T, B, D>` — storage clone (Arc bump) + layout clone; zero-copy on
    data (`tensor/ownership_conversion.rs`, next to the `Tensor`/`TensorCow` impls).
  - **Soundness**: the only safe mutation path is `DataMutAPI::raw_mut`, which is
    `Arc::make_mut` — a shared buffer is copied before a `&mut` is handed out, so two clones can
    never alias mutably through safe code; `DataForceMutAPI::force_mut` keeps its
    caller-uniqueness `unsafe` contract. (`Arc` handles were already obtainable via the public
    `From<Arc<C>>`, so external aliveness is not new.)
  - **Related fix**: `DataArc::into_owned` used `Arc::try_unwrap(..).ok().unwrap()` and panicked
    when the buffer was shared (reachable via `into_owned_keep_layout` / `into_cow`, both
    documented as "moved if possible, or fully cloned"). It now falls back to cloning the data.
- **Tests** (all green, also under miri): bridge — view layout/offset assertion, export outliving
  base + view, foreign-buffer rejection, `TensorArc` base + copy-on-write detach keeping the
  export's bytes; core — 2 unit tests (`DataArc` clone/COW/`into_owned`) and a doctest on the
  `TensorArc` clone (same `raw().as_ptr()`, `strong_count == 2`).
- Python-shim counterpart: the holder keeps the shared tensor (or `TensorArc`) alongside the
  view and calls `to_dlpack_shared_view` per `__dlpack__`.

Closed residuals: `TensorArc` bases (previously "needs a core `DataArc` clone / arc accessor")
now work; the owned-base case stays `into_shared_dlpack_f` (consume) as designed.

## 3. Notes for the record

- `DataDlpack<C, O>` is the same idiom as `DataRef::ManuallyDropOwned(ManuallyDrop<C>)`
  (`storage/data.rs:16-19`; used by `asarray` `:87-100` and `force_mut` `:541-554`) plus an
  explicit owner: **view-like access, owning-handle lifetime** — read-only, never frees the
  buffer, no lifetime parameter; liveness enforced by the owner at runtime instead of the borrow
  checker.
- `TensorDlpack → Tensor`/`TensorView` conversions are rstsr-core's generic ones (no bridge
  code): `.view()` (`ownership_conversion.rs:56`, zero-copy borrow of the handle), `.to_owned()`
  (`:351`) / `.into_owned()` (`:264`) / `.into_shared()` (`:311`) — all copies for this repr (no
  lossless unwrap; `owner()` borrows only). `view_mut` does not compile (no `DataMutAPI`) — the
  type-level read-only guarantee.

## 4. Decision log

- Zero-copy adoption (`from_dlpack_*_adopt_f`) — **rejected** (maintainer): too unsafe; analysis
  kept in §1.
- Owner-less unsafe borrow export — **dropped** (earlier, maintainer).
- `to_dlpack_shared_view_f` (view export) — **implemented**, together with `Clone for
  DataArc`/`TensorArc` and the `DataArc::into_owned` shared-buffer fix.
- Q1 flags (`IS_COPIED` on move export): the cross-framework probes support keeping flags
  advisory — NumPy keys writeability off `READ_ONLY` alone, and torch ignores `READ_ONLY`
  entirely (writes through a read-only NumPy array's buffer).

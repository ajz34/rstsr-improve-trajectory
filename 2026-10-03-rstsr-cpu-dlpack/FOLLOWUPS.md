# Open threads after the post-implementation review (2026-10-03)

> Status: **discussion captured, nothing here implemented.** The crate itself stays as implemented
> (branch `261003/rstsr-cpu-dlpack` of the rstsr repo, uncommitted per policy). These are the
> follow-ups from the review conversation: the maintainer's direction is to handle other aspects
> first and resume the view-export thread afterwards; the adoption decision is still open.

Cross-framework DLPack usage evidence (numpy 2.5.1 / torch 2.14 probes) lives in
`../2026-10-03-rust-numpy-review/experiments/probe_dlpack_python_usage.py` +
`probe-output-python-usage.txt` (commit `21e1b61` of this repo).

## 1. Zero-copy adoption: `TensorDlpack` → owned `Tensor` (unsafe; decision pending)

Motivation: imported tensors are read-only and `!Send`/`!Sync`; `to_owned()` is a gather copy. A
`Vec::from_raw_parts` adoption would drop the copy and give write access + `Send`/`Sync`. The
ecosystem offers no prior art: numpy/torch/cupy all keep the producer's deleter — even
`np.from_dlpack(t, copy=True)` yields `owndata=False` (probe), i.e. the copy stays in the
producer's allocator/deleter domain.

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

Proposed shape, if added:

```rust
/// # Safety (all on the caller)
/// - `flags & IS_COPIED`; `data` is the allocation base; the allocation is a `Vec<T>` from this
///   binary's global allocator with `len == cap`; the layout's span fits in `cap`; the producer's
///   deleter frees only the managed-tensor metadata, never the buffer.
pub unsafe fn from_dlpack_versioned_adopt_f<T, B, D>(
    ptr: *mut DLManagedTensorVersioned, cap: usize,
) -> Result<Tensor<T, B, D>>
```

Sketch: run the usual import validation; `Vec::from_raw_parts(data, cap, cap)`; layout with
`offset' = byte_offset/itemsize + min_index` checked against `cap`; `TensorBase::new_f`; call the
metadata-only deleter. ~60 lines + tests (Rust-made producer; miri-clean — a real `Vec`
allocation, not a fabricated span). Legacy structs have no flags: versioned only.

Options as presented to the maintainer:

- **(a)** the unsafe entry only (usable where the producer is a Rust component you control);
- **(b)** (a) + an `into_dlpack_*_adoptable` export flavor (`data` = allocation base,
  `byte_offset = offset·itemsize`, `cap`/`len` in `manager_ctx`, a private flag bit,
  metadata-only deleter) → self-verifying `export → adopt` round trips within controlled
  pipelines; caveat: a non-adopting consumer (NumPy) leaks the buffer;
- **(c)** neither; document the analysis only.

**Decision: pending.**

## 2. Zero-copy export of basic-indexed views (design ready; resume after other aspects)

Gap: `into_dlpack_f` requires an owned `Tensor`; `to_dlpack_shared_f` requires the whole
`TensorDlpackShared` layout; views go out only through `to_dlpack_copy_f` (a copy). NumPy's
`view.__dlpack__()` works because view objects retain their base (refcount); rstsr views are pure
borrows (`Storage<DataRef<'a, Vec<T>>, T, B>` with `TrueRef(&root)`) carrying no owner — the
exporter must supply one.

All the machinery already exists except one entry point:

- `shared.i((.., ..10, 10..))` **already compiles** and returns `TensorView<'_, T, B, IxD>`
  (indexing is generic over `R: DataAPI<Data = B::Raw>`, `rstsr-core/src/tensor/indexing.rs:176-223`).
- Every view chain collapses to the root container (`view()` re-borrows `raw()`; `DataRef::raw()`
  returns the root for both variants), so a view's layout (offset/strides) is directly
  interpretable against the shared span.
- The repr `Clone` rebuilds a span over the same pointer (`repr.rs:67-72`), so clones of the
  handle share the same root address — `ptr::eq` is exact for all legitimate inputs.

Proposed API:

```rust
pub fn to_dlpack_shared_view_f<T, B, D, D2>(
    shared: &TensorDlpackShared<T, B, D>,
    view: &TensorView<'_, T, B, D2>,
) -> Result<DlpackExport>
```

Implementation: same-root check (`ptr::eq(shared.storage().raw().as_ptr(),
view.storage().raw().as_ptr())`); layout validated against the span through the existing `new_f`
path (`check_strides` + `idx_max <= span.len()`); keepalive = cloned repr (Arc bump);
flags = `READ_ONLY`; `export_from_storage` computes `data = span_ptr + view_layout.offset()`.
After `into_raw()`, both view and handle may be dropped — the `Arc` inside the managed tensor
keeps the buffer alive; repeatable; **no `unsafe`**. Trade-off: the base is frozen once converted
(the shared repr is read-only) — the price of an unbounded consumer lifetime.

Residual cases, **not** closed by the above:

1. base is a plain owned `Tensor` the caller will not give up → consume
   (`into_shared_dlpack_f`) or copy; the unsafe owner-less borrow variant is **dropped**
   (maintainer).
2. `TensorArc` bases: core's refcounted tensor (`into_shared()`,
   `ownership_conversion.rs:311`) is the numpy-style base retention, but the bridge cannot clone
   `DataArc` (no `Clone`, no Arc accessor; only `strong_count`/`weak_count`,
   `storage/data.rs:257-267`). A small core addition (`impl Clone for DataArc` / an arc accessor)
   would enable direct `TensorArc`-based view export; `Arc::make_mut` COW keeps Rust-side writes
   safe afterwards. **Core-side decision, separate.**
3. views over foreign borrowed buffers (`DataRef::ManuallyDropOwned`) — copy or nothing.

Resume plan: implement the entry (+ panic twin; tests: pointer equality
`data == base + offset·itemsize` for a `..10, 10..` slice, `READ_ONLY` flag, repeat export,
wrong-buffer rejection, miri) and note the Python-shim counterpart (the holder keeps handle+view
and `__dlpack__` calls this entry).

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

## 4. Decision log (this conversation)

- Owner-less unsafe borrow export — **dropped** (maintainer).
- `to_dlpack_shared_view_f` (view export) — **deferred**: implement after current aspects.
- Zero-copy adoption — pending (a/b/c).
- Q1 flags (`IS_COPIED` on move export): the cross-framework probes support keeping flags
  advisory — NumPy keys writeability off `READ_ONLY` alone, and torch ignores `READ_ONLY`
  entirely (writes through a read-only NumPy array's buffer).

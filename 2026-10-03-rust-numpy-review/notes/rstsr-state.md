# rstsr workspace state — inventory for a future NumPy ↔ rstsr bridge

Survey target: `/home/a/rstsr_pack/rstsr` at `master` = `f179c46` ("Update to v0.9.0"),
workspace version `0.9.0` (`/home/a/rstsr_pack/rstsr/Cargo.toml:21`), MSRV 1.82.0.
**Working tree at survey time: clean** (`git status --short` → empty; `git rev-parse HEAD` →
`f179c465249c5f444d381743c97702ae91bcba50`).

Crates in scope: `rstsr-core`, `rstsr-common`, `rstsr-dtype-traits`, `rstsr-native-impl`,
`rstsr` (facade), `crates-device/*` (rstsr-aocl, rstsr-blis, rstsr-kml, rstsr-mkl,
rstsr-openblas), plus trait-only crates `rstsr-blas-traits`, `rstsr-linalg-traits`,
`rstsr-sci-traits`, `rstsr-test-manifest`, `crates-plugin/rstsr-tblis`.

Every claim below carries a `path:line` citation. Line numbers are as of the commit above.

---

## A. Storage & ownership (rstsr-core)

### A.1 The storage types

All storage types are generic over a **container** `C` (in practice `C = Vec<T>`) and are
defined in one file, `/home/a/rstsr_pack/rstsr/rstsr-core/src/storage/data.rs`:

| Type | Definition | Owns / borrows |
|---|---|---|
| `DataOwned<C>` | `data.rs:11` `pub struct DataOwned<C> { pub(crate) raw: C }` | owns `C` (drops it) |
| `DataRef<'a, C>` | `data.rs:16` `pub enum DataRef<'a, C> { TrueRef(&'a C), ManuallyDropOwned(ManuallyDrop<C>) }` | borrows, **or** owns-without-managing |
| `DataMut<'a, C>` | `data.rs:22` `pub enum DataMut<'a, C> { TrueRef(&'a mut C), ManuallyDropOwned(ManuallyDrop<C>) }` | mutably borrows, or owns-without-managing |
| `DataCow<'a, C>` | `data.rs:28` `pub enum DataCow<'a, C> { Owned(DataOwned<C>), Ref(DataRef<'a, C>) }` | copy-on-write |
| `DataArc<C>` | `data.rs:34` `pub struct DataArc<C> { pub(crate) raw: Arc<C> }` | shared ownership |
| `DataReference<'a, C>` | `data.rs:39` `pub enum DataReference<'a, C> { Ref(DataRef<'a, C>), Mut(DataMut<'a, C>) }` | internal use by device op parameter structs |

The `DataRef`/`DataMut` doc comments state the contract explicitly:
`data.rs:88-96` — "Wraps data that this wrapper will own without managing. ... The wrapped data
must remain valid for the lifetime `'a` ... The `ManuallyDrop` wrapper guarantees the data is
never freed here." Same wording for `DataMut` at `data.rs:163-170`.
`DataRef::from_manually_drop(data: ManuallyDrop<C>) -> Self` is **safe but contract-bound**
(`data.rs:98`); `DataMut::from_manually_drop` likewise (`data.rs:172`).

`Send`/`Sync` are hand-written with tightened bounds (`data.rs:44-60`):
`unsafe impl<C> Send for DataRef<'_, C> where C: Send + Sync {}` etc., with the comment at
`data.rs:44-50` explaining why the looser bound was a data race.

### A.2 Storage traits

- `data.rs:285` `pub trait DataAPI { type Data; fn raw(&self) -> &Self::Data; fn as_ref(&'_ self)
  -> DataRef<'_, Self::Data>; }`
- `data.rs:293` `pub trait DataCloneAPI where Self: DataAPI, Self::Data: Clone { fn
  into_owned(self) -> DataOwned<Self::Data>; fn into_shared(self) -> DataArc<Self::Data>; }`
- `data.rs:302` `pub trait DataMutAPI: DataAPI { fn raw_mut(&mut self) -> &mut Self::Data; ... }`
- `data.rs:309` `pub trait DataOwnedAPI: DataMutAPI {}`
- `data.rs:311` `pub trait DataForceMutAPI<C>: DataAPI<Data = C> { unsafe fn force_mut(&self) ->
  DataMut<'_, C>; }` — the aliasing escape hatch, with the safety contract spelled out at
  `data.rs:312-318` ("the caller must guarantee unique access ... violating this is an aliasing
  violation (undefined behavior)"). Note `DataForceMutAPI<Vec<T>>` is implemented for
  `DataRef<'_, Vec<T>>` by fabricating a `Vec` over the same pointer
  (`data.rs:541-555`, in particular `Vec::from_raw_parts(ptr as *mut T, len, len)` at `data.rs:551`).

### A.3 The `Storage` struct

`/home/a/rstsr_pack/rstsr/rstsr-core/src/storage/device.rs:14`

```rust
pub struct Storage<R, T, B>
where
    B: DeviceRawAPI<T>,
{
    pub(crate) data: R,
    pub(crate) device: B,
    _phantom: PhantomData<T>,
}
```

Constructed by the public `Storage::new`:

```rust
// storage/device.rs:81
pub fn new(data: R, device: B) -> Self { Self { data, device, _phantom: PhantomData } }
```

and destructured by `pub fn into_raw_parts(self) -> (R, B)` (`device.rs:77`).
Accessors: `device()` (`:61`), `device_mut()` (`:65`), `data()` (`:69`), `data_mut()` (`:73`),
`raw()` (`:144`, requires `R: DataAPI<Data = B::Raw>`), `raw_mut()` (`:154`), `len()`/`is_empty()`
(`:85`/`:89`), `to_cpu_vec`/`into_cpu_vec` (`:93`/`:101`),
`get_index`/`get_index_ptr`/`get_index_mut_ptr`/`set_index` (`:110`/`:118`/`:123`/`:131` — the
raw-pointer access path).

### A.4 Allocation and free

Allocation is **not** a trait; it is a small family of free functions in
`/home/a/rstsr_pack/rstsr/rstsr-common/src/alloc_vec.rs`:

- `alloc_vec.rs:26` `pub unsafe fn uninitialized_vec<T>(size: usize) -> Result<Vec<T>>` — the
  entry point; dispatches to the aligned or unaligned form.
- `alloc_vec.rs:50` `pub unsafe fn unaligned_uninitialized_vec<T>(size: usize) -> Result<Vec<T>>`
  — `Vec::try_reserve_exact` + `set_len`.
- `alloc_vec.rs:91` `pub unsafe fn aligned_uninitialized_vec<T, const N: usize>(size: usize,
  alignment: usize) -> Result<Vec<T>>` — `alloc::alloc::alloc` on a
  `Layout::from_size_align(numbytes, alignment)` (`alloc_vec.rs:66`), then
  `Vec::from_raw_parts(pointer.as_ptr() as *mut T, size, size)` (`alloc_vec.rs:112`) and
  `set_len` (`alloc_vec.rs:113`).
- Gated behind the `aligned_alloc` feature (`alloc_vec.rs:32-35`), 64-byte alignment with
  `N = 128` (`alloc_vec.rs:35`); non-Linux/macOS falls back to unaligned (`alloc_vec.rs:27-28`).

**Free** is implicit: the buffer is a real `Vec<T>`, so deallocation happens through `Vec`'s
`Drop` using the global allocator with `align_of::<T>()`. There is no custom `Drop`/`dealloc`
hook on `Storage` or on any `Data*` type.

Relevant design record: `rstsr-book` ADR-0008 ("Uninitialized allocation: `MaybeUninit` by
default, POD FFI buffers, unsafe `empty`") —
`/home/a/rstsr_pack/rstsr-book/dev/adr/adr-0008-uninitialized-allocation-contract.mdx:10`,
which blesses three allocation patterns (`:40-48`) and treats FFI buffers explicitly
("Pattern 2 — POD FFI buffers for BLAS/LAPACK", `:58-65`), declaring everything else
"forbidden ... a review-reject" (`:45`).

### A.5 Foreign buffers: is there any way to refer to memory rstsr did not allocate?

**Yes — but only through a fabricated non-owning `Vec<T>` wrapped in `ManuallyDrop`, and only
for `Device`s whose `Raw` is `Vec<T>`.** There is no first-class "foreign buffer" storage type.

The mechanism, stated in the crate's own words at `data.rs:88-96` and
`data.rs:130-139` ("a `DataRef::ManuallyDropOwned` `Vec` is fabricated over a buffer that stays
valid for `'a`"). Concretely it is used in three places:

1. **`asarray` from a borrowed slice** — `/home/a/rstsr_pack/rstsr/rstsr-core/src/tensor/asarray.rs:519-543`:

```rust
impl<'a, T, B, D> AsArrayAPI<D> for (&'a [T], Layout<D>, &B)
where T: Clone, B: DeviceAPI<T, Raw = Vec<T>>, D: DimAPI,
{
    type Out = TensorView<'a, T, B, IxD>;
    fn asarray_f(self) -> Result<Self::Out> {
        let (input, layout, device) = self;
        let ptr = input.as_ptr(); let len = input.len();
        let raw = unsafe {
            let ptr = ptr as *mut T;
            Vec::from_raw_parts(ptr, len, len)      // asarray.rs:536
        };
        let device = device.clone();
        let data = DataRef::from_manually_drop(ManuallyDrop::new(raw));
        let storage = Storage::new(data, device);
        let tensor = TensorView::new_f(storage, layout.into_dim()?)?;
        return Ok(tensor);
    }
}
```

   The `&mut [T]` twin (output `TensorMut<'a, T, B, IxD>`) is at `asarray.rs:693-718`
   (`Vec::from_raw_parts` at `asarray.rs:711`, `DataMut::from_manually_drop` at `asarray.rs:714`).

2. **`IntoRSTSR for faer views`** — the closest existing analogue of a DLPack consumer.
   `/home/a/rstsr_pack/rstsr/rstsr-core/src/device_faer/conversion.rs:53-75`:

```rust
impl<'a, T> IntoRSTSR for MatRef<'a, T> {
    type RSTSR = TensorView<'a, T, DeviceFaer, Ix2>;
    fn into_rstsr(self) -> Self::RSTSR {
        let layout = Layout::new([nrows, ncols], [row_stride, col_stride], 0).unwrap();
        let (_, upper_bound) = layout.bounds_index().unwrap();
        let raw = unsafe { Vec::from_raw_parts(ptr as *mut T, upper_bound, upper_bound) }; // :69
        let data = DataRef::from_manually_drop(ManuallyDrop::new(raw));                  // :70
        let storage = Storage::new(data, DeviceFaer::default());                          // :71
        let tensor = unsafe { TensorView::new_unchecked(storage, layout) };               // :72
        return tensor;
    }
}
```

   `ColRef` → `TensorView<'a, T, DeviceFaer, Ix1>` at `conversion.rs:139-157`; `MatMut` →
   `TensorViewMut<'a, T, DeviceFaer, Ix2>` at `conversion.rs:164-184`.

3. The `TensorAny::view()` / `view_mut()` constructors use the borrow (not the ManuallyDrop)
   variant — `ownership_conversion.rs:56-63` and `:102-110`.

**Critical caveat for a bridge.** The same file documents why an *owned* foreign allocation
cannot be re-homed into a `Vec` (`conversion.rs:85-94`): "faer allocates with 64-byte alignment
and pads its row capacity, while a `Vec<T>` always deallocates with `align_of::<T>()` and no
padding. Deallocating with a different layout than the original allocation is undefined
behavior by the allocator contract". This is exactly the hazard a DLPack bridge faces when it
takes ownership of a `DLManagedTensor` whose memory was allocated by another runtime (NumPy,
PyTorch): **accepting the pointer as a non-owning view is safe; adopting the allocation into a
`Vec<T>` is not.** The current re-homing works only because the foreign pointer is wrapped in
`ManuallyDrop` and never freed.

### A.6 Unsafe constructor from `(ptr, layout)`

**There is none.** No `TensorView::from_raw_parts`, no `Tensor::from_ptr`. What exists is a
two-step composition:

```rust
// tensorbase.rs:115
pub unsafe fn new_unchecked(storage: S, layout: Layout<D>) -> Self { Self { storage, layout } }

// tensorbase.rs:195
pub fn new_f(storage: Storage<R, T, B>, layout: Layout<D>) -> Result<Self>
where R: DataAPI<Data = B::Raw>, D: DimAPI, B: DeviceAPI<T>,
// body: layout.check_strides(true)?; then bounds_index() vs storage.len()

// tensorbase.rs:206
pub fn new(storage: Storage<R, T, B>, layout: Layout<D>) -> Self   // panicking form
```

`new_unchecked` is typed over the storage `S`, **not** over a raw pointer; the pointer only
enters through the container `R`. So a bridge must call `Vec::from_raw_parts` itself and then
`DataRef::from_manually_drop` → `Storage::new` → `TensorView::new(_f)`/`new_unchecked`, exactly
as the faer impls do. `rstsr-book` records no ADR on external-buffer ingestion; ADR-0008 is the
only memory/ownership ADR (`adr-0008-...:10`), and `DLPack`/`interop`/`foreign` appear nowhere
in `rstsr-book` (see section G).

---

## B. Layout (rstsr-core / rstsr-common)

### B.1 Definition

`/home/a/rstsr_pack/rstsr/rstsr-common/src/layout/layoutbase.rs:15`

```rust
pub struct Layout<D>
where
    D: DimBaseAPI,
{
    pub(crate) shape: D,
    pub(crate) stride: D::Stride,
    pub(crate) offset: usize,
}
```

- `shape: D` where `D` is a dimension type — `D = [usize; N]` (aliases `Ix0`…`Ix9`) or
  `D = Vec<usize>` (`IxD`/`IxDyn`) — `layout/dim.rs:7-18`.
- **`stride: D::Stride` is `isize`-based**: `dim.rs:37` declares
  `type Stride: AsMut<[isize]> + AsRef<[isize]> + IndexMut<usize, Output = isize> + ...`;
  `dim.rs:60` `type Stride = [isize; N]` for `Ix<N>`; `dim.rs:84` `type Stride = Vec<isize>` for `IxD`.
- **`offset: usize`** — unsigned. Getter `pub fn offset(&self) -> usize` (`layoutbase.rs:53`);
  mutator `pub unsafe fn set_offset(&mut self, offset: usize) -> &mut Self` (`layoutbase.rs:75`,
  documented "We will not check whether this offset is valid").

`Layout` is plain integer data and is unconditionally `Send`/`Sync` (`layoutbase.rs:25-28`).
Getters: `shape()` `:41`, `stride()` `:47`, `offset()` `:53`, `ndim()` `:59`, `size()` `:65`.

Constructors:
- `layoutbase.rs:428` `pub fn new(shape: D, stride: D::Stride, offset: usize) -> Result<Self>`
  — validated: calls `bounds_index()?` then `check_strides(true)?` (`:435-436`).
- `layoutbase.rs:447` `pub unsafe fn new_unchecked(shape: D, stride: D::Stride, offset: usize)
  -> Self` — no validation.
- Order helpers on `DimLayoutContigAPI` (`layoutbase.rs:620`): `new_c_contig(offset)` `:623`,
  `new_f_contig(offset)` `:633`, `c()` `:643`, `f()` `:649`, `new_contig(offset, order)` `:654`.

### B.2 Negative strides — representable and used in practice

`isize` strides are not a formality: `flip` produces them.

- `/home/a/rstsr_pack/rstsr/rstsr-core/src/tensor/manipulation/flip.rs:20` —
  `layout = layout.dim_narrow(axis, slice!(None, None, -1))?;` inside `into_flip_f`
  (`flip.rs:6-25`).
- The negative-step branch of `dim_narrow` is
  `/home/a/rstsr_pack/rstsr/rstsr-common/src/layout/indexer.rs:171-198`: for `step < 0` it
  computes `let offset = (self.offset() as isize + stride[axis] * start) as usize;`
  (`indexer.rs:194`) and `stride[axis] *= step;` (`indexer.rs:196`), returning
  `Self::new(shape, stride, offset)`.
- `bounds_index` explicitly handles the negative-stride case:
  `layoutbase.rs:252-254` — `if stride[i] > 0 { max += stride[i]*(shape[i]-1) } else { min +=
  stride[i]*(shape[i]-1) }`, i.e. negative strides extend the *lower* bound.
- `check_strides` sorts by `stride[k].unsigned_abs()` (`layoutbase.rs:306`), so sign is ignored
  when testing for overlap.
- Other negative-stride handling: `layout/rearrangement.rs:48-53` ("revert negative strides if
  keep_shape is not required", using `dim_narrow(n as isize, slice!(None, None, -1))` at `:53`),
  `indexer.rs:557` test `l.dim_narrow(0, slice!(15, 5, -2))`.

**DLPack flag:** negative *strides* map fine (`isize` ↔ DLPack `int64` strides). The mismatch is
the **offset**: `bounds_index` raises `ValueOutOfRange` if the computed minimum index is
negative (`layoutbase.rs:258` `rstsr_pattern!(min, 0.., ValueOutOfRange)?`), which is consistent
with `offset: usize`. A DLPack tensor whose `byte_offset` is negative therefore **cannot** be
represented as a rstsr `Layout` offset; a bridge must fold the negative byte offset into the
base pointer it hands to `Vec::from_raw_parts` and pass `offset = 0`. (rstsr's own stride unit is
**elements**, not bytes — see `/home/a/rstsr_pack/rstsr-book/docs/numpy-cheatsheet.mdx:78`,
which notes exactly this units divergence against NumPy.)

### B.3 Zero strides (broadcast)

Zero strides are produced by broadcasting: `/home/a/rstsr_pack/rstsr/rstsr-common/src/layout/broadcast.rs:239`
`stride[i] = 0;` (inside `broadcast_layout`), and consumed at `broadcast.rs:266` where
`if self.stride[i] != 0 { ... }` distinguishes a materialised axis from a broadcast one.

They are **allowed**, under a gate: `check_strides(&self, skip_zero: bool)` skips the overlap
test for a zero-stride axis when `skip_zero` is true — `layoutbase.rs:283` signature,
`layoutbase.rs:289-291` the comment "if stride has zero value, `skip_zero` parameter will
determine whether this function will raise error or not", and `layoutbase.rs:333-335`
(`if stride_abs == 0 && skip_zero { continue; }`).

`TensorBase::new_f` calls it with `skip_zero = true` (`tensorbase.rs:197`), and `Layout::new`
likewise (`layoutbase.rs:436`). So **a zero-stride layout is accepted on any tensor, including
writable ones** — the aliasing risk it implies is handled separately, by
`DataForceMutAPI::force_mut`'s caller contract (`data.rs:311-320`) and by the `TensorMutable::
ToBeCloned` design (`tensor/tensor_mutable.rs:15-27`).

### B.4 Row-major / column-major order

Lives in `/home/a/rstsr_pack/rstsr/rstsr-common/src/flags.rs`:

```rust
// flags.rs:42
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FlagOrder {
    #[serde(rename = "RowMajor")] C = 101,   // row-major
    #[serde(rename = "ColMajor")] F = 102,   // column-major
}
// flags.rs:53-54  associated consts
pub const RowMajor: Self = FlagOrder::C;
pub const ColMajor: Self = FlagOrder::F;
// flags.rs:304-305  free-standing aliases
pub use FlagOrder::C as RowMajor;
pub use FlagOrder::F as ColMajor;
```

`Default` is feature-driven: `FlagOrder::F` iff `cfg!(feature = "col_major")`, else `FlagOrder::C`
(`flags.rs:58-64`), with the warning "F-prefer is not a stable feature currently! We develop only
in C-prefer currently" (`flags.rs:36-39`). The two features are mutually exclusive — a
`compile_error!` guard in `rstsr-common/src/lib.rs:28-30`.

`TensorIterOrder` (a separate, richer enum with `C`/`F`/`A`/`K`/`G`/`B` variants,
`flags.rs:75-115`) is a *policy* type, not the memory-order type.

Order is a **property of the device**, not of the layout — `DeviceBaseAPI::default_order()`
(`storage/device.rs:5`) — and the `c()`/`f()` layout constructors take no order argument.

---

## C. Dtype system (rstsr-dtype-traits + rstsr-core)

### C.1 No runtime dtype tag — compile-time traits only

There is **no** `DType`/`Dtype`/`DataType` enum anywhere in `rstsr-dtype-traits`, `rstsr-core` or
`rstsr-common`. The only `DType*` identifiers are traits:
`DTypeIntoFloatAPI` (`promotion.rs:16`), `DTypeCastAPI<T>` (`promotion.rs:21`),
`DTypePromoteAPI<T>` (`promotion.rs:25`). A bridge therefore **cannot ask a tensor what its
dtype is**; it must know `T` statically (or the bridge itself must define the runtime tag).

The crate is a small set of extension traits over concrete Rust primitives. Full module list:
`rstsr-dtype-traits/src/lib.rs:6-18` (`ext_float`, `ext_num`, `ext_real`, `isclose`, `promotion`,
`val_write`).

Traits:

| Trait | Location | Purpose |
|---|---|---|
| `ExtNum: Clone` | `ext_num.rs:5` | `ext_abs`, `ext_sign`, `ext_real`, `ext_imag`, `ext_conj`, `is_nan`; assoc. types `AbsOut`, consts `ABS_UNCHANGED`/`ABS_SAME_TYPE` |
| `ExtReal: Clone` | `ext_real.rs:5` | `ext_floor_divide`, `ext_min`/`ext_max`, `ext_min_value`/`ext_max_value` |
| `ExtFloat: Clone` | `ext_float.rs:2` | `ext_nextafter(self, other)` |
| `DTypePromoteAPI<T>` | `promotion.rs:25` | `type Res`, consts `SAME_TYPE`/`CAN_CAST_SELF`/`CAN_CAST_OTHER`, `promote_self`/`promote_other` |
| `DTypeCastAPI<T>` | `promotion.rs:21` | `into_cast(self) -> T` |
| `DTypeIntoFloatAPI` | `promotion.rs:16` | `type FloatType` |
| `ValWriteAPI<T>` | `val_write.rs:10` | |
| `IsCloseArgs<TE>` | `isclose.rs:21` | `isclose` configuration struct |

### C.2 The dtype enumeration macros (the dtype-mapping source of truth)

Each impl block is generated by `duplicate_item`, and those macro invocations *are* the
supported-type list:

`ExtNum` (`ext_num.rs`):
- unsigned: `ext_num.rs:65` `#[duplicate_item(T; [u8]; [u16]; [u32]; [u64]; [u128]; [usize];)]`
- signed: `ext_num.rs:103` `#[duplicate_item(T; [i8]; [i16]; [i32]; [i64]; [i128]; [isize];)]`
- floats: `ext_num.rs:149` `#[duplicate_item(T; [f32]; [f64];)]`
- half floats: `ext_num.rs:200-201` `#[cfg(feature = "half")] #[duplicate_item(T; [half::f16]; [half::bf16];)]`
- complex: `ext_num.rs:253` `#[duplicate_item(T; [Complex<f32>]; [Complex<f64>];)]`

`ExtReal` mirrors this without complex: `ext_real.rs:32` (u*), `:51` (i*), `:70` (f32/f64),
`:89-90` (half, feature-gated). `ExtFloat`: `ext_float.rs:8` (`f32`), `:14` (`f64`), `:21`/`:28`
(half, feature-gated).

**`bool`**: not in `ExtNum`/`ExtReal`, but it *is* a first-class tensor element — it participates
in promotion/cast (`promotion.rs:124-179`, the `impl_promotion_bool_T!` macro applied at
`:181-…`) and has scalar binary-operator impls (`tensor/operators/op_binary_arithmetic.rs:33`
`impl_arithmetic_scalar_lhs_bool!(bool);`). Bool tensors are real: comparison reductions produce
them (`tensor/reduction.rs:1004` `assert_eq!(a_all.raw(), &[true, true, false]);`).

**No string / object / datetime / void dtype exists.** The full element universe is therefore:
`u8 u16 u32 u64 u128 usize i8 i16 i32 i64 i128 isize f32 f64 Complex<f32> Complex<f64>` plus
`bool`, plus `f16`/`bf16` under the `half` feature.

`half` is an **optional, default-on** dependency of `rstsr-dtype-traits`:
`rstsr-dtype-traits/Cargo.toml:16` `half = { workspace = true, optional = true }`,
`:21` `defaults = ["half"]`, `:22` `half = ["dep:half"]`.

### C.3 What bounds an element type must satisfy in practice

These are two different questions with different answers:

- **Device level: unbounded.** `impl<T> DeviceAPI<T> for DeviceCpuSerial {}`
  (`device_cpu_serial/device.rs:87`) — no bound on `T` at all.
- **Creation/ops: `T: Num`.** e.g. `tensor/creation.rs:820`, `:844`, `:859`, `:1643`
  (`T: Num`), and `:874`/`:886`/`:901` (`T: Num + Clone + Send + Sync`).
- `Tensor<T, B, D>` itself only needs `B: DeviceRawAPI<T>` (`tensorbase.rs:43`).

`half::f16`/`bf16` do implement `num_traits::Num` (`/home/a/.cargo/registry/src/rsproxy.cn-e3de039b2554c837/half-2.7.1/src/num_traits.rs:95`
`impl Num for f16 {`), so they are usable in principle, but the survey found **no test that
builds an `f16`/`bf16` tensor** — their only appearances in `rstsr-core` are scalar-operator
impls (`op_binary_arithmetic.rs:952, 969-974`). Treat f16/bf16 as "declared, lightly exercised".

**DLPack mapping implication:** the bridge must do dtype dispatch on the *Rust generic parameter*
(monomorphised per `T`) or introduce its own runtime tag; DLPack's `DLDataType { code, bits,
lanes }` has no counterpart in this workspace today. Note also DLPack's `lanes` field: rstsr has
no SIMD-lane-carrying element type, and `pack_array` (`tensor/pack_array.rs`, `common/pack_array.rs`)
is a *separate* array-of-arrays abstraction, not a lane dimension.

---

## D. Device model (rstsr-core)

### D.1 The central trait family

All in `/home/a/rstsr_pack/rstsr/rstsr-core/src/storage/` (`mod.rs:3-6`), re-exported via
`prelude_dev.rs:29`:

```rust
// storage/device.rs:3
pub trait DeviceBaseAPI {
    fn same_device(&self, other: &Self) -> bool;
    fn default_order(&self) -> FlagOrder;
    fn set_default_order(&mut self, order: FlagOrder);
}

// storage/device.rs:9
pub trait DeviceRawAPI<T>: DeviceBaseAPI + Clone {
    type Raw;                              // the buffer container, e.g. Vec<T>
}

// storage/device.rs:23
pub trait DeviceStorageAPI<T>: DeviceRawAPI<T> {
    fn len<R>(storage: &Storage<R, T, Self>) -> usize where R: DataAPI<Data = Self::Raw>;
    fn is_empty<R>(storage: &Storage<R, T, Self>) -> bool where R: DataAPI<Data = Self::Raw> { .. }
    fn to_cpu_vec<R>(storage: &Storage<R, T, Self>) -> Result<Vec<T>> where ..;
    fn into_cpu_vec<R>(storage: Storage<R, T, Self>) -> Result<Vec<T>> where ..;
    fn get_index<R>(...) -> T; fn get_index_ptr<R>(...) -> *const T;
    fn get_index_mut_ptr<R>(...) -> *mut T; fn set_index<R>(...);
}

// storage/device.rs:159
pub trait DeviceAPI<T>: DeviceBaseAPI + DeviceRawAPI<T> + DeviceStorageAPI<T> + Clone + Default {}
```

Creation traits live in `storage/creation.rs`: `DeviceCreationAnyAPI<T>` (`:3`, with
`unsafe fn empty_impl(&self, len)`, `full_impl`, `outof_cpu_vec`, `from_cpu_vec`, `uninit_impl`,
`unsafe fn assume_init_impl`), `DeviceCreationNumAPI<T>` (`:36`, `zeros_impl`/`ones_impl`),
`DeviceCreationComplexFloatAPI<T>` (`:41`, `linspace_impl`), `DeviceCreationArangeAPI<T>`
(`:51`, `arange_impl`), `DeviceCreationTriAPI<T>` (`:55`, `tril_impl`/`triu_impl`).

Device transfer: `storage/conversion.rs:3` `pub trait DeviceChangeAPI<'l, BOut, R, T, D>` with
`change_device` / `into_device` / `to_device`.

### D.2 Is a device a ZST / does it carry runtime identity?

**Not a ZST — but also not a tagged device object.** The built-in CPU devices are small structs
whose fields are *behavioural configuration*, not identity:

```rust
// device_cpu_serial/device.rs:5
#[derive(Clone, Debug, Default)]
pub struct DeviceCpuSerial { default_order: FlagOrder }

// feature_rayon/device.rs:40
#[derive(Clone, Debug)]
pub struct DeviceCpuRayon { num_threads: usize, pool: Arc<ThreadPool>, default_order: FlagOrder }

// device_faer/device.rs:5
#[derive(Clone, Debug)]
pub struct DeviceFaer { base: DeviceCpuRayon }
```

There is **no `device_id`, no `device_type` discriminant, no `enum Device`**. The only identity
predicate is `DeviceBaseAPI::same_device(&self, other: &Self) -> bool`, which is field-wise:
`DeviceCpuSerial` compares `default_order` (`device_cpu_serial/device.rs:11-13`); `DeviceFaer`
compares thread count *and* default order (`device_faer/device.rs:47-51`).

**DLPack implication:** mapping to `DLDevice { device_type, device_id }` requires the bridge to
invent the tag (e.g. "everything that is not CPU is unsupported"; CPU ↔ `kDLCPU`), because
rstsr's device carries no such value. Conversely, an incoming `DLDevice` cannot be routed to a
rstsr `B` without a static decision.

### D.3 How the built-in CPU devices are named and constructed

- Default CPU alias, feature-switched — `rstsr-core/src/lib.rs:45-49`:

```rust
#[cfg(feature = "faer_as_default")]
pub type DeviceCpu = device_faer::device::DeviceFaer;
#[cfg(not(feature = "faer_as_default"))]
pub type DeviceCpu = device_cpu_serial::DeviceCpuSerial;
```

- `DeviceCpuSerial::default()` via `#[derive(Default)]` (`device_cpu_serial/device.rs:5`).
- `DeviceFaer::new(num_threads: usize) -> Self` (`device_faer/device.rs:12-16`);
  `Default for DeviceFaer` calls `DeviceFaer::new(0)` (`device_faer/device.rs:40-44`), and
  `0` means "use the current/global rayon thread count" (`feature_rayon/device.rs:60-72`).
- `DeviceCpuRayon::new(num_threads: usize)` builds its own rayon pool
  (`feature_rayon/device.rs:47-51`), documented at `:56-71`. It is deliberately *not* a full
  device: "This device is intended not to implement `DeviceAPI<T>`" (`feature_rayon/device.rs:39`).
- `DeviceRayonAutoImpl` is a **crate-internal alias**: `device_faer/device.rs:10`
  `pub(crate) use self::DeviceFaer as DeviceRayonAutoImpl;`.
- BLAS-backend devices, each `{ base: DeviceCpuRayon }`, defined in their own crate:
  `crates-device/rstsr-openblas/src/lib.rs` (`DeviceOpenBLAS`), `rstsr-mkl` (`DeviceMKL`),
  `rstsr-blis` (`DeviceBLIS`), `rstsr-aocl` (`DeviceAOCL`), `rstsr-kml` (`DeviceKML`). Each also
  does `pub(crate) use DeviceX as DeviceBLAS;` and `pub(crate) use DeviceX as DeviceRayonAutoImpl;`
  inside its own crate. They are surfaced to users under a single `DeviceBLAS` alias selected by
  exactly one backend feature (`rstsr/src/prelude.rs:42-81`). Note the naming: these are
  **CPU** devices with BLAS-accelerated kernels — "BLAS device" does not imply off-device memory.

### D.4 Devices are not `Send`/`Sync`-customised

No `unsafe impl Send/Sync` for any device is present (grep over
`feature_rayon/device.rs` and `device_cpu_serial/device.rs` returned nothing); they rely on
derived impls (`Arc<ThreadPool>` is `Send + Sync`).

### D.5 GPU support: none

A case-insensitive ripgrep for `cuda|hip|rocm|vulkan|metal|gpu|wgpu|opencl` over all `*.rs` and
`Cargo.toml` in the workspace produced **only false positives** — every hit was the substring
"hip" inside "ownership"/"relationship" in doc comments. There is no GPU crate, no GPU feature,
and no GPU device type. (Consistent with `rstsr-book/docs/why-rstsr.mdx:64` and
`docs/warning.mdx:18`, which describe GPU backends as a *future* intention, and with the design
blog's "Heterogeneous backends are not implemented" at
`rstsr-book/blog/2025-01-25-rstsr-second-report.mdx:745`.)

**DLPack consequence:** today a DLPack bridge is CPU-only, and `kDLCPU` is the only reachable
`device_type`. The `Device` trait design does leave room (a device is parameterised by `T` and
owns `type Raw`), but the DLPack `device_type`/`device_id` fields have no home in it.

---

## E. Creation & views — what a bridge crate can actually call

### E.1 Constructors

Free functions (all `Args`/`Inp`-overloaded, all with `_f` fallible twins):
`rt::asarray` (`tensor/asarray.rs:302`), `rt::zeros` (`tensor/creation.rs:2287`),
`rt::ones` (`:1624`), `rt::full` (`:1033`), `rt::arange` (`:139`), `rt::linspace` (`:1431`),
`rt::eye` (`:801`), `rt::tril` (`:2707`), `rt::triu` (`:2971`), plus the `*_like` family
(`zeros_like` `:2481`, `ones_like` `:1809`, `full_like` `:1225`).

Struct constructors: `TensorBase::new_unchecked` / `TensorAny::new_f` / `TensorAny::new`
— see A.6 for signatures.

### E.2 `asarray` overloads relevant to zero-copy

The complete overload list is `asarray.rs:323-889`. The ones a bridge cares about:

| Input | Output | Line |
|---|---|---|
| `(&'a [T], Layout<D>, &B)` | `TensorView<'a, T, B, IxD>` | `asarray.rs:519` |
| `(&'a [T], D, &B)` | `TensorView<'a, T, B, IxD>` | `asarray.rs:546` |
| `(&'a mut [T], Layout<D>, &B)` | `TensorMut<'a, T, B, IxD>` | `asarray.rs:693` |
| `(&'a mut [T], D, &B)` | `TensorMut<'a, T, B, IxD>` | `asarray.rs:721` |
| `(&'a Vec<T>, Layout<D>, &B)` | `TensorView<...>` | `asarray.rs:619` |
| `(Vec<T>, Layout<D>, &B)` | owned `Tensor<...>` (copyless) | `asarray.rs:448` |
| `(&TensorAny<R,T,B,D>, ...)` | owned `Tensor<T,B,D>` (copies) | `asarray.rs:323` |
| `(TensorView<'_, T, B, D>, ...)` | owned `Tensor<T,B,D>` | `asarray.rs:402` |

Note the shape/`D` overloads are **order-dependent**: they pick `.c()` or `.f()` from
`device.default_order()` (`asarray.rs:556-560`, `:731-735`), so a bridge that wants an exact
stride layout must use the `Layout<D>` overload, not the shape overload.

**Crucial gate:** the borrowed-slice overloads require `B: DeviceAPI<T, Raw = Vec<T>>`
(`asarray.rs:522`, `:696`). A bridge device must therefore expose `Raw = Vec<T>` — or the bridge
must copy the foreign buffer into a real `Vec<T>` first and lose zero-copy.

The `asarray` rustdoc itself frames this in bridge terms: the module documents that `&[T]` and
`&mut [T]` inputs yield views (`asarray.rs:78-91`, "Output mutable tensor view `TensorMut`").

### E.3 The `.raw()` / `as_raw` accessors — the "out" direction

There is no `as_raw()`. The accessible surface is:

- `TensorAny::raw(&self) -> &B::Raw` (`tensorbase.rs:226`) and `raw_mut(&mut self) -> &mut B::Raw`
  (`tensorbase.rs:230`, requires `R: DataMutAPI`). For a CPU tensor `B::Raw = Vec<T>`, so
  `tensor.raw().as_ptr()` is the buffer base.
- `TensorAny::as_ptr(&self) -> *const T` / `as_mut_ptr(&mut self) -> *mut T`
  — `tensor/ownership_conversion.rs:629` and `:636`, in an impl gated on
  `B: DeviceAPI<T, Raw = Vec<T>>`:

```rust
// ownership_conversion.rs:629
pub fn as_ptr(&self) -> *const T {
    unsafe { self.raw().as_ptr().add(self.layout().offset()) }
}
// ownership_conversion.rs:636
pub fn as_mut_ptr(&mut self) -> *mut T where R: DataMutAPI {
    unsafe { self.raw_mut().as_mut_ptr().add(self.layout().offset()) }
}
```

  **`as_ptr()` already adds the layout offset**, so the pair it forms with `layout()` is
  `(base + offset, offset)`; a bridge exporting DLPack must emit `data = as_ptr()` with
  `byte_offset = 0`, or `data = raw().as_ptr()` with the element offset scaled to bytes.
- `TensorAny::layout(&self) -> &Layout<D>` (`tensorbase.rs:129`), `shape()` (`:134`),
  `stride()` (`:139`), `offset()` (`:144`), `ndim()` (`:149`), `size()` (`:154`),
  `into_raw_parts(self) -> (S, Layout<D>)` (`tensorbase.rs:164`).
- `Storage::get_index_ptr` / `get_index_mut_ptr` (`storage/device.rs:118`/`:123`) give
  per-element raw pointers.

**Existing precedent of the full "out" direction**: `IntoFaer for TensorView<'a, T, B, Ix2>`
(`device_faer/conversion.rs:10-27`) computes
`ptr = unsafe { self.raw().as_ptr().add(offset) }` (`:22`) and forwards
`(nrows, ncols, row_stride, col_stride)` to `MatRef::from_raw_parts` (`:25`) — a foreign consumer
taking rstsr's pointer + element strides with no copy. `TensorViewMut → MatMut` is the mutable
twin (`:29-46`, `MatMut::from_raw_parts_mut` at `:44`).

### E.4 Summary: is zero-copy expressible today?

- **Zero-copy *in* (foreign buffer → rstsr view): YES**, with three executed precedents
  (`asarray` borrowed-slice, `IntoRSTSR for MatRef`, `IntoRSTSR for ColRef`/`MatMut`), the
  fabrication being `Vec::from_raw_parts(ptr, len, len)` → `ManuallyDrop` → `DataRef/DataMut::
  from_manually_drop` → `Storage::new` → `TensorView::new_f`/`new_unchecked`.
  **Requirements the caller must satisfy itself:** the pointer is valid for `'a`; `len` covers
  the layout's `bounds_index().1`; `B::Raw == Vec<T>`; **the buffer is never freed through the
  `Vec`** (guaranteed by `ManuallyDrop`).
- **Zero-copy *out* (rstsr tensor → ptr + layout): YES** via `as_ptr()`/`as_mut_ptr()` +
  `layout()`, or `raw().as_ptr()` + `layout()`.
- **What does NOT exist:** a `from_raw_parts(ptr, layout)` tensor constructor; a lifetime-free
  (owned) foreign buffer; any deallocation hook for memory rstsr did not allocate; a runtime
  dtype tag; a device tag.

---

## F. Feature / optional-dependency plumbing

### F.1 Manifest inventory

`find /home/a/rstsr_pack/rstsr -name Cargo.toml -not -path '*/target/*'` → **16 manifests**:

`Cargo.toml` (root, virtual — no `[package]`), `rstsr/Cargo.toml` (facade),
`rstsr-core`, `rstsr-common`, `rstsr-dtype-traits`, `rstsr-native-impl`, `rstsr-blas-traits`,
`rstsr-linalg-traits`, `rstsr-sci-traits`, `rstsr-test-manifest`, five device crates
(`crates-device/rstsr-{openblas,mkl,blis,aocl,kml}`), and `crates-plugin/rstsr-tblis`.
There is **no `crates-device/rstsr-faer`** — faer lives inside `rstsr-core` behind a feature.

Workspace members do **not** include `rstsr-test-manifest` (`Cargo.toml:3-18`), which is a
path-only helper crate with `publish = false` (`rstsr-test-manifest/Cargo.toml:4`).

### F.2 `rstsr-core/Cargo.toml` — `[dependencies]` and `[features]`

`[dependencies]` (`rstsr-core/Cargo.toml:13-24`):

```toml
13  [dependencies]
14  rstsr-common = { workspace = true }
15  rstsr-dtype-traits = { workspace = true, features = ["half"] }
16  rstsr-native-impl = { workspace = true }
17  num = { workspace = true }
18  itertools = { workspace = true }
19  half = { workspace = true }
20  rayon = { workspace = true, optional = true }
21  faer = { workspace = true, optional = true }
22  faer-ext = { workspace = true, optional = true }
23  derive_builder = { workspace = true }
24  duplicate = { workspace = true }
```

`[features]` (`rstsr-core/Cargo.toml:36-72`), verbatim (abridged only at `...`, marked):

```toml
36  [features]
37  default = ["row_major", "aligned_alloc", "faer", "faer_as_default"]
38  std = ["rstsr-common/std"]
39  backtrace = ["std", "rstsr-common/backtrace"]
40  rayon = ["dep:rayon", "rstsr-common/rayon", "rstsr-native-impl/rayon"]
41  faer = ["rayon", "dep:faer", "dep:faer-ext"]
42  faer_as_default = ["faer"]
...
48  row_major = ["rstsr-common/row_major"]
49  col_major = ["rstsr-common/col_major"]
...
54  aligned_alloc = ["rstsr-common/aligned_alloc"]
...
60  dispatch_dim_layout_iter = ["rstsr-common/dispatch_dim_layout_iter"]
...
66  [[test]]
67  name = "entry_row_cpu"
68  required-features = ["row_major"]
```

**Heaviestweight finding for a bridge crate:** `rayon`, `faer` and `faer-ext` are optional, but
`faer = ["rayon", ...]` (`:41`) and the **default feature set turns faer on**
(`:37` `default = [..., "faer", "faer_as_default"]`) — so the faer linear-algebra stack plus rayon
is the *default* build of `rstsr-core`. A bridge crate that depends on `rstsr-core` with default
features inherits it.

### F.3 Other crates' features (compact)

- `rstsr-common/Cargo.toml`: optional `rayon` (`:17`); mandatory `rstsr-cblas-base` (`:18`) and
  `serde` (`:19`). Features `:24-45`, and there is **no `default` key**: `std` (`:25`),
  `backtrace` (`:26`), `rayon` (`:27`), `row_major = []` (`:33`), `col_major = []` (`:34`),
  `aligned_alloc = []` (`:39`), `dispatch_dim_layout_iter = []` (`:45`).
- `rstsr-dtype-traits/Cargo.toml`: optional `half` (`:16`). Features `:20-22`:
  **`defaults = ["half"]` (`:21`)** — note the key is literally `defaults`, a typo for
  `default`, so this crate declares **no default feature at all** and `half` is instead pulled in
  explicitly by `rstsr-core` (`rstsr-core/Cargo.toml:15`, `features = ["half"]`). Verified by
  reading the whole file. `half = ["dep:half"]` (`:22`).
- `rstsr-native-impl/Cargo.toml`: optional `rayon` (`:16`); feature
  `rayon = ["dep:rayon", "rstsr-common/rayon"]` (`:19`); no default.
- `rstsr/Cargo.toml` (facade): optional `rstsr-linalg-traits` (`:17`), `rstsr-blas-traits`
  (`:18`), `rstsr-sci-traits` (`:19`), and the five device crates plus `rstsr-tblis` (`:23-29`).
  `default = ["std", "backtrace", "rstsr-core/default", "faer", "faer_as_default",
  "rstsr-openblas?/openmp"]` (`:34`) — the `?` means **no BLAS backend is enabled by default**.
- Device crates (`crates-device/rstsr-{openblas,mkl,blis,aocl,kml}/Cargo.toml`): the FFI binding
  crate is **mandatory**, not optional — e.g. `rstsr-openblas-ffi = { workspace = true }`
  (`rstsr-openblas/Cargo.toml:17`); only `rstsr-linalg-traits` (`:23`) and `rstsr-sci-traits`
  (`:24`) are optional. `default = ["linalg", "openmp"]` for openblas (`:31`),
  `default = ["linalg"]` for the rest.
- `crates-plugin/rstsr-tblis/Cargo.toml`: both `tblis` (`:14`) and `opt-einsum-path` (`:15`) are
  mandatory; feature only `dynamic_loading` (`:22-23`).
- `rstsr-test-manifest/Cargo.toml`: `publish = false` (`:4`), deps `npyz` + `num` (`:13-15`),
  no features. `npyz` is a **pure-Rust `.npy` reader**, not a Python binding.

### F.4 Workspace-wide dependency table

`[workspace.dependencies]` is `Cargo.toml:30-82`. Bridge-relevant entries: all in-tree crates are
`version = "0.9.0", default-features = false` (`:31-50`); `half = { version = "2.7",
default-features = false, features = ["alloc", "num-traits"] }` (`:61`); `num = { version = "0.4",
..., features = ["alloc", "libm"] }` (`:59`); `unsafe`-adjacent dev-only helpers `npyz = { version
= "0.8", features = ["complex"] }` (`:75`) and `criterion` (`:80`).

### F.5 MSRV

Root declaration: `Cargo.toml:20` `[workspace.package]`, `Cargo.toml:23`
`rust-version = "1.82.0"`. `rg -n 'rust-version' --glob 'Cargo.toml'` returns **16 hits**: that
one declaration plus **15 × `rust-version.workspace = true`**, one per crate
(`rstsr/Cargo.toml:5`, `rstsr-core:5`, `rstsr-common:5`, `rstsr-dtype-traits:5`,
`rstsr-native-impl:5`, `rstsr-blas-traits:5`, `rstsr-linalg-traits:5`, `rstsr-sci-traits:5`,
`rstsr-test-manifest:6`, all five device crates `:5`, `crates-plugin/rstsr-tblis:5`).
**No crate overrides the MSRV**; the whole workspace is 1.82.0.
(By contrast, the sibling `dlpack-ffi` declares `rust-version = "1.64.0"`
(`/home/a/rstsr_pack/dlpack-ffi/Cargo.toml:7`), so it imposes no MSRV obstacle on a bridge crate.)

### F.6 Python / NumPy / DLPack dependency check — clean, but with a notable trace

Exact commands and results:

**1.** `rg -n -i 'pyo3|cpython|PyCapsule|capsule' /home/a/rstsr_pack/rstsr --glob '!target'`
→ **no output (0 matches, rg exit 1)**. No PyO3, no CPython embedding, no capsule handling.

**2.** `rg -n -i 'dlpack|DLManagedTensor|DLDevice|DLDataType' /home/a/rstsr_pack/rstsr --glob '!target'`
→ **3 matches, all documentation; zero code, zero dependencies.** Verified independently:

- `/home/a/rstsr_pack/rstsr/rstsr-core/src/tensor/creation.rs:11` — `//! - [ ] ~\`from_dlpack\`~`
  — a **struck-through checklist entry** in the creation-module doc header (surrounded by
  `- [x] empty`, `- [x] eye`, `- [x] full` at `:8-13`). `~…~` marks it as explicitly *not*
  implemented.
- `/home/a/rstsr_pack/rstsr/rstsr-core/src/docs/array_api_standard.md:130` — the array-API
  compliance table row `| D | | [\`from_dlpack\`](…) | Returns a new array containing the data
  from another (array) object with a \`__dlpack__\` method. |`. Status **`D` = dropped**, and the
  "RSTSR counterpart" column is **empty** — i.e. rstsr made a deliberate decision to have no
  `from_dlpack`.
- `/home/a/rstsr_pack/rstsr/rstsr-core/src/docs/array_api_standard.md:379` — the rationale:
  "**Functions related to Python Array API namespace and dlpack.** These routines are mostly for
  forcing other python packages to be compatible to Python Array API. **This is not possible for
  another language currently.**"

  This is the single most important *intent* finding in the survey: the DLPack question was
  already posed, classified as a Python-interop concern, and closed as out-of-scope **on the
  grounds that cross-language interop was not achievable**. Any bridge plan must argue against
  this recorded position, and note that the technical premise ("not possible for another
  language") is exactly what the pure types-only `dlpack-ffi` crate now contradicts.
- No `dlpack-ffi` dependency appears in any `Cargo.toml` or in `Cargo.lock`.

**3.** `rg -n -i 'numpy' /home/a/rstsr_pack/rstsr --glob '!target' --glob '!*.csv'`
→ 614 matches across 82 `.rs`, 17 `.py`, 8 `.md`, 2 `.toml`. All benign, in four categories:
- The **only two `.toml` hits are comments**, not deps: `rstsr-core/Cargo.toml:46` and
  `rstsr-common/Cargo.toml:31`, both
  `# - Row-major convention: similar to NumPy (with same behavior of versatile broadcasting)`.
- `.rs` hits are parity-test module names (`mod numpy_arange`, `static FUNC: &str = "numpy_arange"`
  at `rstsr-core/tests/core_func/creation/test_arange.rs:10-12`) and doc comments citing NumPy
  semantics (e.g. `rstsr-common/src/layout/slice.rs:3`). Token histogram: `NumPy` 414,
  `numpy` 331, remainder module names (`numpy_reshape`, `numpy_argmax`, …).
- `.py` hits are **out-of-band dev/CI scripts**, not build dependencies: 15 BLAS validation
  scripts under `crates-device/*/tests/…` (`# # Driver tests in Python` + `import numpy as np` +
  `import scipy`, e.g. `crates-device/rstsr-openblas/tests/linalg_func/func_validation_f64.py:3-4`),
  plus `rstsr-test-manifest/resources/gen_rand_vec.py:1` which generates `.npy` fixtures.
- No crate named numpy/pyo3/dlpack exists: `rg -n -i 'pyo3|cpython|numpy|dlpack' Cargo.lock`
  matches only `ndarray` (pure-Rust rust-ndarray) and `npyz` (pure-Rust `.npy` reader).

**4.** `rg -n -w -i 'python' /home/a/rstsr_pack/rstsr --glob '!target'`
→ 113 matches across 22 `.rs`, 15 `.py`, 7 `.md`, 1 `.csv`. `.rs` hits are the "Python array API
standard" doc citations (e.g. `rstsr-core/src/tensor/creation.rs:3`), commented-out Python
reference snippets (`rstsr-core/src/format/format_tensor.rs:337`), and one self-description
comment (`rstsr-core/src/lib.rs:4` `// This option is for myself as python-like developer.`).
No Python API calls exist in Rust.

**5.** A fifth datapoint worth recording: `rstsr-core` *does* read NumPy's **on-disk file format**,
but only in a non-core helper — `rstsr-test-manifest` depends on `npyz` (`Cargo.toml:14`) and
loads fixtures via `npyz::NpyFile::new(&bytes[..]).unwrap()` (`src/lib.rs:31`). Core deliberately
opts out: `rstsr-core/tests/CONTEXT.md:62-66` states rstsr-test-manifest is "**Out of scope for
rstsr-core**: core parity tests use `tensor_from_nested!` for small tensors and `rt::asarray` with
a Rust `Vec` otherwise. No `.npy`, no python regen step in core."

**Conclusion:** there is **no Python-interop code of any kind** in the workspace. Python appears
only as (a) reference semantics in docs, (b) out-of-band CI scripts that validate the BLAS device
crates, (c) a stdlib-only NumPy-tracking maintenance script (`rstsr-core/tests/tracking/sync_numpy.py`,
imports `argparse, ast, hashlib, sys` — no NumPy), and (d) the pure-Rust `npyz` reader in the
non-core test-manifest helper. The negative is clean.

---

## G. External references (rstsr-book, campaign corpus, sibling experiments)

### G.1 rstsr-book dev docs and ADRs

`/home/a/rstsr_pack/rstsr-book/dev` contains 14 files: the ADR set `adr-0000`…`adr-0008`
(`adr/adr-0000-charter.mdx` … `adr/adr-0008-uninitialized-allocation-contract.mdx`), plus
`agent-workflow.mdx`, `dev-install.mdx`, `doc-guide.mdx`, `ffi-header-update.mdx`.

**DLPack search: zero hits.** `rg -n -i 'dlpack|DLManagedTensor|DLDevice|DLDataType'
/home/a/rstsr_pack/rstsr-book` (excluding `debug/`, `target/`, `.git/`) → **no matches, rg exit 1**.
DLPack is never mentioned anywhere in the book, including the i18n translations, `docs/`, and
`blog/`.

**Interop / Python / NumPy hits (all parity- or migration-oriented, none a runtime bridge):**

| Path:line | What it says |
|---|---|
| `dev/adr/adr-0003-three-artifact-faithfulness-model.mdx:10-16` | ADR 0003 "Three-artifact NumPy-faithfulness model"; parity tests use "Commented NumPy Python acts as the reference oracle" (`:14`) |
| `dev/adr/adr-0004-creation-from-tensor-category.mdx:15-17,29-34` | Broadens the NumPy-parity audit to `creation_from_tensor`; unimplemented NumPy functions tracked as `todo` |
| `dev/adr/adr-0005-axis-index-errors.mdx:12-16` | `AxisError`/`IndexError` adopted for NumPy parity ("message-only, like Python's") |
| `dev/adr/adr-0008-uninitialized-allocation-contract.mdx:10, 40-48, 58-65, 84` | The only memory/ownership ADR; blesses 3 allocation patterns, has a "Pattern 2 — POD FFI buffers for BLAS/LAPACK" section, and rejects migrating FFI buffers to `MaybeUninit` |
| `dev/adr/adr-0002-entry-binary-test-matrix.mdx:12,16` | Per-(order, device) entry binaries; "The structure is future-ready for `DeviceOpenBLAS` and friends" |
| `dev/dev-install.mdx:52-54` | NumPy installed **only** to generate `.npy` test resources (`python gen_rand_vec.py`) |
| `dev/dev-install.mdx:93` | "The FFI project of RSTSR, which contains binding code for C/C++ libraries" |
| `dev/agent-workflow.mdx:44,87` | NumPy/SciPy/array-api reference checkouts used by the *test* skills |
| `docs/for-numpy.mdx:14-16` | "one of RSTSR's key design goals was to provide a programming experience somewhat similar to NumPy within the native Rust environment" — migration UX, not interop |
| `docs/numpy-cheatsheet.mdx:70,78` | Maps `a.ctypes.data` → raw Rust pointer; **`:78` notes stride units differ (elements vs bytes)** |
| `docs/why-rstsr.mdx:64`; `docs/warning.mdx:18` | "left interfaces for other backends, hoping to implement GPU backends" (future intention) |
| `blog/2025-01-25-rstsr-second-report.mdx:215-221` | "Why Not Consider Using PyTorch's Rust Binding?" — **rejects** `tch-rs` ("PyTorch's runtime is too large ... better to treat PyTorch as an optional backend"); at `:753` "perhaps we can make PyTorch our backend in the future"; at `:745` "Heterogeneous backends are not implemented ... should consider CUDA and HIP" |
| `blog/2024-09-18-rstsr-first-report.mdx:49` | Cites PySCF's Python/C bindings as prior art |

**Negative results, explicitly:** no `pyo3`, no `arrow` (the only "arrow" hit is an npm
`@babel/plugin-transform-arrow-functions` entry in `package-lock.json`), no `zero-copy`/`zero
copy`, no `interop`, no "foreign buffer", and **no plan or proposal anywhere in rstsr-book to
create Python bindings for RSTSR**. The only Python↔RSTSR route ever discussed is the *rejected*
PyTorch-binding idea.

**Takeaway:** rstsr-book has an explicit allocation contract (ADR-0008) but **no ADR on storage
abstraction, external-buffer ingestion, ownership transfer across an FFI boundary, or
cross-language bridging**. That design space is unoccupied — which is good news for a bridge
plan (nothing to undo) and bad news for precedent (nothing to copy).

### G.2 The 2026-09 campaign's original prompt

Directory `/home/a/rstsr_pack/rstsr-improve-trajectory/2026-09-08-plan-prompt/` holds
`initial-prompt.md`, `260908-plan-cpu-serial-efficiency.md`, `260909-campaign-outcome.md`,
`260911-integration-progress.md`, `machine-and-tools.md`, `rstsr-386948b-code-map.md`.

`rg -n -i 'numpy|python|bridge|dlpack|literature|interop' .` run inside that directory: **zero
hits for `dlpack`, `literature`, and `interop`**; the single `bridge` hit is
`rstsr-386948b-code-map.md:141` "Parallel iteration bridged by `ParIterRSTSR`" (rayon plumbing,
unrelated); all NumPy/Python hits are benchmark/oracle references.

The campaign was purely **CPU runtime-efficiency work on rstsr itself** at commit `386948be`
(`initial-prompt.md:3,5-7`: "make a plan on possible efficiency improvement on rstsr
(386948be...)", scope "rstsr-core and rstsr-common, serial part (DeviceCpuSerial) and general
parallel part (DeviceFaer)"), with SIMD dispatch and optimising transpose/vecdot/reduce
(`initial-prompt.md:10-14`). NumPy appears only as an external oracle — e.g.
`260908-plan-cpu-serial-efficiency.md:56` "numpy via `conda activate torch` (einsum for vecdot,
`.T.copy()`, `.sum(axis)`)" — and Python only as the conda tool runner
(`machine-and-tools.md:33-34`). **No Python/NumPy runtime interop, no bridge, no DLPack, and no
literature review appear in that campaign directory.** The "Python bridge" theme does **not**
originate there; it is new with the 2026-10-03 work.

### G.3 The sibling experiment: `dlpack-ffi` (2026-10-03)

Directory `/home/a/rstsr_pack/rstsr-improve-trajectory/2026-10-03-dlpack-ffi/`:
`README.md`, `REPORT.md`, `PLAN.md`, `initial-prompt.md`, `round-1-questions.md`,
`round-1-answers.md`, `round-2-questions.md`, `round-2-answers.md`, `dlpack-ffi.patch`,
`rstsr-agents.patch`. (There is no `SUMMARY.md`; `README.md`/`REPORT.md` play that role.)

**Mission** (`README.md:9-14`): "Redesign and upgrade the `RESTGroup/dlpack-ffi` crate
(dmlc/dlpack bindings): raw types-only C-ABI surface whose API survives DLPack minor-version enum
additions without breaking (bindgen `NewType` style, vendored v1.3 header, generator script,
provenance), plus an `update-ffi-dlpack` skill". Status: delivered 2026-10-03 (`README.md:7`,
`REPORT.md:73-80`).

**What the crate now provides** (crate root `/home/a/rstsr_pack/dlpack-ffi/`):

- **Exactly two newtype enums**, `#[repr(transparent)]` integer wrappers with associated `kDL*`
  constants — chosen so future DLPack minor releases cannot break the API
  (`readme.md:7`: "a `#[repr(transparent)]` integer wrapper with associated constants, so that
  values added by future DLPack minor releases are representable without changing the API";
  `REPORT.md:16`):
  - `DLDeviceType` — `src/lib.rs:58` `pub struct DLDeviceType(pub c_uint)` (repr at `:56`),
    constants `kDLCPU = 1 … kDLTrn = 18` at `src/lib.rs:24-54`.
  - `DLDataTypeCode` — `src/lib.rs:109` (repr at `:106`), constants `kDLInt = 0 …
    kDLFloat4_e2m1fn = 17` at `src/lib.rs:70-104`.
  - Generated with `--default-enum-style newtype`: `scripts/bindgen.py:45`.
- **Vendored header is DLPack v1.3**: `header/dlpack.h:19` `#define DLPACK_MAJOR_VERSION 1`,
  `:22` `#define DLPACK_MINOR_VERSION 3`; provenance "vendored from tag `v1.3`, commit `84d107b`"
  (`readme.md:5`, `REPORT.md:14`).
- **Exported surface** (one generated module; `src/lib.rs:4` is the only `use`, and the crate has
  no functions): `DLPACK_MAJOR_VERSION`/`MINOR` (`src/lib.rs:8-12`), `DLPackVersion` (`:16`),
  `DLDevice` (`:62`), `DLDataType` (`:113`), `DLTensor` (`:124`), `DLManagedTensor` (`:143`),
  **`DLManagedTensorVersioned`** (`:154`), the producer typedefs
  `DLPackManagedTensorAllocator` (`:167`), `DLPackManagedTensorFromPyObjectNoSync` (`:182`),
  `DLPackDLTensorFromPyObjectNoSync` (`:186`), `DLPackCurrentWorkStream` (`:189`),
  `DLPackManagedTensorToPyObjectNoSync` (`:197`), and the v1.3 exchange structs
  `DLPackExchangeAPIHeader` (`:206`) / `DLPackExchangeAPI` (`:215`).
  ABI sizes are asserted in `tests/abi.rs:19-21` (`DLTensor` 48, `DLManagedTensor` 64,
  `DLManagedTensorVersioned` 80) and newtype widths at `:26-29`.
- **v1.3 protocol coverage:** the versioned structs and the exchange API *are* exported
  (`src/lib.rs:154`, `:206`, `:215`), including `DLPackExchangeAPIHeader.version:
  DLPackVersion` (`:208`). But `__dlpack__` / `__dlpack_c_exchange_api__` appear **only inside
  doc comments copied from the C header** (`src/lib.rs:212`; `header/dlpack.h:534-536`) — the
  crate implements no Python-level surface. Negative checks: `kDLConsumer`,
  `legacy_interface`, `UsedDLPackVersion`/`used_dlpack_version`, `__dlpack_device__` → each 0
  hits in `src/lib.rs` + `header/dlpack.h`.
- **Crate metadata:** `Cargo.toml:3` `version = "1.3.0"`, `:6` `edition = "2021"`, `:7`
  `rust-version = "1.64.0"`; `[lib] doctest = false` (`:13-16`); and — the load-bearing fact —
  **the `[dependencies]` section is empty** (`Cargo.toml:18-19`), with no dev-dependencies.

**THE PURITY CONSTRAINT (confirmed, verbatim).** The `dlpack-ffi` crate must remain a *pure*
types-only crate with zero rstsr references, so **bridge glue cannot live in it**:

- `2026-10-03-dlpack-ffi/round-2-answers.md:11` (owner's verbatim answer): "**dlpack-ffi should
  be stay as pure, and not mention "rstsr" or "rest" in dlpack-ffi for any useful purpose.**"
- Same file `:20`: "**dlpack-ffi itself stays pure** — no "rstsr"/"rest" references in its
  content, docs, code, or metadata."
- `/home/a/rstsr_pack/rstsr-agents/skills/update-ffi-dlpack/SKILL.md:13-15`: "**Purity rule**:
  never add "rstsr"/"rest" references to the dlpack-ffi repository (content, docs, code,
  metadata). This skill lives in rstsr-agents and operates on the crate from outside."
- `2026-10-03-dlpack-ffi/PLAN.md:25` (Purity row) and `REPORT.md:38` (verification gate
  "`grep -rin rstsr` over the tree → empty"); `REPORT.md:67-69` "No rstsr-core changes; no
  rstsr-ffi changes; no CI changes; no new dependencies; no `build.rs`."
- Types-only rationale: `SKILL.md:8-11` "Its header declares C types only - no functions - so
  there is no symbol coverage to check"; `/home/a/rstsr_pack/dlpack-ffi/readme.md:5` "**We do not
  add any safe-wrappers or abstractions on top of the FFI layer.**"; `readme.md:9` "It serves as
  a minimal binding layer for users who want to build their own safe abstractions on top of the
  DLPack FFI."
- Independent verification: `grep -rn 'rstsr\|RSTSR' /home/a/rstsr_pack/dlpack-ffi
  --exclude-dir=.git --exclude-dir=target --exclude-dir=debug` → **exit 1, no matches** (the only
  case-insensitive "rest" hit is the org URL `https://github.com/RESTGroup/dlpack-ffi` at
  `Cargo.toml:8`).
- **Nut exempt, but worth knowing:** the crate's hand-written **test suite** does touch tensor
  memory — `tests/ownership.rs:32-60` builds a `DLManagedTensor` over a Rust `Vec<f32>`, reads
  elements through `dl_tensor.data`, and calls the deleter exactly once (versioned variant
  `:63-104`). `readme.md:7` acknowledges "the only hand-written code in the crate is the test
  suite under `tests/`". So "types-only" means *no library code beyond the generated bindings*,
  not "no code at all".

**Where the consumer work was explicitly deferred to:** `SKILL.md:75-77` — an end-to-end interop
check "against `numpy.from_dlpack` needs a consumer-side harness (e.g. pyo3 or a ctypes-loaded
cdylib); it is **out of scope for this generated-only crate** and remains a manual/future check."
Same decision at `PLAN.md:23` ("numpy interop ignored/manual (Q5 c)"), `round-1-answers.md:21`
(consumer work is a separate future task) and `round-1-answers.md:22` (the rstsr-core consumer
task "stays separate/future"). **That deferred consumer crate is precisely what the present
review is planning.**

**Consequence for the bridge plan:** the bridge glue (DLPack ↔ rstsr conversion) must live in a
*new* crate outside `dlpack-ffi` — either in `rstsr` proper or in a new sibling (e.g. an
`rstsr-dlpack`-style crate) that depends on `dlpack-ffi` (C types only) and on `rstsr-core`.
`dlpack-ffi`'s empty `[dependencies]` and the purity rule make it permanently unusable as a
home for any rstsr-aware code.

---

## H. Test / bench infrastructure relevant to a future binding crate

### H.1 Entry-binary pattern

`/home/a/rstsr_pack/rstsr/rstsr-core/tests/` top level holds three test files
(`allocatable_dtype.rs`, `entry_row_cpu.rs`, `tensor_sum.rs`), `CONTEXT.md`, and the directories
`core_func/`, `doc_draft/`, `test_issues/`, `test_utils/`, `tracking/`.

There is **exactly one entry binary today**, `entry_row_cpu.rs`. Verbatim
(`/home/a/rstsr_pack/rstsr/rstsr-core/tests/entry_row_cpu.rs:1-19`):

```rust
// Entry binary: row-major, DeviceCpuSerial.
// See ADR-0002 (entry-binary test matrix) and skill `test-conventions`.
// Gated to `row_major` via [[test]] required-features in rstsr-core/Cargo.toml.
mod core_func;
mod doc_draft;
mod test_issues;
mod test_utils;

pub use rstsr::prelude::*;
pub use std::sync::LazyLock;
pub use test_utils::TestCfg;

pub use DeviceCpuSerial as DeviceType;

pub static TESTCFG: LazyLock<TestCfg<DeviceType>> = LazyLock::new(|| {
    let mut device = DeviceType::default();
    device.set_default_order(RowMajor);
    TestCfg::init(device, vec![], None)
});
```

Wiring (`rstsr-core/Cargo.toml:62-68`):

```toml
62  # Integration test binaries. See ADR-0002 (entry-binary test matrix).
63  # row_major / col_major are compile-time mutually exclusive, so gating each entry
64  # binary on its feature keeps row/col entries from colliding.
65  # Only DeviceCpuSerial row-major is wired now; entry_row_faer / entry_col_cpu follow.
66  [[test]]
67  name = "entry_row_cpu"
68  required-features = ["row_major"]
```

So `entry_row_faer` / `entry_col_cpu` are **stated future work** (`:65`), and only the
`DeviceCpuSerial` × row-major cell of the matrix exists.

The conventions are codified in `/home/a/rstsr_pack/rstsr/rstsr-core/tests/CONTEXT.md`:
"**Entry binary**: A top-level `tests/entry_<order>_<device>.rs` that becomes one cargo test
binary. Defines `type DeviceType` and `static TESTCFG`, then `mod`-includes the body." (`:35-39`);
"**Body** (shared body): Device-agnostic test modules under `tests/` (`core_func/`, `doc_draft/`,
`test_issues/`) holding the test logic. `mod`-included by every entry binary. **Designed to be
symlinked by device crates.**" (`:30-33`). Runtime gating is by the `specify_test!` macro
(`tests/test_utils/mod.rs:58`; documented at `CONTEXT.md:57-60`). CI:
`.github/workflows/rstsr-core-test.yml:31` `run: cargo test -p rstsr-core --test "*" --release`.

Category modules are plain re-exports — `tests/core_func/mod.rs:1-8` lists `creation`,
`creation_from_tensor`, `indexing`, `linalg`, `manipulation`, `math`, `operators`, `reduction`.

### H.2 Where NumPy-parity tests live

- `/home/a/rstsr_pack/rstsr/rstsr-core/tests/core_func/` — **55 `.rs`** = 46 `test_*.rs` + 9
  `mod.rs`. Per category: creation 7, creation_from_tensor 8, indexing 1, linalg 3,
  manipulation 10, math 1, operators 2, reduction 14. This is the parity tree; per
  `CONTEXT.md:11-14` a "**Parity test** ... Lives in `core_func/`. Inherently row-major (NumPy is
  C-order)."
- `/home/a/rstsr_pack/rstsr/rstsr-core/tests/doc_draft/` — **35 `.rs`** = 25 `test_*.rs` + 10
  `mod.rs` (API-docstring examples, `CONTEXT.md:41-43`).
- `/home/a/rstsr_pack/rstsr/rstsr-core/tests/test_issues/` — **4 `.rs`** (`mod.rs`, `issue_7.rs`,
  `issue_77.rs`, `issue_83.rs`) — one regression test per ticket.
- `/home/a/rstsr_pack/rstsr/rstsr-core/tests/test_utils/` — **2 `.rs`** (`mod.rs`, `equality.rs`).
- `/home/a/rstsr_pack/rstsr/rstsr-core/tests/tracking/` — **0 `.rs`**; `doc_coverage.csv`,
  `numpy_coverage.csv`, `numpy_differences.md`, `numpy_differences_resolved.md`, `sync_numpy.py`.

Representative parity-test header (`tests/core_func/creation/test_arange.rs:1-19`) shows the
provenance convention — `#[cfg(test)] mod numpy_arange`, `static FUNC: &str = "numpy_arange";`,
and a per-test comment citing the exact NumPy version/file/line the case was transcribed from
(`:16-17` "NumPy v2.5.2, _core/tests/test_multiarray.py, TestArange::test_start_stop_kwarg (line
10580)").

### H.3 Benchmarks: none

`rg -n '\[\[bench\]\]|criterion|iai' --glob 'Cargo.toml' /home/a/rstsr_pack/rstsr` returns
**exactly two hits, both dependency declarations**: `rstsr-core/Cargo.toml:33`
`criterion = { workspace = true }` and root `Cargo.toml:80` `criterion = { version = "0.5" }`.
No `[[bench]]` target section exists anywhere (the only target section in the workspace is
`[[test]]` at `rstsr-core/Cargo.toml:66`), no `benches/` directory exists, and `iai` appears
nowhere. `criterion` and `cpu-time` are declared dev-deps but **unused** — presumably staged for
the perf campaign. A binding crate would therefore have to bring its own bench harness.

### H.4 Cross-language / FFI tests: none in rstsr-core

`rg -n 'extern "C"|link\(name|ffi|FFI|openblas|mkl|cblas'` over `rstsr-core/tests` and
`rstsr-core/src` returns only `extern crate alloc` declarations (e.g. `storage/data.rs:1`) — no
`extern "C"`, no `#[link]`, no BLAS. The only feature gating in core tests is
`required-features = ["row_major"]` (`rstsr-core/Cargo.toml:68`). And core's dev-dependency on the
facade is deliberately backend-less: `rstsr-core/Cargo.toml:27`
`rstsr = { path = "../rstsr", default-features = false }`.

FFI testing lives in the device crates instead: they link C OpenBLAS at build time
(`crates-device/rstsr-openblas/build.rs:81` `println!("cargo:rustc-link-lib=openblas");`, gated
on `RSTSR_DEV`, `build.rs:1-6`), pair Rust tests with NumPy/SciPy `.py` validation scripts, and
load `.npy` golden data through `npyz` (`rstsr-test-manifest/src/lib.rs:31`).

### H.5 Is there a natural home for a Python-interop test?

**Not inside `rstsr-core`.** Three structural reasons, each citable:

1. Core's tests are *device-agnostic bodies* `mod`-included by an *entry binary* that fixes one
   `DeviceType` (`tests/CONTEXT.md:30-39`). A Python-interop test needs a Python interpreter and
   a `cdylib`/`pyo3` boundary — it is neither device-agnostic nor expressible as a plain
   `#[test]` in that pattern.
2. Core's dev-dependency on the facade is `default-features = false` (`rstsr-core/Cargo.toml:27`),
   deliberately keeping heavy backends out of the core test build.
3. Core already declined NumPy's on-disk format (`tests/CONTEXT.md:62-66` "No `.npy`, no python
   regen step in core"), so the project has an explicit precedent for keeping Python out of
   `rstsr-core/tests/`.

The natural pattern to copy is the **device crates'**: a sibling crate with its own
`tests/` entry binary, an optional feature gate, and out-of-band `.py` validation scripts invoked
from a GitHub workflow (`.github/workflows/rstsr-openblas-test.yml:27` `python gen_rand_vec.py`).
That matches the `dlpack-ffi` experiment's own conclusion
(`skills/update-ffi-dlpack/SKILL.md:75-77`: a consumer-side harness "e.g. pyo3 or a ctypes-loaded
cdylib ... is out of scope for this generated-only crate and remains a manual/future check").
**A new sibling crate (e.g. `rstsr-dlpack`) is the structurally consistent home**, not
`rstsr-core` and not `dlpack-ffi`.

### H.6 Adding a crate to the workspace

There is **no CONTRIBUTING.md** at the repo root, and the readme has no "how to add a crate"
section (`rg -n -i 'new crate|add.*crate|workspace|members' readme.md CONTRIBUTING.md AGENTS.md`
→ no output). The mechanism is the root `members` list (`Cargo.toml:1-18`), verbatim:

```toml
1   [workspace]
2   resolver = "2"
3   members = [
4       "rstsr",
5       "rstsr-core",
6       "rstsr-common",
7       "rstsr-dtype-traits",
8       "rstsr-blas-traits",
9       "rstsr-linalg-traits",
10      "rstsr-native-impl",
11      "rstsr-sci-traits",
12      "crates-device/rstsr-openblas",
13      "crates-device/rstsr-mkl",
14      "crates-device/rstsr-blis",
15      "crates-device/rstsr-aocl",
16      "crates-device/rstsr-kml",
17      "crates-plugin/rstsr-tblis",
18  ]
```

Note the two existing organisational precedents: `crates-device/` (backend implementations that
carry a mandatory `rstsr-*-ffi` dependency) and `crates-plugin/` (e.g. `rstsr-tblis`, mandatory
`tblis` + `opt-einsum-path`). A bridge crate fits either convention — but note there is **no
`exclude` key** and no `[patch]` section in the root manifest, and no crate today depends on a
repo outside this workspace except through published crates.io versions
(`Cargo.toml:52-57`, the `rstsr-*-ffi` crates). A bridge depending on the out-of-tree
`dlpack-ffi` crate would be the **first** workspace member to do so.

---

## Bridge-relevant gaps — what does NOT exist today

Concrete list of things a NumPy ↔ rstsr DLPack bridge would have to build from scratch, because
nothing in this workspace provides them:

1. **No foreign-buffer storage type.** A tensor can borrow a buffer rstsr did not allocate only
   by fabricating a *non-owning* `Vec<T>` (`ManuallyDrop` + `DataRef/DataMut::from_manually_drop`,
   `storage/data.rs:98`/`:172`), used by `asarray` (`asarray.rs:536`, `:711`) and the faer
   `IntoRSTSR` impls (`device_faer/conversion.rs:69`, `:151`, `:178`). There is no `ForeignBuffer`,
   no custom-allocator `Storage`, no `Drop` hook, and — by the crate's own analysis
   (`conversion.rs:85-94`) — **ownership of a foreign allocation cannot be adopted soundly**,
   because freeing it through a `Vec<T>` would deallocate with the wrong size/alignment. Any
   DLPack consumer that takes ownership must therefore implement the deleter protocol itself and
   keep the memory out of `Vec`'s allocator path.
2. **No `(ptr, layout)` tensor constructor.** No `TensorView::from_raw_parts`. Building a view
   from a pointer requires three manual steps plus `unsafe`:
   `Vec::from_raw_parts(ptr, len, len)` → `Data{Ref,Mut}::from_manually_drop(ManuallyDrop::new(..))`
   → `Storage::new(data, device)` → `TensorView::new_f(storage, layout)` or
   `TensorBase::new_unchecked(storage, layout)` (`tensorbase.rs:115`, `:195`). No lifetime-erased
   (owned) variant exists either.
3. **No runtime dtype tag.** `rstsr-dtype-traits` is compile-time extension traits only
   (`ext_num.rs:5`, `ext_real.rs:5`, `ext_float.rs:2`, `promotion.rs:16/21/25`). There is no enum
   to match a `DLDataType` against; a bridge must define its own tag and a
   `match tag -> monomorphised conversion` dispatch over the fixed element set
   (`u8..u128/usize/i8..i128/isize/f32/f64/Complex<f32>/Complex<f64>` + `bool` + feature-gated
   `f16/bf16`).
4. **No `lanes`/item-size metadata** on the element side — DLPack's `DLDataType.bits` and
   `.lanes` have no counterpart; `bits` is recoverable via `size_of::<T>() * 8` only inside a
   monomorphised context, and `lanes > 1` is unrepresentable.
5. **No device tag.** Devices are config structs, not tagged objects
   (`DeviceCpuSerial { default_order }` `device_cpu_serial/device.rs:6`; `DeviceFaer { base }`
   `device_faer/device.rs:6`). No `device_type`, no `device_id`; `same_device` is field-wise
   (`device_cpu_serial/device.rs:11`, `device_faer/device.rs:47`). Mapping to
   `DLDevice { device_type, device_id }` has to be invented. Only `kDLCPU` is reachable — there
   is **no GPU device** anywhere (grep for cuda/hip/rocm/vulkan/metal/gpu/wgpu/opencl: false
   positives only).
6. **No capsule handling.** Zero occurrences of `PyCapsule`, `capsule`, `pyo3`, `cpython`
   anywhere in the workspace; `dlpack-ffi` deliberately implements no Python surface either
   (`src/lib.rs:212` is a doc comment only). The `__dlpack__`/`__dlpack_device__` protocol and
   the `PyCapsule` named `"dltensor"` / `"used_dltensor"` renaming dance must be built by the
   bridge crate (or a Python-side shim).
7. **Offset cannot be negative.** `Layout::offset: usize` (`layoutbase.rs:22`) with
   `bounds_index` rejecting `min < 0` (`layoutbase.rs:258`). A DLPack tensor with a negative
   `byte_offset` must be normalised by folding it into the base pointer. Also note the **unit
   mismatch**: rstsr strides are in *elements* (`[isize]`, `dim.rs:37`), DLPack strides are in
   *bytes* (`book/docs/numpy-cheatsheet.mdx:78`), so `byte_stride / itemsize` needs a
   divisibility check.
8. **No order/stride normalisation contract with NumPy.** `FlagOrder` is a *device* property
   (`storage/device.rs:5`) that silently selects `.c()` vs `.f()` in the shape-based `asarray`
   overloads (`asarray.rs:556-560`); a bridge that wants byte-for-byte NumPy semantics must
   always pass an explicit `Layout<D>` and never rely on the default order.
9. **Zero-stride (broadcast) layouts are accepted and writable.** `TensorBase::new_f` calls
   `check_strides(true)` (`tensorbase.rs:197`), so a zero-stride layout passes validation, and
   broadcast genuinely produces `stride[i] = 0` (`broadcast.rs:239`). A DLPack exporter of such
   a tensor is legal (DLPack allows 0 strides), but a *consumer* writing through it would alias;
   rstsr leaves that to `DataForceMutAPI::force_mut`'s caller contract (`data.rs:311-320`).
10. **No ADR or design doc for external buffers / cross-language ownership.** `rstsr-book`
    contains zero DLPack/interop/foreign-buffer material; ADR-0008
    (`adr-0008-uninitialized-allocation-contract.mdx:10`) is the only memory-ownership ADR and it
    covers *rstsr's own* allocation, not imported memory. A bridge plan should probably propose
    a new ADR, since this territory is unclaimed.
11. **There is an adverse recorded decision to argue against.** `from_dlpack` was explicitly
    evaluated and **dropped**, with a written rationale:
    `rstsr-core/src/docs/array_api_standard.md:130` marks it status `D` (dropped) with an empty
    RSTSR-counterpart column, and `:379` states "Functions related to Python Array API namespace
    and dlpack ... **This is not possible for another language currently.**"
    `rstsr-core/src/tensor/creation.rs:11` carries the struck-through item `~from_dlpack~`. A
    bridge plan is, in effect, a proposal to reverse this. The technical premise it rested on is
    now falsified by the types-only `dlpack-ffi` crate — but the reversal should be explicit.
12. **No cross-language test home yet.** Core's tests are device-agnostic bodies included by a
    single `DeviceType`-fixing entry binary (`rstsr-core/tests/CONTEXT.md:30-39`), its facade
    dev-dep is `default-features = false` (`rstsr-core/Cargo.toml:27`), and it has already
    declined NumPy's `.npy` format (`CONTEXT.md:62-66`). The structural precedent to follow is the
    device crates' (sibling crate + entry binary + out-of-band `.py` scripts invoked from a
    workflow, e.g. `.github/workflows/rstsr-openblas-test.yml:27`). See H.5.
13. **No benchmark harness to reuse.** `criterion` is declared (`rstsr-core/Cargo.toml:33`, root
    `Cargo.toml:80`) but there is **no `[[bench]]` target and no `benches/` directory** anywhere;
    `iai` is absent. Zero-copy/overhead claims for a bridge would need a harness built from
    scratch.
14. **The one asset that does exist:** `/home/a/rstsr_pack/dlpack-ffi` v1.3.0 — a pure
    types-only crate with newtype enums, the vendored v1.3 header, `DLManagedTensorVersioned`
    and `DLPackExchangeAPI` exported, an **empty `[dependencies]`**, and an explicit purity rule
    (`skills/update-ffi-dlpack/SKILL.md:13-15`) forbidding any rstsr reference inside it. Bridge
    glue must therefore live in a *separate* crate that depends on it. Two mechanical notes for
    that crate: (a) it would be the **first** workspace member to depend on an out-of-tree
    rstsr-flavoured crate (today every non-workspace dep is a crates.io release,
    `Cargo.toml:52-57`), and (b) `dlpack-ffi`'s MSRV is 1.64.0
    (`/home/a/rstsr_pack/dlpack-ffi/Cargo.toml:7`) versus the workspace's 1.82.0 — no conflict,
    since the workspace floor is the higher one.

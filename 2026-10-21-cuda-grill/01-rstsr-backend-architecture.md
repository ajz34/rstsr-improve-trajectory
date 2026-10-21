# RSTSR backend architecture — comprehension document for a CUDA backend

**Analyzed commit:** `rstsr` @ `1c3bc11d874e8f196cda4ca71fab5065920b1a7f` (workspace version `0.10.3`,
`/scratch/rstsr_pack/rstsr`). All citations are `path:line` relative to that repository unless the path
starts with `rstsr-agents/` (shared rules) or is a sibling repo; module paths such as
`tensor/...`, `storage/...`, `operators/...`, `feature_rayon/...` and `device_cpu_serial/...` abbreviate
`rstsr-core/src/<path>`.
**Method:** read-only source inspection; no cargo builds were run. Everything below is verified against
the tree at the commit above, not against released docs.
**Audience:** maintainers preparing a `crates-device/rstsr-cuda` backend (see
`2026-10-21-cuda-grill/initial-proposal.md`).

---

## 0. Architecture primer — only the device-integration specifics

The general concepts (tensor = storage × layout, ownership aliases, naming conventions, `_f` twins,
tuple-style overloads) are documented in `rstsr-agents/rules/code-concepts.md`; they are not repeated
here. What matters for a new backend is the seam structure:

```
TensorAny<R, T, B, D> = TensorBase<Storage<R, T, B>, D>          rstsr-core/src/tensorbase.rs:27-32
Storage<R, T, B>      = { data: R, device: B, PhantomData<T> }    rstsr-core/src/storage/device.rs:14-22
R : DataAPI<Data = B::Raw>                                        rstsr-core/src/storage/data.rs:296-302
```

Three consequences drive everything in this document:

1. **The device is a value stored in every tensor.** `B` is cloned into each `Storage` on view
   creation (`rstsr-core/src/tensor/ownership_conversion.rs:56-63`, `:102-113`) and in
   `change_device` (`rstsr-core/src/storage/conversion.rs:13-20`). There is no global device registry
   in core. A `DeviceCuda { ordinal, stream, flags }` therefore travels with the data — which is
   exactly the multi-GPU shape we want.
2. **Element storage is an associated type of the device, not a `Vec<T>`.** `DeviceRawAPI<T>` declares
   `type Raw;` (`rstsr-core/src/storage/device.rs:9-11`) and every device op is written as
   `&<Self as DeviceRawAPI<T>>::Raw` / `&mut <Self as DeviceRawAPI<MaybeUninit<TOut>>>::Raw`
   (e.g. `rstsr-core/src/operators/ops/op_ternary_arithmetic.rs:21-47`). On paper this is
   device-agnostic memory.
3. **Evaluation is eager, but op-by-op.** Every `rt::` function performs its kernel immediately
   through `&self` device methods and returns an owned `Tensor` (see the canonical shape in
   `rstsr-core/src/tensor/operators/op_binary_common.rs:180-187`: allocate `uninit_impl` →
   `device.op_*(...)` → `assume_init_impl`). Nothing in core inspects results, so an async CUDA
   launch can return without synchronizing as long as the *device* side queues and orders work on its
   stream. Fusion (lazy graphs) does not exist and would have to be built inside the device type.

A backend crate implements the device traits for a local device type; Rust coherence permits this for
every trait in core because the device type is local (including conversions keyed on another device,
where the local type appears in the trait's parameter list). The exception is the BLAS trait layer,
whose supertraits pin `Raw = Vec<T>` (see §9, row R10).

---

## 1. Workspace map

The workspace (`/scratch/rstsr_pack/rstsr/Cargo.toml:1-33`) contains 28 member crates in five groups.
Core is `#![cfg_attr(not(test), no_std)]` (`rstsr-core/src/lib.rs:1`); `row_major` / `col_major` are
mutually exclusive compile-time features (`rstsr-common/src/lib.rs:28-29`, `rstsr-core/Cargo.toml:47-52`);
`rayon` gates the whole parallel layer (`rstsr-core/src/lib.rs:31-32`, `feature_rayon/mod.rs`), and
`faer` implies `rayon` (`rstsr-core/Cargo.toml:38`). `DeviceCpu` is `DeviceFaer` under
`faer_as_default`, otherwise `DeviceCpuSerial` (`rstsr-core/src/lib.rs:37-41`).

| Crate | Path | Role | Feature gates that matter to a backend |
|---|---|---|---|
| `rstsr-core` | `rstsr-core/` | `TensorBase`, `Storage`, data reprs, all device/operator traits, `rt::` core functions, reference devices `DeviceCpuSerial` and `DeviceFaer`, rayon auto-impl bodies | `row_major`/`col_major` (exclusive), `rayon`, `faer`, `faer_as_default`, `aligned_alloc`, `dispatch_dim_layout_iter`, `std`/`backtrace` |
| `rstsr-common` | `rstsr-common/` | `RSTSRError`/`Result`, `Layout`/dim types (`Ix0..Ix6`, `IxD`), flags (`FlagOrder`, `TensorIterOrder`), layout iterators, aligned alloc, `par_iter` | `row_major`/`col_major`, `rayon`, `std` |
| `rstsr-dtype-traits` | `rstsr-dtype-traits/` | scalar/dtype traits: `DTypeIntoFloatAPI`, `DTypeCastAPI`, `DTypePromoteAPI`, `ExtNum`/`ExtReal`/`ExtFloat`/`ExtZero`/`ExtSortCmp`/`ExtComplexFloat`, `IsCloseArgs`, `ValWriteAPI`; f16/bf16 impls | `half` (defaulted; `rstsr-core` enables it, `rstsr-core/Cargo.toml:15`) |
| `rstsr-blas-traits` | `rstsr-blas-traits/` | scalar traits `BlasScalar`/`BlasFloat`, 20 public BLAS/LAPACK traits (14 individual drivers + aggregates), `BlasThreadAPI`, `driver_impl/` sources (symlinked into device crates), `blas_tests_impl/` | `ilp64` |
| `rstsr-linalg-traits` | `rstsr-linalg-traits/` | `LinalgAPI` family (19 ops), result/args structs, generic reference implementations over `LapackDriverAPI`; per-device code in `src/blas_impl/` (symlinked) and `src/faer_impl/` | `faer` |
| `rstsr-sci-traits` | `rstsr-sci-traits/` | `CDistAPI` (+~20 metrics), Lebedev quadrature; per-device impls for `DeviceCpuSerial`/`DeviceFaer`/`DeviceRayonAutoImpl` | `faer` |
| `rstsr-native-impl` | `rstsr-native-impl/` | CPU kernels: `cpu_serial/` and `cpu_rayon/` (feature `rayon`); consumed by core and device crates | `std`, `rayon` |
| `crates-device/rstsr-{openblas,mkl,blis,aocl,kml}` | `crates-device/*` | BLAS/LAPACK backends; each device is `DeviceCpuRayon` + FFI, plus symlinked `rayon_auto_impl/`, `driver_impl/`, `linalg_auto_impl/`, `sci_auto_impl/` | per-crate `linalg`, `sci`, `faer`, `dynamic_loading`, `ilp64`, `openmp`, `lapacke` |
| `crates-plugin/rstsr-tblis` | `crates-plugin/rstsr-tblis/` | einsum/tensordot via TBLIS (CPU, `dynamic_loading`) | `dynamic_loading` |
| `crates-interop/rstsr-cpu-dlpack` | `crates-interop/rstsr-cpu-dlpack/` | DLPack import/export; defines a foreign `DataDlpack` data repr; CPU only | `row_major`/`col_major`, `std`, `half` |
| `crates-interop/rstsr-faer-py/*` | 13 crates | pyo3 array-API binding of `DeviceFaer`; dtype-erased `AnyTensor` handle | — |
| `rstsr` | `rstsr/` | facade: re-exports core + member crates as `rt::`, cheat-sheet and API-spec doc pages | `openblas`/`mkl`/`blis`/`aocl`/`kml`, `linalg`, `sci`, `tblis`, `dlpack`, `faer` |
| `rstsr-test-manifest` | `rstsr-test-manifest/` | dev-only: `.npy` resources for driver conformance tests | — |

---

## 2. Full backend trait surface

A device is "functional" when it implements the core families below. The reference implementation is
`device_cpu_serial/` (90 distinct trait names / 102 `impl` sites; many are `#[duplicate_item]` blocks
that expand to dozens more, e.g. the 36 unary kernels), plus `device_faer/` and the symlinked rayon
bodies (`feature_rayon/auto_impl/`, 20 files) for the parallel variant. Sizes below are trait counts,
not method counts.

### (a) Foundational — device, raw storage, creation, conversion

| Trait | Where | Ops/methods | CPU reference |
|---|---|---|---|
| `DeviceBaseAPI` (`same_device`, `default_order`, `set_default_order`) | `storage/device.rs:3-7` | 3 | `device_cpu_serial/device.rs:10-22` |
| `DeviceRawAPI<T> { type Raw; }` | `storage/device.rs:9-11` | 1 assoc. type | `device_cpu_serial/device.rs:24-26` (`Raw = Vec<T>`) |
| `DeviceStorageAPI<T>` (len, is_empty, to_cpu_vec, into_cpu_vec, get_index, get/set ptr, set_index) | `storage/device.rs:23-54` | 7 required + 1 default | `device_cpu_serial/device.rs:28-85` |
| `DeviceAPI<T>` marker (+`Clone + Default`) | `storage/device.rs:159` | — | `device_cpu_serial/device.rs:87` |
| `DeviceCreationAnyAPI<T>` (empty, full, outof_cpu_vec, from_cpu_vec, uninit, assume_init) | `storage/creation.rs:3-34` | 6 | `device_cpu_serial/creation.rs:5-60` |
| `DeviceCreationNumAPI<T>` (zeros, ones) | `storage/creation.rs:36-39` | 2 | `device_cpu_serial/creation.rs:62-75` |
| `DeviceCreationComplexFloatAPI<T>` (linspace) | `storage/creation.rs:41-49` | 1 | `device_cpu_serial/creation.rs:77-90` |
| `DeviceCreationArangeAPI<T>` (arange) | `storage/creation.rs:51-53` | 1 | `device_cpu_serial/creation.rs:92+` |
| `DeviceCreationTriAPI<T>` (tril, triu) | `storage/creation.rs:55-62` | 2 | `device_cpu_serial/operators/op_tri.rs` |
| `DeviceChangeAPI<'l, BOut, R, T, D>` (change/into/to_device) | `storage/conversion.rs:3-21` | 3 + 2 assoc. types | `device_cpu_serial/conversion.rs:3-40` (relabel only) |
| `DeviceRayonAPI` (thread-pool control) | `feature_rayon/device.rs:6-32` | 4 | `feature_rayon/device.rs:98-129` |
| Aggregates `DeviceNumAPI<T,D>`, `DeviceComplexFloatAPI<T,D>` | `operators/combined_trait.rs:6-55` | bound bundles | one empty impl each per device |

### (b) Elementwise

| Family | Traits | Count | CPU reference |
|---|---|---|---|
| Unary math (sin/exp/log/sqrt/abs/conj/real/imag/signbit/ceil/…, plus `OpPositiveAPI`) | `Op*API` generated in `operators/ops/op_binary_common.rs:3-58`, `:117-129` | 36 + 1 | `device_cpu_serial/operators/op_binary_common.rs` |
| Binary arithmetic as `c = a op b` (add/sub/mul/div/rem/bit ops/shift) | `operators/ops/op_ternary_arithmetic.rs:3-48` | 10 traits × 3 arg forms | `device_cpu_serial/operators/op_ternary_arithmetic.rs:1-70` |
| In-place / consuming variants (`a op= b`; `a op b` consuming left/right) | `operators/ops/op_binary_arithmetic.rs:3-109` | 10 + 10 + 10 | `device_cpu_serial/operators/op_binary_arithmetic.rs` |
| `Neg`, `Not` | `operators/ops/op_binary_arithmetic.rs:90-109` | 2 | idem |
| Ternary common (atan2, copysign, comparisons, pow, hypot, logaddexp, maximum, nextafter, ext_* arithmetic) | `operators/ops/op_ternary_common.rs:3-64` | 24 | `device_cpu_serial/operators/op_ternary_common.rs` |
| `isclose` | `operators/ops/op_ternary_common.rs:76-111` | 1 | idem |
| `where(cond,x,y)` | `operators/ops/op_quaternary_common.rs:12-52` | 1 (3 methods) | `device_cpu_serial/operators/op_quaternary_common.rs` |
| `clip(x, lo, hi)` | `operators/ops/op_quaternary_common.rs:67-113` | 1 (4 methods) | idem |
| Assignment (`assign`, `assign_uninit`, `fill`, `assign_arbitary*`) | `operators/assignment.rs:5-53` | 2 | `device_cpu_serial/assignment.rs` |
| Closure ops (`Op_MutC_RefA_RefB_API`, … 7 traits) | `operators/ops/op_with_func.rs:7-129` | 7 | `device_cpu_serial/operators/op_with_func.rs` |

### (c) Reduction

| Family | Traits | Count | CPU reference |
|---|---|---|---|
| `sum/min/max/prod/mean/var/std/l2_norm/argmin/argmax/nanargmin/nanargmax/all/any/count_nonzero` (`func_all` + `func(axes)`) | `operators/reduction.rs:3-35` | 15 | `device_cpu_serial/reduction.rs` (29 impl sites) |
| Output-dtype reductions (`sum/prod/mean/var/std` with explicit `TOut`) | `operators/reduction.rs:37-57` | 5 | idem |
| `unraveled_argmin/argmax` | `operators/reduction.rs:59-79` | 2 | idem |
| `OpSumBoolAPI` | `operators/reduction.rs:81-94` | 1 | idem |
| `OpAllCloseAPI` (`allclose` → host `bool`) | `operators/reduction.rs:96-119` | 1 | idem |
| `OpReduceCustomAPI` (host closures init/fold/combine/finalize) | `operators/reduction.rs:128-166` | 1 | idem |
| Cumulative scans (`cumsum`, `cumprod`, + dtype variants) | `operators/reduction.rs:170-236` | 4 | idem |

`sum_axes(None)` does **not** use the axes kernel: it calls `sum_all` (host scalar) and re-uploads it
(`tensor/reduction.rs:2443-2450`; same pattern at `:581, :693, :781, :1567, :1639, :2219, :2350`).

### (d) Manipulation, indexing, searching, sorting, set

| Trait | Where | Ops | CPU reference |
|---|---|---|---|
| `DeviceIndexSelectAPI` | `operators/adv_indexing.rs:8-24` | 1 (host `&[usize]`) | `device_cpu_serial/adv_indexing.rs` |
| `DeviceTakeAlongAxisAPI` | `operators/adv_indexing.rs:33-50` | 1 | `device_cpu_serial/take_along_axis.rs` |
| `DevicePutAlongAxisAPI` | `operators/adv_indexing.rs:60-77` | 1 | idem |
| `DeviceIndexPutAPI` | `operators/adv_indexing.rs:86-100` | 1 (host `&[usize]`) | `device_cpu_serial/adv_indexing.rs` |
| `DeviceMaskIndexAPI` (mask_select, mask_fill) | `operators/adv_indexing.rs:109-137` | 2 | idem |
| `DeviceArrayIndexAPI` (fancy gather; `ArrayAuxIndexer` carries **device** index storage) | `operators/adv_indexing.rs:150-191` | 1 | `device_cpu_serial/array_indexing.rs` |
| `DeviceArrayIndexAssignAPI` (fancy scatter) | `operators/adv_indexing.rs:207-223` | 1 | idem |
| `OpNonzeroAPI` (count + fill) | `operators/searching.rs:187-204` | 2 | `device_cpu_serial/nonzero.rs` |
| `OpSearchSortedAPI` (host `sorter: Option<&[usize]>`) | `operators/searching.rs:167-181` | 1 | `device_cpu_serial/searching.rs` |
| `OpSortAPI`, `OpArgSortAPI`, `OpSortCustomAPI` | `operators/sorting.rs:68-133` | 4 (incl. 2 closure forms) | `device_cpu_serial/sorting.rs` |
| `OpUniqueAPI` (values / all) | `operators/set.rs:16-43` | 2 | `device_cpu_serial/set.rs` |
| `OpIsinAPI` | `operators/set.rs:48-61` | 1 | idem |
| `OpPackTriAPI`, `OpUnpackTriAPI` | `operators/ops/op_tri.rs:3-30` | 2 | `device_cpu_serial/operators/op_tri.rs` |

Layout-only manipulation (`reshape`/`transpose`/`slice`/`broadcast`/`tile`/`repeat`/`roll`) has **no**
device trait — it is layout algebra plus `assign`/creation (`rstsr-core/src/tensor/manipulation/`).
That part generalizes to any backend for free.

### (e) Linear algebra (core tier, always available)

| Trait | Where | Ops |
|---|---|---|
| `DeviceMatMulAPI` (matmul + matmul_uninit), `DeviceGEMMAPI`, `DeviceSYMMAPI`, `DeviceSYRKAPI`, `DeviceHERKAPI`, `DeviceGEMVAPI` (gemv/gevm), `DeviceInnerDotAPI` | `operators/matmul.rs:7-160` | 7 traits / 8 methods |
| `DeviceVecdotAPI`, `DeviceOuterAPI`, `DeviceCrossAPI`, `DeviceTensordotAPI` | `operators/linalg.rs:3-86` | 4 |
| dtype-promoting variants `DeviceExtMatMulAPI`, `DeviceExtVecdotAPI`, `DeviceExtOuterAPI`, `DeviceExtCrossAPI`, `DeviceExtTensordotAPI` | `operators/ext_linalg.rs:16-157` | 5 |

Optional tiers: `rstsr-blas-traits` adds 20 public traits — 14 individual drivers (GEMM/TRSM/SYHEMM +
SYEV(D)/SYGV(D)/POTRF/GESV/GETRF/GETRI/SYSV/GESVD/GESDD), 3 aggregate bound traits, and 3
scalar/threading traits (`rstsr-blas-traits/src/trait_def.rs:1-76`,
`src/threading.rs`); `rstsr-linalg-traits` adds the 19-op `LinalgAPI` family
(`rstsr-linalg-traits/src/traits_def.rs:8-97`) whose implementations are compiled *inside* the device
crate from symlinked sources.

### (f) Misc

`DeviceChangeAPI` (above), `DeviceStorageAPI::to_cpu_vec`/`into_cpu_vec` (host materialization),
`DeviceRayonAPI` (only if the rayon bodies are reused — a CUDA crate does not need it),
`DeviceStorageAPI::get_index`/`set_index` (host scalar access used by `to_scalar` and fancy-index
resolution).

### Scope estimate

- **Core-functional device (MVP):** creation + assignment + unary/binary/ternary elementwise +
  reduction + core matmul (`DeviceMatMulAPI`/`GEMM`) → roughly 60 of the ~90 reference trait sites.
  Anything not implemented simply fails at the user's call site (traits are per-op bounds), so the
  backend degrades gracefully.
- **Full core parity:** all families above, i.e. ~90 distinct traits (≈150–200 macro-expanded impl
  blocks) plus per-dtype kernel variants. This is the "one to two person-months of trait plumbing"
  order of magnitude, before kernel work.
- **Facade `rt::` surface:** free functions plus associated methods comprise 223 panicking `pub fn`
  names and 181 `_f` twins (`grep` under `rstsr-core/src/tensor`), nearly all routed through the
  traits above. Member-crate surfaces (`rt::linalg`, `rt::sci`, `rt::tblis`, `rt::dlpack`) are
  **optional** facade features (`rstsr/Cargo.toml:49-71`, `rstsr/src/prelude.rs:126-175`).
- **Reuse:** a CUDA crate **cannot** reuse the rayon bodies (`feature_rayon/auto_impl/*` operate on
  `Vec<T>` directly, e.g. `feature_rayon/auto_impl/op_binary_common.rs:39-51`). It must supply its own
  kernels, but can reuse the *symlink + crate-local type alias* mechanism (§8).

---

## 3. Device object model

**Definitions.**

- `DeviceCpuSerial { default_order: FlagOrder }` — a small stateful struct (`FlagOrder` only), `Clone +
  Debug + Default` (`rstsr-core/src/device_cpu_serial/device.rs:5-8`).
- `DeviceCpuRayon { num_threads: usize, pool: Arc<ThreadPool>, default_order: FlagOrder }`
  (`rstsr-core/src/feature_rayon/device.rs:40-45`) — the shared base for all parallel CPU devices.
- `DeviceFaer { base: DeviceCpuRayon }` (`rstsr-core/src/device_faer/device.rs:5-10`);
  `DeviceOpenBLAS { base: DeviceCpuRayon }` (`crates-device/rstsr-openblas/src/lib.rs:22-25`), and the
  same shape for MKL/BLIS/AOCL/KML. All are **stateful, heap-owning, cloneable handles** — cloning a
  device clones an `Arc`, not a pool.

**Trait bounds on the device.**

```
DeviceBaseAPI                       same_device / default_order / set_default_order   storage/device.rs:3-7
DeviceRawAPI<T>: DeviceBaseAPI + Clone                                                storage/device.rs:9-11
DeviceStorageAPI<T>: DeviceRawAPI<T>                                                  storage/device.rs:23-54
DeviceAPI<T>: DeviceBaseAPI + DeviceRawAPI<T> + DeviceStorageAPI<T> + Clone + Default storage/device.rs:159
```

There is **no** `Send`/`Sync`/`PartialEq`/`Debug` bound on `DeviceAPI`; `PartialEq` does not exist on
devices at all (`same_device` is the comparison). `Debug` is required only by the `Debug` impl of
tensors (`format/format_tensor.rs:271`). `Default` is required globally. `Clone` is required because
each tensor owns a device value.

**Global/static state.**

- Core: only `MIN_PRINT`/`MAX_PRINT` (`format/format_tensor.rs:5-6`) and the process-wide
  `DEFAULT_TENSOR_ITER_ORDER: AtomicU8` (`rstsr-common/src/flags.rs:120`, set via
  `ChangeableDefault::change_default`). Neither is device state.
- Device crates: `rstsr-openblas` keeps `static INTERNAL: Mutex<OpenBLASConfig>` and implements
  `with_blas_num_threads` as *set-global → run → restore* (`crates-device/rstsr-openblas/src/threading.rs:100-133`).
  This is a cautionary precedent, not a model: the mutex serializes the two writes but not the scope,
  so concurrent devices can interleave.
- No TLS device state anywhere in core.

**Multiple instances, concurrent threads.** Yes, the design supports it. Each `Storage` owns its device
clone (`storage/device.rs:14-22`), ops dispatch on that clone (`tensor/operators/op_binary_common.rs:181-183`),
and cross-operand mixing is gated by `same_device` (`tensor/operators/op_binary_common.rs:162`; 20+
sites). Two `DeviceCuda { ordinal: 0 }` and `{ ordinal: 1 }` therefore coexist, are stored per tensor,
and can be used from different `std::thread`s provided the device type is `Send + Sync` (a tensor is
`Send`/`Sync` iff its data repr is — `tensorbase.rs:336-353`, `storage/data.rs:44-60`; note `Storage`
is an ordinary struct, so it additionally inherits auto-traits from `B`).

Two design notes for CUDA:

1. `same_device` must encode **ordinal** (and, as a policy, the stream). `DeviceFaer` compares
   `num_threads` *and* order (`device_faer/device.rs:46-51`); copying that pattern gives
   `same_device = ordinal == ordinal && stream_id == stream_id && order == order`. If streams are
   excluded, mixed-stream operands pass the check but need explicit event waits.
2. `set_default_order(&mut self)` mutates only that device value; since each tensor carries its own
   clone, changing a device after tensor creation does not retroactively affect existing tensors.
   This is existing behaviour (CPU included), not a GPU issue, but it means "set the order once on the
   device you construct tensors with".

---

## 4. Storage & raw memory — the key question

### 4.1 What `Raw` is today

Every existing device sets `type Raw = Vec<T>`:

- `DeviceCpuSerial`: `rstsr-core/src/device_cpu_serial/device.rs:24-26`
- `DeviceFaer`: `rstsr-core/src/device_faer/device.rs:62-64`
- `DeviceOpenBLAS`/`MKL`/`BLIS`/`AOCL`/`KML`: `crates-device/rstsr-openblas/src/device.rs:55-57` and
  the identical `device.rs` of each sibling crate.

Note the type is parameterized by `T`: `Raw` for `MaybeUninit<T>` is `Vec<MaybeUninit<T>>`, a distinct
associated type obtained through `DeviceRawAPI<MaybeUninit<T>>` (`storage/creation.rs:17-33`). A CUDA
backend must satisfy the same relation; a `CudaBuffer<T>` (pointer + len + allocator handle +
`PhantomData<T>`) works, and is layout-identical for `T` and `MaybeUninit<T>`.

### 4.2 Every place the core tensor layer assumes host-addressable Raw

`Vec`-specific `Raw = Vec<T>` bounds: 55 occurrences across core, the optional trait crates, and the reference impls themselves:

| Area | Sites | What it provides |
|---|---|---|
| Element iteration | `tensor/iterator_elem.rs:65, 168, 186, 204, 292, 424, 555, 613, 634` | `iter`, `iter_mut`, `indexed_iter*` yield `&T`/`&mut T` from `DataRef<Vec<T>>::as_slice_ref` (`storage/data.rs:130-153`) and `DataMut<Vec<T>>::into_slice_mut` (`:206-229`) |
| Axis iteration | `tensor/iterator_axes.rs:96, 303, 328, 353, 461, 684, 894, 1025, 1052` | `axes_iter*` — **only the constructor** needs it; `IterAxesView` itself is generic (`:17-65`) and the body only calls `DataRef::try_as_true_ref` (`:158-170`), which exists for any `C` (`storage/data.rs:122-127`). This bound looks accidental. |
| Scalar indexing | `tensor/indexing.rs:588, 611, 638, 667` | `Index`/`IndexMut` (`&T` / `&mut T`) and `index_uncheck` |
| Host pointers | `tensor/ownership_conversion.rs:667` | `TensorAny::as_ptr`/`as_mut_ptr` |
| Host-slice `asarray` | `tensor/asarray.rs:522, 549, 568, 622, 636, 696, 725, 744, 800, 814` | zero-copy `TensorView`/`TensorMut` over `&[T]`/`&mut [T]`/`&Vec<T>`; fabricates a `Vec` over the caller's buffer (`asarray.rs:527-543`) |
| Reference rayon device | `feature_rayon/auto_impl/creation.rs:7, 65, 83, 96, 110` | shared bodies, irrelevant to CUDA |
| CPU reference devices | `device_cpu_serial/creation.rs:6, 64, 80, 92`; `device_faer/conversion.rs:12, 31` | reference impls |
| Optional crates | `rstsr-blas-traits/src/trait_def.rs:11`, `src/util/util_tensor.rs:12, 40`; `rstsr-sci-traits/src/distance/*` | BLAS/sci layers |

Additional `Vec`-only helpers a CUDA `Raw` will not have (their absence is a compile-time
subtraction, not a core error): `DataRef::as_slice_ref` / `DataMut::into_slice_mut`
(`storage/data.rs:130-153, 206-229`); `DataForceMutAPI` (`:557-585`) → `TensorAny::force_mut`
(`tensor/ownership_conversion.rs:415-434`, an unsafe escape hatch); `PackArrayAPI`/`UnpackArrayAPI`
(`tensor/pack_array.rs:9-100`).

Host-materialization APIs are **generic** and work through the device traits: `to_raw_f` gathers
on-device and returns `B::Raw` (`tensor/ownership_conversion.rs:440-459` — for CUDA a *device*
buffer); `to_vec`/`into_vec` = `to_raw_f` + `into_cpu_vec` D2H (`:499-546`); `to_scalar` = one
`DeviceStorageAPI::get_index` (`:603-613`); `Tensor::clone` = `to_owned` = device-side gather
(`:356-366`, needs `Raw: Clone`). `Display` additionally needs a
`DeviceChangeAPI<DeviceCpuSerial>` impl (`format/format_tensor.rs:215-239`) and prints a host
`Vec<T>`; `Debug` needs only `Raw: Clone` + `to_cpu_vec` (`:268-291`).

### 4.3 The dlpack precedent

`rstsr-cpu-dlpack` defines a data repr **outside core**:

```rust
pub struct DataDlpack<C, O> { span: ManuallyDrop<C>, owner: O }   // repr.rs:18-21
impl<C, O> DataAPI for DataDlpack<C, O> { type Data = C; ... }     // repr.rs:39-45
pub type TensorDlpack<T, B = DeviceCpu, D> =
    TensorBase<Storage<DataDlpack<Vec<T>, DlpackForeignOwner>, T, B>, D>;  // repr.rs:142
```

It implements only `DataAPI` + `DataCloneAPI` (+ `Clone` for `Vec<T>`, + `DataForceMutAPI<Vec<T>>`,
`:39-91, :172-186`), and it is a **read-only borrowed span over host memory**. The generic data-repr
machinery (`DataOwned<C>`, `DataArc<C>`, `DataRef<C>`, …; `storage/data.rs:10-42`) is fully generic
over `C`, so the trick of "define your own `R`" does generalize. What does **not** generalize is the
`Vec<T>` span: a device-only buffer has no host-addressable `Vec` to fabricate, so the dlpack pattern
must be replaced by setting `Raw` itself to the device buffer type. That is the cleaner route anyway:
`DataOwned<CudaBuffer<T>>` needs no new data repr at all.

### 4.4 Verdict: can `rstsr-cuda` define `Raw` = device-only memory and reuse core?

**Yes for the compute path, with a bounded list of breakages.**

What works unchanged (verified by reading the code): tensor construction, views, layout-only
manipulation, slicing, broadcasting, reshape/COW; every elementwise/ternary/quaternary op,
reduction-to-tensor, matmul/tensordot, sorting, searching and set op (they only pass `&B::Raw` and
`&Layout<D>` into device methods); creation via `DeviceCreation*API` and
`asarray((Vec<T>, &device))` / `tensor_from_nested!` via `outof_cpu_vec` (H2D)
(`tensor/asarray.rs:433-446`, `tensor/creation.rs:3630-3648`); `to_vec`/`into_vec`, `to_scalar`,
`Tensor::clone`, `into_owned`, `change_device` (CUDA→CPU side); `Display`/`Debug` once
`DeviceChangeAPI<DeviceCpuSerial>` and `to_cpu_vec` exist; error mapping, since `RSTSRError` is
`#[non_exhaustive]` with `DeviceError`/`ErrorCode`/`MemoryError` (`rstsr-common/src/error.rs:15-47`).

What breaks (details and severities in §9):

1. `iter`/`iter_mut`/`indexed_iter`, `Index`/`IndexMut`, `as_ptr`/`as_mut_ptr` — unavailable; `Index`
   returning `&T` is **structurally impossible** for device memory. The shared core test body uses
   `a[[i, j]]` in 8 of 85 files (57 occurrences) and tensor `.iter()` in 2 — the shared body will not compile for a
   `Raw != Vec<T>` device as-is.
2. `asarray((&[T], &device))` (zero-copy host view) — unavailable; use the `Vec<T>` owning form.
3. `axes_iter` — needlessly gated (see §4.2); relaxing is a one-line bound change.
4. `unique*` requires the device to *truncate* its `Raw` to the initialized prefix
   (`operators/set.rs:7-15`, `tensor/set.rs:146-186`) — so `Raw` must carry a logical length (a
   `CudaBuffer { ptr, len }` does; a bare `*mut T` does not).
5. `TensorAny::force_mut` and `pack_array` — `Vec`-only, unneeded.
6. CPU→CUDA conversion via `TensorDeviceChangeAPI` has no impl in core (core only writes the
   CPU↔CPU pairs, e.g. `device_cpu_serial/conversion.rs:3-40`); the CUDA crate must write both
   directions itself (coherence allows it — the local device type appears in the trait's `BOut`
   parameter). See §9 R5.

---

## 5. Element-type trait bound inventory

**Where host scalar traits appear.**

- *Device impl level (the rule, and the majority).* `rstsr-dtype-traits` traits and `num` traits are
  bounds on the concrete device impls, e.g.
  `impl<TA,TB,TC,D> OpAddAPI<...> for DeviceCpuSerial where TA: Clone + Op<TB, Output = TC>`
  (`device_cpu_serial/operators/op_ternary_arithmetic.rs:17-21`);
  `T: Clone + DTypeIntoFloatAPI<FloatType: NumTrait>` for unary math, where the macro substitutes
  `ExtComplexFloat` or `ComplexFloat` per op (`device_cpu_serial/operators/op_binary_common.rs:34-38`);
  `T: ExtReal` for min/max reductions (`device_cpu_serial/reduction.rs:38-42`); `T: ExtSortCmp` for sorting
  (`device_cpu_serial/sorting.rs`). The tensor layer carries only device-trait bounds
  (`tensor/operators/op_binary_common.rs:148-157`). **This is exactly the placement a GPU wants:**
  a CUDA impl may use a local `CudaScalar` bound (f32/f64/f16/bf16/…) instead of `num::Float`, and
  nothing in core needs to change.
- *Tensor-tier exceptions* (bounds that the device cannot avoid):
  - `var`/`std` with `correction`: `TOut: Float + FromPrimitive + Send + Sync + 'static`
    (`tensor/reduction.rs:877, 913, 935, 951, 1018, 1159, 1181, 1316, 1348, 1362, 1383, 1397`).
  - `arange` without device uses `T: PartialOrd + Clone + Add<Output=T> + Zero + …`
    (`tensor/creation.rs:226-262`) — the device form only requires `DeviceCreationArangeAPI<T>`.
  - `sort`/`argsort` use a `TypeId` runtime decline for complex (`tensor/sorting.rs:41-51`),
    no scalar bound.
  - `Display`/`Debug` require `T: Clone + Display`/`Debug` (`format/format_tensor.rs:218, 270`).

**Closure parameters** (element callbacks; see §9 R12): `Op_Mut*_API` (`operators/ops/op_with_func.rs:7-129`),
`OpReduceCustomAPI` (`operators/reduction.rs:128-166`), `OpSortCustomAPI` (`operators/sorting.rs:107-133`),
and the tensor-tier `map*`/`mapb*` methods (`tensor/map_elementwise.rs:22-600`), `sort_custom`
(`tensor/sorting.rs:404-666`), and the `var/std` correction rescale (`tensor/reduction.rs:897-903`).
These are host `Fn`/`FnMut` values. They cannot be shipped into a GPU kernel.

**Dtype fast-path dispatch (precedent for dtype-dispatched kernels).** `for_each_fast_dtype!` +
`TypeId` re-typing lists 16 scalar dtypes (incl. `half::f16`/`half::bf16`) and reinterprets
`&Vec<T>` as `&Vec<$ty>` (`device_cpu_serial/set.rs:16-63`); `impl_uninit_dispatch!` does the same
for f32/f64/c32/c64 (`device_faer/matmul.rs:242-266`, `device_faer/ext_linalg/matmul.rs:42-67`);
`TypeId`-guarded arithmetic/remainder (`device_cpu_serial/operators/op_ternary_arithmetic.rs:61-165`);
runtime dtype decline (`tensor/sorting.rs:41-51`). Beyond core, the pyo3 binding carries a fully
dtype-erased `enum AnyTensor { Bool(..), I8(..), … }`
(`crates-interop/rstsr-faer-py/core/src/any_tensor.rs:56-70`).

**What `rstsr-dtype-traits` defines.** `DTypeIntoFloatAPI` (maps integers/bool to an accumulation
float; `promotion.rs:16-118`), `DTypeCastAPI` (`:21-23`), `DTypePromoteAPI` with `SAME_TYPE` /
`CAN_CAST_*` flags (`:25-39`), `ExtNum`/`ExtReal`/`ExtFloat`/`ExtZero`/`ExtSortCmp`/`ExtComplexFloat`
(arithmetic, comparisons, nextafter, zero, total order, complex ops), `IsCloseArgs`, and
`ValWriteAPI<T>` for `T` vs `MaybeUninit<T>` (`val_write.rs:10-28`). There is **no** `DTypeScalarAPI`
in this tree (grep returns nothing) — that name in the task prompt has no counterpart here.

**f16/bf16 support.** Yes, present and non-trivial: `half` is a workspace dep with `num-traits`
(`Cargo.toml:76`) and `rstsr-core` enables `rstsr-dtype-traits/half` (`rstsr-core/Cargo.toml:15`);
the dtype traits implement f16/bf16 (`promotion.rs:94-102, 194-197`, `ext_real.rs:159-160`,
`ext_num.rs:391-392`, `ext_zero.rs:36-37`, `ext_sort_cmp.rs:91-92`, `ext_float.rs:20-133`); core
fast paths list them (`device_cpu_serial/set.rs:33-34`,
`device_cpu_serial/operators/op_ternary_arithmetic.rs:68-69`,
`tensor/operators/op_binary_arithmetic.rs:968-990`); `BlasFloat` covers them
(`rstsr-blas-traits/src/blas_scalar.rs:55-60`), though drivers monomorphize only f32/f64/c32/c64.
`half` implements `num::Num`/`num::Float`, so `DeviceNumAPI` and the `var/std` bounds are satisfiable.

**Assessment.** The bound placement already permits a dtype-dispatched CUDA backend: implement each
`Op*API` for a local scalar bound and dispatch on `TypeId` (or on an explicit dtype tag in the device)
inside the kernel launcher; the existing `TypeId` code is the template. The only forced host-scalar
bounds are the `var/std` `Float + FromPrimitive` ones, which f16/bf16 satisfy.

---

## 6. Host-synchrony inventory

Every API that synchronously touches element data on the host, with its device-trait route.

| API | Site | Route | Sync kind | Avoidable? |
|---|---|---|---|---|
| `to_vec` / `into_vec` | `tensor/ownership_conversion.rs:499, 535-589` | `to_raw_f` (device) → `into_cpu_vec` | D2H copy | explicit read; expected |
| `to_raw` / `into_raw` | `:447-459, 510-525` | device gather, returns `B::Raw` | **no host read** for CUDA | — |
| `to_scalar` | `:603-613` | `DeviceStorageAPI::get_index` (`storage/device.rs:41`) | single-element D2H | structural (returns `T`) |
| `Index` / `IndexMut` | `tensor/indexing.rs:585-630` | host `Vec` slot | per-element D2H | API unavailable for CUDA |
| `iter`/`iter_mut`/`indexed_iter*` | `tensor/iterator_elem.rs:62-646` | `as_slice_ref` | host slice | unavailable for CUDA |
| `axes_iter*` | `tensor/iterator_axes.rs:92-200` | views only | **none** | bound can be relaxed |
| `Display` | `format/format_tensor.rs:215-239` | `change_device(CpuSerial)` + print | full D2H | user-triggered |
| `Debug` | `format/format_tensor.rs:268-291` | `storage.to_cpu_vec()` | full D2H | user-triggered |
| `as_ptr`/`as_mut_ptr` | `tensor/ownership_conversion.rs:663-684` | host pointer | none (but meaningless on GPU) | unavailable for CUDA |
| `asarray((Vec<T>, &dev))`, scalar `asarray`, `tensor_from_nested!` | `tensor/asarray.rs:433-446, 883`; `tensor/creation.rs:3638-3647` | `outof_cpu_vec` | H2D | input creation; expected |
| fancy-index resolution | `tensor/array_indexing.rs:63-102` (`get_index` per entry) + `:469-482` (`outof_cpu_vec`) | element-by-element D2H **then** H2D | O(n) syncs | **yes** — should be a device kernel |
| `index_select` / `index_put` | `operators/adv_indexing.rs:22, 96`; tensor wrappers `tensor/adv_indexing.rs:30-45` | host `&[usize]` | H2D upload | API shape; could take a device tensor |
| `searchsorted` `sorter` | `operators/searching.rs:179` | host `Option<&[usize]>` | H2D upload | API shape |
| `nonzero` | `tensor/nonzero.rs:26-30` | `nonzero_count -> usize` | 1 sync + buffer alloc | data-dependent size; hard to avoid |
| `unique_values/all` | `tensor/set.rs:146-160` | returns `u: usize`; device truncates raw | 1 sync | data-dependent size |
| `argmin_all`/`argmax_all`/`unraveled_*_all` | `operators/reduction.rs:28, 70`; tensor layer `tensor/reduction.rs:2219, 2350` | returns `D`/`usize` then re-uploads via `outof_cpu_vec` | sync + H2D | partially (see below) |
| `*_all` reductions (`sum_all`, `min_all`, …) | `operators/reduction.rs:28` | returns `TOut` | 1 sync | structural |
| `sum_axes(None)` and siblings | `tensor/reduction.rs:2443-2450` (+7 sites) | `sum_all` → `outof_cpu_vec(vec![sum])` | sync **+** H2D | **yes** — call the axes kernel with empty axes |
| `allclose` | `operators/reduction.rs:96-119` | returns `bool` | 1 sync | structural |
| `var`/`std` with `correction` | `tensor/reduction.rs:897-903` | `mapv(host closure)` | sync + host closure | **yes** — device scalar-mult |
| random creation | — | **no RNG exists in core** (grep `rand`/`random`: no hits outside tests) | — | future design point |

Notes:

- There is **no** `PartialEq` for tensors, so `==` is not a sync point (grep: no `impl PartialEq for
  Tensor`); `allclose` is the comparison API.
- The *compute* path never syncs: elementwise, matmul, reductions-to-tensor, sort, search, set,
  creation, and layout ops only enqueue device work. Eager evaluation is compatible with async
  execution **provided** `uninit_impl` / `assume_init_impl` do not synchronize (an allocator concern,
  not a trait concern).
- The two biggest *unnecessary* syncs are the fancy-index host round-trip (O(n) element reads) and
  `*_axes(None)` (scalar round-trip). Both are contained in the tensor layer and fixable without
  touching the device trait surface.

---

## 7. Threading internals

**Where rayon lives.**

| Location | Content |
|---|---|
| `rstsr-core/src/feature_rayon/device.rs:40-129` | `DeviceCpuRayon` + `DeviceRayonAPI`; one `Arc<ThreadPool>` per device; `get_current_pool` returns `None` inside a rayon worker so nested parallelism is avoided (`:122-128`) |
| `rstsr-core/src/feature_rayon/auto_impl/*.rs` (20 files) | blanket `impl … for DeviceRayonAutoImpl` bodies, written against `Vec<T>` |
| `rstsr-core/src/feature_rayon/par_iter.rs` | `IntoParallelIterator` for the element/axes iterators (requires `B::Raw: Send + Sync`, `B: Send`) |
| `rstsr-native-impl/src/cpu_rayon/*` | the actual parallel kernels (`op_with_func.rs`, `matmul_naive.rs`, `tensordot.rs`, `sorting.rs`, …) |
| `rstsr-common/src/par_iter.rs` | `ParIterRSTSR` bridge + parallel layout iterators (feature `rayon`) |

**The `AtomicPtr` hoist.** When a parallel outer loop calls a matmul inside rayon tasks, the crates
cannot hand out `&mut Vec` per task; they hoist the output base pointer into an `AtomicPtr` and
rebuild a task-local `&mut [T]` inside the closure (`crates-device/rstsr-openblas/src/matmul.rs:159`,
reloaded at `:170`). Same pattern in all five device crates (`matmul.rs:159`, `matmul_impl.rs:375`)
and in `rstsr-native-impl/src/cpu_rayon/{adv_indexing.rs:46, tensordot.rs:49, outer.rs:35,
op_tri.rs:44}`. CPU-only workaround; not needed (and must not be copied) for GPU kernels.

**Does anything assume device ops run on the calling thread?** Not in core — ops are `&self` methods
called wherever the user is. But: (1) device ops *are* invoked from rayon workers in the BLAS crates
(the parallel outer loop above), so CUDA must be launchable from any thread and must set the
thread-local context/device (`cudaSetDevice(ordinal)`) on entry (TLS current-device is per host
thread; the per-tensor device value supplies the ordinal); (2) `with_blas_num_threads`-style scoped
global mutation (`crates-device/rstsr-openblas/src/threading.rs:100-133`) is **not** a valid scoping
primitive with two concurrent devices — for CUDA, stream selection and TF32 flags belong in the
device value (or a per-device object), not in process globals.

**Multi-GPU.** One `DeviceCuda` per ordinal with its own stream/context handle, tensors carrying their
device, `std::thread` per device: works with no core change. The device type must be `Send + Sync`
(wrap the raw stream handle in a newtype with `unsafe impl Send/Sync`, or `Arc<Mutex<…>>` if the
stream must be exclusive).

---

## 8. Integration mechanics — the OpenBLAS template

**Crate layout** (`crates-device/rstsr-openblas/`):

```
Cargo.toml   deps: rstsr-core ("rayon"), rstsr-common, rstsr-dtype-traits ("half"), rstsr-blas-traits,
             optional rstsr-linalg-traits / rstsr-sci-traits; dev: rstsr (facade), rstsr-test-manifest
build.rs     RSTSR_DEV-only link wiring (no-op downstream): link-search + -rpath + OpenMP
             (build.rs:37-38, 123, 144-150)
src/lib.rs   DeviceOpenBLAS { base: DeviceCpuRayon } (:22-25); `pub(crate) use` aliases
             DeviceBLAS (:37) and DeviceRayonAutoImpl (:38); mods driver_impl, plus feature-gated
             linalg_auto_impl / sci_auto_impl
src/         device.rs (Raw/Storage/API/Num impls), conversion, matmul, matmul_impl, threading
symlinks     rayon_auto_impl/ -> rstsr-core/src/feature_rayon/auto_impl/*;
             driver_impl/ -> rstsr-blas-traits/driver_impl/* (+mod.rs);
             linalg_auto_impl/ -> rstsr-linalg-traits/src/blas_impl/*;
             sci_auto_impl/ -> rstsr-sci-traits/src/{distance,integrate}/auto_impl_rayon.rs
tests/       entry_row_openblas.rs (type alias + TESTCFG); core_func/ test_issues/ test_utils/
             symlinked from rstsr-core; driver_impl/ linalg_func/ from blas_tests_impl; README.md
```

Key mechanism: shared bodies are **symlinked into the device crate and compiled there**, with
`pub(crate) use DeviceX as DeviceBLAS;` / `as DeviceRayonAutoImpl;` supplying the type the body is
written against (`crates-device/rstsr-openblas/src/lib.rs:36-38`;
`crates-device/rstsr-openblas/src/driver_impl/mod.rs:9-19`; `src/linalg_auto_impl/mod.rs`). This
is the mechanism a CUDA crate can reuse for any body that is genuinely device-generic; it is *not*
usable for the `Vec<T>` kernels.

**Feature wiring in the facade** (`rstsr/Cargo.toml:36-71`): each device gets
`<name> = ["dep:rstsr-<name>", "use_blas_traits"]`, and `linalg` / `sci` forward optional
per-device features. The prelude nests member crates:

```rust
rstsr_traits::linalg     -> rstsr_linalg_traits::prelude::rstsr_traits   // rstsr/src/prelude.rs:3-6
rstsr_structs::dlpack    -> rstsr_cpu_dlpack::prelude::rstsr_structs    // :88-91
rstsr_funcs::dlpack      -> rstsr_cpu_dlpack::prelude::rstsr_funcs      // :111-114
pub mod dlpack { … }                                                    // :147-152
pub mod rt { … pub use super::dlpack; }                                 // :154-181
```

Note the asymmetry: BLAS device *types* are exported flat (`rstsr_structs::DeviceOpenBLAS`,
`DeviceMKL`, …, plus a selected `type DeviceBLAS`, `rstsr/src/prelude.rs:36-86`), while opt-in member
crates get a namespace (`rt::linalg`, `rt::sci`, `rt::tblis`, `rt::dlpack`).

**Where `rt::cuda` would live.** Follow the member-crate pattern: add
`cuda = ["dep:rstsr-cuda"]` to `rstsr/Cargo.toml`, re-export the device type in
`rstsr_structs` (flat, like `DeviceOpenBLAS`) or under `rstsr_structs::cuda`, and add
`pub mod cuda { pub use rstsr_cuda::prelude::*; }` + `pub use super::cuda;` inside `mod rt`
(mirroring `:147-181`). Device-specific utilities (streams, TF32 control, memory pool) naturally
live in `rt::cuda::`.

**Tests.** Device entries are plain `tests/*.rs` files; the shared bodies (`core_func/`,
`test_issues/`, `test_utils/`) are directory symlinks
(`crates-device/rstsr-openblas/tests/entry_row_openblas.rs:1-14`,
`tests/README.md`). Contrary to `rstsr-agents/rules/testing.md` ("`[[test]]` section in the crate's
Cargo.toml with `required-features`"), the device crates in this tree declare **no** `[[test]]`
sections — they rely on auto-discovery, and `tests_linalg_func.rs` gates itself with
`#[cfg(feature = "linalg")]`. `rstsr-test-manifest` supplies `.npy` resources
(`rstsr-test-manifest/src/lib.rs:3-32`).

**Implication for CUDA:** the entry file and symlinks are reusable; the *shared body itself* is not
compilable against a device whose `Raw != Vec<T>` (§4.4, §9 R1). The device crate will need either
(a) a core change adding host-scalar element access (`get`/`set`) and migrating the shared body, or
(b) a dedicated test body (the "dedicated-file exception" of ADR-0012, recorded in `tests/README.md`).

---

## 9. Restrictive-assumption inventory

Severity: **blocker** = cannot implement the advertised functionality without a core/API change;
**annoyance** = implementable with a workaround or a documented limitation; **non-issue** = verified
compatible.

| # | Assumption | Where | GPU impact | Severity | Candidate change |
|---|---|---|---|---|---|
| R1 | Element access requires `Raw = Vec<T>` (`iter`, `iter_mut`, `indexed_iter`, `axes_iter`, `Index`, `as_ptr`) | `tensor/iterator_elem.rs:65,168,186,204,292,424,555,613,634`; `tensor/iterator_axes.rs:96,303,…`; `tensor/indexing.rs:588,611,638,667`; `tensor/ownership_conversion.rs:667` | unavailable for device memory; shared test body uses `[[i,j]]` in 8/85 files (57 occurrences) and tensor `.iter()` in 2 | **blocker** for test parity; annoyance for users | add `TensorAny::get/set` via `DeviceStorageAPI::get_index`; relax `axes_iter` (its body only needs `try_as_true_ref`); introduce a `HostSlice` marker for the genuinely host-only APIs |
| R2 | `Index`/`IndexMut` return `&T`/`&mut T` | `tensor/indexing.rs:585-630` | structurally impossible for device memory | **blocker** (for that API only) | replace with `get`/`set` in docs/tests; keep `Index` as a CPU-only convenience |
| R3 | `axes_iter` constructor bound is `Raw = Vec<T>` although the body needs nothing host-specific | `tensor/iterator_axes.rs:96-170` vs `storage/data.rs:122-127` | needless loss | annoyance (easy win) | drop the bound; `B: DeviceAPI<T>` + `R: DataAPI<Data=B::Raw>` suffices |
| R4 | `asarray((&[T], &device))` fabricates a `Vec` over the caller's slice | `tensor/asarray.rs:519-543` | zero-copy host view impossible | annoyance | keep `Vec<T>`/`&[T]` forms CPU-gated; CUDA uses `Vec<T>` (H2D) |
| R5 | `DeviceChangeAPI` is keyed on the **source** device, and core only implements CPU↔CPU pairs | `storage/conversion.rs:3-21`; `device_cpu_serial/conversion.rs:3-40` | CPU→CUDA and CUDA→CPU must each be written in `rstsr-cuda`; coherence permits it (local type in the `BOut` trait arg), but the asymmetry is easy to miss | annoyance | optionally add a generic `impl<'a, BOut: DeviceCreationAnyAPI<T> + DeviceStorageAPI<T>, …> DeviceChangeAPI<'a, BOut, …> for DeviceCpuSerial` in core |
| R6 | `DeviceStorageAPI::get_index`/`set_index` are infallible (`-> T`, no `Result`) | `storage/device.rs:41-53` | CUDA D2H can fail; impl must panic/unwrap | annoyance | add fallible twins (`get_index_f`) or document panic-on-error |
| R7 | `get_index_ptr`/`get_index_mut_ptr` return host-dereferenceable `*const T`/`*mut T` | `storage/device.rs:45-50` (unused in core) | on CUDA this is a device pointer; the signature invites misuse | annoyance | add a distinct `device_ptr()` API and document the host-pointer contract |
| R8 | `DeviceAPI: Default` | `storage/device.rs:159` | `DeviceCuda::default()` must panic or silently pick ordinal 0 | annoyance | acceptable; add `try_default`/`DeviceCuda::new(ordinal)` |
| R9 | `DeviceRawAPI<T>: Clone` and device cloned into every `Storage` | `storage/device.rs:9`, `:14-22` | clone must be cheap (Arc/stream handle) | non-issue | keep `Arc` inside `DeviceCuda` |
| R10 | `BlasDriverBaseAPI` requires `Raw = Vec<T>` (and `DeviceRayonAPI`) | `rstsr-blas-traits/src/trait_def.rs:10-41` | the entire BLAS/LAPACK driver layer (14 driver traits + aggregates, 35 driver files) cannot be implemented for CUDA | **blocker** for the BLAS-based `linalg` tier | cuBLAS/cuSOLVER must be a separate trait surface (core `DeviceMatMulAPI` or new `Device*` traits), or the BLAS layer must be generalized |
| R11 | sci distance impls are hard-coded to CPU devices | `rstsr-sci-traits/src/distance/auto_impl_rayon.rs:20`, `impl_cpu_serial.rs:17`, `impl_faer.rs` | `rt::sci::cdist` unavailable on CUDA | annoyance (optional feature) | implement `CDistAPI<DeviceCuda>` separately |
| R12 | Host-closure ops: `Op_Mut*_API`, `OpReduceCustomAPI`, `OpSortCustomAPI` | `operators/ops/op_with_func.rs:7-129`; `operators/reduction.rs:128-166`; `operators/sorting.rs:107-133` | `Fn` cannot be a kernel; `.map()`, `.sort_custom()`, custom reductions impossible | **blocker** for those methods | omit on CUDA (trait bound unsatisfied → compile error at call site) with clear docs, or a host round-trip fallback |
| R13 | `var`/`std` with `correction` is implemented as a host `mapv` rescale | `tensor/reduction.rs:871-905` | forces the closure op above; a sync + host closure | **blocker/annoyance** (single feature) | use a device scalar-multiply (`OpMul` with a scalar) instead of `mapv` |
| R14 | `*_axes(None)` reductions round-trip through the host | `tensor/reduction.rs:2443-2450` + 7 more `outof_cpu_vec(vec![…])` sites | one sync + H2D whenever all axes are reduced into a 0-d result | annoyance | call the axes kernel with an empty axes list, or write the device scalar into a device buffer |
| R15 | Fancy indexing resolves each device index entry through `get_index` and re-uploads | `tensor/array_indexing.rs:63-102`, `:469-482` | O(n) D2H + H2D per `array_index` call | **blocker for performance** (correct but slow) | make resolution a device op (bounds-check on device), returning a device `usize` buffer; the `ArrayAuxIndexer` design already anticipates device index storage (`operators/adv_indexing.rs:139-160`) |
| R16 | `index_select`/`index_put` take host `&[usize]`; `searchsorted` takes host `Option<&[usize]>` | `operators/adv_indexing.rs:22,96`; `operators/searching.rs:179` | H2D upload per call; cannot keep an index tensor on device | annoyance | add device-index overloads |
| R17 | `nonzero`/`unique` return data-dependent lengths as host `usize` | `operators/searching.rs:193`; `operators/set.rs:23-42`; `tensor/set.rs:146-160` | one sync + allocation sized on host | annoyance (structural for the current `Tensor` return type) | accept; or return a device count and defer |
| R18 | `unique*` require the device to truncate `Raw` to the initialized prefix | `operators/set.rs:7-15`; `tensor/set.rs:146-186` | `Raw` must carry a logical length | non-issue if `CudaBuffer` is a struct; blocker if `Raw` is a bare pointer | design `Raw` as a (ptr, len, alloc/stream) struct |
| R19 | `transmute<&mut Raw<TC>, &mut Raw<MaybeUninit<TC>>>` in the closure-op driver | `tensor/operators/op_with_func.rs:44-48` | requires `Raw<T>` and `Raw<MaybeUninit<T>>` layout-identical | non-issue for `PhantomData`-parameterized buffers; must be documented | — |
| R20 | Device ops are generic over `D: DimAPI` | every device trait, e.g. `operators/ops/op_ternary_arithmetic.rs:17-20` | a naive per-`D` impl explodes; implement generically and convert to `IxD` inside, or only for `IxD` (breaks fixed-dim tensors) | annoyance | canonical kernel entry takes `Layout<IxD>`; a generic impl forwards |
| R21 | Negative strides (`flip`) and arbitrary offsets/strides | `rstsr-common` `Layout` (`stride: Vec<isize>`, `offset: usize`); kernel index math e.g. `device_cpu_serial/set.rs:56` | kernels must do signed-stride address computation | non-issue (implement it) | — |
| R22 | `same_device` is the only compatibility check | `storage/device.rs:4`; 20+ call sites | must encode ordinal (and stream policy) or mixed-GPU/stream operands pass | annoyance | `same_device = ordinal && stream && order` |
| R23 | Process-global `DEFAULT_TENSOR_ITER_ORDER` | `rstsr-common/src/flags.rs:120-152` | shared across devices; one thread can change global policy | non-issue (document) | — |
| R24 | Device crates use process-global scoped state (`with_num_threads`) | `crates-device/rstsr-openblas/src/threading.rs:100-133` | unsafe to copy for multi-device CUDA | non-issue (avoid) | keep stream/TF32 state in the device value |
| R25 | `rstsr-core` is `no_std`; `std` is optional | `rstsr-core/src/lib.rs:1`, `Cargo.toml:31` | CUDA runtime needs `std` | non-issue | enable `rstsr-core/std` in `rstsr-cuda` |
| R26 | Eager evaluation, no graph/fusion | §0.3 | no kernel fusion across `rt::` calls; async still possible per op | non-issue (by design) | optional: a stream-capturing device mode later |

**Positive findings (things that surprisingly do generalize):**

- **`Raw` is an associated type** (`storage/device.rs:9-11`) that the compute path passes around
  without inspecting — device-only memory is the intended extension point.
- **The device is per-tensor state** (a value in `Storage`), so multi-GPU/multi-stream needs no core
  change and no global registry.
- **`Raw` is parameterized by `T`**, so `MaybeUninit<T>` storage is a separate associated type — the
  `uninit_impl`/`assume_init_impl` contract (ADR-0008) maps onto `cudaMalloc` + "write-only kernel"
  with no host zero-fill.
- **Element-trait bounds live on device impls** (`device_cpu_serial/operators/op_binary_common.rs:34-38`
  and every sibling), so CUDA can define its own `CudaScalar` bound without touching core.
- **TypeId dtype dispatch is established practice** (`device_cpu_serial/set.rs:16-63`,
  `device_faer/matmul.rs:242-266`) — the shape a dtype-dispatched CUDA launcher needs. f16/bf16 are
  already first-class in dtype traits and `BlasFloat` (§5).
- **Errors are extensible**: `RSTSRError` is `#[non_exhaustive]` with `DeviceError`/`ErrorCode`/
  `MemoryError` (`rstsr-common/src/error.rs:15-47`).
- **`change_device` is a real seam** with free `Repr`/`ReprTo` types (`storage/conversion.rs:3-21`);
  both directions are implementable in the backend crate (R5 is ergonomics, not coherence).
- **Symlink + crate-local type alias** is an established way to compile one body per device
  (`crates-device/rstsr-openblas/src/lib.rs:36-38`); **`dlpack` proves out-of-core data reprs work**
  (`rstsr-cpu-dlpack/src/repr.rs:18-45, 142`), and the generic `DataOwned<C>` means a CUDA crate need
  not add a new repr at all.
- **The barrier for the `rt::` surface is per-op**: unimplemented families fail at the call site, so
  a partial CUDA backend is a coherent deliverable.

---

## 10. Open questions for the maintainers

1. **Element access API.** Do we add `TensorAny::get(index) -> T` / `set(index, T)` (routing through
   `DeviceStorageAPI::get_index`), and migrate the shared test body from `a[[i, j]]` to it? Without
   this, a CUDA device cannot run the symlinked core tests unchanged (8/85 files use bracket
   indexing, 57 occurrences; 2 use tensor `.iter()`; 3 use `TensorAny::as_ptr`).
2. **`DeviceChangeAPI` direction and `Raw` shape.** Leave both transfer directions to each backend,
   or add a generic `impl … for DeviceCpuSerial` in core? And should a backend's `Raw` be mandated to
   be a length-carrying handle (needed by `unique*` truncation), with a `device_ptr()` accessor
   replacing the host-pointer `get_index_ptr`?
3. **Allocator.** `uninit_impl`/`empty_impl` are called per op; `cudaMalloc` is expensive and
   serializing. Should the device own a caching pool, and is `assume_init_impl` allowed to be a no-op
   (it is a type-level relabel for CUDA)?
4. **Async/streams.** Is per-op async launch (no sync until an explicit host read) acceptable given
   eager evaluation, or do we want a documented "synchronize on `to_vec`/`to_scalar`/scalar
   reductions" contract? What is the policy for tensors produced on one stream and consumed on
   another when `same_device` ignores streams?
5. **BLAS tier.** Wire cuBLAS through the core `DeviceMatMulAPI`/`GEMM` traits (and cuSOLVER through
   a new linalg surface), or generalize `rstsr-blas-traits` away from `Raw = Vec<T>`?
6. **Host-closure ops.** Omit `map`/`sort_custom`/custom reductions on CUDA (compile-time absence), or
   provide an explicit host round-trip fallback?
7. **Dtype set, dispatch, and TF32 policy.** Which dtypes for v1 (f32/f64/f16/bf16/complex?); local
   `CudaScalar` + `TypeId` dispatch (existing precedent) or a device-carried dtype tag; and does the
   TF32/FP16 policy live in the device field, a per-call argument, or both?
8. **NVRTC vs precompiled kernels.** The traits are monomorphized per `(T, D, op)`; NVRTC-at-runtime
   would need a kernel-cache keyed by dtype/rank/op. Is a compiled-kernel crate (one generic kernel
   per op, dtype as a runtime tag) acceptable for v1?
9. **Testing.** Dedicated CUDA test body vs core changes for R1; and is a CPU↔CUDA parity harness
   (copy to CPU, compare) the intended tier-2/3 CI shape (there is no GPU CI vendor today;
   `rstsr-agents/rules/testing.md` tiers T3/T4)?
10. **Multi-GPU API and `rt::cuda` surface.** One `DeviceCuda` per ordinal created by the user, or a
    `rt::cuda::device(i)` registry (the per-tensor device storage supports the former without core
    changes)? Confirm the member-crate namespace (`rt::cuda::*`, `rstsr_structs::cuda`) plus a flat
    `DeviceCuda`, mirroring `rt::dlpack` / `DeviceOpenBLAS`.

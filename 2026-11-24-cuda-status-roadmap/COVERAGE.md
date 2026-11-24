# Common-usage coverage matrix

Function-level answer to "will ordinary `rstsr`/Array-API code run on
`DeviceCuda`?" at `9c7d49a`. Device-level detail is in
[STATUS.md](STATUS.md); priorities are in [ROADMAP.md](ROADMAP.md).

**Legend**

| Code | Meaning |
|---|---|
| `Y` | works on `DeviceCuda` (subject to the dtype limits of [STATUS.md](STATUS.md) §6) |
| `P` | partial: works only for some dtypes/layouts, or there is no sugar and a workaround is needed |
| `N` | not implemented; the use site fails to compile |
| `B` | blocked by a `rstsr-core` restriction, not by the CUDA crate |
| `X` | impossible or intentionally excluded (host closure / accelerator-unsafe API) |

Unless a note says otherwise, `Y` was established by reading the bound chain
and the existing hardware tests, not by a fresh compile of every row. The
device-trait inventory in [STATUS.md](STATUS.md) §3–§5 is the authoritative
"what is missing"; this file maps it to user-facing names.

---

## 1. Tier A — everyday array code

### Creation

| Function | State | Note |
|---|---|---|
| `empty`, `zeros`, `ones`, `full`, `uninit` | Y | kernel fill / memset; `full` covers every `CudaDType`. |
| `zeros_like`, `ones_like`, `full_like` | Y | creation + assign. |
| `arange`, `linspace` | Y | device sequence kernels; f32/f64/integer paths exercised, complex not tested. |
| `eye` | Y | `DeviceCreationNumAPI` + `OpAssignAPI`. |
| `asarray` from `Vec`/slice/scalar | Y | host→device copy. |
| `asarray` from nested `Vec<Vec<T>>` | P | host flattening path; not tested on CUDA. |
| `from_scalar` | Y | |
| `from_dlpack` | N | no DLPack bridge in `rstsr-cuda`. |
| `tril`, `triu` | **N** | `DeviceCreationTriAPI` missing. |
| `meshgrid`, `diag`, `atleast_1d/2d/3d` | Y | compositions of reshape/assign. |
| `concat`, `stack`, `hstack`, `vstack` | Y | `OpAssignAPI`; each input is a copy. |
| `unstack` | Y | slicing views. |

### Layout / manipulation (no data or a copy)

| Function | State | Note |
|---|---|---|
| `reshape`, `change_shape`, `to_shape` | Y | layout-only when possible, otherwise a copy via assign. |
| `transpose`, `permute_dims`, `swapaxes`, `moveaxis` | Y | layout-only. |
| `squeeze`, `expand_dims`, `unsqueeze` | Y | layout-only. |
| `broadcast_to`, `broadcast_arrays` | Y | layout-only (stride-0). |
| `slice`, `i`, `i_mut`, `diagonal` | Y | layout-only. |
| `flip`, `roll` | Y | copy via slicing/assign. |
| `repeat`, `tile` | Y | assign-based copies. |
| `to_contig`, `to_layout`, `to_prefer` | Y | copy kernel. |
| `diff` | Y | slice + sub + assign. |
| `pack_array`, `unpack_array` | P | bit-packing; not exercised on CUDA. |

### Element access and iteration

| Function | State | Note |
|---|---|---|
| `to_scalar` | Y | one D2H + stream sync; fine for a single element. |
| `a[[i, j]]` (`Index`) | **B** | impl requires `DeviceAPI<T, Raw = Vec<T>>`; CUDA `Raw = CudaBuffer<T>`. |
| `a[[i, j]] = v` (`IndexMut`) | **B** | same gate. |
| `iter()`, axis iterators, `as_ptr` | **B** | all gated on `Raw = Vec<T>`. |
| `get_index`/`set_index` (device trait) | Y | per-element sync; debug-grade only. |
| `Display` / `Debug` | Y | visible-gather D2H, truncation-aware. |

### Arithmetic and elementwise

| Operator / function | State | Note |
|---|---|---|
| `+ - * / //` binary and compound assign | Y | same-dtype; `ext_add/sub/mul/div` follow promotion. |
| `%` (`rt::rem`) | Y | NumPy-parity remainder kernel. |
| `**` (`pow`) | Y | same-dtype (CPU promotes); one negative-exponent guard. |
| unary `+x` | Y | |
| unary `-x`, `rt::neg` | **N** | `OpNegAPI` missing — confirmed compile error. |
| `!x`, `~x` | Y | |
| `&`, `\|`, `^`, `<<`, `>>` | Y | same-dtype integer/bool. |
| `ext_bitand`, `ext_bitor`, `ext_bitxor`, `ext_shl`, `ext_shr` | **N** | mixed-integer op traits missing. |
| `< <= > >= == !=` | Y | complex supports `==`/`!=` only, as on CPU. |
| `abs`, `sign`, `conj`, `real`, `imag`, `reciprocal`, `square` | Y | real+complex where meaningful. |
| `sqrt`, `exp`, `exp2`, `expm1`, `log`, `log2`, `log10`, `log1p` | Y | f32/f64/complex; integers refused (CPU widens). |
| `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2` | Y | f32/f64/complex. |
| `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh` | Y | f32/f64/complex. |
| `degrees`, `radians` | Y | f32/f64. |
| `floor`, `ceil`, `trunc`, `round` | Y | f32/f64. |
| `isnan`, `isinf`, `isfinite`, `signbit` | Y | float and complex where applicable. |
| `maximum`, `minimum` | Y | same-dtype; NaN-propagating. |
| `hypot`, `copysign`, `nextafter`, `logaddexp`, `floor_divide` | Y | same-dtype. |
| `where(cond, x, y)` | P | same-dtype only; scalar tensor operands tested. |
| `clip(x, lo, hi)` | P | same-dtype only; scalar bounds fast path. |
| `mapv` and other host-closure helpers | **X** | a Rust closure cannot cross into a kernel. |

### Reductions and statistics

| Function | State | Note |
|---|---|---|
| `sum`, `prod` (f32/f64) | Y | deterministic tree, not CPU-bit-identical. |
| `sum`, `prod` (integers) | **P** | only through `sum_dtype::<f64>` / `prod_dtype`; plain `.sum()` fails. |
| `sum` (bool) | Y | count kernel. |
| `sum`/`prod` (complex) | **N** | `ReduceAcc` is f32/f64 only. |
| `mean`, `l2_norm` | P | f32/f64 only. |
| `min`, `max` | P | f32/f64 and integers; not bool/half/complex. |
| `all`, `any` | Y | bool. |
| `count_nonzero` | Y | ints and floats. |
| `var`, `std` (+ dtype variants) | P | f32/f64 only; `correction` handled on the device. |
| `argmin`, `argmax` | Y | ordered dtypes; raveled index contract. |
| `nanargmin`, `nanargmax` | Y | f32/f64; D2H read-back for the NaN scan. |
| `unraveled_argmin`, `unraveled_argmax` | **N** | op traits missing. |
| `cumsum`, `cumprod` (free `cumulative_sum`/`cumulative_prod`) | P | ints/f32/f64; complex refused. |
| `cumsum`/`cumprod` with `dtype=` | **N** | `OpCumSumDtypeAPI`/`OpCumProdDtypeAPI` missing. |
| `reduce_custom`, custom reducers | **X** | host closure. |
| `searchsorted`, `isin` | **N** | see §4. |

### Comparison of arrays

| Function | State | Note |
|---|---|---|
| `equal`, `not_equal`, ordering | Y | elementwise. |
| `isclose` | **N** | `OpIsCloseAPI` missing. |
| `allclose` | **N** | `OpAllCloseAPI` missing; `allclose_axes` is separately unimplemented on CPU. |
| `array_equal`-style `(a == b).all()` | Y | composition. |

---

## 2. Tier A — indexing and masking

| Function | State | Note |
|---|---|---|
| basic slicing, `diagonal`, `i`/`i_mut` | Y | views. |
| `index_select` | Y | host index slice; gather kernel. |
| `index_put` | Y | host index slice; scatter, last-write-wins order. |
| `array_index` (fancy indexing) | **N** | `DeviceArrayIndexAPI` missing; core already provides device index buffers. |
| `array_index_assign` | **N** | `DeviceArrayIndexAssignAPI` missing. |
| `mask_select` (boolean `x[mask]`) | **N** | `DeviceMaskIndexAPI` + `OpNonzeroAPI` missing. |
| `mask_fill` (`x[mask] = v`) | **N** | `DeviceMaskIndexAPI` missing. |
| `nonzero` | **N** | `OpNonzeroAPI` missing (count + scatter compaction). |
| `take_along_axis` | **B** | core bound demands `Raw = Vec<usize>` (host index tensor). |
| `put_along_axis` | **B** | same. |
| `where(mask, x, y)` as a mask read | P | same-dtype only. |

The boolean-mask family is the single largest ergonomic hole in Tier A: a user
who writes `x[x > 0]` gets a compile error, not a slow path.

---

## 3. Tier A — linear algebra products

| Function | State | Note |
|---|---|---|
| `matmul`, `matmul_from` (f32/f64) | Y | cuBLAS primary; `RSTSR_CUDA_MATMUL=tiled` forces the tiled kernel. |
| `matmul` (complex) | **N** | `DeviceMatMulAPI` bound is `CublasScalar` (f32/f64). cuBLAS itself supports `Zgemm`/`Cgemm`; the bound is the blocker. |
| `matmul` (integers, half) | **N** | same bound; tiled kernel not reachable. |
| `tensordot` (f32/f64, view-only layouts) | Y | delegates to the shared GEMM planner; refuses non-GEMM contractions. |
| `tensordot` (complex) | **N** | same `CublasScalar` bound. |
| `outer`, `cross`, `vecdot` | Y | `CudaArith`; complex and integers included. |
| `ext_matmul`, `ext_vecdot`, `ext_tensordot`, `ext_outer`, `ext_cross` | **N** | mixed-dtype `DeviceExt*API` missing. |
| `trace`, `matrix_transpose`, `diagonal` | Y | reduction/layout composition. |
| `matrix_power` | **N** | linalg-trait entry; no cuSOLVER/linalg tier. |

---

## 4. Tier B — sorting, searching, set operations

| Function | State | Note |
|---|---|---|
| `sort` | **N** | `OpSortAPI` missing. |
| `argsort` | **N** | `OpArgSortAPI` missing. |
| `sort_custom`, `argsort_custom` | **X** | host comparator. |
| `unique_values`, `unique_counts`, `unique_inverse`, `unique_all` | **N** | `OpUniqueAPI` missing (sort + dedup + inverse). |
| `searchsorted` | **N** | `OpSearchSortedAPI` missing. |
| `isin` | **N** | `OpIsinAPI` missing. |

This tier is entirely missing. None of it is blocked by core; it is a kernel
family (bitonic/radix sort plus the derived scans).

---

## 5. Tier C — domain linear algebra (cuSOLVER)

All 19 `rstsr-linalg-traits` entries are absent for `DeviceCuda`. The
`blas_impl` bodies exist but are written against the compile-time `DeviceBLAS`
alias, so they do not apply to a new device.

| Group | Functions | State |
|---|---|---|
| Decomposition | `cholesky`, `qr`, `svd`, `svdvals`, `eig`, `eigh`, `eigvals`, `eigvalsh` | **N** |
| Solve / inverse | `solve_general`, `solve_symmetric`, `solve_triangular`, `inv`, `pinv`, `matrix_power` | **N** |
| Scalar-valued | `det`, `slogdet`, `matrix_rank` | **N** |
| Norms | `matrix_norm`, `vector_norm` | **N** |

---

## 6. Tier D — device, dtype, interop, streaming

| Function / API | State | Note |
|---|---|---|
| `to_device` CPU→CUDA / CUDA→CPU | P | works but **blocking** (pageable path). |
| async pinned `to_device` both ways | Y | `DeviceCudaPinned`; non-blocking. |
| `to_scalar` | Y | one D2H + sync. |
| `to_vec` / `into_vec` | **P** | `to_vec` returns `Raw` = `CudaBuffer<T>`, **not** `Vec<T>`; `into_vec` (consuming) returns a host `Vec`. `to_cpu_vec` exists on `Storage`, not on the tensor. |
| `to_dlpack`, `from_dlpack` | **N** | no DLPack bridge for CUDA or pinned host. |
| `astype` / tensor-level dtype cast | **N** | absent **core-wide**; cross-dtype `assign` works, but there is no sugar. |
| `TensorArc` / shared storage | P | `CudaBuffer` is `Send + Sync`; not exercised. |
| `new_stream`, `with_stream`, `set_stream` | Y | per-device-value execution stream. |
| `CudaEvent` record/wait/sync/query | Y | user-side ordering. |
| `CudaSyncAPI::synchronize` | Y | device, CUDA tensor, pinned tensor. |
| core `DeviceSyncAPI` | **N** | trait does not exist yet; crate-local substitute. |
| pinned `PinnedBuffer`, `host_view` | Y | zero-copy host access. |
| `h2d_pinned_async`, `d2h_pinned_async` | Y | non-blocking edges with per-buffer event. |
| chunked 3-stream pipeline | P | demonstrated by `bench_streaming`; no library helper. |
| allocator configuration (`RSTSR_CUDA_ALLOC_CONF`, `empty_cache`) | **N** | D7 not implemented. |
| CUDA graphs | **N** | deferred. |
| multi-GPU peer copy | **P** | cross-ordinal copy exists but blocking and without peer access. |
| stream priorities | **N** | not exposed. |
| DLPack `kDLCUDA`/`kDLCUDAHost` | **N** | no bridge. |

---

## 7. What "common usage" would require

A reasonable definition of done, in increasing cost:

1. **Compile-and-run the Tier A list for f64/f32**, including `-a`, `tril`,
   boolean masking, `nonzero`, `array_index`, `allclose`, and complex `matmul`.
   This is the minimum for "I can port a CPU `rstsr` script".
2. **Add Tier B (sort/search/set)** so data-analysis code runs.
3. **Add Tier C (cuSOLVER)** so domain linear algebra runs.
4. **Close the core gates** (`Index`/`iter`, `take_along_axis`/
   `put_along_axis`) so idiomatic element access and the shared test bodies
   work.
5. **Streaming parity**: allocator framework, DLPack, multi-GPU peer copy,
   graphs, and a GPU CI/perf gate.

The first item is small and should be the next wave; items 2–3 are each a
coherent kernel family; item 4 is core work that benefits every accelerator.

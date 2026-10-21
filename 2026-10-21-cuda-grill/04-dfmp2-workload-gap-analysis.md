# DF-MP2 pair-energy workload: gap analysis against a future RSTSR CUDA backend

| | |
|---|---|
| RSTSR commit analyzed | `1c3bc11d874e8f196cda4ca71fab5065920b1a7f` (2026-10-20), workspace version `0.10.3`, `git describe` = `rstsr-v0.10.3-1-g1c3bc11` |
| Workload | `/home/a/rest_pack/rest/src/ri_pt2/py/dfmp2_addons.py` (1625 lines, mtime 2026-09-24) |
| Method | Full read of the workload; every RSTSR claim below was verified by `grep`/read of the checked-out source at the commit above (no builds run, no branch assumptions) |
| Legend | `L<n>` = line in the workload; `file:line` = RSTSR source. Device shorthand: **CS** = `DeviceCpuSerial`, **F** = `DeviceFaer`, **R** = the rayon auto-impl (`DeviceRayonAutoImpl`, which *is* `DeviceFaer`), **BLAS** = the five `crates-device` backends, which symlink the same `feature_rayon/auto_impl/*` files |
| Scope | Capability inventory, precision/TF32 design, async feasibility, multi-GPU threading model, verdict |

The workload is eager, autograd-free, f32-or-f64, one **single large 2-D GEMM** per
macro-batch step (deliberately not `bmm`, L580-584), a scalar fold with an f64
accumulator, and per-device cluster results assembled once. It is a good first
target for a CUDA backend because it uses a narrow slice of PyTorch — but that
slice includes several corners RSTSR has never had to express.

---

## 1. Capability table

### 1.1 Creation, transfer, dtype

| # | Torch usage (workload) | RSTSR today (item; file:line; devices) | Status | Severity | Notes for a CUDA backend |
|---|---|---|---|---|---|
| 1 | `torch.zeros([nocc,nocc], dtype=f64, device=dev)` (L106-110, 160-161, 213, 535, 539-540, 841, 856, 1225-1227) | `rt::zeros((shape, &device))` `rstsr-core/src/tensor/creation.rs:2688`; `(D,&B)` impl at `:2738` calls `DeviceCreationNumAPI::zeros_impl` (`:2721`); also `empty` `:432`, `ones` `:1944`, `full` `:1161`. CS/F/R/BLAS. | EXISTS | — | `zeros_impl` maps to `cudaMemsetAsync` on the device stream. dtype is static (inferred from the binding), not a runtime argument — see row 5. |
| 2 | `torch.as_tensor(np.asarray(x), dtype=, device=)` = `_as_torch` (L28-32, 93-95, 567-568, 824-829, 970-971, 1210, 1255) | `rt::asarray((vec, &device))` / `(&[T], &device)` / `(slice, layout, &device)` `rstsr-core/src/tensor/asarray.rs:302`; impls `:433`, `:565`; all route through `DeviceCreationAnyAPI::outof_cpu_vec` (`rstsr-core/src/storage/creation.rs:8`). CS/F/R/BLAS. | EXISTS (sync semantics) | workaround-exists | `outof_cpu_vec(&self, Vec<T>)` consumes and drops the host Vec, so it *must* be a blocking H2D. That matches the workload's own measurement that non-pinned `torch.as_tensor` is host-blocking and a side stream never helps (L481-491). Async H2D needs a new pinned-staging API (section 3). |
| 3 | `.to(device=)` on an already-resident tensor (no-op when same) (L31, 93, 528, 824, 970) | `to_device` / `into_device` / `change_device` `rstsr-core/src/tensor/device_conversion.rs:8-24`; device trait `rstsr-core/src/storage/conversion.rs:3-19`. Implemented **only among CPU devices**: CS↔F↔F (`rstsr-core/src/device_faer/conversion.rs:190-196`), CS→CS (`rstsr-core/src/device_cpu_serial/conversion.rs:3-35`), and CS/F↔BLAS↔BLAS (`crates-device/rstsr-openblas/src/conversion.rs:41-47`, same file in mkl/blis/aocl/kml). | PARTIAL | blocker (for `.cpu()`) | Today `change_device` is a *relabel* (same `Vec<T>`, new device field), not a copy — all CPU devices share `Raw = Vec<T>`. A real CUDA transfer impl must be written; the trait shape (consuming/borrowing, `Result`, associated `Repr`) is suitable. `Display` requires this impl too (row 24). |
| 4 | `.to(dtype)` elementwise cast (L156-157, 179, 195-196, 229, 243, 554-555, 573, 615, 637-638, 852-854, 872, 888) | **Nothing at tensor level.** `DTypeCastAPI` is a scalar trait only (`rstsr-dtype-traits/src/promotion.rs:21`), used inside device kernels. `astype` exists only in the Python shim `crates-interop/rstsr-faer-py/create/src/creation.rs:537`, whose header records "rstsr 0.9.0 has no tensor-level dtype conversion; gap register G-007" (`:1-3`, `:23-26`). A scan of every local branch (`git grep 'fn astype' <branch>`) finds no core implementation anywhere. | MISSING | **blocker** (hot path) | Workaround exists: `let mut o: Tensor<f64,B,D> = rt::zeros((shape,&dev)); o.assign(&x);` — `assign` casts on the fly (`rstsr-core/src/tensor/assignment.rs:24`, device bound `TA: DTypeCastAPI<TC>` at `rstsr-core/src/device_cpu_serial/assignment.rs:6`). Cost: a zero-fill pass plus a strided read/write. Acceptable for the two per-call energy vectors, not for `g_ab.to(f64)` (L179, 615) which is `jb*nvir*nvir` per step. |
| 5 | Runtime dtype drives the algorithm: `_is_f64()` and `mat_dtype = f64 if ... else f32` (L525, 665-669, 820, 1117-1122) | RSTSR is statically typed; dtype is a generic parameter, no runtime dtype object. | PARTIAL (by design) | workaround-exists | The Rust port must dispatch once per entry point (`match` on an input dtype enum → monomorphized f32/f64 path) or be generic over `T`. This is a porting pattern, not an RSTSR gap. The kernel bodies stay identical. |
| 6 | `.cpu().numpy()` (L319-320, 729-730, 928, 994-996, 1289-1298, 1600-1607) | Host-read primitives exist: `Tensor::into_vec` (1-D only, `rstsr-core/src/tensor/ownership_conversion.rs:530-546`), `Storage::to_cpu_vec` / `into_cpu_vec` (`rstsr-core/src/storage/device.rs:93`, `:101`), `to_raw_f` (1-D gather, `:447`). But no N-D `to_cpu` and no CUDA→CPU `change_device`. | PARTIAL | blocker | Recommended port shape: `t.to_contig(RowMajor).into_shape([-1]).into_vec()` (contig is a no-op for C-contiguous data), or implement `DeviceChangeAPI<DeviceCpuSerial>` for `DeviceCuda` and use `.to_device(&DeviceCpu)`. The latter also unblocks `Display`. |
| 7 | Explicit `del cderi_task` before the next upload (L644, 1244, 1257); allocator note `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` (L479-480, 1090-1091, 1418-1419) | Rust `Drop` on `Storage`/`DataOwned`; no allocator configuration anywhere. | PARTIAL | workaround-exists | `drop()` timing is deterministic (end of scope), matching the comments' intent. Without a pooling allocator, the per-task ~3.5 GB `cudaMalloc`/`cudaFree` would dominate the run, so a caching allocator is a practical requirement, and an allocator knob is needed for the memory-wall case the file documents. |

### 1.2 Layout, broadcasting, arithmetic

| # | Torch usage (workload) | RSTSR today (item; file:line; devices) | Status | Severity | Notes for a CUDA backend |
|---|---|---|---|---|---|
| 8 | Basic-slice read views `x[i:i+1]`, `x[j0:j0+jb]`, `occ_energy[j0:j0+jb]` (L135-136, 143, 172, 176, 221, 225, 569, 583, 862, 869) | `rt::slice` / `TensorAny::slice` `rstsr-core/src/tensor/indexing.rs:58`, `:95`; scalar select `i` `:117`; macro `s![..]`. CS/F/R/BLAS (layout-only). | EXISTS | — | Pure metadata; no sync, no allocation. |
| 9 | `transpose(-1,-2)` and `.mT` (L136, 171, 187, 220, 227, 575, 592, 623, 861) | `rt::swapaxes(&a, -1, -2)` `rstsr-core/src/tensor/manipulation/transpose.rs:635`; `rt::matrix_transpose` (`.mT`) `rstsr-core/src/tensor/linalg/matrix_transpose.rs:107`, implemented as `into_swapaxes_f(-1, -2)`. Also `transpose([...])` `transpose.rs:194`, `permute_dims` `:288`. | EXISTS | — | View only. |
| 10 | `.contiguous()` (L137, 172, 221, 576, 862) | `to_contig(order)` `rstsr-core/src/tensor/manipulation/to_contig.rs:151` → `TensorCow`; consuming `into_contig` `:205`. | EXISTS | — | Order argument is explicit (`FlagOrder::default()` is RowMajor in default features, `rstsr-common/src/flags.rs:58-64`). Pass `device.default_order()` for a torch-equivalent. Requires an `OpAssignArbitaryAPI` impl on the backend (the copy kernel). |
| 11 | `.reshape(naux,nvir)`, `.reshape(jb*nvir,naux)`, `.reshape(jb,nvir,nvir)` (L139, 143, 177, 226, 584, 870) | `into_shape` (consuming, may copy) `rstsr-core/src/tensor/manipulation/reshape.rs:857`, method `:1303`; `to_shape` (Cow) `:1324`; `reshape` `:1247`. Copy-on-write machinery in `reshape.rs:71-240`. | EXISTS | — | On a contiguous buffer all three reshapes are view-only (no copy), exactly as in torch. Note the flattening assumes row-major (section 3.4 of `order_semantics`); a CUDA device should default to `RowMajor`. |
| 12 | Broadcasting arithmetic with `(B,1,1)`, `(1,N,N)`, 0-d scalars: `occ[i] + occ[j0:j0+jb,None,None] + d_vv[None]`, `-vir[:,None]-vir[None,:]` (L112, 181-184, 189, 230-233, 238, 544, 595-607, 616-619, 627-631, 844, 873-883) | `rt::add/sub/mul/div` + Rust operators `+ - * /` `rstsr-core/src/tensor/operators/op_binary_arithmetic.rs:4-6` (trailing-axis alignment under RowMajor), generated at `:218-222`; `expand_dims` (None/newaxis) `rstsr-core/src/tensor/manipulation/expand_dims.rs:185`; rank-0 padding in `rstsr-common/src/layout/broadcast.rs:231-235`. | EXISTS | — | Row-major broadcasting matches NumPy/torch. Under `ColMajor` broadcasting aligns *leading* axes (`op_binary_arithmetic.rs:4-6`) — another reason the CUDA device should pin `RowMajor`. Rank-0 operands are legal. |
| 13 | Unary negation `-vir_energy[...]`, `-g_ab` implied (L112, 544, 844) | `rt::neg` / `Neg` `rstsr-core/src/tensor/operators/op_unary_arithmetic.rs:93`, `:128`. | EXISTS | — | — |
| 14 | Elementwise `sub`, `div`, `mul` on `(jb,nvir,nvir)` (L185-187, 190, 227, 235, 239, 592, 878, 884) | Same family as row 12. CS/F/R/BLAS via `OpSubAPI`/`OpDivAPI`/`OpMulAPI`. | EXISTS | — | These are the bandwidth-bound part; a fused kernel (`(t*g).sum(axes)`) is the obvious CUDA win, but the eager form is expressible. |
| 15 | 2-D `torch.matmul(b_2d, a_t)`: one `(jb*nvir, naux) @ (naux, nvir)` GEMM (L144, 177, 226, 584, 870) | `rt::matmul(&a, &b)` `rstsr-core/src/tensor/linalg/matmul.rs:133`, method `:529`, operator `%`; device trait `DeviceMatMulAPI` `rstsr-core/src/operators/matmul.rs:7`; dispatch `op_refa_refb_matmul` `matmul.rs:279-304`. Implemented by CS (naive, `device_cpu_serial/linalg/matmul.rs:10`), F (faer, `device_faer/matmul.rs:274`), BLAS (`crates-device/rstsr-openblas/src/matmul.rs:284`, same file in mkl/blis/aocl/kml). BLAS fast path dtypes: f32, f64, c32, c64 (`:273-276`), generic fallback otherwise. | EXISTS | — | The reshape trick is **fully expressible**: `b_2d = cderi.slice(s![j0..j0+jb]).into_shape([jb*nvir, naux])` (view, contiguous) → `rt::matmul(&b_2d, &a_t)` → `.into_shape([jb,nvir,nvir])` (view). Output allocation is `uninit_impl` (`matmul.rs:303`), i.e. no zero-fill, matching torch. CUDA: `cublasGemmEx`/`cublasLtMatmul` on the row-major buffer. |
| 16 | Batched matmul is *not* used (deliberately avoided, L167-168, 580-584) | `rt::matmul` does support batch dims with broadcasting (rule table `rstsr-core/src/tensor/linalg/matmul.rs:25-37`; layout logic `rstsr-common/src/layout/matmul.rs:285`). | EXISTS (unused) | — | No action needed; noted so nobody "optimizes" the port back to `bmm`. |

### 1.3 Reduction and host scalars

| # | Torch usage (workload) | RSTSR today (item; file:line; devices) | Status | Severity | Notes for a CUDA backend |
|---|---|---|---|---|---|
| 17 | `.sum(dim=(1,2))`, output dtype = input dtype (L186-187, 195-196, 236, 243, 601, 612, 622-623, 637-638, 879, 888) | `rt::sum_axes(&x, [1,2])` or `rt::sum_with_args(&x, ([1,2], false))`; generated at `rstsr-core/src/tensor/reduction.rs:741`; device trait `OpReduceAPI` `rstsr-core/src/operators/reduction.rs:23-36` with `TOut = T` for CS (`device_cpu_serial/reduction.rs:12`). `AxesIndex` accepts `[1,2]`, `vec![1,2]`, `(…)` (`rstsr-common/src/axis_index.rs:5-81`). CS/R/F/BLAS. | EXISTS | — | Multi-axis in one call is native (no sequential reduce, no intermediate `(jb,nvir)` tensor). |
| 18 | `.sum(dim=(1,2), dtype=torch.float64)` — f32 elements accumulated in f64 (L192-193, 241, 610, 634-635, 886) | `rt::sum_with_dtype::<f64>(&x, ([1,2], false))`; tensor layer `rstsr-core/src/tensor/reduction.rs:761-857` (dispatch `:773`, instantiation `:971`); device trait `OpReduceDtypeAPI` `rstsr-core/src/operators/reduction.rs:46-56`. **Merged to master** (commit `f736987`, PR #110, ancestor of HEAD); tests `rstsr-core/tests/core_func/reduction/test_reduce_dtype.rs:14`, `test_cumulative.rs:168`. CS/R/F/BLAS. | EXISTS | — | This is exactly the "f32acc" fold: the CPU kernel accumulates with `|acc, x| acc + x.into_cast()` (`rstsr-core/src/device_cpu_serial/reduction.rs:509-531`, closure at `:528`), i.e. per-element cast, no input-sized f64 copy. A CUDA kernel must accumulate in f64 registers (not f32-then-cast) to match. |
| 19 | `.sum()` to a Python float (numpy side, L997-999, 1306-1308, 1615-1618) and any `.item()` need | Whole-array: `rt::sum(&t)` → host `T` (`rstsr-core/src/tensor/reduction.rs:619-624`, calling `sum_all` `:592-597`; device op returns `Result<Self::TOut>`, `operators/reduction.rs:28`). Element: `TensorAny::to_scalar` `rstsr-core/src/tensor/ownership_conversion.rs:654`, reads at the layout offset via `Storage::get_index` (`:603-613`, `storage/device.rs:108`). | EXISTS | — | Both are host-read sync points by design (`func_all` returns a host scalar from the device trait — a CUDA backend cannot make a whole-array reduce fully async, and should not try). Minor wart: `sum_axes(..., AxesIndex::None)` produces a 0-d tensor by H2D-ing the host scalar (`reduction.rs:579-583`, and `reduce_dispatch_output!` `:781`) — two transfers for `sum()`. Prefer `rt::sum(&t)` (row above). |
| 20 | `torch.cuda.synchronize()` around timed regions (L130, 145, 163, 202, 215, 247, 558, 647, 891) | Nothing. No stream, no sync, no event anywhere in core (a repo-wide grep for `stream|synchronize|cuda|non_blocking|pinned` over `rstsr-core/src`, `rstsr-common/src`, `rstsr/src` returns zero hits). | MISSING | blocker | Needs `rt::cuda::synchronize(&device)` (and a stream handle if more than one stream ever exists). |

### 1.4 Writes into sub-regions

| # | Torch usage (workload) | RSTSR today (item; file:line; devices) | Status | Severity | Notes for a CUDA backend |
|---|---|---|---|---|---|
| 21 | Symmetric slice assignment `eng_pair_bi1[i, j0:j0+jb] = e` and `eng_pair_bi1[j0:j0+jb, i] = e` (L197-200, 244-245, 624-625, 639-640, 889, 1231) | Mutable basic-index view `slice_mut` `rstsr-core/src/tensor/indexing.rs:145` (`i_mut` `:188`, `diagonal_mut` `:564`) + whole-view `assign` `rstsr-core/src/tensor/assignment.rs:102` (broadcast + dtype cast; rejects broadcasted destinations `:125`). CS/R/F/BLAS via `OpAssignAPI` (`rstsr-core/src/operators/assignment.rs:30`; impls `device_cpu_serial/assignment.rs:29`, `feature_rayon/auto_impl/assignment.rs:29`). | EXISTS | — | `eng.slice_mut(s![i, j0..j0+jb]).assign(&e_bi1)`. This is a strided scatter kernel on CUDA (one row/column), not a memcpy. |
| 22 | Long-index-tensor assignment `eng_pair_ss[i, jcols] = e_ss` (L613, 624-625, 639-640), `jcols` a `torch.long` device tensor (L572, 585) | `rt::array_index_assign(&mut a, (i, jcols), &e)` `rstsr-core/src/tensor/array_indexing.rs:1113` (method `:1167`); `ArrayIndexer::ArrayIndex(Tensor<isize,B,IxD>)` is a first-class variant (`rstsr-core/src/tensor/array_indexer.rs:27-47`, line 34). Alternative: `rt::index_put(&mut a, axis, indices, value)` `rstsr-core/src/tensor/adv_indexing.rs:1395`, but its indices are **host lists only** (`I: TryInto<AxesIndex<isize>>` `:1276`; device trait takes `&[usize]` `rstsr-core/src/operators/adv_indexing.rs:92`). Device op `DeviceArrayIndexAssignAPI` `operators/adv_indexing.rs:207`; implementations CS + R only. | PARTIAL | workaround-exists | Two caveats. (a) Even with a device index tensor, the tensor layer *resolves indices on the host*: `array_indexing.rs:458-482` walks the index entries into a `Vec<usize>` and re-uploads via `outof_cpu_vec` — a D2H+H2D round trip per call. (b) In this workload `jcols` is always a contiguous global range (built as `[nocc_full + i for i in range(...)]`, L395-401), so a plain `slice_mut(s![i, jstart..jend]).assign(&e)` (row 21) is a faithful and cheaper replacement. Treat row 22 as "supported on paper, avoid in practice". |
| 23 | `torch.as_tensor(occ_idx_task, dtype=torch.long, device=dev)` (L572) | `rt::asarray((vec_of_isize, &device))` `rstsr-core/src/tensor/asarray.rs:433`; `isize` is a supported element type, and `isize` index tensors are what `ArrayIndexer::ArrayIndex` wants. | EXISTS | — | Needed only if row 22's device-index route is taken; the host list is already available in the Python code, so the Rust port can skip the tensor entirely. |
| 24 | Assembly on host: `pair[slc, slc] = intra.cpu().numpy()`, `pair[slc, :] += inter_np`, `pair[:, slc] += inter_np.T` (L1289-1299, 1600-1607) | Same primitives on device: `slice_mut(...).assign(&intra)`, in-place `rt::add_assign` / `+=` (`rstsr-core/src/tensor/operators/op_binary_assign.rs:58`, `:121`, module doc `:1-6`), transpose view for `inter.mT`. | EXISTS | — | The whole assembly can stay on the GPU (no D2H), which removes the file's single-sync-point design constraint (L1276, L1589). Only the final scalar needs a host read (row 19). |
| 25 | `del cderi_task, b_2d` before the next iteration allocates (L644) | `drop()` / end of scope. | EXISTS | — | Ordering is explicit in Rust; pair with a caching allocator. |

### 1.5 Device, thread, introspection

| # | Torch usage (workload) | RSTSR today (item; file:line; devices) | Status | Severity | Notes for a CUDA backend |
|---|---|---|---|---|---|
| 26 | `torch.cuda.is_available()` / `device_count()` (L85, 523, 818, 963, 1126, 1451, 1387) | Nothing. | MISSING | workaround-exists | `rt::cuda::device_count()`, `is_available()`. Small, but every entry point of the file branches on it. |
| 27 | `torch.cuda.mem_get_info(d)[0]`, `min(...)` across devices (L1139, 1368) | Nothing. | MISSING | blocker for the auto-batching heuristic | `rt::cuda::memory_info(&device) -> (free: usize, total: usize)` via `cuMemGetInfo`; the whole `nbatch_from_avail*` logic (L420-433, L1020-1044) consumes exactly this. |
| 28 | `with torch.cuda.device(dev_id):` per worker (L1196-1200, 1521-1525) | No global current device: the device travels inside every tensor (`Storage` owns `device: B`, `rstsr-core/src/storage/device.rs:15-21`; `TensorAny::device()` `rstsr-core/src/tensorbase.rs:210`). | PARTIAL | workaround-exists | The RSTSR model is strictly better for this workload (no hidden state). What is still needed is a per-thread *CUDA context binding* (equivalent to `torch.cuda.device(id)`) because the driver's "current context" is thread-local; put it in `DeviceCuda::bind_to_current_thread()` or do it on `DeviceCuda::new`. |
| 29 | TF32 toggled through the process-global `torch.backends.cuda.matmul.allow_tf32`, hoisted around the parallel region to avoid races (L35-50, 314, 715, 918, 967, 1262, 1577) | Nothing; no notion of a math mode on any device. | MISSING | blocker (for the f32 path) | See section 2. The per-device design also *removes* the race the file works around (L40-43, L1260-1262). |
| 30 | One worker thread per GPU, each pinned to its device, results stay on device (L1374-1392, L1193-1258, L1518-1573) | Frame-level: no Send/Sync requirement on the device trait (`DeviceAPI: Clone + Default`, `rstsr-core/src/storage/device.rs:159`); tensors are movable (`DataOwned<C>: Send where C: Send`, `rstsr-core/src/storage/data.rs:51`); no globals in core apart from the print limits (`rstsr-core/src/format/format_tensor.rs:5-6`). | PARTIAL | workaround-exists | See section 4. Needs `DeviceCuda: Send + Sync` and a `Raw` buffer type that is `Send + Sync`; core imposes nothing, so this is entirely in the backend's control. |

### 1.6 Notes on the four verifications the brief called out

**Reduction with accumulate dtype — merged, tested, on master.** `git merge-base
--is-ancestor f736987 HEAD` succeeds; `sum_with_dtype` / `prod_with_dtype` /
`mean_with_dtype` / `var_with_dtype` / `std_with_dtype` exist
(`rstsr-core/src/tensor/reduction.rs:967-975`), with the element-cast-fold
contract documented at `:761-766`. `ReduceArgs` accepts `([1,2], false)`,
`Vec<isize>`, `[isize; N]`, bare axis, `()`, `None`, and `bool` keepdims
(`reduction.rs:176-330`). No branch check is needed: the feature is not sitting
anywhere.

**Dtype cast — missing everywhere.** The only implementation of a tensor cast in
this clone is the Python-facing shim (`crates-interop/rstsr-faer-py/create/src/creation.rs:537`),
which carries its own cast table precisely because the core has none (G-007).
`DTypeCastAPI` (`rstsr-dtype-traits/src/promotion.rs:21`) is a *scalar*
conversion trait and is used inside kernels (`device_cpu_serial/assignment.rs:6`,
`device_cpu_serial/reduction.rs:511`). There is no `rt::astype`, no `Tensor::to`,
no `cast`. Workaround: allocate + `assign` (row 4).

**Writes into sub-regions — three distinct levels exist.** (i) Whole-tensor
`assign`/`fill` (`assignment.rs:102`, `:241`); (ii) mutable basic-index views +
`assign` (`slice_mut` `indexing.rs:145`, `i_mut` `:188`, `diagonal_mut` `:564`),
which is the exact analogue of `x[i, j0:j1] = e`; (iii) fancy/index writes:
`index_put` (host index list, inverse of `index_select`), `array_index_assign`
(host lists *and* device index tensors), `mask_assign`/`mask_fill`,
`put_along_axis` (`rstsr-core/src/tensor/adv_indexing.rs:1264`, `:1395`,
`:1412`, `:1089`; `array_indexing.rs:901`). The one semantic gap is that
`array_index_assign` resolves index arrays on the host before dispatching
(`array_indexing.rs:458-482`), so a device-resident index tensor buys ordering
convenience but not a device-only path.

**matmul.** One trait (`DeviceMatMulAPI`), three implementation families (serial
naive, faer, BLAS-the-crate-family shared by five crates), f32/f64/c32/c64
on the BLAS fast path, generic fallback otherwise, batched broadcast per the
array-API rule table, `alpha`/`beta` in `matmul_from`, and an uninitialized-output
path used by the allocating form. The `(jb*nvir, naux) @ (naux, nvir)` reshape
trick is expressible with zero copies on the operands (contiguous slices) and
zero copies on the result (contiguous output). The only implicit assumption is
row-major flattening, which is also RSTSR's default order.

---

## 2. Precision and dtype

**f32 / f64.** Fully supported end to end; this workload needs nothing else on
the numeric path.

**f16 / bf16.** Present but thin: the `half` crate is a default feature of
`rstsr-dtype-traits` (`rstsr-dtype-traits/Cargo.toml:16`, `:21-22`) and core
depends on it with `features = ["half"]` (`rstsr-core/Cargo.toml:15`).
Trait impls exist for `ExtZero` (`ext_zero.rs:36-37`), `ExtFloat`
(`ext_float.rs:20-30`), `ExtReal` (`ext_real.rs:159-160`), `ExtSortCmp`
(`ext_sort_cmp.rs:91-92`), and promotion (`promotion.rs:94-95`). What does **not**
exist is a fast matmul path: the BLAS dispatch lists only `f32, f64, c32, c64`
(`crates-device/rstsr-openblas/src/matmul.rs:273-276`), so half precision falls
into the naive scalar kernel. For CUDA that is the *opposite* of the desired
behavior (tensor cores are f16/bf16-native). A CUDA backend should add its own
dispatch to `cublasGemmEx` with `CUDA_R_16F`/`CUDA_R_16BF` inputs and
`CUDA_R_32F` compute — the type-level plumbing is ready; only the dispatch is
missing. Complex (c32/c64) is fully supported and needs no work here.

**TF32, dynamically, not via environment.** The workload needs a *runtime*
toggle (L35-50, L283-285) because the same process runs f32 contractions both
ways. RSTSR has no math-mode concept anywhere. Three design options, in RSTSR
terms:

1. **Field on a stateful `DeviceCuda`.** This is the framework's stated
   philosophy (device-specific state lives in the device), and there is a direct
   precedent: `DeviceFaer { base: DeviceCpuRayon { num_threads, pool,
   default_order } }` (`rstsr-core/src/device_faer/device.rs:6-8`,
   `rstsr-core/src/feature_rayon/device.rs:41-51`), mutated through
   `set_num_threads(&mut self)` (`feature_rayon/device.rs:8`) and compared in
   `same_device` (`device_faer/device.rs:46-51`).
   The catch: operations never receive the caller's device — they call
   `a.device()` on the *input tensor* and clone it (`matmul.rs:295-304`), and
   every `Storage` owns its own device clone. A plain `math_mode: MathMode`
   field would therefore be a *snapshot*: toggling the original device would not
   affect tensors created earlier, and the workload reuses `cderi_ovl` across
   calls with different TF32 settings (L314, 715, 918, 967). Conclusion: if it
   lives in the device, it must be shared state, e.g.
   `DeviceCuda { state: Arc<CudaDeviceState { ordinal, math_mode: AtomicU8,
   allocator, ... }> }`, with `set_math_mode(&self, ...)` (interior mutability,
   *not* `&mut self`) so a `&DeviceCuda` obtained from a tensor can flip it.
2. **Per-call argument.** Precedent: `change_shape_with_args`/`ReshapeArgs`
   (`rstsr-core/src/tensor/manipulation/reshape.rs:194-240`), `SortArgs`,
   `SearchSortedArgs`. This is the cleanest stateless design and maps 1:1 to
   `cublasLtMatmul`'s per-call `computeType`. But `rt::matmul` has no args
   parameter today (`matmul.rs:133-148`), and `%` is also matmul
   (`op_binary_arithmetic.rs:21-27`), so it means either a new entry point
   (`matmul_with_args`) or a breaking signature change. It also does not match
   how the workload uses the flag (a region, not a call).
3. **Context-scoped guard.** `let _g = rt::cuda::with_math_mode(&device, Tf32);`
   mirrors the Python `with_tf32` (L35-50) most directly and is RAII-safe on
   panic. On its own it still needs shared state underneath (option 1), but it
   gives the ergonomics the port wants.

**Recommendation.** Option 1 + 3: shared per-device state with an atomic,
exposed as `rt::cuda::set_math_mode(&DeviceCuda, MathMode)` and a
`with_math_mode` guard. Do **not** include `math_mode` in `same_device` (the
`DeviceFaer` precedent includes `num_threads`, but including a precision knob
would reject `a × b` across a toggle, which is exactly what the workload does).
Underneath, prefer per-call compute types (`cublasLtMatmul` with an explicit
`computeType`, or `cublasGemmEx` with `CUBLAS_COMPUTE_32F_FAST_TF32`) over
`cublasSetMathMode`, which is per-handle state; a handle shared by two worker
threads would reintroduce the very race the Python file documents at L40-43.
Per-device state also *fixes* that race: two GPUs can hold different modes
concurrently, unlike torch's single process-global flag. cudarc ships a
`cublaslt` module; confirm at implementation time that the compute-type field is
surfaced in the Rust wrapper before committing to this route.

**The "f32acc" fold.** `MP2_FOLD=f32acc` (L114-117, L191-193) is, in RSTSR
terms, `sum_with_dtype::<f64>` over an f32 tensor: elements cast per element
inside the accumulate closure, accumulation in f64, no input-sized f64 copy
(`rstsr-core/src/device_cpu_serial/reduction.rs:509-531`). That design is
**merged and tested on master**, so the fold is a solved problem; the CUDA
kernel must implement `OpReduceDtypeAPI` with f64 accumulators to inherit it.
The other two modes map to `MP2_FOLD=f32` → `sum` + a cast (blocked by the
missing cast) and `MP2_FOLD=f64` → a cast of `g_ab` before the fold (also
blocked).

---

## 3. Streaming / async

RSTSR is host-synchronous eager today: no stream type, no event, no async copy,
no `synchronize`. The question is whether the *API shape* would survive an
asynchronous CUDA backend. Analysis, with no code changes assumed:

**What the tensor layer reads.** Every operator takes `&B::Raw` plus
`&Layout<D>` and returns device storage (e.g. `matmul.rs:294-307`: layouts and
`same_device`, then `uninit_impl`, then the device op, then
`assume_init_impl`). Creation allocates with `uninit_impl`
(`rstsr-core/src/storage/creation.rs:14-19`). There is no per-op host read. So
async launches are semantically identical as long as the per-device stream
orders the kernels and the allocator is stream-ordered.

**Enumerated host-read points** (the complete list, from a grep of the tensor
layer):

1. `to_scalar`/`to_scalar_f` (`ownership_conversion.rs:603-654`) — one element
   via `Storage::get_index` (`storage/device.rs:108`).
2. `to_vec`/`to_raw_f`/`into_vec`/`into_raw` — 1-D only
   (`ownership_conversion.rs:447-546`), plus `Storage::to_cpu_vec` /
   `into_cpu_vec` (`storage/device.rs:93`, `:101`).
3. `Debug` (`format/format_tensor.rs:268-284`) and `Display` (`:215-235`, which
   additionally requires a `DeviceChangeAPI<…, DeviceCpuSerial, …>` impl, i.e. a
   mandatory D2H path just to print).
4. Whole-array reductions: `OpReduceAPI::func_all` returns a *host* scalar
   (`operators/reduction.rs:28`); the tensor layer then re-uploads it to build a
   0-d tensor (`tensor/reduction.rs:579-583`, `:781`).
5. Data-dependent output shapes: `nonzero` (`tensor/nonzero.rs:8`, `:147`),
   `mask_select`/`bool_select` (`tensor/adv_indexing.rs:797`, `:350`),
   `unique_*` (`tensor/set.rs`). These need a count back on the host before
   allocating.
6. Index-array resolution inside `array_index`/`array_index_assign`
   (`tensor/array_indexing.rs:458-482`) — a D2H/H2D round trip even when the
   index tensor is already on the device.
7. H2D uplinks: `asarray` from host (`asarray.rs:442`, `:457`, `:883`) and the
   `outof_cpu_vec` calls above.

**What would silently over-synchronize.** Nothing on the compute path. The
risks are: (a) `Display`/`Debug` on a large tensor (a full D2H; `Debug` also
requires `Raw: Clone`, i.e. potentially a D2D copy, `format_tensor.rs:270`); (b)
`Storage::to_cpu_vec` requires `Raw: Clone` (`storage/device.rs:93-99`) — for a
CUDA buffer that is a device-to-device copy if `Clone` is a memcpy; (c) the
reduction-to-0-d double transfer (point 4); (d) `array_index*`'s host
round-trip; (e) `op_with_func`, which hands a **host Rust closure**
`FnMut(&mut MaybeUninit<TC>, &TA, &TB)` to the device
(`tensor/operators/op_with_func.rs:11-20`) — not implementable on a GPU without
either D2H or a kernel compiler, so it must stay `Unimplemented` (or host-fallback
with a sync). Nothing in this workload uses it.

**Drop / allocator.** A device buffer dropped while its last kernel is still in
flight is safe if free and reuse are stream-ordered on the *same* stream (the
cudaMallocAsync/caching-allocator model). Cross-stream reuse needs
`record_stream`-style tracking; since the workload is single-stream per device,
a simple pool keyed by the device stream is enough. The file's explicit `del`
(L644) becomes deterministic `drop()`, and the pooling question is the same one
that `PYTORCH_CUDA_ALLOC_CONF` answers today (L479-480, L1090-1091).

**Recommendation on `Raw`.** Make `DeviceCuda::Raw` a cheaply clonable *handle*
(e.g. `Arc`-backed buffer descriptor with an owning drop), not a bare
`CudaSlice`. `Raw: Clone` is required by `to_cpu_vec`, `Debug`, `to_vec` and
several `DataCloneAPI` paths; if `Clone` means a memcpy, printing or reading
becomes quadratic in a way that is easy to miss.

**Verdict.** The current eager API *does* admit a cupy-style async CUDA backend
without changing any tensor-layer signature. The minimal additions are a
`Stream` handle (or one implicit stream per device), `rt::cuda::synchronize`,
pinned-host staging for *optional* async H2D, `rt::cuda::memory_info`, and an
allocator knob. One API shape must be respected: `outof_cpu_vec(Vec<T>)` must
remain blocking (it consumes and drops the source buffer); true async upload
needs a `PinnedBuf<T>` type whose lifetime outlives the copy.

---

## 4. Multi-GPU

**The workload's driver model** (L1193-1258, L1518-1573): one worker thread per
entry of `device_list`, each entering `torch.cuda.device(dev_id)`; each worker
owns the clusters at positions `pos + k*ndevice`; the cluster's cderi is
uploaded once and freed before the next; per-cluster pair matrices stay on the
device; the single assembly loop pulls each block to CPU once (L1276-1299,
L1589-1608). TF32 is hoisted around the whole parallel region because the flag
is process-global (L1260-1262).

**Mapping to Rust.** `std::thread::scope` with one `DeviceCuda` constructed and
bound inside each worker is a direct translation. What core requires, verified:

- `DeviceAPI<T>: DeviceBaseAPI + DeviceRawAPI<T> + DeviceStorageAPI<T> + Clone + Default`
  (`rstsr-core/src/storage/device.rs:159`). **No `Send`/`Sync` bound.** A CUDA
  backend must add `unsafe impl Send/Sync` (or rely on its fields) and accept the
  `Default` requirement — `DeviceCuda::default()` must pick ordinal 0 (a
  fallible `new(ordinal)` is fine, but `Default` still has to exist). If a
  fallible or handle-typed default is undesirable, relaxing `Default` on
  `DeviceAPI` is a small framework change worth considering.
- A device is carried by value inside every `Storage` (`storage/device.rs:15-21`,
  `tensorbase.rs:210`), so there is no "current device" to fight over; a tensor
  moved to another thread carries its device handle with it. `same_device` must
  compare the **ordinal** (not the context pointer, not the stream), otherwise
  two handles for GPU 1 fail the `DeviceMismatch` asserts in `matmul.rs:295`,
  `assignment.rs:122`, `op_binary_common` etc.
- No global state in core: a grep for `static`/`thread_local!`/`lazy_static`
  finds only `MIN_PRINT`/`MAX_PRINT` (`format/format_tensor.rs:5-6`). Rayon
  pools are per device *instance* (`Arc<ThreadPool>` in
  `feature_rayon/device.rs:41-51`), which is exactly the precedent a CUDA
  backend should follow: no process-global pool, no TLS execution context beyond
  the CUDA driver's thread-local current context (bind it per worker, like
  L1196-1200).
- Carrying results across threads: move the `Tensor<f64, DeviceCuda, IxD>` out
  of the worker (`DataOwned<C>: Send where C: Send`,
  `storage/data.rs:51`); use `TensorArc` (`tensorbase.rs:82`,
  `DataArc<C>: Send + Sync where C: Send + Sync`, `storage/data.rs:57-58`) if
  several threads must share one result. Both are backend-controlled, since they
  reduce to `Raw: Send`/`Raw: Send + Sync`.
- Assembly can stay on the device (`slice_mut(...).assign`,
  `rt::add_assign`/`+=`, row 24), so the file's "pull to CPU at the single sync
  point" can be dropped entirely; only the final scalar sum syncs.

**What could still fight per-thread devices.** (i) A CUDA backend that caches a
single "current stream/context" in a process-global; (ii) a global allocator
lock that serializes workers (use per-device pools); (iii) a shared cuBLAS
handle across workers (make it per device, like the per-device `Arc<ThreadPool>`).
None of these are imposed by RSTSR.

---

## 5. Verdict

**(a) Can this file be ported to RSTSR-CUDA on today's API, as-is? No.**
The *computational* surface is nearly all present (matmul, broadcasting,
multi-axis reductions with an accumulate dtype, slice assignment, in-place
assembly), but three API-level holes sit directly in the inner loop or on the
boundary of the process: no tensor cast, no cross-device transfer, no
stream/sync, plus the entire CUDA backend itself. Note that "no CUDA backend" is
not an API gap — the framework's device abstraction is genuinely sufficient; the
gaps below are the ones a backend author cannot paper over locally.

**(b) Ordered blocking gaps.**

1. **No CUDA device** (the project itself) — and with it, no `Raw` buffer type,
   no allocator, no kernels. Everything below is dwarfed by this.
2. **No tensor-level dtype cast** (`astype`/`to`); gap G-007, absent from every
   branch of this clone. Blocks `MP2_FOLD=f32` (L195-196) and `MP2_FOLD=f64` (L179, L615)
   outright, and makes the mandatory `mat_dtype` conversions (L156-157, 554-555,
   852-854) allocate-and-fill instead of cast. This is the one *pure API* gap
   that lands inside the hot loop.
3. **No cross-device transfer** — `DeviceChangeAPI` is implemented only among
   CPU devices and is a relabel, not a copy
   (`device_cpu_serial/conversion.rs:3-35`, `device_faer/conversion.rs:190-196`,
   `crates-device/rstsr-openblas/src/conversion.rs:41-47`).
   Needed for `.cpu()` (L319, 994, 1289, 1600) and, because `Display` requires
   it, for `println!` (`format/format_tensor.rs:215-235`).
4. **No stream / synchronize / event API** (zero hits in core). The workload
   times regions with `torch.cuda.synchronize()` (L130, 145) and relies on
   default-stream ordering everywhere.
5. **No device introspection** — `memory_info` and `device_count` drive the
   `nbatch` heuristics (L420-433, L1020-1044, L1139, L1368); both are one driver
   call each but have no home today.
6. **No math-mode / TF32 notion**, and no per-call precision argument on matmul.
   Blocks the f32 fast path the workload exists to exercise (section 2).
7. **`DeviceAPI: Default`** forces a `DeviceCuda::default()`; minor, but it
   should be a deliberate decision (ordinal 0) rather than an accident.
8. **Data-dependent-shape and host-read paths are inherently synchronous**
   (reductions to a scalar, `nonzero`, `mask_select`, `Display`). Not blockers
   for this workload — but a port must know that `sum()` and `to_scalar()` are
   its sync points, and that `array_index_assign` round-trips indices through
   the host (`array_indexing.rs:458-482`).
9. **Half precision has no matmul path** (BLAS dispatch is f32/f64/c32/c64
   only). Irrelevant to this file, relevant to the stated FP16/BF16 goal.

**(c) Already surprisingly fine (all verified in source).**

- The whole contraction is a functional, allocation-per-op eager graph — the
  same shape RSTSR has.
- The single-large-GEMM trick is exactly expressible with no hidden copies
  (contiguous slice → `into_shape` view → `matmul` → `into_shape` view).
- `sum_with_dtype::<f64>` is precisely `sum(dim=(1,2), dtype=f64)` including the
  per-element cast fold, merged and tested on master.
- Multi-axis reduction in one call; no sequential-reduce workaround.
- Row-major broadcasting of 0-d, `(B,1,1)`, `(1,N,N)` and `(B,N,N)` matches
  torch; `swapaxes`, `matrix_transpose`, `to_contig`, `into_shape` cover
  `.transpose(-1,-2)`, `.mT`, `.contiguous()`, `.reshape`.
- Slice assignment and in-place `+=` on mutable views make the whole assembly
  doable on device — arguably *better* than the file's D2H assembly.
- Creation with a device argument and tuple overloads is idiomatic
  (`rt::zeros((shape, &device))`, `rt::asarray((vec, &device))`).
- `to_scalar` was deliberately made device-generic (`ownership_conversion.rs:603-613`
  reads one element rather than cloning storage).
- The device model (state in the device, `Clone` per storage, no globals) fits
  one-device-per-thread exactly; DLPack already has a documented seam for
  accelerator backends (`crates-interop/rstsr-cpu-dlpack/src/device.rs:11-29`:
  "an accelerator backend … would provide its own implementations in its own
  bridge crate"), and the facade already has a namespaced member-crate pattern
  (`rt::dlpack`, `rstsr/src/prelude.rs:5-14`, feature at `rstsr/Cargo.toml:75`)
  that `rt::cuda::` can copy.
- No autograd, no `bmm`, no dtype promotion in anger: the numeric contract is
  simple.

**(d) Estimated `rt::cuda::` surface.** Roughly 8-12 public items plus 3-4 trait
impl families:

- `DeviceCuda` (`ordinal`, context, per-device stream, shared `math_mode`,
  allocator handle) + `Default`; `rt::cuda::device(ordinal)`; thread binding.
- `rt::cuda::synchronize(&device)`; `Stream` + `stream()`/`with_stream` (can be
  deferred if one implicit stream per device is the v1 model).
- `rt::cuda::memory_info(&device) -> (free, total)`; `device_count()`;
  `is_available()`.
- `MathMode` enum + `set_math_mode`/`math_mode`/`with_math_mode` guard.
- `PinnedBuf<T>` + async H2D (`copy_h2d_async`), plus a blocking default path
  via the existing `outof_cpu_vec`.
- Allocator configuration (`set_allocator` or `RSTSR_CUDA_ALLOC_CONF`).
- Trait impls: `DeviceChangeAPI` for CUDA↔CPU and CUDA↔CUDA (also unlocks
  `Display`); `DeviceCreationAnyAPI`/`DeviceCreationNumAPI`; `DeviceMatMulAPI`
  (with the math-mode read); `DeviceReduceDtypeAPI`; `OpAssignAPI` +
  `DeviceArrayIndexAssignAPI` (needed for row 22 if that route is taken).
- Optional: a DLPack bridge crate for torch/NumPy interop, following
  `rstsr-cpu-dlpack`.

The largest *engineering* items are not on this list — they are the elementwise
kernel set (nvrtc or precompiled) and the caching allocator. On the API side the
work is small; the two that matter are the cast and the cross-device copy.

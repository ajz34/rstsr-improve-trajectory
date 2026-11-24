# Roadmap — from a workload slice to a general CUDA backend

Priorities answer one question: **what makes ordinary `rstsr` code run, then
run fast, then run safely at scale?** Sizes are relative (S ≤ a day of focused
work, M a few days, L a wave). Every item names an acceptance test that can be
checked on the 8 GiB card unless stated otherwise.

Baseline: `261021/cuda` @ `9c7d49a`. Coverage evidence:
[COVERAGE.md](COVERAGE.md). Defects to fix along the way:
[STATUS.md](STATUS.md) §9.

---

## P0 — "port a CPU script" wave (W1)

These are small, independent, and remove the most common compile errors. They
can land as one PR.

### [ ] P0-1. Implement `OpNegAPI`

- **Why**: `let b = -&a;` and `rt::neg` do not compile on `DeviceCuda`.
- **Work**: add the negate row to the elementwise op table (`operators.rs` /
  `kernels/elementwise.rs`); `CudaArith` already covers every dtype that has
  `Neg` on the CPU; make the kernel use `0 - x` for unsigned integers if the
  CPU does.
- **Accept**: compile probe `-&a` for f64/f32/i32/`Complex<f64>`; parity with
  `DeviceCpuSerial` on the existing elementwise wave; `test_ops_wave` gains a
  neg case.
- **Size**: S.

### [ ] P0-2. `tril` / `triu` via `DeviceCreationTriAPI`

- **Why**: standard Array-API creation; every triangular-mask script needs it.
- **Work**: implement `DeviceCreationTriAPI<T>` for `DeviceCuda` with the same
  out-of-place fill semantics as the CPU (kernel writes only the selected
  triangle; the rest is the identity/zero fill the trait specifies).
- **Accept**: `tril`/`triu` parity with the CPU for square, rectangular,
  offset ±k, stacked matrices, integer and complex dtypes; `eye` still green.
- **Size**: S.

### [ ] P0-3. `isclose` / `allclose`

- **Why**: every numerical test and many user checks use them.
- **Work**: `OpIsCloseAPI` as an elementwise predicate (broadcast, tolerances,
  `equal_nan`), then `OpAllCloseAPI` as a reduction of that predicate (or a
  fused kernel); match the CPU's `atol + rtol*|b|` formula and NaN rules.
- **Accept**: parity with `DeviceCpuSerial` for default and explicit
  tolerances, `equal_nan` true/false, broadcasting, empty arrays; a test that
  `allclose(a, a)` is true and one that a one-ulp difference flips it.
- **Size**: S–M.

### [ ] P0-4. Fix the `lib.rs` "Absent, and why" section

- **Why**: it currently misleads readers about `allclose`, `take_along_axis`
  and `neg` (see [STATUS.md](STATUS.md) §9.3), and it labels common operations
  "deliberate, not pending".
- **Work**: correct the three claims; separate "impossible on an accelerator"
  (host closures) from "not yet implemented" (sort/mask/etc.); state the
  reduction dtype restriction and the layout-refusal policy.
- **Accept**: the section's claims match the code; no new feature work.
- **Size**: S.

### [ ] P0-5. `nonzero` + `mask_select` / `mask_fill`

- **Why**: boolean masking is ordinary use; `nonzero` is also the primitive
  behind `mask_select` and part of `array_index`.
- **Work**: implement `OpNonzeroAPI` (count/prefix-sum + scatter to a device
  buffer, with the same per-axis output convention as the CPU);
  `DeviceMaskIndexAPI` for select/fill; reuse the compaction primitive.
- **Accept**: parity with the CPU for 1-D/2-D/3-D masks, non-contiguous
  inputs, all-true/all-false/empty masks, and the mask-shape contract
  (broadcast rules); `mask_fill` is a no-op on an all-false mask.
- **Size**: M.

### [ ] P0-6. `array_index` / `array_index_assign`

- **Why**: integer-array indexing is a core Array-API feature; the core layer
  already resolves and uploads indices as device buffers, so the device side
  is one gather/scatter.
- **Work**: `DeviceArrayIndexAPI` (gather from the resolved aux index buffers
  in the requested iteration order) and `DeviceArrayIndexAssignAPI`
  (scatter; define duplicate-target ordering — last-write-wins serial is the
  safe default). Coordinate with `OpNonzeroAPI` for bool indexers.
- **Accept**: parity with the CPU for single/multiple array indexers, mixed
  basic+array, broadcasting of index arrays, negative indices, row-major and
  column-major default orders; duplicates in `array_index_assign` follow the
  documented policy.
- **Size**: M.

---

## P1 — general backend waves

### W2 — sorting, searching, set (P1-1)

- **Why**: the whole Tier B of [COVERAGE.md](COVERAGE.md) is missing.
- **Work**: a sort family (bitonic for power-of-two blocks or a radix sort for
  fixed-width keys), then `argsort`, `unique_*` (sort + adjacent dedup +
  inverse scatter), `searchsorted`, `isin`. `sort_custom` stays host-only.
- **Accept**: parity with the CPU for f32/f64/int dtypes, ascending/descending,
  NaN ordering (`ExtSortCmp`), stable/not-stable flag, axes and flattened
  forms, 0-length and 1-element inputs; a 1e6-element f64 sort bench recorded
  in this repository.
- **Size**: L.

### W3 — cuSOLVER linalg tier (P1-2)

- **Why**: the domain (electronic structure) cannot multiply its way out of
  `solve`/`eigh`/`svd`.
- **Work**:
  1. cuSOLVER driver wrappers (`potrf`, `getrf`, `getri`, `gesv`, `gesvd`,
     `gesdd`, `syev`, `syevd`, `sygv`, `sygvd`) mirroring
     `rstsr-blas-traits`' `LapackDriverAPI`.
  2. Generalize `rstsr-linalg-traits::blas_impl` off the compile-time
     `DeviceBLAS` alias (generic over `B: LapackDriverAPI<T>`), or add a CUDA
     implementation module with the same bodies.
  3. Wire the tensor-tier entries: `cholesky`, `solve_*`, `inv`, `det`,
     `slogdet`, `pinv`, `matrix_rank`, `matrix_power`, `qr`, `svd*`, `eig*`,
     `matrix_norm`, `vector_norm`.
- **Accept**: parity with `DeviceFaer` on small/medium matrices for each
  entry, batched (n-D) forms, row-major and column-major orders, f32/f64 and
  complex where LAPACK supports it; singular/non-PSD error paths return the
  same error class, not a panic.
- **Size**: L (two waves: solves/inverse first, then eig/svd).

### W4 — dtype breadth for GEMM and products (P1-3)

- **Why**: `Complex<f64>` matmul is the default in the target domain, and
  integer/half matmul are expected by Array-API users.
- **Work**: add `Zgemm`/`Cgemm` to `CublasScalar` (or move to `gemmEx` with a
  compute type); fix the tiled kernel's complex zero spelling; add
  `DeviceExt*API` for mixed-dtype `matmul`/`vecdot`/`tensordot`/`outer`/`cross`
  (route-B `promote_pair`); add half GEMM through `gemmEx` if measured to be
  worth it.
- **Accept**: complex `matmul`/`tensordot` parity with the CPU across
  transpose/conjugate-transpose/batch cases; integer matmul exactness;
  `ext_*` parity with `DeviceCpuSerial`; a bench cell for c64 matmul.
- **Size**: M–L.

### W5 — reduction and scan completeness (P1-4)

- **Why**: current reductions are f32/f64-only and layout-restricted; common
  code reduces integer and complex arrays, and slices/views.
- **Work**:
  1. A generic fallback kernel that accepts any layout (one thread per output,
     strided walk) so "refuse" becomes "slow but correct". Keep the fast plan
     for classified layouts.
  2. Complex `sum`/`prod`/`mean` (accumulator `T`, not `Real`) and
     complex `cumsum`/`cumprod`.
  3. `cumsum_dtype`/`cumprod_dtype`.
  4. Unraveled `argmin`/`argmax`.
  5. Integer `sum`/`prod`/`mean` at the tensor tier (either a `ReduceAcc` impl
     for ints with wrapping semantics, or a documented `*_dtype` requirement).
- **Accept**: parity for every reduction row on a layout matrix
  (contiguous/transposed/sliced/negative-stride/broadcast), integer and
  complex dtypes; the fallback is selected only when the fast plan refuses and
  is marked in the launch trace.
- **Size**: M.

### W6 — performance: TF32, cuBLASLt, benches (P1-5)

- **Why**: PR #4 shipped no throughput numbers for reductions/linalg/indexing,
  and the elementwise table predates the merge.
- **Work**:
  1. cuBLASLt path for f32 (and f16/bf16) with the `ComputeType32F` selection;
     keep typed cuBLAS as the pedantic path.
  2. Benchmarks for reductions (all rows, 1-D/2-D/3-D), scans, matmul (square,
     tall-skinny, batched, transposed), tensordot, index_select, sort.
  3. Re-run the 19-cell elementwise table at `9c7d49a` and compare with
     CuPy/torch.
  4. A `criterion` or script-level regression gate with a documented threshold
     (e.g. no cell >1.15× the recorded baseline on the same machine).
  5. **Decide `compute_type_32f`'s fate**: wire it through the cuBLASLt path
     (the `Fast*` variants are the only reason the flag exists), or delete the
     enum, env var, getter and `same_device` clause. A silent no-op is the one
     unacceptable outcome.
- **Accept**: a table in this repository with the same format as
  `2026-10-21-cuda-decisions/03-elementwise-kernel/REPORT.md`, plus one
  sentence per family on whether it is at cuBLAS/CuPy parity.
- **Size**: M.

### W7 — streaming completeness (P1-6)

| Item | Why / work | Accept | Size |
|---|---|---|---|
| Allocator framework (D7) | `CudaAlloc` trait, `RSTSR_CUDA_ALLOC_CONF`, release threshold, `rt::cuda::empty_cache`, OOM force-trim-retry, typed `MemoryError` | config parsed and applied; `empty_cache` trims; OOM path returns the typed error instead of aborting; a test with a tiny pool | M |
| Multi-GPU peer copy | `cuCtxEnablePeerAccess` per pair, `cuMemcpyPeerAsync` (or documented fallback), async on the source stream, `change_device`/`into_device`/`to_device` use it | with two GPUs visible: cross-ordinal `to_device` bitwise correct and non-blocking; without peer access: documented host-staged fallback; skip cleanly on one GPU | M |
| DLPack `kDLCUDA` / `kDLCUDAHost` | export/import for `CudaBuffer` and `PinnedBuffer`, owner/deleter rules, stream semantics | round-trip `Tensor<f64, DeviceCuda>` → Python (torch/cupy) → back, zero-copy; pinned host capsule recognized | L |
| CUDA graphs | capture a repeated op sequence, replay; expose a small `CudaGraph` wrapper | a 100-op sequence replays with lower launch overhead than eager, outputs bitwise equal; opt-in only | L |
| Stream priorities | priority constructor and a query | a high-priority copy stream preempts a low-priority compute stream in a synthetic test | S |
| Pinned allocator pool | cache `cuMemHostAlloc` blocks keyed by size; reuse after event completion | pipeline with per-chunk allocation does not grow host RSS and does not stall; bench before/after | M |
| Per-stream event pool | reuse `CUevent`s instead of create/destroy per cross-stream op | cross-stream microbench shows no event-alloc cost; no semantic change | S |

### W8 — packaging, facade, CI (P1-7)

| Item | Work | Accept | Size |
|---|---|---|---|
| `rstsr` facade | `cuda` feature, optional `rstsr-cuda` dep, `rt::cuda` namespace, re-export `DeviceCuda`/`DeviceCudaPinned` | a doc example builds with `rstsr = { features = ["cuda"] }` and no direct `rstsr-cuda` import | M |
| Crate metadata | `readme.md`, `publish = false` until release, release-plz exclusion if needed | `cargo package` is either intentionally blocked or clean | S |
| GPU CI | a self-hosted or rented GPU job: hardware suite + arch matrix + bench smoke | green on a PR with a GPU label; host-only job stays for cheap checks | M |
| Book page | "CUDA backend" chapter: device model, streaming recipe, dtype/layout limits, migration notes | page builds and is linked from the book index | M |

---

## P2 — ergonomics and hardening

- [ ] **P2-1. Core `get`/`set` element access and `Raw = Vec<T>` migration.**
  Add `TensorAny::get`/`set` that go through `DeviceStorageAPI` instead of a
  host slice; migrate `Index`/`IndexMut`/`iter`/`as_ptr`/`as_mut_ptr` and the
  shared test bodies. This unblocks idiomatic element access on CUDA and any
  future accelerator. **Size**: L, core-wide; coordinate with the core owner.
- [ ] **P2-2. `take_along_axis` / `put_along_axis` core signature.** Relax the
  `Raw = Vec<usize>` bound to `DeviceAPI<usize>` + a device trait (as
  `array_index` already does), then implement the CUDA gather/scatter.
  **Size**: M core + S device.
- [ ] **P2-3. Half arithmetic.** Native or float-promoted `+ - * /` and
  transcendentals for f16/bf16, plus half reductions through an f32
  accumulator. **Size**: M.
- [ ] **P2-4. Batched element access.** A `get_many`/`set_many` or device-side
  gather so reading 1e5 scalars does not sync 1e5 times. **Size**: M.
- [ ] **P2-5. High-level streaming helper.** A small `chunked_pipeline`
  utility that encodes the bench's laws (issue per chunk, read-event on D2H,
  keep buffers alive). Optional if the bench is the documentation. **Size**: S.
- [ ] **P2-6. Performance regression harness.** Record baselines in the
  trajectory repo and compare on demand; no CI gate until a GPU runner exists.
  **Size**: M.
- [ ] **P2-7. Error-path parity.** Ensure CUDA errors map to the same
  `RSTSRError` classes as the CPU (shape/layout/device/value), and add tests
  for the refused-layout messages. **Size**: S.
- [ ] **P2-8. `Send`/`Sync`/`TensorArc` audit.** Document the threading model;
  add a multi-threaded producer/consumer test over one ordinal. **Size**: S.

---

## Suggested PR sequence

| PR | Contents | Risk | Depends on |
|---|---|---|---|
| A | P0-1…P0-4 (neg, tril/triu, isclose/allclose, docs) | low | — |
| B | P0-5/P0-6 (nonzero, mask, array_index) | medium | A (docs corrected) |
| C | P1-1 sort/search/set | medium | B (compaction primitives reused) |
| D | P1-2 cuSOLVER (solves/inverse first) | high | W4 dtype work for complex |
| E | P1-3 complex/int GEMM + `ext_*` | medium | — |
| F | P1-5 TF32/cuBLASLt + benches | medium | E |
| G | P1-6 streaming (allocator, multi-GPU, DLPack, graphs) | high | — |
| H | P1-7 facade/packaging/CI | low | A |

Do **not** bundle D with A–C: cuSOLVER needs a trait-layer refactor, and
mixing it with small correctness fixes makes review harder than it needs to be.

---

## Definition of done

### "Common usage" gate (after W1–W5)

- [ ] A port of a representative CPU script (elementwise + reductions +
  boolean masking + `matmul` + `solve`) runs unchanged except for the device
  value and one `to_device`.
- [ ] Every row in [COVERAGE.md](COVERAGE.md) Tier A is `Y`, or `P` with a
  documented dtype/layout limit.
- [ ] Cross-device parity tests compare CUDA against `DeviceCpuSerial` for
  every implemented family with a tolerance (no `==` on floats).
- [ ] The `lib.rs` coverage table matches reality.

### "Streaming" gate (after W7)

- [ ] `bench_streaming` stays ≥2.2× the naive flow with 3 streams.
- [ ] The pipeline works with a per-chunk pinned pool and does not allocate
  per iteration.
- [ ] Pinned↔CUDA transfer is non-blocking in both directions; the only host
  waits are explicit `synchronize`/`host_view`.
- [ ] DLPack round-trips to at least one external consumer without a copy.
- [ ] Multi-GPU copy is either peer-async or a documented host-staged path.

### "Shippable" gate (after W8)

- [ ] `rstsr` feature works; crate has a README and is deliberately published
  or deliberately excluded.
- [ ] GPU CI runs the hardware suite and the arch matrix.
- [ ] The book has a CUDA page with the streaming recipe and the numeric
  contract.

# rstsr code map — commit `386948be819baa334b8da02232f3a1944e5447d5`

Produced 2026-09-08 by two exploration agents for the efficiency-improvement
planning session. Purpose: let future sessions working against this commit
orient in the rstsr workspace quickly, especially anyone hunting performance
in the CPU serial / rayon paths.

Companion files: [machine-and-tools.md](./machine-and-tools.md),
[initial-prompt.md](./initial-prompt.md). The plan itself:
[./260908-plan-cpu-serial-efficiency.md](./260908-plan-cpu-serial-efficiency.md).

## 1. Commit & workspace

- Commit `386948be` ("rstsr-core: API doc campaign (doctest-verified) with
  meshgrid/eye/to_contig behavior fixes (#99)", 2026-09-07). It is **2 commits
  ahead of tag `rstsr-v0.8.0`** — the published crates.io 0.8.0 is NOT this
  commit; pin via local path dep (sibling checkout) or git rev dep.
- 18-crate workspace. Efficiency-relevant crates:
  - `rstsr-common` — layout machinery only (no numeric kernels).
  - `rstsr-native-impl` — **the actual raw kernels** over `(&[T], &Layout<D>)`,
    `cpu_serial/` and `cpu_rayon/` (9 files each). `#![no_std]` when not testing.
  - `rstsr-core` — device structs + operator traits + user-facing tensor API;
    thin wiring that calls `rstsr-native-impl`.
  - `rstsr-dtype-traits` — `ExtNum`/`ExtReal`/`ExtFloat`, promote/cast traits.
  - `rstsr-linalg-traits` — higher linalg drivers (not in the elementwise/reduce hot path).

## 2. Where each hot op lives

### (a) Transpose / data movement
- `TensorBase::t()` / `transpose(None)` (`rstsr-core/src/tensor/manipulation/transpose.rs`,
  `t()` ~line 438) — **view only**, permutes shape/strides, no copy.
- The real copy: `change_layout_f` (`rstsr-core/src/tensor/manipulation/to_layout.rs`
  lines 8–37) → allocates new storage → `device.assign_arbitary_uninit(...)` →
  generic assign kernels `assign_arbitary_*_cpu_serial` / `_cpu_rayon`
  (`rstsr-native-impl/src/cpu_serial/assignment.rs`, `cpu_rayon/assignment.rs`).
  `#[duplicate_item]`-generated 4 variants; contiguous case is a zipped slice
  iterator, strided case is zipped layout iterators with per-element `clone()`.
- **A blocked 2-D transpose kernel exists but is NOT used by the tensor path**:
  `orderchange_out_r2c_ix2_cpu_serial` (`rstsr-native-impl/src/cpu_serial/transpose.rs`,
  BLOCK_SIZE=64, nested block loops, bounds-checked writes; rayon twin
  parallelizes over blocks with raw-pointer writes). Its only callers are
  LAPACK drivers (`rstsr-blas-traits/driver_impl/lapack/{svd/gesvd.rs,solve/potrf.rs}`).
  So `a.t().to_contig(RowMajor)` currently copies via the slow generic strided
  assign, not the blocked kernel.

### (b) vecdot / inner product
- `vecdot_naive_cpu_serial` (`rstsr-native-impl/src/cpu_serial/vecdot.rs`):
  splits summed/remaining axes via `get_axes_composition`, then 3 branches —
  contiguous-summed (good: `unrolled_binary_reduce`); **contiguous-remaining
  (poor: per-element `c.write(c.assume_init_read() + val)` MaybeUninit
  read-modify-write inside the contraction loop)**; general (worst: per-element
  index arithmetic fold).
- `vecdot_naive_cpu_rayon` (`cpu_rayon/vecdot.rs`, PARALLEL_SWITCH=512) mirrors it.
- `inner_dot_naive_cpu_rayon` (`cpu_rayon/matmul_naive.rs` ~line 57+): the
  vector·vector path of `%` under DeviceFaer — full `la.index_uncheck(&[i])`
  unravel arithmetic per element, scalar fold, no unrolling.
- Device wiring: `rstsr-core/src/device_cpu_serial/linalg/vecdot.rs`,
  `rstsr-core/src/device_faer/rayon_auto_impl/vecdot.rs`;
  tensor entry `rstsr-core/src/tensor/linalg/vecdot.rs`.

### (c) Reductions (sum/min/max/mean/var/norm, sum_axes, argmin/argmax)
- `rstsr-native-impl/src/cpu_serial/reduction.rs`:
  - `unrolled_reduce` / `unrolled_binary_reduce` (lines 44–140): 8-lane
    unrolled accumulators copied from ndarray's `numeric_util.rs`; relies on
    LLVM autovectorization.
  - `reduce_all_cpu_serial`: col-major translate + contig-run split; runs ≥
    CONTIG_SWITCH(32) use `unrolled_reduce`, else per-element fold.
  - `reduce_axes_cpu_serial` (lines 180–370): 3 branches by
    `get_axes_composition` — summed-contiguous (unrolled inner);
    **remaining-contiguous (CHUNK=48, `*acc = f(acc.clone(), x.clone())`
    per element — clone-heavy)**; plain fold. Broadcast duplication is a
    scalar clone loop; final non-K-order fixup is a full extra
    `op_muta_refb_func` copy.
  - argmin/argmax: `reduce_all_unraveled_arg_cpu_serial` — `Option<(D,T)>`
    accumulator, per-element closure calls, no contig fast path.
- Rayon twin `cpu_rayon/reduction.rs`: `into_par_iter` + fold/reduce,
  PARALLEL_CHUNK_MAX=1024, PARALLEL_SWITCH=1024.
- sum/min/max/mean/var/l2_norm are closures fed into these two generic kernels
  from `rstsr-core/src/device_cpu_serial/reduction.rs` (l2_norm folds
  `acc + (x*x.conj()).re()` then sqrt).

### (d) Elementwise binary/unary/scale
- Kernels: `rstsr-native-impl/src/cpu_serial/op_with_func.rs` —
  `op_mutc_refa_refb_func_cpu_serial` (+ numb/numa/muta variants):
  `translate_to_col_major(&[lc,la,lb], K)` + contig split at CONTIG_SWITCH=16;
  **contiguous branch is `for i in 0..size_contig` with index arithmetic +
  bounds checks per element** (no `chunks_exact` slice zips); strided branch
  zips layout iterators calling `f` per element.
- All concrete arithmetic (add/sub/mul/div/…) are macro-generated closures in
  `rstsr-core/src/device_cpu_serial/operators/op_binary_arithmetic.rs` and
  `device_faer/rayon_auto_impl/op_binary_arithmetic.rs` (`#[duplicate_item]`),
  each doing `assume_init_read`/`write`/`clone` per element through
  `&mut MaybeUninit<T>`. Unary math funcs in `op_binary_common.rs` /
  `op_unary_common.rs`.
- Tensor level: `rstsr-core/src/tensor/map_elementwise.rs`,
  `tensor/operators/*` broadcast then call `device.op_mutc_refa_refb_func`.

### (e) fill / creation
- `fill_promote_cpu_serial` (`cpu_serial/assignment.rs` lines 129–153):
  contiguous branch is `for i in 0..size_contig { c[idx+i] = fill.clone() }` —
  a clone per element; used by zeros/ones creation.

## 3. Dispatch & specialization machinery

- **Cargo features** (rstsr-core): `std`, `backtrace`, `rayon`, `faer`,
  `faer_as_default`, `row_major`, `col_major`, `aligned_alloc`,
  `dispatch_dim_layout_iter`; default = `row_major + aligned_alloc + faer +
  faer_as_default`. rstsr-common: same minus faer pair; **no default features**.
  `row_major`/`col_major` are compile-time mutually exclusive.
- `dispatch_dim_layout_iter` — runtime `match la.ndim() { 1=>Ix1 … 6=>Ix6 }`
  monomorphizing `IterLayoutColMajor` (`rstsr-common/src/layout/iterator.rs`
  lines 838–1031; parallel twin `par_iter.rs` lines 89–279). Off by default.
- **Dtype dispatch today** is pure trait-generics monomorphization. The single
  runtime dtype dispatch is `gemm_faer_ix2_dispatch`
  (`rstsr-core/src/device_faer/matmul.rs`): `TypeId`-based `same_type::<TA,$ty>()`
  macro over f32/f64/Complex<f32>/Complex<f64> — the template a future
  `dispatch_simd` scheme should follow.
- Layout dispatch: runtime CONTIG_SWITCH thresholds (16 elementwise/assign,
  32 reductions) + `get_axes_composition` / `translate_to_col_major_with_contig`
  computing contiguous run lengths at runtime.
- `aligned_alloc` — 64-byte allocation for vectors >128 elements
  (`rstsr-common/src/alloc_vec.rs`).

## 4. Layout representation

- `Layout<D> { shape: D, stride: D::Stride, offset: usize }`
  (`rstsr-common/src/layout/layoutbase.rs`); `D` is `[usize; N]` (Ix0–Ix9) or
  `Vec<usize>` (IxD) (`layout/dim.rs`).
- **No layout-category enum**; contiguity computed on demand
  (`ndim_of_f_contig`, `f_contig`, `c_prefer`, …). Strides are `isize` and
  **may be 0 (broadcast)**.

## 5. DeviceFaer parallel wiring (non-gemm)

- `DeviceFaer` (`rstsr-core/src/device_faer/device.rs`) wraps
  `base: DeviceCpuRayon` (`feature_rayon/device.rs`) owning
  `Arc<rayon::ThreadPool>`.
- Non-gemm ops: `rstsr-core/src/device_faer/rayon_auto_impl/*.rs` calling the
  matching `*_cpu_rayon` kernel with `pool.install(...)`; `get_current_pool()`
  returns None inside a rayon worker (no nested pools).
- Parallel iteration bridged by `ParIterRSTSR` (`rstsr-common/src/par_iter.rs`),
  a rayon Producer over `IterLayoutColMajor::split_at`.
- Serial-fallback thresholds: vecdot 512, reduction 1024, assignment 16384,
  transpose 16·64².
- Note: `rstsr-core/src/feature_rayon/auto_impl/` is a byte-identical copy of
  `device_faer/rayon_auto_impl/` but **not declared in `feature_rayon/mod.rs`**
  — dead code at this commit.

## 6. Existing bench/timing artifacts

- **No benches/ directories, no [[bench]], criterion 0.5 declared but unused**
  (workspace `[workspace.dependencies]` + rstsr-core dev-deps; `cpu-time` likewise).
- Only timing artifact: `rstsr-core/tests/tensor_sum.rs` — one-shot
  `std::time::Instant`, rstsr serial vs default-device(rayon) vs ndarray
  `sum_axis` on [4,512,512] f64, correctness vs ndarray asserted.
- Functional test matrix: `rstsr-core/tests/entry_row_cpu.rs` + `core_func/**`
  (ADR-0002). Fast verified test command:
  `cargo test -p rstsr-core --test entry_row_cpu --no-default-features --features "backtrace row_major"`.

## 7. Ranked optimization targets (from code shape)

1. `vecdot_naive_cpu_serial` contiguous-remaining branch
   (`cpu_serial/vecdot.rs` ~117–137): uninit RMW per element per contraction
   step → local accumulator restructure.
2. `inner_dot_naive_cpu_rayon` (`cpu_rayon/matmul_naive.rs`): per-element
   index-unravel scalar fold → contiguous unrolled path.
3. `op_mutc_refa_refb_func_cpu_serial` contiguous branch
   (`cpu_serial/op_with_func.rs` ~32–36): index loop → `chunks_exact` zips;
   highest-traffic kernel in the library (all add/mul/scale/map).
4. `reduce_axes_cpu_serial` remaining-contiguous branch
   (`cpu_serial/reduction.rs` ~279–305): clone-heavy CHUNK=48 accumulation,
   broadcast dup loop, extra order-fixup copy.
5. Generic strided assign / `change_layout_f` transpose copy: wire the unused
   64×64 blocked `orderchange_out_r2c_ix2_cpu_serial` kernel into the tensor
   path; contiguous assign via slice copy.
6. argmin/argmax (`reduce_all_unraveled_arg_cpu_serial` ~427+): no unrolling,
   no contig fast path, `Option<(D,T)>` + two closures per element.
7. vecdot general (strided) branch (`cpu_serial/vecdot.rs` ~141–162):
   per-element offset arithmetic → stride-based inner loop.
8. `fill_promote_cpu_serial` (`cpu_serial/assignment.rs` ~129–153): per-element
   clone → slice fill.

## 8. Gotchas

- `.t()` is free (view); any "transpose benchmark" must force the copy
  (`.t().to_contig(...)`).
- Broadcast strides can be 0 — kernels and benches must cover it.
- Everything runs through `MaybeUninit` + closure indirection at the
  rstsr-core layer; LLVM may or may not inline all of it (check in profile).
- Release profile is stock cargo defaults (no LTO config) — benches in the
  trajectory repo should state their profile explicitly.
- Workspace pins `channel = "nightly"` without a date (MSRV 1.82); record
  `rustc --version` with every benchmark.

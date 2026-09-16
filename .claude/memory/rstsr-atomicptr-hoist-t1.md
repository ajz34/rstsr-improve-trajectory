---
name: rstsr-atomicptr-hoist-t1
description: T1 §4.1 write-through-as_ptr UB remediated via AtomicPtr hoist — A-B shows neutral-or-faster (cdist −24%); numbers and porting state
metadata:
  type: project
---

T1 unsafe-audit §4.1 (write-through `c.as_ptr() as *mut` in rayon kernels,
Stacked-Borrows UB) was **remediated 2026-09-15** on rstsr branch
`260915-unsafe-soundness-3` (working tree, not yet committed there): 36 sites in
`rstsr-native-impl/src/cpu_rayon/*` + 4 sites beyond audit scope in
`rstsr-sci-traits/src/distance/native_impl.rs` (`cdist_rayon`/`cdist_weighted_rayon`,
same pattern, found by workspace re-grep). Pattern: `AtomicPtr::new(x.as_mut_ptr())`
before the parallel region + `load(Ordering::Relaxed)` at the old derivation point;
`Relaxed` suffices (publication via rayon spawn; pointer never reassigned).

A-B bench (`2026-09-15-atomicptr-hoist-ab/`, 6 interleaved rounds, DeviceFaer API):
geomean 0.968; per-task cases 0.97–1.03 (noise); in-place blocked-2D −5%,
naive matmul −2%, **cdist −24%** (old inner loop re-derived `dists.as_ptr()` per
element through 2 dependent loads; hoist leaves register-resident base).

Lessons for future edits to these kernels:
- The tensor-facing rayon device is **DeviceFaer** (`DeviceRayonAutoImpl` alias);
  `DeviceCpuRayon` is a base type with deliberately no `DeviceAPI<T>` — cannot
  create tensors on it. matmul-naive is reached via faer-device i64 fallback;
  cdist via faer-device `cdist((xa.view(), xb.view(), MetricEuclidean))`.
- Closure shape gotcha: inner `into_par_iter` closures must capture `&AtomicPtr`
  (Sync) and load inside; capturing the raw pointer (even `move`-copied) breaks
  Send/Sync, and `move` also demotes outer `Fn` to `FnOnce` by moving `f`.
- Shapes in `full`/`asarray` need `vec![...]` (tuples don't impl DimAPI);
  strided-add cases need b.t() shape-compatible (use transposed (n,m) source);
  `sum_axes(&[0])`, `permute_dims(&a, (2,0,1))`, `vecdot(&a, &b, (-1,-1))`.
- ±10–20% run transients on ~100 µs and fresh-2^24 cases confirmed (see [[rstsr-vecdot-t3]], [[rstsr-alloc-pagefault-t7]]); 6 interleaved rounds settle medians.
- "before" worktree kept at `/home/a/rstsr_pack/tmp/aptr-before` (aa24643) for re-runs.

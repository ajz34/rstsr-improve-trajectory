---
name: rstsr-broadcast-write-gates-t1
description: T1 §4.2 resolved via capability model — broadcast layouts are read-only by write-path enforcement, not by construction; check_strides zero-alloc rewrite
metadata:
  type: project
---

T1 audit §4.2 (stride-0 axis → aliasing `&mut`) resolved 2026-09-15/16, rstsr branch `260915-unsafe-soundness-3`, commit `6ad5e84` (user-approved design):

- **Capability model**: a `TensorMut` with a stride-0 layout is legal to hold and read (inert); UB only materializes at write sites. Every write path gates on the pre-existing `Layout::is_broadcasted()` (= `stride().contains(&0)`; `assign`/`fill` already enforced it, docs promised it).
- Gated: `add_assign` family (both tensor-ref and scalar impls, `Err(InvalidLayout)`); `mapi_f`/`mapi_fnmut_f`; matmul output driver `op_mutc_refa_refb_matmul`; `iter_mut`/`indexed_iter_mut` constructors; `axes_iter_mut`/`indexed_axes_iter_mut` guard the **iterated-axes layout only** — yielded items keep non-iterated broadcast axes as inert views, and writes through items hit the item-level gates. `index_mut`/`IndexMut` need no gate (borrow checker serializes single-element access).
- Unary owned in-place ops (`op_unary_arithmetic.rs` AND the identical pattern in `op_unary_common.rs`) FALL BACK to the fresh-output view impl instead of erroring — same policy as binary-op consume-reuse (`is_broadcasted` → don't reuse). Don't "fix" these to error.
- Same commit: `check_strides` rewritten — insertion sort into `[(usize,usize); 8]` stack buffer (bound chosen by user), heap `Vec` fallback beyond 8 non-degenerate axes, checked span arithmetic (`InvalidLayout` on overflow). Shape-1 axes are filtered before the stride loop, so `check_strides(false)` still accepts stride-0 on shape-1 axes; `is_broadcasted()` is the stricter conservative gate. `Layout::new` 3-D 33.5→8.0 ns/call, 10-D 75.9→46.4 ns.
- Test recipe for broadcast-owned tensor: `let a = arange((3.0, &device)); let (storage, _) = a.into_raw_parts(); Tensor::new(storage, Layout::new([2, 3], [0, 1], 0))` — rows broadcast (`[0,1,2,0,1,2]`); strides `[1,0]` broadcasts columns instead.
- Gotchas hit while testing: `to_vec()` is 1-D only; `TensorMut` has no `.iter()` (use `.view().iter()`); non-Send `FnMut` methods need explicit `DeviceCpuSerial`, not the `DeviceCpu` alias (rayon feature swaps the default device).

Related: [[rstsr-soundness-t1-unsafe-audit]], [[rstsr-atomicptr-hoist-t1]] (§4.1), [[rstsr-tensor-extraction-quirks]].

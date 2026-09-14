---
name: rstsr-soundness-t1-unsafe-audit
description: T1 unsafe-soundness audit at acfa93e - 4 real bugs found (full() OOB, iterator lifetime UAF, alloc overflow, DataRef Send bounds); as_ptr-write rayon pattern flagged
metadata:
  type: project
---

# T1 unsafe-soundness audit (2026-09-14, rstsr acfa93e)

Campaign dir: `2026-09-14-soundness-check/T1-unsafe-audit/` (README + fixes.patch + safety-comments.patch; worktree tmp/snd-unsafe, patches reverse-apply verified).

## Real bugs (all fixed + tested)
1. `full((Layout, fill))` allocated `layout.size()` not `bounds_index().1` → heap OOB for offset/negative-stride layouts (rstsr-core tensor/creation.rs). siblings empty/zeros/ones were correct.
2. `iter()/indexed_iter()/axes_iter()` on OWNED tensors: constructors transmuted a `&self` borrow to a free impl lifetime `'a` → iterator outlives tensor = UAF (verified garbage output). Fix = return `<'_>`. Mut variants were safe (`&'a mut self`). Cost: `a.t().iter()` on a temporary no longer compiles (bind the view first) — ndarray-style.
3. `aligned_uninitialized_vec`: `size * sizeof` wrapped (release) → tiny alloc + huge Vec. Fixed with checked_mul; unaligned path was already safe (try_reserve_exact).
4. `DataRef/DataCow/DataArc/DataReference` unsafe Send impls with only `C: Send` — a `&C` is Send only if `C: Sync`; Arc needs Send+Sync. Tightened; par_iter.rs needed `B::Raw: Sync` added.

## Design-level flags (not fixed, in README §4)
- rayon kernels write through `as_ptr() as *mut` (~30 sites) — disjoint but Stacked-Borrows-UB; sound fix = `AtomicPtr::new(x.as_mut_ptr())` hoist (pattern already used in cpu_rayon/op_tri.rs + reduction.rs arg paths). Raw ptrs are !Sync which is why the pattern exists.
- `axes_iter([])` underflow-panics (`axes_check.len() - 1`).
- stride-0 axes pass `check_strides(true)` → stride-0 MUTABLE tensor reachable via asarray(&mut, custom layout) → axes_iter_mut could yield aliasing &mut.
- faer Mat→Tensor takes ownership of faer alloc as a Vec (dealloc-layout mismatch if faer over-aligns).
- `Layout::size()` doc claims cached but recomputes (and can silently wrap).

## Lesson
`unsafe impl Send for Wrapper<C> where C: Send` over an enum that can hold `&C` is the classic bound bug; and any `transmute` extending a borrow past an impl-generic lifetime is unsound whenever R (data repr) has no lifetime parameter.

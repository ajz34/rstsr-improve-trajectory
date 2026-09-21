---
name: rstsr-soundness-t1-unsafe-audit
description: T1 unsafe-soundness audit at acfa93e + 2026-09-17 full-workspace recheck - 5 bugs then R1/R2 residue fixed (fb45e78); R3 BLAS3/getrf/getri/gesvd fixed via PR #106 (2026-09-21); still open tblis shared-ptr output + vendor thread state
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

## Full-workspace recheck 2026-09-17 (branch 260915-unsafe-soundness-3, base 31ccac4)

Extended to the never-audited crates (blas-traits, tblis, 5x crates-device,
sci-traits follow-up) via 4 parallel audit agents, findings hand-verified.
README §8 is the full record.

- **Fixed, rstsr commit fb45e78** (14 files, +158/-44): R1 = §4.1-class
  residue — parallel-outer batched gemm fabricated per-task full-length `&mut`
  from `c.as_ptr()` in device_faer + all 5 device crates, syrk write-back
  wrote through `as_ptr().add() as *mut` x5; AtomicPtr(as_mut_ptr) hoist
  applied (device_faer's T1 SAFETY comment there was WRONG — argued
  disjointness only). R2 = §4.2 gate gaps — `*_with_output` family
  (op_mutc_refa_refb), op_with_func drivers x3, vecdot_from_f lacked the
  `is_broadcasted()` gate; all gated. Core lib 129 green both feature sets;
  device-crate tests compile-checked only (no cblas link on this machine).
- **R3 MOSTLY FIXED (PR RESTGroup/rstsr#106, branch 260918/lapack-fix,
  2026-09-21, awaiting review)**: BLAS3 offset/ld (acfe875), getrf ipiv
  min(m,n) + getri len check, gesvd order inversion, empty-matrix
  superb underflow (wrapper + driver), gesvd driver RowMajor path
  (404ee21) — see 2026-09-18-lapack-view-fixes/README.md. STILL OPEN:
  tblis output through shared-derived ptr; blas_int truncation;
  blis/aocl/kml unguarded vendor-global thread state; syhemm dead-API
  slot/n convention bugs (no driver, untestable).
- **R4 notes**: aligned_uninitialized_vec dealloc-layout mismatch under
  aligned_alloc feature (faer-§4.4 class); Raw<T>→Raw<MaybeUninit<T>>
  transmutes assume layout identity; stale REVIEWME in op_binary_arithmetic.
- **Trajectory-independence**: rstsr/book/agents working trees CLEAN (all
  file types); git history carries refs (b700fc8 cites the hoist A-B bench;
  4 commits say "T1 unsafe-audit") — owner informed, no rewrite requested;
  fb45e78 written without trajectory references.
- **Test-writing gotcha hit**: rstsr-core `#[cfg(test)]` mods using
  `use rstsr::prelude::*` (facade dev-dep) get "multiple versions of crate
  rstsr_core in the dependency graph" when calling INTERNAL fns with
  facade-typed views — use `crate::prelude_dev::*` + a separate test mod
  instead. Also: `into_shape` yields IxD, `Layout::new([a,b],..)` yields Ix2
  — align dims explicitly or DimMaxAPI inference fails.

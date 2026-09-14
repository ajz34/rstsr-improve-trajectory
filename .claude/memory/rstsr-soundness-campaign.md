---
name: rstsr-soundness-campaign
description: 2026-09-14 soundness check outcome at rstsr acfa93e — 13 bugs fixed (iterator lifetime UAF, beta=0 uninit reads, Send bounds, reshape/axes panics), col_major feature wired (entry_col_cpu + col_func track), 460 unsafe sites SAFETY-commented; patches in 2026-09-14-soundness-check/
metadata:
  type: project
---

# rstsr soundness campaign 2026-09-14 (base acfa93e)

Deliverables in `2026-09-14-soundness-check/` (this repo): `combined-all.patch`
(90 files, +3364/−154, green on BOTH row/col feature contracts) + per-theme
patches in T1–T4 subdirs. Read `README.md` there first.

Key facts to remember:

- **Col-major builds**: `cargo test -p rstsr-core --no-default-features
  --features "col_major,aligned_alloc,faer,faer_as_default,std"` (std required;
  no rstsr/col_major dev-dep feature needed). Alternating row/col recompiles the
  dep subtree (~50 s warm). Shared parity body pins RowMajor per test, so col
  coverage = new `tests/col_func/` track (66 tests).
- **Worst bugs found** (don't re-introduce): owned-tensor `iter()` used a free
  impl lifetime via transmute (UAF); naive matmul kernels read uninit output at
  beta=0; `DataRef/Arc Send where C: Send` too loose (needs +Sync); `rt::full`
  with user layout allocated size() not bounds.
- **Unsafe comment convention adopted**: `// SAFETY: <concrete invariant>` 1–3
  lines above the block; ~460 sites covered, 303/513 unsafe lines have one
  within 6 lines. T1 README §4 lists 7 design-level items left for owner
  (rayon as_ptr write-through pattern = Stacked Borrows issue, suggest
  AtomicPtr hoist).
- **DeviceFaer surface**: matmul f32/f64/c32/c64→faer gemm/syrk; 10 linalg
  drivers in `rstsr-linalg-traits/src/faer_impl/`; slogdet/solve_symmetric
  BLAS-only (missing); inv = svd().inverse() (LU would do); singular solve
  silently ±inf (faer SVD contract).
- **Worktree protocol that worked**: detached worktrees per agent
  (`git worktree add --detach`), patches via `git apply -3`, thematic order
  T3fix→T1fix→T4fix→T2fix→T2wire→T1comments→T4comments; 3 documented overlap
  resolutions (iterator_elem next_back combine, faer comments keep T4,
  axes_iter test order-independent expectation).
- Layout facts: `Layout::size()` is NOT cached (doc lies); check_strides
  heap-allocs per Layout::new; attempt_nocopy_reshape verified vs numpy shape.c.

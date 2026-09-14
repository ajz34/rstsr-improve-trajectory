---
name: rstsr-t3-layout-theory
description: T3 layout/index-math theory audit at acfa93e COMPLETE — 5 bugs fixed in fixes.patch (reshape -1 div0, fn_b 0-dim panic, Order::A prefer copy-paste, 5x indexed next_back wrong index, axes_iter neg-axis check), col-major broadcast = Julia-style intentional, overflow margins unreachable
metadata:
  type: project
---

# T3 layout-theory audit (2026-09-14 soundness campaign, base acfa93e)

Report + `fixes.patch` in `2026-09-14-soundness-check/T3-layout-theory/`; work done in
worktree `tmp/snd-theory` (never commit there). 10 in-crate regression tests added;
`cargo test -p rstsr-common` (39+4 doc) and `-p rstsr-core --lib` (116) green, default
features.

Bugs fixed (all fail-before/pass-after):
1. `reshape_substitute_negatives` (common/layout/reshape.rs): 0 among known dims +
   `-1` ⇒ `size % 0` panic; numpy raises ValueError. Fixed with checked_mul fold +
   `size_neg > 0` guard.
2. `translate_to_col_major_unary` fn_b (rearrangement.rs): 0-dim layout with
   Order::A/B panicked `shape[0]` on empty array (c_contig() is true for 0-dim).
3. `translate_to_col_major` Order::A multi-layout checked `c_contig/f_contig` in
   slots named c_prefer/f_prefer (copy-paste; falls to cargo-feature default for
   prefer-not-contig arrays). Efficiency/intent only.
4. Indexed iterator `next_back` returned `index_start` instead of post-back
   `index_end` in 5 impls (common IndexedIterLayout; core IndexedIterVecView/Mut,
   IndexedIterAxesView/Mut) — every backward (index, elem) pair had the wrong index.
   Public API, no in-repo caller exercised rev() — hence survived.
5. `axes_iter*` (core iterator_axes.rs, 4 sites): only FIRST axis checked for
   over-negative ⇒ `shape_full[(-3) as usize]` panic; empty axes ⇒ `len()-1`
   underflow panic. Fixed with `.any()` + `windows(2)`.

Key invariant facts worth remembering:
- Soundness gate = `TensorBase::new_f` (tensorbase.rs): check_strides(skip_zero=true)
  + bounds_index max < storage len. bounds_index early-returns (s,s) on any 0 dim, so
  stride-0 broadcast dims never inflate bounds.
- `attempt_nocopy_reshape` is numpy-faithful (diffed against
  ~/Git-Others/numpy/.../shape.c) and SAFER than numpy: numpy reads newdims[-1] when
  ni==0 (C UB), rstsr guards it.
- Col-major broadcasting is intentionally Julia-style (fastest-axis-left-aligned,
  error on numpy-valid combos) — enforced by broadcast_shape erroring; do not
  "fix" it to numpy semantics (tracked in tests/tracking/numpy_differences.md).
- Overflow in bounds/stride math (isize products) wraps in release at ≥2^63
  elements — unreachable; documented, debug_asserts left as owner-judgment item.
- `to_vec()` is 1-D only (runtime ndim check) — don't call on N-D views in tests.

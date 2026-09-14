# T3 — Layout & index-math theoretical correctness audit (S1)

- **Date**: 2026-09-14
- **rstsr base**: `acfa93e` (master), audited in detached worktree `/home/a/rstsr_pack/tmp/snd-theory`
- **Deliverable**: `fixes.patch` (5 behavioral fixes + 10 regression test functions, all
  in-crate `#[cfg(test)]` mods), this README.
- **Verification**: `cargo test -p rstsr-common` → 39 lib + 4 doc tests green;
  `cargo test -p rstsr-core --lib` → 116 green / 3 ignored (pre-existing), default features.

## 1. Invariant model reconstructed

Everything below was verified against the code, and — where indicated — against numpy's
`numpy/_core/src/multiarray/shape.c` (local checkout at `~/Git-Others/numpy`).

**Layout** `L(d, t, s)`: shape `d` (usize, `d_k ≥ 0`), stride `t` (isize, element units,
`≠ 0` only enforced for `d_k > 1`), offset `s ≥ 0`. Element address is
`z(i) = s + Σ i_k t_k`. Negative user indices `ĩ` normalize as `i = ĩ + d_k`.

**Soundness gate (the load-bearing invariant).** A container is safe iff

1. `bounds_index()` min ≥ 0 (no element before storage start),
2. `bounds_index()` max < storage length,
3. no two indices map to the same element (`check_strides`, sorted by |stride|,
   `elem_cum < |stride_next|`, size-1/size-0/stride-0 axes exempt).

`TensorBase::new_f` (rstsr-core/src/tensorbase.rs:195) enforces 1–3; every `new_unchecked`
call site must be justified by construction from an already-valid layout (audited: all
such sites either split/subset a valid layout, or get validated downstream by `new_f`).

**Bounds math.** `bounds_index` (layoutbase.rs:237) folds `min/max += t_k (d_k − 1)` per
axis, early-returns `(s, s)` for any `d_k = 0` (empty ⇒ zero-width bound), so broadcast
stride-0 dims and size-1 dims cannot inflate bounds. Verified: bounds are exact for
negative steps, permutations, diagonals, and sliced layouts (test suite values
cross-checked against numpy `nditer` output in existing tests).

**Reshape.** `quick_check` computes the output size product with `checked_mul` (gh-7455
already fixed upstream); size-0/1 shapes short-circuit to stride-1 layouts;
`attempt_nocopy_reshape` is a faithful port of numpy's `_attempt_nocopy_reshape`
(C-order contiguity check `t[ok] == d[ok+1]·t[ok+1]`, F-order
`t[ok+1] == d[ok]·t[ok]`, stride-composition loops, trailing-1 handling). It is
**stricter than numpy** in one spot: numpy multiplies `last_stride *= newdims[ni-1]`
outside the `ni ≥ 1` guard (reads `newdims[-1]`, C UB) — rstsr guards it correctly.

**Broadcasting.** `broadcast_shape` = numpy semantics under `RowMajor`; under `ColMajor`
the shapes are reversed first ⇒ broadcasting aligns **fastest axes first and appends
new dims at the slow end** (Julia semantics). `update_layout_by_shape` right-aligns old
strides in row space; the col-major path reverses layout+shape+types, reuses the row
path, and reverses back — verified correct for every broadcast that passes
`broadcast_shape` (old dims are exactly the suffix in reversed space, so the
`stride[n−n_old..n]` shift is aligned with the reversed `BroadcastType` vector).

**Iteration.** `IterLayout{ColMajor,RowMajor}` are odometer iterators over an
exclusive-end index (`index_end = unravel(size)`, intentionally out-of-range;
`index_uncheck` returns `isize` to tolerate this). `next_iter_index`/`back_iter_index`
carry the offset incrementally and compensate wraps with `− d_k·t_k` — exact for
negative and zero strides. `dim_dispatch_{1,2,3,2diff}` zip iterators, so unequal-size
`2diff` pairs stop at the shorter one (contract).

**Degenerate shapes.** 0-dim ⇒ size 1; any `d_k = 0` ⇒ size 0 and every transform that
matters short-circuits (`check_strides`, `bounds_index`, `greedy_layout`, iterators,
`dim_narrow`). Stride-0 dims may only be introduced by `update_layout_by_shape`
(Expand/Upcast) or explicit user layouts (accepted by `check_strides(skip_zero=true)`);
consumers detect them via `is_broadcasted()` / `get_axes_composition`.

## 2. Findings

### [BUG-fixed] 1. `reshape_substitute_negatives` divides by zero when a known dim is 0

- **Where**: `rstsr-common/src/layout/reshape.rs:141` (pre-fix line numbers)
- `acc * v` over all non-`-1` dims; if any known dim is `0`, `size_neg == 0` and
  `size_in % size_neg` **panics** ("attempt to calculate the remainder with a divisor of
  zero") instead of returning `Err`. numpy raises `ValueError: cannot reshape array of
  size 5 into shape (0,newaxis)` — i.e. a clean error is the expected behavior.
  Reachable from the public API: `arange(5).reshape([0, -1])`.
- **Fix**: `try_fold` with `checked_mul` (also turns isize product overflow into
  `InvalidValue` instead of a debug panic/release wrap) + `size_neg > 0` assertion with
  the same `-1 could not be determined` message.
- **Test**: `layout::reshape::test::test_reshape_substitute_negatives` — 11 cases
  (general, size-0, non-divisible, two `-1`, non-`-1` negative, `0`-dim with `-1`
  ×2, isize-overflow).

### [BUG-fixed] 2. Buffer-order translation (`Order::A`/`Order::B`) panics on 0-dim layouts

- **Where**: `rstsr-common/src/layout/rearrangement.rs:172` (`fn_b` in
  `translate_to_col_major_unary`)
- For a 0-dim layout, `ndim_of_c_contig() == 0 == ndim` ⇒ `c_contig()` is true, so
  `Order::A` routes to `fn_b`, which executes `shape[0] = l.size()` on an **empty**
  shape array → index-out-of-bounds panic. `Order::B` hits the same line directly.
  Reachable from the public API: any op on a scalar tensor that uses buffer-order
  translation (e.g. `Tensor::iter_with_order(TensorIterOrder::A)` on a 0-dim tensor).
- **Fix**: early-return the layout unchanged when `ndim() == 0`.
- **Test**: `layout::rearrangement::test::test_translate_to_col_major_zero_dim` — all
  six orders, unary and multi-layout entry points, offset preserved.

### [BUG-fixed] 3. `translate_to_col_major` (`Order::A`, multi-layout) checks contiguity where prefer is documented

- **Where**: `rstsr-common/src/layout/rearrangement.rs:257` (pre-fix)
- The doc comment says "A: B if contiguous, C if c-prefer, F if f-prefer; otherwise
  default", and the unary twin (line 199) implements exactly that. The multi-layout
  version was copy-pasted with `c_contig()`/`f_contig()` in the slots named
  `c_prefer`/`f_prefer`. Consequence: for arrays that are prefer-but-not-contiguous
  (e.g. any sliced array), both flags come out `false` and the choice silently falls to
  the cargo-feature default instead of the prefer check. Pure iteration-order/efficiency
  intent bug (both branches are valid layouts; no memory-safety impact), and under
  `RowMajor` default it made f-prefer strided arrays iterate C-reversed.
- **Fix**: use `c_prefer()`/`f_prefer()`.
- **Test**: `layout::rearrangement::test::test_translate_to_col_major_order_a_prefer` —
  a c-prefer and an f-prefer non-contiguous layout must translate to their prefer
  orders (this test fails pre-fix under default `RowMajor` features).

### [BUG-fixed] 4. Indexed iterators return the *front* index from `next_back`

- **Where** (5 impls, all the same copy-pasted pattern):
  - `rstsr-common/src/layout/iterator.rs` — `IndexedIterLayout::next_back`
  - `rstsr-core/src/tensor/iterator_elem.rs` — `IndexedIterVecView::next_back`,
    `IndexedIterVecMut::next_back`
  - `rstsr-core/src/tensor/iterator_axes.rs` — `IndexedIterAxesView::next_back`,
    `IndexedIterAxesMut::next_back`
- `next_back()` moves the internal cursor **before** yielding, but all five impls
  cloned `index_start` (the front cursor) instead of reading `index_end` **after**
  `next_back()` (which by then holds the index of the element actually yielded).
  Concretely, `a.indexed_iter().rev()` on `[3, 2]` yields `(index [0,0], element 5)`
  — index and value refer to different elements. Every yielded pair of a backward
  indexed iteration was wrong (element correct, index always the front's). This is
  public user-facing API (`TensorAny::indexed_iter().rev()` etc.); no in-repo caller
  exercised the backward path, which is why it survived.
- **Fix**: call `next_back()` first, then clone `index_end()`.
- **Tests** (all fail pre-fix): `layout::iterator::test_col_major::test_indexed_iter_next_back`
  (3 layouts × 2 flag orders), `tensor::iterator_elem::tests_serial::test_indexed_iter_rev`,
  `...::test_indexed_iter_mut_rev`, `tensor::iterator_axes::tests_serial::test_indexed_axes_iter_rev`
  (orders C and F), `...::test_indexed_axes_iter_mut_rev` (also verifies write-through
  lands on the indexed views).

### [BUG-fixed] 5. `axes_iter` axis normalization only checks the first axis for over-negative values, and underflows on empty axes

- **Where**: `rstsr-core/src/tensor/iterator_axes.rs` — 4 sites
  (`axes_iter_with_order_f`, `axes_iter_mut_with_order_f`,
  `indexed_axes_iter_with_order_f`, `indexed_axes_iter_mut_with_order_f`)
- Two defects in the same block:
  1. `if axes.first().is_some_and(|&v| v < 0)` — only the **first** element is checked.
     `a.axes_iter([0, -5])` on a 2-D tensor normalizes to `[0, -3]`, passes the check,
     and then indexes `shape_full[(-3) as usize]` → index-out-of-bounds panic with a
     meaningless offset instead of the intended `InvalidValue` error.
  2. `for i in 0..axes_check.len() - 1` — subtract-with-overflow panic (debug) /
     huge-range index panic (release) for an **empty** axes list, e.g.
     `a.axes_iter(())`, which is a meaningful request (iterate zero axes ⇒ yield the
     whole tensor once, the empty product).
- **Fix**: `axes.iter().any(|&v| v < 0)`; duplicate check via
  `axes_check.windows(2).any(...)` (no arithmetic on length).
- **Tests**: `test_axes_iter_invalid_axes` (too-negative at second position → clean
  `Err` on all four constructors; valid negative axis still `Ok`; duplicates `Err`),
  `test_axes_iter_empty_axes` (zero axes yields exactly one full view).

### [HAZARD-documented] 6. isize/usize overflow margins in bounds/stride math (theoretical only)

All of the following wrap silently in release and panic in debug. None is reachable
without an allocation of ≥ 2^63 elements (≈ 8 EiB for f64), i.e. not practically
triggerable; the danger would be a wrapped value *laundering* an invalid layout through
the `new_f` gate, which requires contrived exact-collision values:

- `bounds_index` (layoutbase.rs:255-257): `stride[i] * (shape[i] as isize − 1)` can wrap
  for `new_unchecked` layouts with astronomic shape/stride. The wrapper consumer
  `into_owned`/`into_shared` (ownership_conversion.rs:255,302) compares the bound to the
  storage length and falls back to the safe gather path on any mismatch, so a wrapped
  bound degrades to a copy, not to UB.
- `check_strides` (layoutbase.rs:317): `(shape−1) * |stride|` usize product can wrap;
  wrap-down could in principle let an overlapping layout pass. Requires
  Σ-products ≥ 2^64.
- `index_f` / `index_uncheck` (layoutbase.rs:215-219, 538-541): `strd * idx` isize
  accumulation — same class.
- `stride_f_contig`/`stride_c_contig` (shape.rs:77-94): `d as isize` and stride products
  overflow for shape products > 2^63.
- `unravel_index_{f,c}` (shape.rs:104-178) divide by `dim`, so a call with `index ≥
  size` (which includes any index into a size-0 tensor) is a divide-by-zero panic —
  documented unsafe contract ("does not check whether index is out of bounds");
  all in-crate callers (iterator construction, `split_at`) respect `index ≤ size`.

**Recommendation (not applied)**: adding `debug_assert!`s on the `checked_*` variants in
`bounds_index`/`check_strides` would convert the release-wrap case into a debug-time
signal at zero hot-path cost. Left out of the patch because the margins are
unreachable and the maintainers may prefer to keep these functions branch-free.

### [HAZARD-documented] 7. Fixed-dim vs dynamic contiguous strides disagree on zero dims

- `stride_f_contig`/`stride_c_contig` for `Ix<N>` use `.max(1)` (shape.rs:80, 91), so a
  zero dim contributes stride factor 1; the `IxD` versions (shape.rs:187-204) omit
  `.max(1)`, so `[2, 0, 4]` yields `[1, 2, 0]` (fixed) vs `[1, 0, 0]` (dynamic).
  Benign today: zero-size layouts never dereference, `check_strides` short-circuits on
  `size == 0`, `Layout::eq` ignores strides of size-0/1 dims, and every consumer that
  normalizes strides goes through order-independent paths. Worth unifying if any future
  code starts comparing fixed/dyn contiguous strides directly.

### [INTENTIONAL-divergence-confirmed] 8. Col-major broadcasting is Julia-style ("limited"), and the contract is enforced by erroring

- `rstsr-core/Cargo.toml:45-46` documents "Col-major convention: similar to Julia
  (limited broadcasting)"; `tests/tracking/numpy_differences.md` records it as the
  intentional rstsr extension "ColMajor broadcast applies from the left".
- Verified end-to-end: `broadcast_shape` reverses shapes, applies numpy's
  right-alignment in reversed space (= left/fastest-alignment in stored space), and
  **errors** (`InvalidLayout, "Broadcasting failed."`) on numpy-valid but Julia-invalid
  combos (e.g. `(8,…,1)` × `(7,1,3,5)` in col mode fails at dim0 8 vs 7 — correct per
  the contract). `update_layout_by_shape`'s col path was verified stride-exact for
  every broadcast that passes the shape check (old dims are the reversed-space suffix,
  so the right-shift and reversed `BroadcastType` stay aligned; traced the
  `[7,1,3,5].f()` × `[8,1,6,3,1]` case by hand — the shape check rejects it before the
  layout path, and the sibling case `col (2,3) × col (2,)` produces exactly the
  Julia-correct strides).
- Zero stride is introduced only for `Expand`/`Upcast` dims; `Preserve` keeps old
  strides, so pre-broadcast broadcast views compose correctly.

### [INTENTIONAL-divergence-confirmed] 9. Slice clamping matches numpy everywhere probed

`dim_narrow` (indexer.rs:120-199) verified case-by-case against numpy slice semantics:
negative start/stop folding (`max(0)` / `max(-1)` clamps), step>0 `start>stop ⇒ empty`,
`start` clamped to `len` (empty), step<0 `start` clamped to `len−1`, `stop=−1` sentinel
for "before element 0", ceil-division shapes `(stop−start+step±1)/step` exact for both
step signs (checked against `slice!(5,15,2)`, `(5,16,2)`, `(15,5,-2)`, `(15,4,-2)`,
`[::-1]`, `[:-2:-1]`, `[1:-2:-1]`, length-1 axes, and the
`fix_too_strided_stride_check` regression already in the suite). Offsets stay
non-negative by construction in both branches, so `Layout::new`'s bound check never
rejects a valid numpy slice. The documented divergences in
`tests/tracking/numpy_differences.md` (default order, no A/K orders for reshape/ravel,
unified error taxonomy, `...` expansion, etc.) were read first and are unaffected.

### [INTENTIONAL-divergence-confirmed] 10. `attempt_nocopy_reshape` port is numpy-faithful (and safer than numpy in one corner)

Diffed line-by-line against numpy `_attempt_nocopy_reshape`
(`~/Git-Others/numpy/numpy/_core/src/multiarray/shape.c`): 1-removal, product-matching
loop, both contiguity checks, stride composition, trailing-1 strides — identical
semantics. rstsr additionally bounds-checks the `nj`/`oj` cursors (returns `None` →
copy path) where numpy relies on the outer loop, and guards the `ni == 0` `last_stride`
case that numpy reads out-of-bounds. `layout_reshapeable` never reaches
`attempt_nocopy_reshape` unless `quick_check` has already proven
`size_in == size_out` (checked product), so the usize products in the matching loop
cannot overflow through public APIs. Negative-stride and stride-0 (broadcast view)
inputs fall out correctly (contiguity check fails ⇒ copy).

### Verified-clean areas (no findings)

- **matmul.rs**: all 7 rules, fixed-dim and dyn, both orders (col via
  reverse-axes-and-swap delegation); the shipped col-major tests mirror the row-major
  table exactly; `layout_matmul_dyn_row_major_with_lc` broadcasts *to* the caller's
  `lc` and splits real layouts, so no fabricated strides.
- **iterators, dispatch conditions**: dispatch macros preserve layout validity via
  `to_dim` (static/dynamic match enforced); 0-dim dispatch calls `f(offset)` once
  (size-1 correct); empty tensors iterate zero times; `2diff` zips to the shorter.
- **format_layout.rs / `Debug for Layout`**: pure formatting over `c_contig`/`f_contig`
  predicates, which short-circuit on size 0 — no crash surface for 0-size or huge
  strides (the integer values are only `{:?}`-printed).
- **`get_layout_for_binary_op` / `get_axes_composition`**: output strides are
  re-derived (broadcast axes of either operand get real strides in the output; `a0_o`
  is the intersection, so an axis broadcast in only one operand is re-strided); shape-1
  axes take neighbor strides preserving prefer-order; offset 0 is correct because this
  describes a fresh output.
- **`TensorBase::new_f`** gate + `Index`/`IndexMut` (`index()` bounds-checks each
  component and final sign; `Vec` indexing re-checks against storage).
- **`asarray` with explicit layout** goes through `Tensor::new_f` ⇒ bounds + stride
  validation (the doc example's `Layout::new([3,2],[2,7],5)` sub-view case is gated).
- **`dim_split_at`** allows `axis == ndim` (documented `[-n, n]` range); the trailing
  0-dim "rest" layout is consumed only by reduction code that treats it as the
  zero-axis remainder — `dim_split_axes` (the checked variant) is what reductions use.
- **`tensordot_to_einsum`**: per-side `normalize_axes_index` bounds both axes lists;
  the removed dead asserts (per the in-file comment) were indeed unreachable; label
  exhaustion (>52 dims) is a clean `Err`.
- **`normalize_axes_index`**: duplicate detection correct with and without `sort`.

## 3. Efficiency notes (stumbled upon, not chased)

1. `Layout::size()` is a fresh `iter().product()` on every call despite the doc note
   "uses cached size" (layoutbase.rs:67) — there is no cache; hot paths
   (`f_prefer`/`c_prefer`/iterators) recompute it. A small struct cache or a
   `size_non_zero` fast path could help iterator-heavy code.
2. `check_strides` allocates two `Vec`s and sorts on every `Layout::new` — fine for
   construction, but `Tensor::new_f` runs it on every container build; a stack
   small-vec for `ndim ≤ 8` would remove the heap churn.
3. `f_prefer`/`c_prefer` loop twice over shape/stride where a single reversed pass with
   an early exit would do (micro; the compiler likely already handles it).
4. `translate_to_col_major` `Order::K` computes `size_non_broadcast()` for every layout
   twice (`max`/`min` iterators) — trivial.
5. `greedy_layout` with `keep_shape=false` returns shape-1/stride-0 axes squeezed to
   `(d=1, t=0)` via `new_unchecked`; since `d` was already 1 or `t` already 0, the
   write-back is a no-op for `d` — only kept for symmetry; harmless.

## 4. Tests added (10 new `#[test]` functions)

| Crate | Module | Test |
|---|---|---|
| rstsr-common | `layout::reshape::test` | `test_reshape_substitute_negatives` |
| rstsr-common | `layout::rearrangement::test` | `test_translate_to_col_major_zero_dim` |
| rstsr-common | `layout::rearrangement::test` | `test_translate_to_col_major_order_a_prefer` |
| rstsr-common | `layout::iterator::test_col_major` | `test_indexed_iter_next_back` |
| rstsr-core | `tensor::iterator_elem::tests_serial` | `test_indexed_iter_rev` |
| rstsr-core | `tensor::iterator_elem::tests_serial` | `test_indexed_iter_mut_rev` |
| rstsr-core | `tensor::iterator_axes::tests_serial` | `test_axes_iter_invalid_axes` |
| rstsr-core | `tensor::iterator_axes::tests_serial` | `test_axes_iter_empty_axes` |
| rstsr-core | `tensor::iterator_axes::tests_serial` | `test_indexed_axes_iter_rev` |
| rstsr-core | `tensor::iterator_axes::tests_serial` | `test_indexed_axes_iter_mut_rev` |

## 5. Open items for owner judgment

1. **Overflow `debug_assert!`s** (finding 6): cheap hardening, but touches hot
   predicate functions — owner's call whether to add `checked_mul`-based debug asserts
   in `bounds_index`/`check_strides`/`index_uncheck`.
2. **`axes_iter(())` semantics**: the fix makes zero-axes iteration yield one full
   view (the empty-product convention). If the project prefers an explicit error for
   empty axes, the `windows(2)` change should be replaced by an
   `InvalidValue`-raising check instead.
3. **Fixed-vs-dyn `stride_contig` zero-dim mismatch** (finding 7): unifying is a
   one-line change (add `.max(1)` to the `IxD` impls) but alters dyn layouts of
   zero-size tensors that any test might compare verbatim; left as documented.
4. Only the default (`row_major`) feature configuration was tested here; T2 covers the
   col-major build. The fixes in this patch are order-symmetric by construction
   (indexed `next_back` uses whichever cursor the variant maintains; `fn_b`/prefer fix
   are order-independent), so no col-specific behavior change is expected.

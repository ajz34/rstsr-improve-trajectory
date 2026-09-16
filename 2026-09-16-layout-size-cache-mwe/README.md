# MWE — does caching `Layout::size()` pay? (T1 unsafe-audit §4.5)

- **Date**: 2026-09-16
- **rstsr inspected**: commit `aa24643` (branch `260915-unsafe-soundness-3`);
  `Layout<D>` (`rstsr-common/src/layout/layoutbase.rs`) is semantically
  identical to master `acfa93e` for this question: `{ shape: D, stride:
  D::Stride, offset: usize }`, `size() = shape.as_ref().iter().product()`,
  recomputed per call; doc falsely says "uses cached size".
- **Question**: audit §4.5 flags that `size()` recomputes the shape product per
  call (unchecked, doc/impl mismatch). Caching would add a derived `size`
  field to `Layout` — is the efficiency win worth the redundant field?
- **Answer**: **No.** Recompute costs 0.36–0.68 ns per call (heap-`Vec` shape,
  ndim 2–8) and ≈0 ns for inline arrays; every live `size()` call site in
  rstsr is once-per-op, so the total win is ≪0.1% of any real op. The cache
  costs +8 B per stored layout, +1.3–2.7% on clone-heavy paths, and makes
  construction *slower* (one product either way — caching moves it, then adds
  a field). Only a hypothetical per-element `size()` caller would profit
  (−33% in the extreme bench), and no such caller exists; if one appears, the
  local fix is hoisting `let size = l.size()` out of the loop, no API change.

## Environment

- AMD Ryzen 9 9950X3D ×16, Linux; rustc 1.97.1 stable (rustup default);
  criterion 0.5.1, `--release` bench profile. Single-threaded benches.
- MWE is fully synthetic (`LayoutVec*` mirrors `IxD`, `LayoutArr*` mirrors
  `Dim<[usize; N]>`); no rstsr dependency. Call-frequency claims cross-checked
  against real code by grepping every `.size()` site at `aa24643`.

## Call-frequency reality in rstsr (grounding)

All `.size()` occurrences in `rstsr-native-impl/src/cpu_{serial,rayon}` and
`rstsr-core/src/tensor` are **once per operation** (dispatch, allocation
sizing, validation) — none inside a per-task closure, per-step iterator
`next()`, or per-element loop. `check_strides` (the other hot candidate) calls
it once per layout validation (and its zero-alloc rewrite landed separately,
see memory `rstsr-broadcast-write-gates-t1`). So the realistic frequency
benchmark is `kernel_per_task`, not the frequency sweep.

## Results

Loop floor (cached-load benches bottom out here): ~0.18 ns.

### Raw `size()` call (`size_call_raw`)

| case | recompute | cached | delta |
|---|---|---|---|
| vec, ndim 2 | 0.541 ns | 0.177 ns | **+0.36 ns** |
| vec, ndim 4 | 0.856 ns | 0.177 ns | **+0.68 ns** |
| vec, ndim 8 | 0.784 ns | 0.177 ns | **+0.61 ns** |
| arr `[usize; 4]` | 0.177 ns | 0.177 ns | **≈ 0** (floor) |

The inline-array regime — the common `Dim<[usize; N]>` case — already costs
nothing measurable: the product is a few in-register multiplies. Re-run
transient: raw-recompute varied ±15% between runs (0.78–0.92 ns at ndim 8);
always sub-ns.

### Creation path, fairly paired (`construct_plus_one_size`)

Both variants clone shape+stride; recompute pays the product at the single
`size()` call, cached pays it in the constructor. One product either way.

| ndim | recompute | cached |
|---|---|---|
| 2 | 13.44 ns | 14.31 ns |
| 4 | 13.73 ns | 15.67 ns (±9% transient) |
| 8 | 14.20 ns | 16.01 ns (−6.2% transient) |

Caching does **not** make creation cheaper — it makes it ~1–2 ns slower with
worse run-to-run variance (extra field write, larger struct).

### Struct-size cost of the derived field

| type | recompute | cached |
|---|---|---|
| `Layout` (Vec, `IxD`) | 56 B | 64 B |
| `Layout` (`[usize; 2]`) | 40 B | 48 B |
| `Layout` (`[usize; 4]`) | 72 B | 80 B |
| `Option<Layout>` (Vec) | 56 B | 64 B |
| `&Layout` (args) | 8 B | 8 B |

`Option<Layout>` stays at base size in both variants (`Vec`'s `NonNull` niche
hosts the tag). The +8 B applies wherever a layout is *stored*: `TensorBase`,
iterator structs, `Vec<Layout>`.

### Downstream effects of the bigger struct

| bench | recompute | cached | Δ |
|---|---|---|---|
| `clone_layout` vec (2 Vec clones + POD) | 12.95 ns | 13.26 ns | +2.4% |
| `clone_layout` arr4 (pure memcpy) | 1.132 ns | 1.163 ns | +2.7% |
| `per_step_clone` ×1000 (`IterAxesView::next` analog) | 14.23 µs | 14.42 µs | +1.3% |
| `kernel_per_task` 1e6 f64 adds, 64 tasks, 1 `size()`/task | 353.5 µs | 355.5 µs | +0.55% (noise) |
| `walk_size_every` 256 (1e6 elems) | 1.987 ms | 1.991 ms | parity |
| `walk_size_every` 64 | 1.906 ms | 1.948 ms | parity (noise) |
| `walk_size_every` 1 (**per element**) | 2.793 ms | 1.864 ms | **−33%** |

### Break-even

Recompute costs ~0.4–0.7 ns/call (heap shape) and ~0.2 ns (inline). A kernel
loses >1% of its time to recompute only if it calls `size()` more often than
once per ~40–70 ns of work — i.e. once per **~8–24 elements** for simple flop
kernels. Current rstsr's densest real frequency is once per task (thousands of
elements): measured parity.

## Conclusion

1. **Do not add a cached `size` field to `Layout`.** The win at rstsr's actual
   call frequencies is unmeasurable; the costs (+8 B/layout, +1.3–2.7% on
   clone-heavy paths, slower and noisier construction) are real. The owner's
   reluctance to add a derived field is validated by measurement.
2. **Do fix the doc**: `Layout::size`'s "# Note: This function uses cached
   size" is false — the product is recomputed. A one-line doc correction (and
   optionally `checked_mul` paranoia from §4.5, separate decision) is free.
3. **If a future hot path needs `size()` per step/per element**, the right fix
   is local hoisting (`let size = layout.size();` before the loop — sound
   because layouts are immutable during kernels), not a struct-wide cache.
4. The unchecked-wrap aspect of §4.5 (shape product overflowing `usize`) is a
   soundness-paranoia question, not an efficiency one, and is orthogonal to
   caching: a `checked_mul` recompute costs the same ~0.4 ns for realistic
   shapes and keeps the struct unchanged.

## Reproduce

```sh
cd 2026-09-16-layout-size-cache-mwe
cargo run --release            # struct-size table
cargo bench                    # all groups (criterion, ~10 min)
cargo bench --bench bench -- "size_call_raw"   # one group
```

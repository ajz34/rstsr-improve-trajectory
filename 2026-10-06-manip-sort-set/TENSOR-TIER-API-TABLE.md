# Tensor-tier API table — manip/sort/set wave

Scratch report, 2026-10-07. Scope: the 16 functions added relative to `main`
(`e7cdc6a`) on branch `261006/manip-sort-set`, **tensor tier only**
(`rstsr-core/src/tensor/**`): the `rt::` free functions and the associated
methods (the `TensorAny` surface). Device kernels, device-crate auto-impls and
the faer-py bindings are out of scope.

Every call form below was checked by compiling it (probe files
`rstsr-core/tests/zz_forms*.rs`, since deleted); "accepted input" means it
compiles. `x` = the input tensor, `T/B/D` = dtype/backend/dim as usual.

State: everything is as committed in `2aacd5e` **except** the method forms of
`isin` and `diff`, which were missing and were added 2026-10-07 in the working
tree (uncommitted).

## 1. The table

| # | function | returns | free fn (panicking) | free fn (fallible) | method (panicking) | method (fallible) |
|--|--|--|--|--|--|--|
| 1 | `repeat` | `Tensor<T,B,IxD>` | `rt::repeat((x, repeats))`<br>`rt::repeat((x, repeats, axis))` | `rt::repeat_f(x, repeats, axis)` | `x.repeat(repeats, axis)` | `x.repeat_f(repeats, axis)` |
| 2 | `roll` | `Tensor<T,B,D>` | `rt::roll((x, shift))`<br>`rt::roll((x, shift, axis))` | `rt::roll_f(x, shift, axis)` | `x.roll(shift, axis)` | `x.roll_f(shift, axis)` |
| 3 | `tile` | `Tensor<T,B,IxD>` | `rt::tile((x, repetitions))` | `rt::tile_f(x, repetitions)` | `x.tile(repetitions)` | `x.tile_f(repetitions)` |
| 4 | `sort` | `Tensor<T,B,IxD>` | `rt::sort((x, args))` | `rt::sort_f(x, args)` | `x.sort(args)` | `x.sort_f(args)` |
| 5 | `argsort` | `Tensor<usize,B,IxD>` | `rt::argsort((x, args))` | `rt::argsort_f(x, args)` | `x.argsort(args)` | `x.argsort_f(args)` |
| 6 | `sort_custom` | `Tensor<T,B,IxD>` | `rt::sort_custom((x, axis, cmp))` | `rt::sort_custom_f(x, axis, cmp)` | `x.sort_custom(axis, cmp)` | `x.sort_custom_f(axis, cmp)` |
| 7 | `argsort_custom` | `Tensor<usize,B,IxD>` | `rt::argsort_custom((x, axis, cmp))` | `rt::argsort_custom_f(x, axis, cmp)` | `x.argsort_custom(axis, cmp)` | `x.argsort_custom_f(axis, cmp)` |
| 8 | `searchsorted` | `Tensor<usize,B,IxD>` | `rt::searchsorted((x1, x2, args))` | `rt::searchsorted_f(x1, x2, args)` | `x1.searchsorted(x2, args)` | `x1.searchsorted_f(x2, args)` |
| 9 | `take_along_axis` | `Tensor<T,B,IxD>` | `rt::take_along_axis((x, indices, axis))` | `rt::take_along_axis_f(x, indices, axis)` | `x.take_along_axis(indices, axis)` | `x.take_along_axis_f(indices, axis)` |
| 10 | `isin` | `Tensor<bool,B,IxD>` | `rt::isin((x1, x2, invert))` | `rt::isin_f(x1, x2, invert)` | `x1.isin(x2, invert)` | `x1.isin_f(x2, invert)` |
| 11 | `diff` | `Tensor<T,B,IxD>` | `rt::diff((x, axis, n, prepend, append))` | `rt::diff_f(x, axis, n, prepend, append)` | `x.diff(axis, n, prepend, append)` | `x.diff_f(axis, n, prepend, append)` |
| 12 | `nonzero` | `Vec<Tensor<usize,B,IxD>>` | `rt::nonzero(x)` | `rt::nonzero_f(x)` | `x.nonzero()` | `x.nonzero_f()` |
| 13 | `unique_values` | `Tensor<T,B,IxD>` | `rt::unique_values(x)` | `rt::unique_values_f(x)` | `x.unique_values()` | `x.unique_values_f()` |
| 14 | `unique_counts` | `UniqueCounts<T,B>` | `rt::unique_counts(x)` | `rt::unique_counts_f(x)` | `x.unique_counts()` | `x.unique_counts_f()` |
| 15 | `unique_inverse` | `UniqueInverse<T,B>` | `rt::unique_inverse(x)` | `rt::unique_inverse_f(x)` | `x.unique_inverse()` | `x.unique_inverse_f()` |
| 16 | `unique_all` | `UniqueAll<T,B>` | `rt::unique_all(x)` | `rt::unique_all_f(x)` | `x.unique_all()` | `x.unique_all_f()` |

Method *mechanism* (invisible in the call syntax, but visible to the compiler
and to rustdoc):

- rows 1–11: inherent methods on `TensorAny<R,T,B,D>` (`&self` receiver,
  rustdoc-addressable as `TensorAny::sort`, …);
- rows 12–16: methods come from the exported API traits (`NonzeroAPI`,
  `UniqueValuesAPI`, `UniqueCountsAPI`, `UniqueInverseAPI`, `UniqueAllAPI`)
  implemented for `&TensorAny<R,T,B,D>` **and** for `TensorView<'_,T,B,D>` —
  i.e. they need the trait in scope (the prelude re-exports them) and the
  `TensorView` impl takes `self` **by value**, so
  `let v = x.view(); v.nonzero(); v.nonzero();` does not compile (the view is
  moved), whereas `v.sort(()); v.sort(());` does.

## 2. Accepted input forms (free functions)

`TensorViewAPI` in the fallible column means the parameter accepts
`TensorAny` by value (which covers `Tensor`, its ownership aliases, and
`TensorView` by value), `&TensorAny` (covers `&Tensor` and `&TensorView`) and
`&mut TensorAny` — the three impls in `tensor/ownership_conversion.rs`. The
concrete forms `&x` and `x.view()` were compile-checked.

| function | panicking free fn input | fallible free fn input |
|--|--|--|
| `repeat`, `tile`, `sort`, `argsort`, `sort_custom`, `argsort_custom`, `searchsorted`, `take_along_axis`, `isin` | `&TensorAny`, `TensorView`, `&TensorView` | `impl TensorViewAPI` (incl. `TensorView` by value) |
| `roll`, `diff` | `&TensorAny`, `&TensorView` (**not** a `TensorView` by value) | `&TensorAny` (incl. `&TensorView`; **not** a view by value) |
| `nonzero`, `unique_*` | `&TensorAny`, `TensorView` | `impl TensorViewAPI` (incl. `TensorView` by value) |

## 3. Consistency findings

### 3.1 Panicking free fn is tuple-dispatch, fallible free fn is positional (11/16)

```
rt::sort((&a, ()))        // panic: one tuple argument, dispatched via SortAPI
rt::sort_f(&a, ())        // fallible: plain positional arguments
rt::repeat((&a, 2, None)) // vs
rt::repeat_f(&a, 2, None)
```

Affects `repeat`, `roll`, `tile`, `sort`, `argsort`, `sort_custom`,
`argsort_custom`, `searchsorted`, `take_along_axis`, `isin`, `diff`. The
mirror-image call `rt::sort_f((&a, ()))` does **not** compile.

Existing families pick *one* style and use it on both members:

| family (pre-wave) | panicking | fallible |
|--|--|--|
| `flip`, `transpose`, `reshape`, `index_select`, `take`, `bool_select`, `nonzero`, `unique_*` | positional | positional |
| `concat`/`concatenate`, `diag`, `unstack` | tuple-dispatch | tuple-dispatch |

So the mixed pair introduced by this wave has no precedent in the tree.

### 3.2 Free-fn arity overloads are inconsistent

| function | overloads offered by the free fn | method |
|--|--|--|
| `repeat` | `(x, repeats)` and `(x, repeats, axis)` | only `(repeats, axis)` |
| `roll` | `(x, shift)` and `(x, shift, axis)` | only `(shift, axis)` |
| `tile` | `(x, repetitions)` | `(repetitions)` |
| `searchsorted` | `(x1, x2, args)` only — args mandatory (pass `()`) | args mandatory |
| `take_along_axis` | `(x, indices, axis)` only | axis mandatory |
| `isin` | `(x1, x2, invert)` only — `invert` mandatory | `invert` mandatory |
| `diff` | 5-tuple only — `axis`, `n`, `prepend`, `append` all mandatory | all mandatory |

Rust has no default arguments, so the axis-less / default-arg call cannot be
expressed as a second method of the same name. Consequence: the short forms
that *do* exist (`rt::repeat((&a, 2))`, `rt::roll((&a, 1))`) have no method
counterpart; the method-side way to say "no axis" is the explicit
`x.repeat(2, None)` / `x.repeat(2, ())`. `x.repeat(2)`, `x.isin(&b)`,
`x.diff(-1)` do not compile.

### 3.3 `roll` / `diff` reject a `TensorView` by value in the free form

```
rt::sort((a.view(), ()))              // ok
rt::roll((a.view(), 1))               // error: (TensorView, i32): RollAPI not satisfied
rt::diff((a.view(), 0, 1, None, None)) // error: (TensorView, ...): DiffAPI not satisfied
```

`roll_f` needs `DataIntoCowAPI` and `diff_f` needs `DataCloneAPI` on the
storage, which a `DataRef` view does not provide; the other tuple families got
`TensorView` tuple impls during R2. `&a.view()` works for both. The methods do
accept views (inherent, `&self`).

### 3.4 Method receiver semantics differ between groups

Inherent methods (rows 1–11) borrow (`&self`) → a view can be reused. Trait
methods (rows 12–16) are `self`-by-value on the `TensorView` impl → the view is
consumed. `a.nonzero()` on an owned tensor still only autoborrows (the impl is
on `&TensorAny`), so this only bites when the receiver is a view value.

### 3.5 Documentation vs. reality

- `api_specification.md` marks these functions `assoc/fn` ("both associated
  method and function"), which is true now, but the table cannot express the
  tuple-vs-positional split of §3.1.
- The anchors' `Overloads Table` / examples only ever show the panicking tuple
  form; the fallible free function is documented as "fallible version" with no
  signature, so the positional form is not discoverable from the docs.
- Rows 12–16 did not list their associated methods in
  `## Variants of this function`; added 2026-10-07 (working tree).

## 4. Suggested directions (for discussion, not implemented)

1. **Pick one style per family and use it for both members.** Either give the
   `_f` twins tuple-dispatch overloads (`rt::sort_f((&a, ()))`, matching
   `concat_f`), or promote the positional call to the anchor and keep the tuple
   form as the overload. The former is the smaller change: add
   `XAPI::x_f` tuple impls so `rt::foo_f((args))` dispatches like `rt::foo((args))`.
2. **Align the default-argument story**: either always require the full
   argument list in both free and method forms (drop the 2-tuple `repeat`/`roll`
   overloads), or add `Args` structs with `From<()>` (like `SortArgs`,
   `VarArgs`) so one form covers "given" and "default".
3. **Give `roll`/`diff` the `TensorView` tuple impls** the other families got
   (or document why they cannot have them).
4. Decide whether rows 12–16 should become inherent methods too, for a uniform
   mechanism (`&self`, no trait import, no view consumption) — at the cost of
   duplicating the API-trait surface.

# Manual Review -- Round 2

Note for manual review, you are able to edit codes and run tests, but prohibited to auto git commit.
This review also involves code style discussion, and you may browse the project for comprehension. And you may also take notes on code style to rstsr-agents related skills if I'm correct.

rstsr commit a90b4130d06cd2d8f6323efd1e5cfbd73792d666, compared to main e7cdc6a6c96727bb9d3b4559930cb0f1a56854c8.

This is some personal ideas. If the following is not clear or not correct, you can rebute me.
Also, the problems listed here may be partial. You should also see the whole project to find similar problems, and not only limited the current commit to main branch. You can also check if existing main branch has similar problems.

## 1. Named tuple of unique related functions

For example of rstsr-core/src/tensor/set.rs (a90b413), you can implement `Into` like similarly this way

```rust
impl From<(Tensor, Tensor)> for UniqueResult
```

and so that `let (values, counts) = rt::unique_counts(&tsr).into();` is probably valid. Similar unique functions also applies.
Well the inverse implementation `impl From<UniqueResult> for (Tensor, Tensor)` is also valid, but actually not necessary. You can even remove them.
And also you can add documentation at docstring on the `.into()` to tuple usage.

## 2. Usage of `impl TensorViewAPI` instead of `&TensorAny`

For example of code

```rust
pub fn searchsorted_f<R1, R2, T, B, D1, D2, AArg>(
    x1: &TensorAny<R1, T, B, D1>,
    x2: &TensorAny<R2, T, B, D2>,
    args: AArg,
) -> Result<Tensor<usize, B, IxD>>
```

I'm not sure if it is possible to use TensorViewAPI to 1) simplify trait bound 2) make view also applicable.

This usage is not generally applied to everything, but seems to be possible here. Some examples can be

```rust
pub fn allclose_all_f<TA, TB, TE, B, DA, DB>(
    tensor_a: impl TensorViewAPI<Type = TA, Backend = B, Dim = DA>,
    tensor_b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
    isclose_args: impl Into<IsCloseArgs<TE>>,
) -> Result<bool>
```

## 3. Expand implementation on arguments

Example of code rstsr-core/src/tensor/sorting.rs, line 147

```rust
impl<R, T, B, D, AArg> ArgSortAPI<()> for (&TensorAny<R, T, B, D>, AArg)
```

You can also try implement `(TensorView<T, B, D>, AArg)` for `ArgSortAPI<()>`. Things for others like `SortAPI` and others are similar.

## 4. Visit order of flattened iteration form

Related code: rstsr-core/src/tensor/manipulation/repeat.rs, line 222

Well for those code that relates to flattened visit order, I suggest you change the current behavior. Let row-major to flatten in row-major order, and column-major to flatten in column-major order. So that row/col major may be different in output results (you can check that `reshape` is a good example of this, and `flatten` is simply `reshape(-1)`).

From implementation view of point, you may
- dispatch a function for row-major only (following current implementation);
- if row major, just call that function; if column major, first reverse-axes the input tensor (by view, `tsr.reverse_axes()`), then call that function, and finally reverse-axes the output tensor (by into, `tsr.into_reverse_axes()` to avoid unnecessary copies).

I'm not sure if this is achievable, and you may gonna test this (by set device's default order, not by cargo feature).

May perhaps other functions like `roll` also affected, and you may also apply this kind of change.

## 5. Usage of NDIndex

Related code: rstsr-core/src/tensor/manipulation/tile.rs, line 83

I suggest that there is probably similar structs in `IterLayoutRowMajor` of `rstsr-common/src/layout/iterator.rs`? Probably they are similar, and if they are, you can try use rstsr-common's.

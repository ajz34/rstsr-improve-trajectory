# Manual Review -- Round 1

Note for manual review, you are able to edit codes and run tests, but prohibited to auto git commit.
This review also involves code style discussion, and you may browse the project for comprehension. And you may also take notes on code style to rstsr-agents related skills if I'm correct.

rstsr commit 6f2bc9eb3d738eea44f250fc41b82f1539de1c83, compared to main e7cdc6a6c96727bb9d3b4559930cb0f1a56854c8.

This is some personal ideas. If the following is not clear or not correct, you can rebute me.
Also, the problems listed here may be partial. You should also see the whole project to find similar problems, and not only limited the current commit to main branch. You can also check if existing main branch has similar problems.

## 1. Error handling in api.py

Currently some functions implemented in api.py have complicated error handling logic, some of them may be possible to merged into rust side code.

- Example: rstsr/crates-interop/rstsr-faer-py/python/rstsr_faer/api.py, line 1395, `indices.ndim != h.ndim()` in `take_along_axis`.
- Note: crates-interop/rstsr-faer-py/python/rstsr_faer/api.py, line 1446, repeat function requires arguments that in python side to be int, so type check needed. This is naturally handled in rust side, so you can state these value check reasons in one-liner comment in python side.

## 2. Type trait of Tensor and Op

Example code at `rstsr-core/src/tensor/nonzero.rs`, line 14, which includes unexpected type trait requirements on `T: ExtZero` at tensor level. The expected behavior at here is either `T: Clone` or nothing.

In my consideration, type trait handling should be at device implementation level, not at tensor level. See rstsr-core/src/tensor/operators/op_binary_common.rs for example. In `tensor/operators/op_binary_common.rs`, you may see that though different binary functions requires different type trait to `T` (tensor element type), but there is no real type trait confinement at tensor level. The type trait confinement is at operator implementation at device level (like `rstsr-core/src/device_cpu_serial/operators/op_ternary_common.rs`).

There are two exceptions on this rule:
- Arithmetic `rstsr-core/src/tensor/operators/op_binary_arithmetic.rs` where we need type traits to overload rust's arithmetic operators like `+-*/`. But still note, we use the trait bound on type `T` is for operator implementation, not for how it is implemented in rust std or CPU level. Reason is explained later in "Back to nonzero".
- `rstsr-core/src/tensor/operators/op_with_func.rs`, which is general form and pass function `&mut F` as argument. However, this is not common use case, and is CPU only.

Speaking to CPU-only and device-non-dependent tensor implementation, first we have actually no implementation on GPU, so the discussion is pure theoretical. We hope most of rstsr's functions can be gpu-acceptable in future, but several functions are surely not gpu-acceptable.
For example of `rt::sort_custom`, which includes `f: Fn(&T, &T) -> ...` or similar which is a user-specified function. This inclusion will make gpu-non-acceptable but we have no other ways to implement it. Similar cases can be `rt::reduce`, or mapping functions like `mapv`. However, we hope `rt::nonzero` and `rt::sort` to be some kind of gpu-acceptable. No `f: Fn(&T, &T) -> ...` is only a step of gpu-acceptable, but all 3-tier architecture should be incropperated.

Back to nonzero, we see its trait function defined as
```rust
    fn nonzero_count(
        &self,
        a: &<Self as DeviceRawAPI<T>>::Raw,
        la: &Layout<D>,
        is_nonzero: &dyn Fn(&T) -> bool,
    ) -> Result<usize>;
```

There is `is_nonzero` function here. Why that's bad? If we will implement in GPU device, the function of `&dyn Fn(&T) -> bool` is not meaningful. This function should be device-dependent/specific.

And for implementator `rstsr-core/src/device_cpu_serial/nonzero.rs`, we do not see type trait confinement on `T`, but it is expected to add trait bound here.

## 3. Tensor-layer should be thin wrapper of device-layer

Also `rstsr-core/src/tensor/nonzero.rs`.

We first say that there is 3-tier architecture:
- Tensor layer: Tensor trait (to let associated function applicable, like `TensorATan2API`) and exportable functions (`rt::atan2`)
- Device Operator layer: Operator trait `OpATan2API` that is universal for devices, but without implementation. Tensor layer will call this trait function.
- Device implementation layer: Implement the operator trait for specific device. This is device dependent.

For functions that applies with 3-tier architecture, the tensor layer should be thin wrapper of device layer. It handles basic layout check, dimension check, input argument sanity check, allocates output tensor buffer (via something like `device.uninit_impl(lc.bounds_index()?.1)?;`), and then call the device operator trait function.

Return to nonzero implementation, I'm not sure but you seems to called `outof_cpu_vec`, which should be used as last resort. The output buffer should be either pre-allocated by tensor layer (you need change way of implementation), or allocated by device operator layer (also need change to device operator trait, which can return some storage instead of returning only `Result<()>`/`Result<(usize)>`). We may have very few cases that device returns output buffer (since most cases we require tensor-layout generic algorithm implementation), but in some cases this can be changed (but still not preferred this way, since row-major and column-major will have different layout preferrences).

## 4. Usage of `use_fast_path`

Refer to `rstsr-core/src/device_cpu_serial/set.rs`.

In grill document, I said that you can implement as PartialEq (by definition any types that defines equality is valid to be appliable for unique_* functions), and dispatch some kind of types by fast path.
However, In current status, you required `T: PartialEq + ExtSortCmp`, which is not expected. What I hope is `T: PartialEq`, and dispatch/unroll each type listed (like f64, bool, i8, ..., u128, f64, *f32*) with unsafe transmute casts. In this way, I suppose word "ExtSortCmp" is not going to be appear in this file. Also recall faer's and blas's way of implementing f32/f64/c32/c64 matmul. Matmul usually only requires `T: Copy + Add + Mul`, but we dispatched/unrolled those four types with transmute casts, to use the faster blas implementation.

Also, notice that if I'm correct, `half` is cargo-feature-gated, so you need use some `cfg(feature)` trick for half::f16/bf16.

## 5. Docstrings should not refer to local directories

See `rstsr-core/src/tensor/set.rs`, line 213. There are some notices like `tests/tracking/numpy_differences.md`, which is not expected. You either include them as docstring code blocks, or use plain language if it is easy to describe.

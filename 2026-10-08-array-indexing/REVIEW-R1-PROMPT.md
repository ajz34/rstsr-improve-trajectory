This review comment is of manual human review, related to `ArrayAuxIndexer` and row/col-major implementation issue.

I think this still should be defined as

```rust
pub struct ArrayAuxIndexer<'a> {
    pub src_axis: usize,
    pub indices: &DeviceRawAPI<usize>::Raw,
    pub layout: Layout<IxD>,
}
```

Note indices should be at device-level (not host-level, though for CPU host/device are both CPU).

And we do not require the indices to be row-major; but for any layout, as long as `(indices, layout)` represents the same tensor, the indexing operation will be the same (given the default order; note different default order will still give different results). This is important and may affect what you are doing. Anyway, indices can be iterated with given layout. That is what I mean "layout-generic".
And, probably you need to be noticed, that array indexing is default-layout aware. You need to pass default layout to `DeviceArrayIndexAPI::array_index`, and while in this function's CPU serial/rayon-auto implementation, for column-major case, you need to either 1) revert input `ArrayAuxIndexer.layout` axes order to row-major layout, apply row-major indexing; 2) use column-major indexing iterator. You should try both ways to see if you get the same result, and choose one way that you prefer.

And since indices are using `DeviceRawAPI<usize>::Raw`, the function implementattion at rstsr-core/src/tensor/array_indexing.rs will need some refactoring. You are not going to extract indices from tensor to slice, but put slice/vec data to tensor by outof_cpu_vec or from from_cpu_vec to device storage.

Anyway, we deferred column-major tests, so you should change column-major behavior, but still make row-major tests pass.

Q3':

On astype, you can also try implement in this stage.

Well for reduction, especially cast-then-reduce mechanism, I have something to say. Remember that we have

```rust
// cpu_rayon/reduction.rs
    I: Fn() -> TS + Send + Sync,
    F: Fn(TS, TI) -> TS + Send + Sync,
    FSum: Fn(TS, TS) -> TS + Send + Sync,
    FOut: Fn(TS) -> TO + Send + Sync,
```

And now the TI should be input type, TS should be the casted type, and TO should be the output type. It is probably that TS usually same to TO, so identity.
The reduction operation should usually not create a new tensor. In most cases, we just go through multiple numbers into a single number, and in this case additional memory is strictly O(1). In some cases of strided sum, we may add tensors by array, but that additional memory will not exceed output tensor, or it is actually O(1).
So if cast-then-reduce on element, that's okay (where the casting procedure may happen in function `F: Fn(TS, TI) -> TS`). But if that cast on a tensor, then a whole new tensor will be created, and that's not acceptable.
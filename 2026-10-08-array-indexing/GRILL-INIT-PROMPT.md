You are going to implement array indexing (fancy indexing).

- We will formally name it as "array indexing" in this context, in rstsr. We name "advanced indexing" for boolean/mask/integer indexing, which is one kind of advanced indexing for numpy-like packages. "array indexing" is more like fancy indexing.
- We may need a new type along with `AxesIndex`.
    
    - Recall `AxesIndex<Indexer>`
    
    ```rust
    pub enum AxesIndex<T> {
        None,
        Val(T),
        Vec(Vec<T>),
    }

    pub enum Indexer {
        Slice(SliceI),
        Select(isize),
        Insert,
        Ellipsis,
    }
    ```

    These are defined in rstsr-common. However, for array indexing, we may need a new indexer enum:

    ```rust
    pub enum ArrayIndexer<B> {
        Basic(Indexer),
        ArrayIndex(Tensor<isize, B, IxD>),
        ArrayBool(Tensor<bool, B, IxD>),
        OneDimIndex(Vec<isize>),  // will cast to ArrayIndex when parsing, only for `From` trait implementation
        OneDimBool(Vec<bool>),  // will cast to ArrayIndex when parsing, only for `From` trait implementation
    }
    ```

    where `B` is the backend type, `IxD` means we only handle dynamic dimensionality for ArrayIndexer.
    You may take `TryFrom<AxesIndex<ArrayIndexer<B>>, Error: ...>` as the input type of array/fancy indexing.
    We have defined various `From` trait for `ArrayIndexer`, and so do we need to do that for ArrayIndexer. Note that `Vec/&Vec<isize/usize/i64/u64/i32>` (as 1-d), `&[isize/usize/i64/u64/i32]` (as 1-D), `[isize/usize/i64/u64/i32; const N as usize]` (as 1-d), `Tensor/TensorView/&TensorAny<isize, B, IxD>` are all valid array indexers. We may need to implement `From` trait for them.

- We do not implement even more advanced indexing (like numpy's `tensor[1:3, ([0, 1, 2], [0, 2, 1])]`). But we are going to implement schemes like numpy's `tensor[1:3, [0, 1, 2], [0, 2, 1]]` (all indexers are not grouped by parentheses).

- This feature requires device implementation. You need to create a new operator (maybe say `OpArrayIndexAPI`) for array indexing. But note that at device implementation level, there should be of no tensor, so you may also need an auxiliary enum struct like

    ```rust
    pub enum ArrayAuxIndexer<V> {
        Basic(Indexer),
        Array(V, Layout<IxD>)
    }
    ```

    where `V` is the storage or data of the tensor (on CPU, be storage CPU, or simply the raw data type `Vec<T>`).

    Recall 3-tier (tensor-layer, device-interop, device-impl). You may need to first cast `ArrayIndexer<B>` to `ArrayAuxIndexer<V>` and determine output tensor shape and pre-allocate output tensor storage at tensor tier, pass `ArrayAuxIndexer<V>` at tensor-layer or device-interop, and implement the indexing algorithms at device-impl.

- The signature of tensor-layer may be

    ```rust
    pub fn array_index_f(tensor: impl TensorViewAPI, indexer: impl TryFrom<AxesIndex<ArrayIndexer<B>>, Error: ...>) -> Result<Tensor<...>> // or Result<TensorCow<...>>

    pub fn array_index(...) -> Tensor<...> // or TensorCow<...>
    {
        array_index_f(tensor, indexer).rstsr_unwrap()
    }

    impl TensorAny {
        pub fn array_index_f(&self, indexer: impl TryFrom<AxesIndex<ArrayIndexer<B>>, Error: ...>) -> Result<Tensor<...>> // or Result<TensorCow<...>>

        pub fn array_index(&self, indexer: impl TryFrom<AxesIndex<ArrayIndexer<B>>, Error: ...>) -> Tensor<...> // or TensorCow<...>
        {
            self.array_index_f(indexer).rstsr_unwrap()
        }
    }
    ```

    `OpArrayIndexAPI` may be defined as

    ```rust
    pub trait OpArrayIndexAPI: DeviceAPI {
        fn array_index_impl(
            &self,
            c: &DeviceRawAPI::Raw,
            lc: &Layout,
            a: &DeviceRawAPI::Raw,
            la: &Layout,
            indexer: &[ArrayAuxIndexer<DeviceRawAPI::Raw>]
        )
    }
    ```

    However, depending on whether you performed all possible basic indexing before array indexing, the array index implementation can pass `indexer: &[(isize (axis), V (array data), Layout<IxD> (array layout))]` instead of `ArrayAuxIndexer<V>` that shipped along with basic indexer. I prefer you to choose the `indexer: &[(isize (axis), V (array data), Layout<IxD> (array layout))]` way, that is probably more difficult but better.
    Or say explicitly: For case like `tensor[1:3, [0, 1, 2], [0, 2, 1]]`, you can perform all possible basic indexing first, let `t = tensor[1:3]`, then pass `t[:, [0, 1, 2], [0, 2, 1]]` to the device implementation (note that `t` in this case is still 3-d, so it is different to `t[[0, 1, 2], [0, 2, 1]]`, so that if basic indexing is performed first, then we need to explicitly pass the axis of the array indexer to the device implementation). Anyway, this proposal may not fit to real situation, but it is a proposal and you need to be very careful on implementing.

- Our final target is the similar to numpy's advanced indexing, pass array-api tests, and also transfer and pass most numpy's advanced indexing tests, in row-major.

- You are also noted that column major may have different behavior than row-major, in that the indexing order will be different. This will be real-different on >1-dimensional array indexing. For example of the above example, in rust, the user can write as

    ```rust
    let t = tensor.array_index((1..3, [0, 1, 2], [0, 2, 1]));
    ```

- If you are sure on yourself, when encountering array indexing that can be reduced to basic indexing, you can return viewed tensor. And notice `TensorCow` can be used to return either viewed tensor (no real array indexing) or owned tensor ().

This is grill dialog, and you have chance to question me and polish thoughts on this non-trivial feature.

Also, you can refer to local directories ~/Git-Others for numpy and cupy source code as reference.

If you need subagents to browse project and other source code, propose grill questions after these subagents are finished and you have comprehensed their results (not to grill me during you are spawning subagents).

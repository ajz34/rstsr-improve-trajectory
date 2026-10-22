R2-Q1:

I may defend more on that flags should be light and compared part.
And comparasion should be applied to both light and stream (that not gpu ordinal-based, one ordinal may have two streams or flags).

R2-Q1-S1:

Let it fails.

Before uploading CPU tensors, the device should be pre-built. The math mode should be changed at this time. And then use something like `a.to_device(&device)` and `b.to_device(&device)` to move the tensors to the same device.

For math mode, I re-thinked the way:
```rust
pub static flags_default: Arc<Mutex<CudaRuntimeFlags>> // anyway, make it possible to be changed by some safe way.

pub struct DeviceCuda {
    shared: Arc<CudaOrdinalObj>, // only compare ordinal number
    stream: Arc<CudaStreamObj>, // better to be comparable
    /// flags to override; user can use `to/into/change_device` or some other way to change the device flags_override
    /// or simply let it to be None and use the default flags that can be changed by some safe way.
    flags_override: Option<CudaRuntimeFlags>, // comparable, deep clone
    default_order: FlagOrder,
}
```
By this way, we may defer `with_` functions, or `with_` functions only apply to the default flags, not device level.

This may add some complexity, so maybe some device functions like `device.get_flag(&str)` need to be added. Set flags use other ways, probably not `device.set_flag(&str, flag)` but `device.set_flags_override(flags_override)` and directly change `flags_default`.

I think for R2-Q1 we have not reached a final decision. Still challenge me.

R2-Q2:

The note on "Cross-stream operands are legal", I may disagree, or agree but on condition that I feel need to log some warning message (not error) when cross-stream operands are used. I think since rstsr tensor explicitly need device parameter on generation, and `to_device` should be returning view (or Cow's view) instead of copy, then changing stream is available and should be explicit. Stream only on LHS may pose some design issue on user side.

The note on "Stream selection = device-value transformation", I think that it is better `let mut d2 = d.clone(); d2.set_stream(stream2);`, like that of set default order.

The note on "Async H2D/D2H on side streams", note `outof_cpu_vec` is naive function that use device's own stream. We may need `rt::cuda::async_h2d` or something that can be used to truely do async H2D/D2H on side streams.

For question "What this makes revertable vs not", I am considering `CudaOrdinalObj` should have a field of `default_stream: Arc<CudaStreamObj>`, and `override_stream: Arc<Option<CudaStreamObj>>`. The function `with_stream` will change the shared `override_stream` status, and if the computation detects `override_stream` set, enforce all computation (not tensor creation) to use the `override_stream`. So maybe we can support some flexible stream selection: shared override_stream > device stream > shared default stream.

I think on how stream handled we have not reached a final decision. Still challenge me.

R2-Q3:

Previous discussion not matching the ground of your question. Need further discussion.

R2-Q4:

Okay on 5.1. nd 5.2. A private `CudaAlloc`. Make cuda allocation be it's own.

R2-Q5:

Good and since it's separated now, I think it is only technical issue. Change on formatting will not touch the core part of cuda device/storage definition.

R2-Q6:

Okay. This problem is some detailed and probably not such intutive. Your direction is correct.

Change device is something similar to `change_shape` or other `change_` functions, that if input view, return view; if input owned, return owned without clone. Try preserve the original lifetime/ownership. And can use cloned owned as fallback.

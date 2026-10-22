Note though we are going to do things step-by-step, we need to make the first version (your "v1") to be better polished.
If any idea that is not easily revertable, you are suggest to better deeper challenge ourselves.

Also note I made changes on `Raw: Clone` trait bound in last commit in RSTSR. You need to check that.

F1:

Your insight is correct. And this is actually some pitfall in original design of RSTSR structure.

I made changes on `Raw: Clone` trait bound in last commit in RSTSR. But anyway, I still think that cuda data should implement `Clone` trait, and follow what your thought that Clone is actually deep copy. 

As a reference, though we haven't implemented hdf5 backend (which only available for assign and load, not computation). But if we have in future, I will probably not implement `Raw: Clone` on that.

F6:

We may or may not use cudarc's safe wrapper. If cudarc's wrapper is too restrictive, we simply call the underlying cuBLAS/cuBLASLt functions, just like what we did in CPU BLAS backends. So make sure not to be limited by cudarc's safe wrapper. We use cudarc is that it is well maintained, and its cuda-version cargo-feature gate is quite useful, and its CudaSlice is convenient and well designed. But not for its safe wrapper.

Q1:

Okay. Seems we need this way for types that not well defined in cudarc.

Also, the data's stream may need to introduced, or may ignored. You may seems to noticed this issue and this can be adjusted if we reached shared understanding.

Q1-p1: Make clone to be deep copy. We removed clone trait bound on DeviceAPI, but I will still suggest you implement this on cuda device. This clone may not be frequently used, but it is still useful for some cases.

Q1-p2: Okay, for your prefer. But I want to make sure if it is easy to revert to real 0-byte alloc.

Q1-p3: Well I also recommend not doing that. Also note we are not framework of deep learning; Arc thing is something commonly used in deep learning framework, but we may not.

Q2:

For mempool, I also have some other idea.
I think that is that possible to add a `DeviceAllocatorAPI`, for both CPU and cuda?
For CPU, it is common sense that `Vec<T, A = Allocator>` (which is commonly used in rust or C++). However, I'm not sure (since this is really irregular, but I'm not sure) if allocation/deallocation can add a trait, and by default it does nothing (rust Vec can be used as default allocator, for CPU). There may be some specialized cases, even in CPU, that we need some custom allocator, and if trait design can be used to achieve this.

Q2-p1: Follow pytorch/cupy, default cache. You can add threshold, default (None) to use up all GPU memory. Well but if it is possible that if GPU becomes idle, then release? But this may really cause some performance issue, and I'm not sure if this is possible.

Q2-p2: I think we need OOM-retry, as soon as possible, until everything is released but still memory not enough, then OOM.

Q2-p3: Well I think environment variable is better early designed, and I will accept this. Further more, we can also add device-dependent environment struct that stores these things, and the program can force write them (like atomic boolean flag write). But maybe environment variable should be stored at device (device object at runtime) or at global static but mutable atomic variable?

Q3:

Well I feel maybe (c) is an option. Maybe it's worst, and you need to explain to me in more detail.

This also affects that how to compare equal between two devices, and how clone performs.

- Heavy weight device object will probably only one per device (gpu(0), gpu(1), etc). Equal compare always be true if they are the same device, and clone will be shallow like pointer copy (if it is sound to you).
- Light weight device will contain flags (though flags can be very much?). Equal compare will only be true if all flags are the same, and clone will be deep copy.

Q3-p1: Well I think it is not needed to lazy device. If on CUDA-less machine, we can just panic. Or the user should not create a CUDA device.

Q3-p2: My previous comment on heavy/light weight may states this, though may not be clear or deeply thoughted. In this way, math_mode is considered to be light weight and should be compared. Ordinal objects may be heavy weight, always use same pointer, and should not be compared or compared to be true when on same gpu ordinal. I'm not sure how stream should be treated, probably this is heavy-weight but something need to be compared (as a third category)?

Q3-p3: Well yes, and better to also have common display format that only prints ordinal (like `cuda:0` with no flags), and debug to be more detailed.

Q4:

As heading comment, I'm suspecious to notations like "v1". If this decision is hardly changed or reverted, then we need to discuss more.

I prefer option (b) that makes streams user-visible from day one.

Re challenge me on this question.

Q5:

As my suggestion, that maybe we need to split cuda device to 2-3 parts, heavy-weight/light-weight/stream parts (if stream to be in the device or other heavy-weight components that prone to comparasion).

I'm in doubt on "no env var". Surely env var may need to be stored in some mutable atomic static variable, but I'm not sure whether I need entries to also be in device object, and if None we read the static variable.

Q5-p1: Agree, and should try to let be comparable.

Q5-p2: I prefer split cuda device to parts, let heavy part to be global, let lightweight part to be per device. And I think in this scheme, TLS mode is strightforward (though heavy part is actually global, it can be Arc/ArcMutex wrapped).

Q6:

As long as the previous device split scheme also fits this kind of design, I think it is okay.
But if math mode is not shared state, maybe Closure scoped way is the only way? If not, you can also try (a).

Q7:

Please note that we have changed display implementation in last commit, you need to check that and see if full D2H is avoidable. Efficiency is important but less memory copy is more important for printing.

Q8:

Well note just a note that `change_device` from device to device, if on the same device (ordinal number same), then data copy should be avoided (but flag copy of device is required). Different device (ordinal number different), then data copy is required. If required, make output type to be Cow type instead of owned or view type, and let the caller to choose whether `.into_owned()` or `.view()`.

Q9: Okay. And just want to make sure, if it's MathMode, or we may add some flags other than math?
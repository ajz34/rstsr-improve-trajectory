# Decision points: Answers

Q1: (c).
Q2: (c).
Q2 Additional: We may need different layers for interchange. dlpack can be simple one, and the user can use that for non-heavy data transfer interfacing. User even can use dlpack as C interface to rstsr. And heavier one can be implementing rust-numpy traits.
Q3: (a). Rename as rstsr-cpu-dlpack/rstsr-cpu-python/rstsr-cpu-pyo3/etc. We have not implemented cuda backend, and hence have not decided how to implement in GPU. Note this may need trait implementation for all representation as `Vec<T>` (CPU storage), so DeviceCpuSerial/Faer/OpenBLAS/etc can all use such kind of representation.
Q4: May need rediscussion. For rstsr, I'm not going to really write a new python package that driven by rust; just make data transfer from/to rust/python be possible is the goal.
Q5: Well this is tricky. View is surely okay, but I'm still considering mutable view. Note view may have different semantics on mutability.
Q6: Well this is another note, that device_faer's comment may wrong and you should check that. The UB is for device_faer specific, that faer uses different alignment on its owned matrix, and casting it may possibly be UB. This may or may not happen for numpy buffer -> rstsr.
Q7: Though for myself I only concern f64/32/i64/32/complex. So for non-common types, follow your recommendation; I'm not of that care of at current time, but design as early as possible is good.
Q10: (a). Anyway, we have faer in crate, which I remember is 1.84. There was example on this.
Q12: As mentioned, try trait implement all that `DeviceAPI<Raw = Vec<T>>`.


Q8/Q9/Q11/Q13: Okay.

It is very possible that I'm not considering correctly. You should point out if you think wrong or inconsistent.

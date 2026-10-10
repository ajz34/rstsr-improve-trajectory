Something before answering:

Take tblis tensordot as learning reference, but forget about it.
We just implement the array API and rstsr-core, but `rstsr::tblis::tensordot` will be a standalone plugin/extension. We do not require tblis implementation to mangle here.

Also, you may need to first pull main branch and stay at #133 before doing this task.

Q1:
Well if you are sure on hybrid case that you are able to implement, you can do that. Otherwise I prefer (a).
But it is very important that for "reshape" step, if the reshape will make clone to data, then we prefer not to use matmul consequently; we use naive kernel instead (to avoid unnecessary memory allocation as first priority). I think rstsr has some utilities to check if reshape will clone or not.
Re-iterate, we do not use einsum. This requires external library.
And also, naive reshape may fail (like cases "abcd, adbe -> bcae" in row-major case; but if we naively iterate "ab" and perform "a x b" times of matmul of "cd, de -> ce", which is able to be accelarated GEMM since they are c-preferred or f-preferred (most contiguous stride is 1), then we can still use matmul to accelarate this kind of tensordot). And if two indices are to be contracted, things will be more tricky. So I actually don't think (b) is strictly less code and faster, we can take many tricks and boundary situations into account to make tensordot really faster with proper usage of GEMM.

Q3: If I'm correct, vecdot faces similar problem, and we use `(0, 1)` (a 0 vs b 1) to distinguish `[0, 1]` (a 01 vs b 01). You should double check if I'm correct.

Q5: Agreed, but `axes=None` seems not an option on array-api. For numpy the default is `axes=2` and does not accept None. You must double check this behavior.

Q6/7: Yes. Two commits, but you can first implement everything, and make two commits for separated files. You first make rstsr-core one works, no commits to rstsr until my manual check.

Agreed: Q2/4.

Evidence: For tensor view, you may noted TensorViewAPI, that introduces both `&TensorAny`/`TensorView`. I'm not sure if that is applicable.

Q1: Well you can include diff. For others like clip/log1p/dtype/linalg these are out. fft is not in scope for this project at this current time.

Q2: For this time delivery in multiple commits, and not in PR. You will be granted automatically do things, but prohibited to push to remote and automatically gh pr.

Q3: Well for unique_*4 and isin, I think it's better to have device-dependent 3-tier implementation. Composition may be somehow more costly and memory-intensive, if my comprehension is correct. I'm not sure what's nonzero.

Q3 additional: Notice that while tier-3 implementing, the low-level rstsr-native-impl functions usually come with `&mut output` and `&inputs`. Other than these parameters, the memory consumption is better not exceeding O(1), unless efficiency is a concern (example like matmul that needs to explicitly copy input when input matrix is not contiguous in the most contiguous dimension, i.e. minimal stride is not 1; but these kind of example is very exceptional, and you can also check the current status of rstsr). Not introducing more memory than inputs and output is a good practice.

Q4: 
- Background: you can achieve source code of numpy and array-api from `~/Git-Others`.
- Rename `Repeats` to `RepeatArg`, and I'm not sure if `RepeatArg` need some enum for tensor input. I suspect that with the declaration of python array api, actually `Vec<usize>` is something 1-d tensor. And also, since it involves broadcastable thing, row-major and col-major can be different for this function.
- Be clear that some functions argument axis (exactly one axis), instead of axes (none/one/multiple axis). If possible, you can create a new type `AxisIndex` along with `AxesIndex` to avoid confusion.
- Things like SearchSide, you can make proper into/tryinto impl, and use `impl SearchSide` as parameter type.
- For struct `UniqueCounts`/`UniqueInverse`/`UniqueAll`, they are something named tuples in python. You can implement `Into<(Tensor, Tensor)>` for these structs (if 2-tuple). You are right that positional `Vec` is not suited here.

Q5:
- sort/argsort: right. do not complicated things. no complex. this is intentional deviation to numpy, but should try follow array-api.
- NaN/±0.0 in sort: emmm well you are going to make compilent with array-api at first priority. If this causes very low efficiency, then write a side function to handle. If python array api and its tests does not specify this issue, you can follow your thoughts.

Q4/5 additional:
- You are suggested to create a function (just like rt::reduce), that user can feed some custom partial order function into rt::sort_custom, then custom sort can be implemented. Complex float sort can be achieved in this way (but you are not going to write a function that really do complex float sort in rstsr; just an example that can be documented in docstring and tested).
- NumPy sort function have optional parameter `kind`.
- Sort output tensor can be always row-major/column-major (depending on the default device layout). You may be noticed that many binary/ternary functions that output layout is sometimes follows input layout, but sort function does not need to follow this.
- You may allowed to edit rstsr-dtype-traits, or even add new traits and implementations, to make sure float point comparasion is correct. You can make API breaking changes when you feel proper. If complex sort/isin/searchsorted is something python array api requested, you may need new traits to support them.

Q6: (b). You see if it is better to be overloadable argument, or a must-have argument entry.

Q7: Okay.

Q8: Yes. Correctness comes first. Do this with your leasure.

Q9: Okay. Notice rstsr-core tests also needs numpy tests parity. See rstsr-agents for more details.

A note for future implementation: This is extremely huge task. Your working progress may become a main session with subagents that do real implements and checks. Things are not going to do, or not supposed to do in parallel; but main session should be aleviated with token, so to spawn subagents.

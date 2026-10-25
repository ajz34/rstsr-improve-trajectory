You are then going to plan implement and perform benchmark on elementwise operations.
Before you write the plan, you can ask me questions about the task.

This task can be a somehow large task. And note this session is not going to be detailed on technical aspects (you are going to take over technical parts), but rather on the overall design and establishing benchmark guidelines on this task, to make a plan.
Correctness in every case, and efficiency, are both important.

This task is going to achieve
- All unary, binary, ternary elementwise operations on GPU (that supported on CPU)
- Efficient implementation of contiguous, strided-contiguous, fully-strided tensors and broadcasted cases (do what CPU device can do, and better make it fast)
- Assign arbitary (something used as like reshape).
- Test correctness (layout can be strided, transposed, offsetted; binary/ternary can have different layouts even with the same shape).
- Make comprehensive efficiency benchmark on different senarios in f64 type. Especially case of 
  - `A + B.T` and `A + A.T` (A and B are both contiguous and square, but B is transposed during computation)
  - `A + B.swapaxes(0, 1)` for 3-dim tensors A and B (both c-contiguous, but B is transposed during computation)
  - `a[None, :] + b[:, None]` for 1-dim tensors a and b (outer-cross)
  - `A[:, :, :] + B[:, None, :]` broadcasted case with partial strided and `A[:, :, :] + B[:, :, None]` not strided
  - sliced tensors like `A[:, 10:1010] + B[:, 50:1050]` (which have some contiguous axes, but the tensor is not contiguous in all dimensions).

Notes
- You are going to implement type promotion, test one or two, but not necessarily efficiency benchmark them.
- You are not going to limit the code size, try better efficiency now. But note try limit code size per file, and you can create directories for some kernel collections.
- We expect at least on-par with cupy's efficiency (tolerance 1.2x). Maximum tolerance is 2x, but for such case must noted and explained in the report, show what means tried and why not worked. We are not too strict on code complexity at this time. Note cupy, pytorch and numpy code are available at ~/Git-Others as reference.
- It is also suggested to test pytorch's efficiency.
- Note you can take rstsr-improve-trajectory as scratch and report area. For benchmark, let rstsr only have code and explanation, but not compare and mention other libraries. For report, you can compare and mention other libraries.
- If some code *strongly* uses concepts of cupy or pytorch's implementation on elementwise operators, you can briefly mention in code comments.
- Though not likely, if CPU part have error, you can report it.
- If you feel CPU should have better implementation (that's possible in that some GPU algorithms can also be applied to CPU), you can also suggest it (but not required to implement it, we would not limit GPU's code complexity but CPU is not to be complicated, due to that usual rayon multi-threading is completely DRAM bound, so CPU does not expect to be much faster for more advanced algorithms).
- Note you can just focus on elementwise operations (also assign), but not other operations like reduction, matrix multiplication, set functions, etc.
- You are not going to auto git commit. Still work on current branch.

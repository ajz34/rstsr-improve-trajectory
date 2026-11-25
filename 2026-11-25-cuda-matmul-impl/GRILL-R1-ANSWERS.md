Q1: Note that we may accept, or try accept any combination by naive way. Only and try best to utilize the cuBLAS/Lt/Ex drivers as the acclarated driver.
FP8 is probably a special case, that it is not previously supported by rstsr. Introducing FP8 may be a separate decision.
On other aspects, in principle generally accept (a).

Q4: Okay but this is device-level internal, so may name as `matmul_with_mode_inner`, and see if this really need to be in `mod.rs` (which I prefer to be a pure import and utility file instead of real implementation in mod.rs). Also introduce a function like `matmul_with_mode` in tensor way, which signature is going to be similar to `matmul_with_output` but also with mode.

Q7: Well I think probably we need to add an entry to `CudaRuntimeFlags` to enable/disable the driver status. Maybe in this case using environment variable is not good idea, and better to refactor these environment variables out. You can think how to do this, or propose your suggestion. Note API shape is not fixed, so any change for better API and functionality shape is welcome.

Q8: As said on Q4, mod.rs should be lightweighted. Also, you may need to implement a naive matmul in a separate file. Also, tiled matmul is not fully tested on correctness and efficiency, and this is a separate issue but related here. Also said in Q7, API shape is not fixed.

Q10: Refer to CPU implementation, that with arbitary layout (can be strided, not c/f-prefer, even negative/zero strided), we can still do matmul. Also, support dtype promotion is better, but that's not a must in cuda device (probably first cast to output type then perform matmul is better, and seems that's how cupy/pytorch does). Please also check how tiled SIMT kernel works.

Q2/Q3/Q5/Q6/Q9: For your recommendation.
Q1: 

Firstly is that, the lapack-translation thing is sololy to lapack. If we have lapacke as feature we surely won't do that, but many BLAS vendors do not ship lapacke in its shared library. So we should assume that most users do only have lapack and no lapacke. And lapack is col-major-only (in most cases, at least its wrapper does not allow row-major). So I will choose (c).
It is possible that results (c) is different to (b). This happened once, and finally find that's lapacke's own bug. So also take this into account, though lapacke's bug is rare.

Take rstsr-blas-traits/driver_impl/lapack/eigh/syev.rs as example.
There is some code named `orderchange_out_r2c_ix2_cpu_serial`. And I think this function serves similarly to `dge_trans` in LAPACKE. We do not apply something like `dsy_trans` at current time.
But also note, this will be applied as single-threaded. I'm not sure if this should be expanded to multi-threaded, and in which policy do the number of thread controls to be injected.

Q2: Well I think firstly `_work` is not purely caller-of-lapack. It handles query and computation together. But the no `_work` version as thin wrapper, is able to separate query call and computation call. So there is the separation, but your tone is somehow different to me. You can do some kind of splitting and mirror that, just place in one file.

Q3: Ah good. You can do that.

Q4:

Well you are going to help me check what scipy and numpy do. And then re-ask this question.

Also, there is some cases that important: if input tensor is mutable (especially solve_general that is a lapack-call (different to solve_triangular that is a BLAS call)), then if input matrix is f-prefer but default-device-order is row-major, then if I choose (b) way, we must make transpose to c-prefer and increasing memory budget; but if I choose (c), we can still stay at fortran-order and no memory budget occurs. If input / user-provided-output is f-prefer, then we should maximally utilize this f-prefer. That's what I suppose (c) is better.

Q5: Ah I'm not familiar to this. If you can clearly comprehense that and make that sound, do that.

Q6: Well keep. Also, make usual lapack / openblas as first priority if any distribution-related problems occurs.

Q7: Emmm. If no testing issues, (a).

Q8: Emmm well I really noticed this point, but I just want to check that, whether NaN input check is for uninitialized memory-unsafe C (safety reason), or some arithmetic reason. We can assume safety reason is gone for rust, but if arithmetic reason then we still think nan check is sounded, but just we deviates from lapacke by intention at this time.

Q9: Emmm this is some sounding check and well, I'm not sure these techniques are too defensive. But if budget of redundant allocation is small O(1), then we agreed and accept that.

Q10: OpenBLAS only. Also note, we only have OpenBLAS in CI. Others are leave as untested (lower-priority) but usually of no problem.

Q11: Okay. But note for this case, you are not going to automatically git commit. You are suggested to split jobs into separate small parts, and each time a part is done, stop and let me review.

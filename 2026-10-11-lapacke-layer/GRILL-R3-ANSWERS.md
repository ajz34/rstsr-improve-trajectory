Q1: Well good. Just a notice that this way is implemented in many elementwise code, but not in lapacke translation code like `rstsr-blas-traits/driver_impl/lapack/eigh/syev.rs`. So I think (ii) is probably a new deviation to existing codebase.

Q2: well that driver function is at trait level, and it can't be modified (or at least I'm not going to refactor driver trait definition at this current time). However, the `dgesvd_work` is auxiliary function, and it can be placed outside trait function (you should declare it as public, but it is actually some internal private purpose function in this project).

Q3: Okay.

Q4: Well I think feature lapack and feature lapacke is clashing? Anyway, the current codebase should have already tests for drivers and real runs. If "feature lapack" passes and "feature lapacke" also passes, then it is good. Usually "feature lapacke" will be more easy to pass, since it does not involve code effort; but rare case that lapacke itself is buggy. And you can always use numpy/scipy to reproduce correct behavior.

Q5: Okay. My original SVD is not good at handling these values, and you can really help me improve. Current space setting is probably too conservative and memory costly.

Q6: Yes, (ii) is good. And this comes from two reasons: dsy/tr trans does not cost less or more memory, just adding memory footprint; and the footprint is O(N^2), less than O(N^3) of the main computation for most lapack functions.

Q7: Okay. Follow your recommendation.

Q8: Well, I'm rethinking of this. If introducing nan check, we must let user to either allow or disallow nan check (not always on); and that requires us to change trait functions.

Well this is another question: our library is not popularly used, and even if we used this library, the user is usually calling high-level functions, but not such low-level lapack drivers. So probably driver trait changes is feasible. And if this kind of trait fn restriction breaks, what would you recommend for new features or arguments to be introduced?

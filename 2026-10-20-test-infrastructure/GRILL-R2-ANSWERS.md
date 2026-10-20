Q18: I think it is better to absorb. The name iteself `test-conventions` is misleading in both its active field (rstsr-core only) and its content (now conventions are at rule level).

Q19: (a).

Q20: Well I think this only leaves as future implement. Will be in crates-bench, but may need different crate names and multiple crates to be benchmarked for different scenarios and devices.

Q21: Good.

Q22: Emmm. Add T4 layer, and many devices are only tested when time proper, for big version change or simply manually trigger (not pre-release change). T1 and T2 can be triggered by github. Part of T3 can also be triggered by github. You can also change the file names of github action scripts to `tier1-...yml`. Cargo clippy / rustfmt still as is, it is default tested.

In many cases, T2 is not triggered, so it is more like T3 not T1.

core col cpu to be T1, but we may not implement it now.
cpu-dlpack, openblas ARM to be T2.
no_std builds, rstsr-faer-py to be T3.
benchmark to be T4.
mkl/blis/aocl/kml devices to be T4 for now.

Probably cuda to be T3, or if we have new CI vendors like CircleCI, we can move it to T2.

Q23: Okay.

Q24: Okay. Note probably you need multiple commits and new branch for that. You are going to do in main branch, but not auto commit; stop for one task and let me review.

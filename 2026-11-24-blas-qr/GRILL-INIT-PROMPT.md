# Grilling init prompt — QR decomposition on the BLAS devices

Recorded 2026-11-24. Human prompt (verbatim):

> We are going to implement the QR decomposition for BLAS devices. Plan this feature.
>
> This will involve many places to implement. We do not test MKL/AOCL/... but only
> OpenBLAS with openmp, but all them should work in principle. QR is similar but
> still different to other devices, in that it have different modes like full /
> reduced decomposition. This may also introduce difficulty on signature design.
> Please refer to existing function implementation, and related skills, to make
> sure you have a full picture. Our signature will be similar to numpy/scipy.
> Note that local repos should be available at ~/Git-Others.
>
> Follow the grilling session convention described in ruaa-traj.
> Note this session disallow subagents. You need to do this task in main session only.

Session constraint: no subagents; every fact was looked up in the main session.

## State at the start of the grill

- rstsr: `c9fcc20` on `main` (v0.10.4 + stacked BLAS linalg #168 + gemv #167),
  clean tree. rstsr-ffi: `83efe50`, clean. **No rstsr file has been edited.**
- rstsr-improve-trajectory: `main`, clean before this directory.

## What the prompt already settles

1. Target: the five BLAS device crates, through the shared trees in
   `rstsr-blas-traits` / `rstsr-linalg-traits`.
2. Only OpenBLAS (OpenMP build) is exercised locally; the others must work in
   principle.
3. The signature follows NumPy/SciPy; modes (full/reduced) are expected.

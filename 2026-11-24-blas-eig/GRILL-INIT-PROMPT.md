# Grilling init prompt — eig / eigvals (complex) on the BLAS devices

Recorded 2026-11-24 (Asia/Shanghai). Human prompt (verbatim):

> We are going to implement the eig/eigvals (complex) for BLAS devices. Plan this feature.
>
> This will involve many places to implement. We do not test MKL/AOCL/... but only
> OpenBLAS with openmp, but all them should work in principle. eig/eigvals is
> different to other devices in that it returns complex matrices on eigenvector,
> and also involves probably 3 kinds (if I remember correctly, you need do fact
> check) of different generalized eigen problems, as well as the standard eigen
> problem.
> - See scipy.linalg.eig for possible argument overload options
> - See rt::linalg::eigh for similar functionality, and the driver selection (can
>   choose divide-concour/usual)
> - See faer for existing eig implementation, but just note faer may only
>   implemented very few options; we are suggested to also implement the
>   generalized eig.
> Please refer to existing function implementation, and related skills, to make
> sure you have a full picture.
> Our signature will be similar to numpy/scipy.
> Note that local repos should be available at ~/Git-Others.
>
> Follow the grilling session convention described in ruaa-traj.
> Note this session disallow subagents. You need to do this task in main session only.

Session constraint: no subagents; every fact was looked up in the main session.

## State at the start of the grill

- rstsr: `9e8dac4` on `261124/blas-qr` (= `main` `c9fcc20` + QR `fb6f921` + two
  BLAS-entry docstring fixes `8ebd56a`, `9e8dac4`), clean tree, QR PR not opened.
- rstsr-ffi: `83efe50`, clean. **No rstsr file has been edited.**
- rstsr-improve-trajectory: clean before this directory.

## What the prompt already settles

1. Target: `eig` and `eigvals` on the five BLAS device crates, through the
   shared trees in `rstsr-blas-traits` / `rstsr-linalg-traits`.
2. Only OpenBLAS (OpenMP build) is exercised locally; the others must work in
   principle.
3. The signature follows NumPy/SciPy; the generalized eigenproblem is in scope.
4. Facts the prompt asks to check: how many generalized problem kinds exist,
   SciPy's argument surface, the `eigh` driver-selection precedent, what faer
   implements.

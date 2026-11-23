# Grilling init prompt — BLAS-2 GEMV dispatch for matmul (openblas / faer)

Recorded 2026-11-18. Verbatim human prompt (typos preserved):

> I want to check on the behavior on current rayon parallel gemv efficiency on
> row vector or column vector on device OpenBLAS or device Faer.
>
> The related file is crates-device/rstsr-openblas/src/matmul.rs.
> For BLAS deivce, probably we have not dispatched to GEMV function, and we may
> need to do similar work on rstsr-blas-traits. And GEMM of OpenBLAS is not very
> efficient on some mxk kx1 matmul.
> You are going to plan and /grill-me
>
> Some notes:
> - On `match (la.ndim(), lb.ndim(), lc.ndim()) {`, only pair (1, 2, 1), (2, 1, 1),
>   (2, 2, 2). Note that (2, 2, 2) means matrix multiplication, but need to
>   additionally check if one axis is of shape 1 to reduce to either (1, 2, 1) or
>   (2, 1, 1) and apply this.
> - This may involve adding a BLAS-2 function into rstsr-blas-traits/src, which is
>   somehow new.

Follow-up instruction in session: "Also note rstsr-traj and follow that
convention to create grilling files." — hence this directory and the
`GRILL-R<N>-QUESTIONS.md` / user-owned `GRILL-R<N>-ANSWERS.md` files.

## State at the start of the grill

- rstsr: `d5c9b86` on `261023/astype`, working tree clean. **No rstsr file has
  been edited for this campaign.**
- `rstsr-improve-trajectory`: `d97449e`, clean.
- Nothing measured yet on the rstsr side; the only numbers in `GRILL-R1-QUESTIONS.md`
  are a 30-line C probe against the system OpenBLAS, explicitly not the campaign
  measurement.

## What the prompt already settles (not open for the grill)

1. The dispatch shape handled inside `matmul_row_major_blas` is the
   `(la.ndim(), lb.ndim(), lc.ndim())` match, and the campaign adds the pairs
   `(1, 2, 1)` and `(2, 1, 1)`, plus a degenerate check inside `(2, 2, 2)` that
   reduces a size-1 matrix axis to one of those two vector shapes.
2. A BLAS-2 function is to be added to `rstsr-blas-traits/src`, not invented
   elsewhere.
3. The openblas device crate is the reference implementation; the work is
   "similar work on rstsr-blas-traits" for the driver plumbing.
4. The framing is a planning/grilling session first; execution is gated by the
   answers.

## What R1 must settle (open)

Campaign phasing and the measurement surface; the BLAS-2 API shape in
`rstsr-blas-traits`; where the layout normalisation lives; which layouts take the
fast path; batched threading; propagation to the five device copies; what
happens to the dead `DeviceGEMVAPI`; verification and deliverables.

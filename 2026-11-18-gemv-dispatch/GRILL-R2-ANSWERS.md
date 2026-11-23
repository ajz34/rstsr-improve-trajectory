Q1: (a).

Q2: (a).

Q3: (b). Please re-check the details on flip_trans. It returns TensorCow (not necessary owns data), and if input c-prefer, it will automatically handle N/T cases. For C (complex conjugate), it may perform a conjugate operation and return a owned tensor wrapped by TensorCow. So at least for f32/f64, and c32/c64 if no conjugate, any input 2-dim f-prefer or c-prefer tensor should make no copy and call correct GEMV operation with proper trans code. This is also the most important and very careful code in GEMM or TRSM, so you may need to re-read those part of code, and transfer your learning on GEMV.

Q4: This is approximately correct. Just note
- For usual cases, please refer Q3's answer and carefully check how `flip_trans` works on matrices.
- `flip_trans` do not need to be operated on vectors, and as your recommendation, pass the stride size.

Q5: I think probably we already have some functions? Like `dim_chop` which is probably just for this usage.

Q6: Okay. Just note this is at matmul implementation function, not in `GEMV_` of BLAS wrapper.

Q7: (a).

Q8: Okay, but (iii) CHANGELOG.md is only updated when version updates, not now.

Q9: (a).

Q10: Okay, two patches.


Q8: Qkay.

Q9: Okay. Usual syrk (2-dim matrix) should be still (first fill upper/lower triangle, then copy to the other triangle), which is exactly the CPU BLAS implementation. But the batched syrk will just follow batched gemm now.

Q10: Okay. Anyway, try a micro-benchmark before implementation.

Q11: Well I think that if unless cublasDotEx is significantly faster than our own implementation (which is not likely), we can use the own kernel.

Q12: Okay.

Q13: Okay. Follow similar approach to what BLAS device implementation on matmul.
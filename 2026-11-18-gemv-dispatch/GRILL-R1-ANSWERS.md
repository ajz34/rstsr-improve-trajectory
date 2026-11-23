Q1: (a). The sentence of `we probably never dispatch to GEMV, and OpenBLAS gemm is not efficient on some m×k, k×1 matmul` should be updated after GEMV abstraction landed.

Q5: Okay. Note that benchmark details should not leave into rstsr itself, but only in ruaa-traj or some local place. Let rstsr only handle code work and it is published as opensource.

Q6: Also a case on 256 x 452676 and its transpose. This should substentially exceeds L3 cache.

Q9: (a). But note be very careful on how to handle dimension, tensor view/mut/own relations, to avoid explicit copy unless very necessary, let both contiguous/c-or-f-preferred/strided tensors also able to give correct results. The GEMM/TRSM should be good example, but that requires very careful design.

Q2-Q4, Q7-Q8: Follow your recommendation.

Q10: Okay. And try (b) since this is achievable in principle.

Q11: I noticed this issue. So probably (b), make gemv specific (I think you are suggesting that probably in any situation use parallel, and since it's memory-bounded, parallel/serial should theoretically give similar performance, so accept parallel).

Q12: You can show me the drift, and let me decide.

Q13: (a). This is a historical reason, may be removed in future.

Q14: Okay.


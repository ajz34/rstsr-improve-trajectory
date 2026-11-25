Q1: Oh I forgot a case that of batched GEMM. This can be used on broadcasted multi-dimension gemm. I'm not sure if this is related, or can be deferred in future that will not cause too much difficulty on API design.

Q2: Okay. And before implementation, I may wish you to check vecdot efficiency against GPU memory bandwidth (not FLOPs).

Q3-Q5:
- Note cuBLAS have similar function on cublasDotEx. This should be take into consideration.
- Also, cuBLAS have `cublas<t>gemv`, batched version, or similar functionalities.
- You need to take a comprehensive research on these functionalities.

Q6: Okay.

Q7: Yes.

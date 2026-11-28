Follow your recommendation: Q2, 3, 4, 7, 8, 9, 10, 12, 13, 14, 15

Q1: Okay. Your propose on both layer and plan re-validation is also sounding and better than my thought in my mind. Also, on one-shot layer, I think we simply only accept IxD (dynamic dimensionality), not dispatching `DA, DB, DD`, to make things simpler (and anyway, cuTENSOR I think it also only accepts IxD, using some fixed dimensionality does not give efficiency gains).
Q5: Well I think better to gate on cuda/cutensor version at runtime. We should allow stride = 0 on cutensor version gate for input (point 1 of yours, not output that point 2 of yours). Overlap check is probably too defensive and I think not overcheck that. Others follow your recommendation.
Q6: Well I think directly call cudarc is not a problem, as long as it i-s not unsafe.
For compute descriptor, I will suggest you also use cudarc, not wrap it now.
But I still have compute descriptor wrapping in mind, that follow what matmul do, we may have enum type that binds to output data type. But I'm not sure if that is really needed now. Maybe when we move on to some utility of `rt::cuda::einsum` then handle such kind of thing.
Q11: Yes, prefer (b).

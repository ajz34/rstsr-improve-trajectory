You mentioned wave 1 and wave 2. I will decline this and should let this time to consider as more as we can.

Q1: Accept copy on these situations.
Q2: Yes. And just note, we introduce `einsum_with_conj`, and let `einsum` to be usual einsum without conjugation. Since `einsum_with_outpupt` is also introduced, a combination `einsum_with_output_conj` is also introduced. I'm not sure if `einsum_from` or similar have in `rt::tblis`.
Q3: (a) or (c). You see if (c) is easy to be implemented.
Q4: I'm not sure the exact meaning. You can explain more to me.
Q5: Accepted. and also notice Q2. We will usually not use tensordot, and let it work correctly is good; einsum is more general and need to be carefully designed when grill session finished and planning.
Q6: (a).
Q7: (a).
Q8: Okay.
Q9: Approximately yes.
Q10: Well I think as long as opt-einsum-path accepts and cutensor returns correct value, we can allow them. But I'm not sure if cutensor will give correct value for these reduction/transposition/diagonal cases, so you may need fact check.
Q11: Okay.
Q12: Approximately yes.

Q1: Okay.

Q2: If recipe is possible, prefer recipe. But also rationale is needed, since there may be some cases recipe not covered. And in that case, it is better to notice the user that ADR/skill is lack of, and need update.

Q3: Yes as your recommendation. You are probably need to do something that make code modification to validate your thoughts.

Q4: Okay. This is something basic yet important. Always do matrix clone when encountered non f-prefer tensors, including c-contig/c-prefer/non-contiguous. If input is f-prefer, then it should pass by reference; but there is some cases that the lapack function will overwrite input; and for these cases, input tensor is to be cloned anyway, whatever its contiguity is. And also, LAPACKE do similar job; what's different for LAPACKE is that input tensor must be either c-prefer or f-prefer, but we may also handling tensor that is fully non-contiguous.

Q5: We truely didn't implemented such kind of test policy. You can also try to build that test policy. And also, for these tests, the test case and data should be not that large (usually no more than 20 in row/columns).

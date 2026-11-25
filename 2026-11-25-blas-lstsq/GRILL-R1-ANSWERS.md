Follow your recommendation: Q1, 2, 4, 6, 7, 8, 9

Q3: Note that existing code should be different on tensor default order. The current scheme is for row-major; but for column major, it may be a `(M, N, ...)`, b `(M, [K, ] ...)`, i.e. `...` at right side for col-major. Please make fact check on this behavior on existing BLAS solve plan.

Q5: Okay. And I'm not expert on that, I just feel program efficiency take the lead, so I also want to consult if GELSD is better for efficiency.

Q10: Please also introduce python fingerprint tests; you need also create new fingerprint value tests in python, and test rust side compared to those fingerprints. Also note that only test few cases on fingerprints (see existing fingerprint tests, should usually be only of 2-5 cases for one function), to test correctness on large 2-dim matrix.

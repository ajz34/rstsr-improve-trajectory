Additional notes:

Q1/2/4/5: Agreed.

Q3: Okay. Take integer-array first, and if boolean array too difficult, you can skip it.

Q6: We always not say that for column-major, we are the same to numpy, and even Julia (Julia does not support some kinds of broadcasting), and we are very clear that reshape is different for row/col-major. So divergence is expected, and you can add some tests to see if these divergence is reasonable, not buggy.

Q7: Yes, that should be rstsr-core, and maybe need a new file for this like `array_indexer.rs`, along with some `array_indexing.rs`.
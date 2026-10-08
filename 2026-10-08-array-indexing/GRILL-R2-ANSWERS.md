Q1: Prefer no transpose copy or view. The output tensor shape is possibly computable, and use that to construct the output layout and storage directly. However, also note the OpArrayIndexAPI should be better implemented in a way of layout-generic (not assuming row/column-contiguous).

Also note on need transpose case: we have function `into_transpose` that can convert owned tensor to a transposed layout. So transpose copy is surely not necessary, and that can still give the owned-version TensorCow.

Q2: Follow NumPy. Also note that you are not going to make 0-d integer to be `trait From`-able to `ArrayIndexer::ArrayIndex`.

Q3/4/5: Agreed.

Q6: Agreed and for numpy parity tests, you need to check existing memory or skills.

Q7: You have new branch name "261009/array-indexing", where previous code fixes merged. You can auto commit, but not push to remote or pr.

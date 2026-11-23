Q1: (a). This task is not related to faer. You can use faer for benchmark, but this is only on BLAS, and do not try to improve efficiency using means other than BLAS in normal cases.

Q2: On batched cases, please do simple iteration (in main thread) to use multi-threaded BLAS, not using `with_num_threads` to spawn multiple threads to run single-threaded BLAS in each rayon spawned thread. Do this in every case where `n_task > 1`, even larger `n_task` will also use simple iteration to use multi-threaded BLAS.

Q3: Okay.
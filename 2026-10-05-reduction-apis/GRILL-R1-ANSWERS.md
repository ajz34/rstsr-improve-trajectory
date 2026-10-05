Q1/7: Okay.

Q2/3: I'm considering that use a new function `sum_with_dtype` along with other existing functions, so that not going to make breaking changes. `sum_with_dtype` should be similar to `sum_with_args` that pass similar arguments, but with different type handling, and `sum_with_dtype` can have its own `TOut` parameter generic.

Q4/5: Well good, I didn't thought of that before. Well you can create ReduceArgs/VarArgs/NormArgs. Also, `axes` should be also a field to these structs (not spell `axis` like numpy at rust side rstsr, this is intentional).

Q5: Yes, defer SVD-dependent functions.

Q6: Whatever that follows array-api.

Q8: Well if you think 3 PRs are better, then you may need three branch names. But anyway, you should stop at the point on each task, and I will manually review and merge them.

Q9: Maybe you need a fresh new sub-agent on this task. Also, you may note in some noticeable place that code style may change during development, and this is just as reference, and future tasks should not strictly follow this style: if existing rstsr-code-style is out-of-time or not documented correctly, you should ask the user what is divergent, and ask whether an update is needed.

It may happen that I am not directly answering to your question. Re-ask if you still feel blur.
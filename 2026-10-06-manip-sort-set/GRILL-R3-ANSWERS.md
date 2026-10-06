Q1: Okay. Note that though device implementations are layout-generic, the layout of output tensor is of disposal by high-tier. So for row-major default device, output tensor is C-contiguous preferred; for col-major default device, output tensor is F-contiguous preferred.

Q2: Okay.

Q2/3: For complex `unique_*`, perhaps I need additional discussion with you.
I thought that `unique_*` is possible to relax the trait bound to something PartialEq; but given additional memory, and proper types (like integer and floats), a set of unique values can be sorted, hence finding the unique values can be much more faster.
So, I think here is some way. In real implementation, you can use naive PartialEq for common types and complex; but for integer and floats and bools that can be compared, you can dispatch to a more efficient implementation. Anyway, we persue correctness first, and efficiency is not the most of concern. So if in doubt, or efficient implementation can be too complicated, you can just use naive PartialEq for all types.

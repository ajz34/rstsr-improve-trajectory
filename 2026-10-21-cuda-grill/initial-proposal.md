# Initial Proposal for RSTSR CUDA Backend

This is a proposal, and maybe then implementation decisions and plans, for RSTSR CUDA backend.

This is probably a large and multi-step project. I may even not prepared to implement it. So there are some tasks that we need to do. You are suggested to summarize comprehension documents in 2026-10-21-cuda-grill for me or you in future for better understanding.

- You are going to understand current RSTSR codebase, understand its architecture and understand how a new backend can be implemented.

- I'm going to enhance some shared understanding with you, that what RSTSR, in its framework, may lack of for implementing a new backend.

  Note RSTSR is designed to be very flexible and extensible framework, only restricting serval operations by traits and then use these traits for general computation.
So if anything device-specific, we can encode them in device field. This is runtime changeable, and `change_device` can do this without changing tensor's data and its bounded lifetime. However, it is still possible that our framework is making too restrictive assumptions, and you are going to point out them if any.

  Also, our framework is eager evaluation. I know that some GPU streaming techniques can be lazy, and you are also going to check (without code compile) if such techniques are applicable here.

- My initial idea is that

  - We may wish to write a cupy-like backend library, but in rust. We may further extend available devices, but cuda is a good starting point.

  - You are suggested to read the cupy codebase, and summarize the architecture difference to numpy, which may be helpful for RSTSR.

  - For most operators, we implement traits, with type different (not something like CPU that can use `num::Float` kind of trait bound, for GPU it is difficult; we may need one, but implementation is not trait-based but dtype dispatched). Prefer nvrtc way instead of cubecl way, which should be more flexible and probably more performant. But I'm not sure, and I'm not very expert, so you can point out errors in my understanding.

  - For specific operators, and utilities, follow similar way of cupy or pytorch, we have `rt::cuda::` namespace, and implement them in device-specific way.

- My initial project that may transfer to rust, refer to `/home/a/rest_pack/rest/src/ri_pt2/py/dfmp2_addons.py`. I hope this file's usage of pytorch (no autograd) can be implemented in RSTSR CUDA backend in future. See if there is some gaps that RSTSR framework will be difficult. Note that we also need something that dynamically (not like cupy use environment variable) to achieve TF32 matmul. We are currently not supporting, or not strongly supporting, FP4 or INT4 (if not naively supported). FP16/BF16 is suggested.

- For available resources:

  - cupy, pytorch, candle (rust) source code at ~/Git-Others. Note pytorch is shallow clone and you are assumed not to clone more; but if you requires, you can ask me.

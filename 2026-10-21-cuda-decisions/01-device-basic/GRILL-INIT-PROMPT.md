Lets prepare plan on implementing device and data storage schemes on rstsr-cuda.

Related preparation work at rstsr-improve-trajectory/2026-10-21-cuda-grill (which is not truly grill, but useful resources).
Your record should at rstsr-improve-trajectory/2026-10-21-cuda-decisions/01-device-basic.

You are going to challenge me. Just note not challenge everything that already proposed at 2026-10-21-cuda-grill: we do this step by step, and something may changed and reverted during implementation.

If you need spawn subagents before grill questions for background knowledge, you should proposing questions after all subagents converges.

------

We are going to initialize rstsr-cuda, first on basic definitions: device and data representation.

Some notes, that I thought of and you are also going to challenge me in more detail and propose your thoughts, in supplement of your findings:
- Both device and raw data (`DeviceAPI::Raw`) can contain data. Raw data may or may not be only CudaSlice or some handle, and device object can also contain flags. How to separate them is also a question.
- Some flags may be None and have default values; are these necessarily to be some global state with device mutable?
- Where may we need to put streams or other related objects?
- For usage of python's `with` statement on streams, what is your preferred rust design?

Q14: Well I think probably we do not need creation traits to be defined? Especially arange and linspace. DeviceCudaPinned is just going to be storage of pinned memory, additionally transform it to tensor view/mut; other usage is not required, unless necessary.

Q15: Okay, try that.

Q16, Q18, Q19: Okay.

Q17: Emmm, well, if you can find useful test cases, may try allow all to be implemented. Also `DeviceCuda::new_stream_with_ordinal(ordinal)` and `DeviceCuda::new_stream(&self)` probably better to my feel.

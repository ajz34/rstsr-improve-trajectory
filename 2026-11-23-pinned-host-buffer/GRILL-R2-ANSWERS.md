Okay. This was my under consideration.
May name to `DeviceCudaPinned` (also Q3), which may better on future distinguishing to different gpu backends.

Continue grill R1 questions. Following answers to R1.

Q3: May better name `DeviceCudaPinned`. So first it's pinned memory instead of usual memory. Second this is cuda's handling.
And you may also check if PinnedBuffer have it's origin in cudarc.

Q4-Q6: Okay.

Q7: Sorry that maybe `device: &B` is better, not `device: B`, following many RSTSR functions that pass device as reference.
And, just for confirm, you mean "same contract as `asarray`", so should probably be ManuallyDrop instead of usual reference of DataRef/DataMut.

Q8-Q10: Okay.

Q11: Sorry though probably I will accept your's, but that I also may need your explanation on pool and allocator. I feel I need to recap the related concepts.

Q12: Okay.

Q13: Yes, but just a more question on your tone: if we can make non-minimal edits, what you will do?
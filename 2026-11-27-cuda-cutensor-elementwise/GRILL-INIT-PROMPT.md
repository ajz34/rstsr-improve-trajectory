# GRILL-INIT-PROMPT.md

Prompt that opened this experiment (2026-11-27), verbatim:

> You are going to see if cuTENSOR can perform better efficiency on elementwise unary/binary/ternary
> evaluation than current hand-made kernel.
> Also, I want to know that if cuTENSOR must pass input/output types to be the same dtype (does
> cuTENSOR handles some subset of dtype promotion, or never handle any dtype promotion).
> Note that not only test contiguous tensors, but also transposed, sub-window viewed, sliced
> situations.
> The test are going to be in rstsr-improve-trajectory, not in rstsr itself.
> You can discuss with me, since the instruction is not clear enough, and I'm also not fully
> familiar to cuTENSOR.

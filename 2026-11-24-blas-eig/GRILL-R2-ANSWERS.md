Q17:
- Well I still think it should be called `EigSide`, not `EigVectors`, which I comprehense as input not output.
- And also, probably we need to think of how to properly cast `&str` and `FlagSide` to `EigSide`, and introduce them in the overloading table.
- I may think probably we simply return EigBothResult, not EigResult, in BLAS devices. This is a lack-of-consideration on previous faer implementation of eig function. And we may rename `EigResult` to `<W, VL, VR>`, and let `EigRightResult<W, VR>` to be the previous one-sided form, or even discard `EigRightResult<W, VR>`. I mean, you can make API breaking change on faer, and make the EigResult to uniformally describe three possible outputs.

Q20: As Q17 discussion, faer can make refactor on output type. We do not introduce other changes on faer implementation.

Q23: Given previous discussion, you just evaluate this. You can close if you are confident.

Q18/19/21/22: Follow your recommendation.

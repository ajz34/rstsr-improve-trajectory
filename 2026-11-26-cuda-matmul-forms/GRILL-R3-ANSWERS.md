Q14: Okay. Drop.

Q15: Okay. But, well, I am also curious that if inner dot can be a case of gemm that in the chain of gemm driver? Inner dot can be a special case of gemm. CPU not using this way is because OpenBALS may have large overhead, and naive rayon implementation is sufficiently fast. But for cuda this is not that important.

Q16: Well I think you can just gate the F32 case on `max(k, n) >= 1024`; for f64 and complex just simply use syrk (since I think f64's 256 is 0.97, not that slow but just not fast enough).

Q17: Okay.

Q18: Well currently leave the other reduction operations as is. Innerdot only implemented in matmul, and can treat differently.

# Grill round 4: dot as GEMM, and the f32 syrk gate

Numbers: `B0-REPORT.md` §1b and §3. Answer on screen or in `GRILL-R4-ANSWERS.md`. "Yes" accepts
the recommendation.

## Round 3 answers (user, on screen; transcribed by the agent)

- **Q14** (drop gemv): okay, drop.
- **Q15** (two-stage `vecdot`): okay, "but I am curious whether the inner dot can be a case of
  gemm in the chain of gemm drivers ... CPU does not do this because OpenBLAS has a large
  overhead ... for CUDA this is not that important". Measured: `B0-REPORT.md` §1b.
- **Q16**: "gate the f32 case on `max(k, n) >= 1024`; for f64 and complex simply use syrk (f64's
  256 is 0.97, not that slow but just not fast enough)".
- **Q17** (batches): okay.
- **Q18**: "leave the other reduction operations as is; inner dot is implemented in matmul only
  and can be treated differently".

---

❓ **Q19 - Inner dot as a GEMM on the chain**: measured, the chain on `1 x k @ k x 1` equals the
two-stage `vecdot` within 5 % and reaches 92–97 % of the ceiling. So the rank-1 case of
`resolve_geom` is lifted to that GEMM (a size-1 axis inserted on each operand and on the rank-0
output) and nothing else changes: no `Dot` form, no scratch, no epilogue kernel. `alpha`, `beta`,
the driver flags (the dot follows them like any GEMM, which replaces D4) and mixed dtypes come
from the chain. One limit: with the naive driver forced, the dot is one thread, like every
`1 x 1` product there.

➡️ Yes: the GEMM route. It is less code than Q15 and equally fast.

---

❓ **Q20 - The f32 gate `max(k, n) >= 1024` loses on five measured cells**: with that gate, syrk
is taken where the probe measured a loss: `n = 256, k = 2048` (0.53), `n = 512, k = 4096`
(0.58), `n = 512, k = 512` (0.69), `n = 256, k = 256` (0.70), `n = 1024, k = 128` (0.61). The
losses follow `n`, not `max(k, n)`: the output is only `n^2` elements, and below `n = 2048` the
`f32` syrk kernels do not fill the GPU. Gate on **`n >= 2048` for `f32`** instead: every
measured `f32` cell with `n >= 2048` wins (1.21–2.96) and every cell with `n < 2048` would lose.

➡️ Yes, `f32`: `n >= 2048`.

---

❓ **Q21 - `f64` and complex always take syrk**: agreed, with one cell on record: `f64`,
`n = 256, k = 32` measures 0.45 (syrk 33 us against the chain's best 16 us; absolute loss 19 us).
Every other `f64` cell is 0.97–2.02 and complex is 1.76–2.82.

➡️ Yes, always (the 19 us cell is accepted).

---

❓ **Q22 - Batches with the GEMM route**: B1 is now small: lift the rank-1 case in `geom.rs`,
  update its host test and the `inner_dot` error test, add dot rows to the tests. Then B2 syrk
  (condition as the CPU devices, gates of Q20/Q21, one `?syrk_v2` call plus the mirror kernel),
  then B3 matrix, bench, docs. The rank-0 reductions outside matmul stay as they are (Q18) and
  are noted in the README only.

➡️ Yes.

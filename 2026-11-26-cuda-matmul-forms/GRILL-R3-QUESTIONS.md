# Grill round 3: after the probe

Numbers: `B0-REPORT.md`. Answer on screen or in `GRILL-R3-ANSWERS.md`. "Yes" accepts the
recommendation.

## Round 2 answers (user, on screen; transcribed by the agent)

- **Q8** (batched gemv in scope): okay. **Q9** (defer batched syrk and gapped batches): okay;
  "usual syrk (2-dim matrix) should still be: first fill one triangle, then copy to the other,
  exactly the CPU BLAS implementation; batched syrk follows batched gemm now."
- **Q10** (bandwidth bar): okay; "try a micro-benchmark before implementation".
- **Q11**: own kernel unless `cublasDotEx` is significantly faster (not expected).
- **Q12** (gemv types): okay. **Q13** (syrk scope): okay; "follow the BLAS device implementation
  on matmul".

*(Correction recorded after the round: `cublasSetPointerMode_v2` is bound after all, so B0 timed
the cuBLAS dot in device mode as well.)*

---

❓ **Q14 - gemv is dropped**: on all 22 large cells the chain equals raw `?gemv` within 3 %, and
batched is identical; the only gap is one launch-bound `f32` cell of 1.9 us. D9's 10 % bar is not
met, so no gemv routine is written. Batch B2 is removed; the CPU precedent's win came from a gemm
that did *not* dispatch to gemv kernels, which cuBLAS's does.

➡️ Yes, drop it and record the numbers.

---

❓ **Q15 - Inner dot as a two-stage `vecdot`**: today's kernel runs the whole dot on one block
(7 % of the bandwidth ceiling). Splitting `k` into `B = clamp(k / 4096 rounded up to a power of
two, 256, 4096)` partials with the existing kernel, then reducing the partials, reaches
101–110 % of the ceiling for `k >= 2^22` and equals the cuBLAS dot. No new kernel: the `Dot`
form calls `classify_reduce` + `vecdot_launch` twice (a small device scratch of `B` elements),
then the one-element `alpha`/`beta` epilogue.

➡️ Yes.

---

❓ **Q16 - syrk size rule by output type**: syrk is taken only when it wins in the probe:
`f64` output from `n >= 512`, `f32` output from `n >= 2048`, complex outputs at any `n` (1.8–2.8x
from `n = 256`). Below the threshold the gemm chain runs. `c64` takes the `f64` threshold
 untimed. The thresholds are constants in the classic driver, documented next to the routine.

➡️ Yes.

---

❓ **Q17 - Batches now**: B1 inner dot (Q15), B2 syrk (Q16, CPU rule for the condition, one
`syrk` call plus a mirror kernel), B3 matrix, bench, docs. Each is one rstsr commit on your go,
in that order. The first go starts B1.

➡️ Yes.

---

❓ **Q18 - Follow-up handed to you, not in this plan**: every CUDA reduction with a rank-0
output (whole-tensor `sum`, norms, today's dot) runs on a single block, for the same reason.
Fixing it belongs to the reduce kernel (a two-stage plan for rank-0 outputs), not to matmul; it
is recorded in the README under "What is left", with the dot's numbers as the evidence.

➡️ Yes, record it as a follow-up.

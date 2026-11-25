# Grill round 2: CUDA matmul special forms

Survey: `RESEARCH-cublas.md` (it corrects the binding table of `GRILL-INIT-PROMPT.md`).
Answer on screen or in `GRILL-R2-ANSWERS.md`. "Yes" accepts the recommendation.

## Round 1 answers (user, on screen; transcribed by the agent)

- **Q1** (forms, no new drivers): not objected to; added a case: *batched GEMM*, "can be used on
  broadcasted multi-dimension gemm", unsure whether related or deferrable "without too much
  difficulty on API design".
- **Q2** (inner dot on `vecdot`): okay; "before implementation, I may wish you to check vecdot
  efficiency against GPU memory bandwidth (not FLOPs)".
- **Q3–Q5**: noted that cuBLAS has `cublasDotEx`, batched `cublas<t>gemv` "or similar
  functionalities"; asked for a comprehensive research of these first.
- **Q6** (measure first, 10 % bar): okay. **Q7** (bookkeeping): yes.

## What the survey changed

1. `cublas{S,D,C,Z}gemvStridedBatched` is bound (CUDA 11.6+): batched gemv needs only the batch
   steps the geometry already has.
2. No batched syrk exists in cuBLAS 13.2; no half syrk; no Lt gemv/syrk/dot.
3. `cublasDotEx` is bound. *(Corrected after the round: `cublasSetPointerMode_v2` is bound too,
   CUDA 13 name; in device mode the dot is asynchronous. The question is settled by B0's numbers,
   see D3.)* In host mode the dot returns through a host pointer, i.e. it blocks.
4. Half gemv (`cublasHSHgemv...`, `cublasTSSgemv...`) exists but is not bound.
5. Non-affine batches of gemm (and gemv) have a cuBLAS route: `cublasGemmBatchedEx` /
   `cublasGemmGroupedBatchedEx` / `cublas{t}gemvBatched`, all with device arrays of pointers.

---

❓ **Q8 - Batched gemv in scope**: the gemv form takes affine batches through
   `cublas{S,D,C,Z}gemvStridedBatched` (a broadcast operand is a step of 0), the same batch rule
   the gemm drivers use. Unbatched and batched gemv come together, since the addressing is shared.

➡️ Yes.

---

❓ **Q9 - Batched syrk and non-affine batches deferred**: `a[p, m, k] @ a[p, m, k].t()` stays on the
   gemm chain (cuBLAS has no batched syrk; a loop of `p` calls is not worth a probe). Gapped
   multi-axis batches of gemm or gemv stay on the tiled/naive fallback for now: the pointer-array
   entries are a separate item inside `prepare`, with no interface impact, so deferring costs
   nothing later. Both recorded as follow-ups in the plan.

➡️ Yes, defer both.

---

❓ **Q10 - Inner dot bar is bandwidth**: B0 measures `vecdot` on `k` from 2^20 to 2^26 (`f32`,
   `f64`), the device-to-device copy on the same bytes as the achievable ceiling, and
   `cublasDotEx` as the library anchor. Acceptance: `vecdot` reaches at least 75 % of the copy
   ceiling for `k >= 2^22`; below that the reduce plan is tuned first (or the dot gets its own
   kernel, a round-3 question with the numbers). Small `k` is launch-bound and not gated.

➡️ Yes.

---

❓ **Q11 - cuBLAS dots are anchors only**: `cublasDotEx`/`?dot_v2` are measured in B0 but not
   implemented: blocking result, no bound pointer-mode setter. The dot stays on the own kernel
   whatever the numbers, unless Q10 fails in a way the kernel cannot fix.

➡️ Yes.

---

❓ **Q12 - gemv types**: typed `f32`, `f64`, `c32`, `c64`, unbatched and affine-batched. Half gemv
   is out (entries exist, not bound; "no new FFI" stands). Rest of Q4 unchanged: `m == 1` or
   `n == 1`, `k > 1`, free vector strides, matrix needs one unit-stride axis, fall to the gemm
   chain otherwise.

➡️ Yes.

---

❓ **Q13 - syrk scope stands**: Q5 as asked (unbatched `a @ a.t()`, fresh output, one `?syrk` plus
   a mirror kernel; everything else on the gemm chain). `?syrkx` and `?herk` have no caller here.

➡️ Yes.

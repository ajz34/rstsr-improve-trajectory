# Grill round 1: CUDA matmul special forms

Facts: `GRILL-INIT-PROMPT.md`. Draft plan: `PLAN.md`. Answer on screen or in
`GRILL-R1-ANSWERS.md`. "Yes" accepts the recommendation.

---

❓ **Q1 - Forms before the chain, no new drivers**: classify each problem as `Dot`, `Gemv`, `Syrk`
or `Gemm` after the geometry is resolved; gemv and syrk are routines of the **classic cuBLAS
driver**, the gemm chain is the fallback when a form routine declines. `GemmDriver`,
`MatMulDrivers` and the tensor-level entries stay as they are; `trace_matmul` names the routine.
The alternative is new driver variants with flag bits (`Gemv`, `Syrk`), which is a public, though
non-breaking, change and lets a user enable gemv without gemm, which cuBLAS gives no reason to
want.

➡️ Yes: forms, no interface change. Disabling the classic driver is the switch that turns the
forms off.

---

❓ **Q2 - Inner dot on the existing `vecdot` kernel**: `[k] @ [k]` goes to `classify_reduce` +
`vecdot_launch` (multi-block, strided, every CUDA dtype) with a one-element epilogue for `alpha`
and `beta * c`. Not the naive GEMM kernel (its `1 x 1 x k` tile is one thread, so it is serial in
`k`), not `cublas?dot_v2` (the result comes back through a host pointer, which blocks the caller;
device pointer mode is handle-global state; complex is not bound).

➡️ Yes.

---

❓ **Q3 - The dot ignores `MatMulDrivers`**: the dot is a reduction, not a GEMM driver, so it runs
on `vecdot` whatever the enabled set (even `MatMulDrivers::none()`); the trace says `vecdot`. The
alternative is to require the `NAIVE` bit, which would make the flags mean two different things.

➡️ Yes.

---

❓ **Q4 - gemv scope**: unbatched only (`n_batch == 1`), `m == 1` or `n == 1`, `k > 1` (`k == 1`
is an outer product and stays gemm, as on the CPU devices); typed `f32`/`f64`/`c32`/`c64`; the
vectors keep any stride through `incx`/`incy` (no copy, unlike the CPU rule); the matrix needs one
unit-stride axis. Batched, half and inexpressible layouts fall to the gemm chain.

➡️ Yes.

---

❓ **Q5 - syrk scope**: only `a @ a.t()` detected by buffer identity and transposed strides; only
with a fresh output (`beta == 0`); one `?syrk` call plus a mirror kernel for the other triangle.
`beta != 0` (mirroring is wrong for a non-symmetric old `c`), batched, `a @ a.h()` and half types
fall to the gemm chain. Alternative: two syrk calls, which needs no mirror kernel but does the
same flops as gemm and so gains nothing.

➡️ Yes.

---

❓ **Q6 - Measure first, 10 % bar**: B0 is a probe in this directory (raw `?gemv`/`?syrk` anchors
against the current chain, correctness gate before timing) and runs before any driver code. A form
is implemented only if it is at least 10 % faster than the chain's choice on the cells where it
applies and not slower elsewhere; the inner dot is implemented whatever the numbers (it is a
functionality gap). If gemv wins only on some shapes, the rule (or dropping it) is a round-2
question with the numbers on the table.

➡️ Yes.

---

❓ **Q7 - Bookkeeping**: this directory for the probe, the numbers and the patches; the rstsr work
stays on branch `261021/cuda` (unpushed, carries the matmul commits) as separate commits, one per
batch.

➡️ Yes.

---
name: cuda-matmul-order-output-before-launch
 description: rstsr-cuda matmul drivers call order_output before the launch, so in-place matmul across two streams reads back stale data; elementwise/adv-index kernels call it after the launch (correct)
metadata:
  type: project
---

Found in the 2026-11-26 review (`2026-11-26-cuda-matmul-review/REVIEW.md`, C2), reproduced (probe T5):

- `kernel::order_output` records its event **at call time** (`kernel.rs` `order_output_raw`). The matmul
  drivers call it before anything is enqueued: `cublas::prepare` (shared by Lt, Ex, classic), `syrk.rs` (syrk
  and the mirror kernel), `tiled.rs`, `naive.rs`. The allocation stream never waits for the product.
- Symptom: output allocated on the default stream and re-tagged onto a second device value
  (`into_device(&d2)`), `rt::matmul_from(&mut c, &a, &b, 1.0, 1.0)` in a loop, `to_cpu_vec` after each call:
  30 of 30 iterations stale, all 2 359 296 elements exactly one update behind (chain, cuBLASLt, tiled).
  One stream: 0 of 10.
- The correct pattern exists in the crate: `kernels/elementwise.rs` and `kernels/adv_index.rs` call
  `order_output` after `launch()`. Matmul is the exception (classic already did it at `c75aecd`).
- With `beta != 0` the product also reads `c` unordered (`prepare` has no `order_input(c)`; only a pending
  allocation is covered by `read_ptr_on`).
- Fix: move the `order_output` calls after the enqueue and order `c` as input when it is read; regression
  test = probe T5. Fresh-output entries are safe (output allocated on the executing stream).
- **FIXED 2026-11-26**, same changeset as the broadcast fix: all five matmul drivers (prepare
  for Lt/Ex/classic, syrk typed, tiled, naive) now `order_input(c)` before the enqueue (it is a
  write target and, with `beta != 0`, a reader) and `order_output(c)` **after** the enqueue, on
  `Verdict::Ran` only (a refusal enqueues nothing). The syrk mirror needs no own markers: it
  runs on the same stream right after the syrk, and the caller orders `c` around both. Regression:
  hw `test_matmul_modes.rs::inplace_cross_stream_is_ordered` (exact accumulation, re-tagged
  output, n = 1024); probe T5 0 of 30 stale on chain/cuBLASLt/tiled.

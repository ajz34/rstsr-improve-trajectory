---
name: cpu-reference-default-order-must-match-device
 description: In rstsr-cuda hardware tests the CPU reference device must take the same default order as the CUDA device, or under ColMajor the two "same" tensors are laid out differently and views stop meaning the same thing
metadata:
  type: project
---

Found while fixing T2 of the 2026-11-26 matmul review (test-only, `syrk_views` rewrite):

- `DeviceCpuSerial::default()` and `DeviceCuda::new(0)` each carry a **default order** that
  decides how `asarray((v, &dev)).into_shape([...])` lays the flat data out. A test that builds
  its CPU reference and its device operand under different default orders gets two logically
  different tensors under `ColMajor` — views of them are not the same submatrix, and comparisons
  fail with plausible wrong values (rel err ~1.0e0).
- The crate's convention (`matrix`, `dot_matrix`): inside the `for order in [RowMajor, ColMajor]`
  loop, set `dcpu.set_default_order(order)` **and** `d.set_default_order(order)` before any
  `asarray`, so both sides interpret identically. Transferring with `to_device` also preserves
  logical layout.
- Related trap: `b.t().t()` is simplified back to the original layout, so a test that means to
  exercise `OP_T` must present a genuinely transposed layout (store `[n, k]`, use one `.t()`),
  not a double transpose.
- Another trap: a macro/vec of test cases that mixes owned tensors and `.t()` views will not
  type-check (DataOwned vs DataRef) — the borrowed-view case needs its own test function (the
  layout-matrix row stayed an owned "transposed copy" look-alike, named honestly, and the form's
  view coverage lives in `syrk_views`).

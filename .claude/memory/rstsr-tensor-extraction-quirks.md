---
name: rstsr-tensor-extraction-quirks
description: rstsr API quirks at commit 386948be - to_vec is 1-D only, asarray with shape returns IxD, a.t() output shape, a+b.t() broadcast rule
metadata:
  type: reference
---

rstsr-core at commit 386948be, tensor data-extraction and shape quirks hit when
writing benches/correctness code:

- `tensor.to_vec()` **only supports 1-D tensors** (panics `InvalidLayout:
  to_vec currently only support 1-D tensor` for 0-D/2-D). Use
  `.reshape(-1).to_vec()` for n-D owned/contiguous tensors; 0-D results
  (e.g. 1-D `rt::vecdot`) also need `.reshape(-1)`.
- `rt::asarray((vec, [m, n], &device))` returns a **`Tensor<_, _, IxD>`**
  (Vec-dim), NOT Ix2. Annotating `Tensor<f64, _, Ix2>` fails to unify; leave
  dims inferred.
- `rt::zeros(([m, n], &dev))` needs `Tensor<f64, _>`-style annotation (T must
  be pinned; B inferred, D defaults IxD).
- `a.t()` is a view with shape **[n, m]**; its `to_contig(RowMajor)` copy
  flattens as `out[j*m + i] == a[i][j]`. Non-square `a + a.t()` is a
  broadcast ERROR (shapes [m,n] vs [n,m] don't broadcast) - benches must pair
  an [m,n] with a transposed [n,m].
- `rt::zeros` on a fresh [m,n] shape is **calloc-lazy** (~2 µs for 32 MiB, no
  page touching) - it does NOT benchmark fill; use `rt::full` to measure the
  fill_promote path.
- `Vec::zeros`/creation zero-fill never touches pages; first-touch page-fault
  cost appears in the op that writes (add/full), not in the allocation call.

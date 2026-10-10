---
name: rstsr-linalg-batching-r1
description: "faer linalg batched (n-dim) entries — the array-API R1 lever: device-order batch walk, in-place solve writes into b, the square-check regression, conformance 1296/53/33, and the wheel-glob build gotcha"
metadata:
  type: project
---

2026-10-10, branch `261010/linalg-nbatch`, PR RESTGroup/rstsr#136. Generalizes
the faer linalg column to `(..., M, N)` / `(..., M)` / `(..., M, K)` stacks —
the "dominant lever" for the array-API `linalg` extension (stacked inputs are
the contract). 11 single-operand entries + 2 solves.

- **Device-order rule (critical):** row-major → matrix axes **last**, batch
  leading; column-major → matrix axes **first**, batch trailing. The batch walk
  peels the **leading** axis under `RowMajor` and the **trailing** axis under
  `ColMajor`. **Do NOT** use `reverse_axes` (the old slogdet trick) — it
  transposes the matrix and is only safe for transpose-invariants like `det`.
- **Shared helpers** (`rstsr-linalg-traits::linalg_util`, now `pub`):
  `map_batch_matrices` / `map_batch_square_matrices` (adds the square check),
  `map_batch_solve_into_output` / `map_batch_solve_inplace` (broadcast-aware),
  `solve_plan`, `split_shape`.
- **In-place solve** (owned / `&mut` `b`) writes into `b`'s own buffer (no
  copy; the output buffer *is* `b`); the allocating path copies the broadcast
  `b` into the output **once**, then runs the same in-place walk. Column-major
  **stacked** solve errors (the broadcast would need left-alignment); 2-D
  col-major works.
- **`clone_to_mut` is a side effect, not a caller duty.** `faer_impl_solve_*_f`
  ends with `Ok(b.clone_to_mut())`; the write-back (`arr_view.assign(&arr_owned)`)
  happens *inside* the callee, so a caller that drops the returned
  `TensorMutable` loses nothing. A review flagged the nd solve kernels as
  "dropping the write-back" — **false positive**, verified with a step-sliced
  view whose slice is `ToBeCloned` (both `f_prefer`/`c_prefer` false).
- **Review-caught regression:** the n-dim retrofit routed every square entry
  through `batch_and_matrix_shape` (checks only `ndim >= 2`), so a non-square
  `det`/`cholesky`/`inv`/`eigh`/`eigvalsh`/`slogdet` **panicked** inside faer
  (`determinant.rs:8 Assertion failed: mat.nrows() == mat.ncols()`) instead of
  returning `Err(InvalidLayout)`; for `ndim >= 3` it was a brand-new abort path.
  Fixed via `map_batch_square_matrices`; regression-tested.
- **Conformance** `1294 / 55 / 33` → **`1296 / 53 / 33`** (+9 flips:
  `cholesky, det, eigh, eigvalsh, inv, svdvals` + the 3 `slogdet` nodes). `pinv`
  and `svd` stay red on faer SVD `NoConvergence` (a 2-D robustness limit, not
  batching); `solve` stays red on G-009 mixed dtype.
- **Wheel-build gotcha:** `$HOME/.cache/rstsr-wheels` accumulated two
  `rstsr_faer_py-*.whl` (a stale `manylinux_2_34` + the new `manylinux_2_39`);
  `pip install "$WH"/rstsr_faer_py-*.whl` matched **both** → pip
  `ResolutionImpossible` and a **silent stale-wheel suite run** that reports the
  old numbers. Always install an explicit wheel path.

See `2026-10-10-arrayapi-linalg/CHECKLIST.md` (R1) and
[[arrayapi-convergence-harness]]. Related: [[rstsr-faer-py-wrapper-only]],
[[rstsr-tensor-extraction-quirks]].

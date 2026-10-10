---
name: rstsr-ext-linalg-r2
description: "ext_linalg mixed-dtype products (array-API R2): rt::ext_matmul/vecdot/tensordot, route-B in-kernel promote_pair, cpu_serial+faer only, the required shim conj fix, conformance 1300/49/33"
metadata:
  type: project
---

2026-10-10, branch `261010/ext-matmul` (off post-`#135` main `2f5cf1d`), PR
RESTGroup/rstsr#137 (CI 13/13). Closes the array-API **R2** G-009 promotion gap
for the whole product family — mixed-dtype `matmul` / `vecdot` / `tensordot`.

- **Shape**: an `ext_linalg/` directory per layer, one file per op (matching the
  earlier `ext_matmul` layout): device traits `DeviceExtMatMulAPI` /
  `DeviceExtVecdotAPI` / `DeviceExtTensordotAPI` in `operators/ext_linalg.rs`;
  tensor tier `tensor/ext_linalg/{matmul,vecdot,tensordot}.rs`; kernels in
  `rstsr-native-impl/src/{cpu_serial,cpu_rayon}/ext_linalg/`; device impls in
  `device_cpu_serial/ext_linalg/` and `device_faer/ext_linalg/`.
- **Route B (chosen over A)**: promotion is fused into the kernel inner loop
  (`TA: DTypePromoteAPI<TB, Res = TC>` + `promote_pair`), not done by promoting
  into intermediate buffers (route A, rejected). No temporaries; the cost is a
  thin per-dtype copy of the kernel. `ext_tensordot` reuses the view-only GEMM
  fast path by calling `ext_matmul_uninit`.
- **`ext_vecdot` conjugates** the first operand in the *promoted* type
  (`TC: ExtNum::ext_conj`, identity for reals) — the array-API contract.
- **Scope: `DeviceCpuSerial` + `DeviceFaer` only.** Two traps: (1) the BLAS
  crates' test binaries symlink `rstsr-core/tests/core_func` with
  `DeviceType = DeviceBLAS`, so an ext test there breaks all five BLAS crates —
  coverage lives in device unit tests; (2) `DeviceFaer` **is**
  `DeviceRayonAutoImpl`, so a shared `feature_rayon/auto_impl/ext_linalg.rs`
  would collide with the explicit faer impl — that is why the trait is not
  universal. Consequence: the BLAS devices expose the same-dtype entries but
  **not** the `ext_` forms (a known asymmetry, unresolved).
- **Shim (`rstsr-faer-py`)**: `linalg_matmul` / `linalg_vecdot` /
  `linalg_tensordot` route both same- and mixed-dtype pairs through the `ext_`
  entries via `dispatch_bin_promote_arith!` (extended with a trailing
  `$($extra:expr),*` so vecdot/tensordot can pass `axis`/`axes`). The binary
  dispatch was **missing the `i8 × i16` arm** (99/100); fixed — but it moved 0
  conformance nodes.
- **`conj` on integers is required, not cosmetic**: the suite's vecdot reference
  computes `xp.conj(int)`, so the shim's `conj` now returns a `deep_copy` for
  bool/int instead of promoting to float (the array-API rule). Without it the
  vecdot nodes died in the reference, not in vecdot.
- **Conformance** `1294 / 55 / 33` → `1296 / 53 / 33` (matmul) →
  **`1300 / 49 / 33`** (vecdot + tensordot), 0 regressions. Flipped:
  `test_matmul`, `test_linalg_matmul`, `test_vecdot`, `test_linalg_vecdot`,
  `test_tensordot`, `test_linalg_tensordot`. This branch carries **no batching**
  (that is R1, [[rstsr-linalg-batching-r1]]) — not yet combined.
- **Row/Col fact (empirical)**: matmul's result *layout* follows the operand
  layouts, not the device order — a 2-D call gives `Cc` in both orders for
  row-major operands. The real order difference is the **axis-role** convention
  (row: last-two-axes `[2,3,4]@[2,4,5]→[2,3,5]`; col: first-two-axes
  `[3,4,2]@[4,5,2]→[3,5,2]`). Used for the ext_matmul per-order doc example.

See `2026-10-10-arrayapi-linalg/CHECKLIST.md` (R2). Related:
[[arrayapi-convergence-harness]], [[rstsr-linalg-batching-r1]],
[[rstsr-faer-py-wrapper-only]].

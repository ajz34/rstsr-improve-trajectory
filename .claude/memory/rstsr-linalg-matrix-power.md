---
name: rstsr-linalg-matrix-power
description: "matrix_power (array-API linalg, the R4 deferred member): faer-only MatrixPowerAPI ((Tr, isize)), composes batched inv_f with the tensor-tier matmul, numpy-parity n=0/n<0/squaring, merged #141, conformance 1309/40/33 -> 1312/37/33"
metadata:
  type: project
---

2026-10-11, branch `261011/matrix-power` (off `2e93deb`, i.e. post-`#140` batched
linalg), **merged as RESTGroup/rstsr#141** (squash `bbc6a19`). The R4 member the
earlier note said was *deferred, not dropped* — and it landed exactly where that
note predicted: on the **`rstsr-linalg-traits`** side, **not** rstsr-core (see
[[rstsr-linalg-outer-r4]] for the deferral reasoning). Suite
**`1309 / 40 / 33` → `1312 / 37 / 33`**, 0 regressions (flipped:
`test_has_names[linalg-matrix_power]`,
`test_extension_func_signature[linalg.matrix_power]`,
`test_linalg.py::test_matrix_power`).

- **Faer device only** (no BLAS column). New `MatrixPowerAPI<Inp>` in
  `rstsr-linalg-traits/src/traits_def.rs` with the receiver `(Tr, isize)` (the
  exponent travels with the tensor, per the "argument groups travel as one tuple"
  convention); kernel `faer_impl/matrix_power.rs`.
- **Composition, not a new kernel**: it composes the **batched `inv_f`** (R1,
  [[rstsr-linalg-batching-r1]]) with the **tensor-tier `matmul_f`**. The dependency
  direction is why it cannot live in rstsr-core: `n < 0` needs the matrix inverse,
  and the real `inv` lives in `rstsr-linalg-traits`, which *depends on*
  rstsr-core — so the entry belongs on the linalg side. (rstsr-core's `rt::inv` is
  the element-wise reciprocal.)
- **NumPy-parity semantics**: `n = 0` → identity (not `A^0` by multiplication),
  `n < 0` → invert **once** then the positive power of the inverse, and the
  general case by **repeated squaring LSB→MSB**. `(Tr, isize)` accepts the
  array-API `n` directly.
- `matrix_power` is now exposed by the shim (all three of its nodes pass), so it
  is no longer an absent `xp.linalg` member.

See `2026-10-10-arrayapi-linalg/CHECKLIST.md` (R4). Related:
[[rstsr-linalg-norms-r5]], [[rstsr-linalg-outer-r4]],
[[rstsr-linalg-batching-r1]].

---
name: rstsr-colmajor-soundness-t2
description: col_major cfg is fully wired and green at acfa93e — working test invocation, order_semantics.md contract doc, body self-neutralizes col (RowMajor hardcoded), faer matmul col twin fix, argmax row-flat contract holds on non-contig too
metadata:
  type: project
---

rstsr col_major soundness (T2 of 2026-09-14 campaign) at commit acfa93e,
patches in `2026-09-14-soundness-check/T2-col-major/`:

- Working col test invocation: `cargo test -p rstsr-core --no-default-features
  --features "col_major,aligned_alloc,faer,faer_as_default,std" --test
  entry_col_cpu`. `std` is REQUIRED (--no-default-features drops
  rstsr-common/std). No `rstsr/col_major` dev-dep feature needed: workspace
  declares rstsr-core `default-features = false`, so the umbrella dev-dep
  never leaks row_major into unification (compile_error! never fires).
- The predicted stale-cfg compile friction did NOT exist — all
  `#[cfg(feature = "col_major")]` branches (op_binary_assign,
  op_binary_common, adv_indexing, map_elementwise, reduction, iterator_*,
  op_tri, manipulation, op_binary_arithmetic) compile and pass at acfa93e.
- The contract spec for order behavior is `rstsr-core/src/docs/
  order_semantics.md` (+ per-function Row/Column Major Notice docstrings);
  col doctests 183/183. Col convention = FIRST-axis (leading) broadcast
  alignment (Julia rule): [2,3]+[3] errors; [3,2]+[3] broadcasts axis 0.
- The numpy-parity body (core_func/doc_draft/test_issues) pins
  `set_default_order(RowMajor)` in essentially every test (73 files), so the
  col entry re-ran it under row semantics — the body self-neutralizes col
  cfg. Col behavior was covered only by in-source col unit tests + (now)
  tests/col_func/ (66 plain-#[test] fns, own explicit devices, no
  specify_test!) wired via entry_col_cpu.rs + [[test]] required-features.
- FIXED in fixes.patch: device_faer/matmul_impl.rs test_minimal_correctness
  was `#[cfg(not(col_major))]`-silenced (faer matmul had zero col coverage).
  Col twin expects sum −78+282i (row twin −78+270i); derivation sum(AB)=
  Σ_k colsum(A)_k·rowsum(B)_k.
- argmin/argmax flat index = ROW-major flat of the logical tensor under BOTH
  cfgs, including non-contiguous views (transposed strides [6,2,1], partial
  slices) — verified, NOT a buffer offset. Mechanism: reduce_all_arg_cmp_cpu_serial
  (rstsr-native-impl/src/cpu_serial/reduction.rs:832) ravels the unraveled
  fold index through a RowMajor pseudo-layout.
- Test-authoring API gotchas (col build): `rt::full((shape, value, &dev))`;
  `rt::eye((r, c, k, &dev))`; `rt::tril((&a, k))`; `rt::vecdot(&a, &b, None)`
  returns a TENSOR (.to_scalar()); `rt::empty` is unsafe (use zeros);
  `to_vec` is 1-D only (reshape(-1) first); `.iter()` yields refs (use
  `.copied()` before collect-compare); `Tensor<T, B = DeviceCpu, D = IxD>`
  alias takes THREE params; fn-return tensors need concrete types
  (`Tensor<f64, DeviceCpuSerial>`), `_` not allowed in return position.

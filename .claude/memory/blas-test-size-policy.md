---
name: blas-test-size-policy
description: "User rule for device-tier tests (2026-11-25): large matrices (256 or 1024) only for fingerprint tests, one full-size solve per fingerprint; overload / parameter / side / layout / relation tests use the smallest workable matrix (3-5). The eig linalg subset went from ~500 s to ~23 s per device after restructuring."
metadata:
  type: feedback
---

Owner's rule, given while reviewing the BLAS eig tests: "large 256 or
1024 matrices tests are for fingerprint tests; overload/parameter tests
should be as small as the matrix can be (like 3-5 size matrix)". A suite
that takes minutes for one function family is read as over-tested.

**What was wrong (eig, `linalg_func/func_{f64,c64}/{eig,eigvals}.rs`)**
- Eight 1024 tests re-solved the same 1024x1024 matrix for every overload,
  side, homogeneous marker and conversion: ~34 full solves per dtype, 68 in
  all, 490 s on OpenBLAS (89% of the whole linalg suite), 434 s on AOCL.
- `--test-threads=1` was not the cause; the solves were.

**Shape that follows the rule**
- Fingerprint tests: exactly one full-size solve per routine/dtype
  (`eig((a, EigSide::Both))` gives `w`, `vl`, `vr` in one call). GEEV keeps
  1024 (3 s); GGEV moved to 256 leading blocks (a full-size QZ solve is far
  costlier: 27-35 s per solve at 1024) — regenerate the SciPy constants and
  the validator blocks when the size changes.
- Overload/parameter tests on a 4x4 block of the manifest data: every
  operand kind (owned, `&t`, view, `view_mut()`) x every form (side enum,
  string, `FlagSide`, `Homogeneous`, 3-tuples), compared against the small
  `Both` result (nearest-neighbour spectrum pairing, `allclose` blocks).
  Mutable operands are reset (`clone_from`) before each form.
- Driver tier: layout matrix n in {1,2,3,5,6}; relation tests n = 6.
- Measure with `cargo test ... -- --test-threads=1 -Z unstable-options
  --report-time <filter>` (nightly) before and after.

**Result**: eig subset 503 s -> 22.7 s on OpenBLAS, driver eig 34 tests in
8.8 s; validators run in 6 s / 10 s instead of 63 s / 84 s.

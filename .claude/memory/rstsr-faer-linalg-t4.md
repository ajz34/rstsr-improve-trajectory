---
name: rstsr-faer-linalg-t4
description: T4 faer-linalg audit results — beta=0 uninit-read bug fixed in all 6 naive kernels, faer_impl error-path fixes, DeviceFaer surface map, patches in 2026-09-14-soundness-check/T4-faer-linalg
metadata:
  type: project
---

T4 (faer+linalg soundness) at acfa93e, COMPLETE 2026-09-14. Patches in
`2026-09-14-soundness-check/T4-faer-linalg/`: fix-01-naive-beta0-uninit-read,
fix-02-faer-linalg-error-paths, safety-comments-and-tests. Key facts:

- Naive matmul/inner-dot kernels (native-impl cpu_serial + cpu_rayon, 6 fns)
  used to READ the uninitialized `empty()` output even at beta=0 → NaN/Inf
  garbage could propagate (empirically caught on 31x2@2x5). Fixed with
  `beta.is_zero()` branches; `TC: Zero` bound now required by the 4 serial
  kernels (rayon ones already had it). NaN-canary regression test pattern:
  pre-fill output with NaN, assert full overwrite.
- DeviceFaer surface: matmul in device_faer/matmul.rs (f32/f64/c32/c64→faer,
  rules 3–7 via batched layouts, parallel-outer at n_task>4*nthreads);
  linalg = 10 faer-native drivers in rstsr-linalg-traits/faer_impl;
  slogdet + solve_symmetric are BLAS-only (missing on DeviceFaer);
  vecdot/norm via rayon auto_impl symlinks.
- faer_impl error paths leaked `faer::set_global_parallelism` (global state!)
  via `?` — fixed by compute→restore→map_err; generalized eigh wrapped in
  closure. Non-square inputs panicked in faer asserts → now rstsr errors;
  eigh/eigvalsh 0x0 panicked in faer EVD → now numpy-compatible empty results
  (guard n==0). Singular solve/inv still silently returns inf (faer SVD
  contract, pinned by test, owner to decide).
- rstsr→faer stride map: shape [m,n], stride[0]→row_stride, stride[1]→
  col_stride — NO transposition. faer 0.22 accepts stride-0 (broadcast) and
  arbitrary positive strides.
- ColMajor matmul gotcha: creation (asarray/arange/into_shape) on a
  ColMajor-default device yields f-order VALUES — tensors compared across
  devices must be built on one device and `to_device`-transferred, else
  values differ (false "bug"). Col-major matmul rules 5–7 are expression-
  INCOMPATIBLE with numpy (batch axes trailing).
- Owned faer Mat→rstsr Tensor conversion (conversion.rs L73) is formal
  dealloc-layout UB (faer over-aligns/pads; Vec drop uses T align) —
  "works" on glibc, matches rstsr's own uninitialized_vec convention;
  Miri-unsafe. Owner-judgment item.
- rstsr-linalg-traits tests need `python rstsr-test-manifest/resources/
  gen_rand_vec.py` once (npy resources not in git); run with
  `cargo test -p rstsr-linalg-traits --features faer`.

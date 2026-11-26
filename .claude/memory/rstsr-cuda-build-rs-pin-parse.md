---
name: rstsr-cuda-build-rs-pin-parse
 description: rstsr-cuda build.rs (261021/cuda) requires a 6-character CUDARC_CUDA_VERSION but cudarc spells 5 ("12060", "13000"), so any pinned build has neither cublas_math_mode nor cublas_fixedpoint_math
metadata:
  type: project
---

Found in the 2026-11-26 review (`2026-11-26-cuda-matmul-review/REVIEW.md`, C4), verified with
`cargo rustc -p rstsr-cuda --lib -- --print cfg`:

- `build.rs:38` `v.len() == 6 && v[1] == '0' && v[5] == '0'` is never true -> `return None` (also skips the
  nvcc fallback) -> no cfg emitted. With `CUDARC_CUDA_VERSION=13000` (the value `matmul/mode.rs` recommends
  for the emulation) the classic driver refuses Pedantic/Fast*/EmulatedFixedPoint blaming the cudarc bindings.
  Unset (nvcc 13.2 detected) both cfgs are emitted. The CI pin 12060 is right by accident.
- Fix: parse cudarc's spelling (`format!("{major}0{minor}0")`: 5 chars, one-digit major, two-digit minor).
- The gating idea is right, see `cudarc-cublas-version-gating.md`.

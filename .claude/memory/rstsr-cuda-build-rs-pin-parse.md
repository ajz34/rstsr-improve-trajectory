---
name: rstsr-cuda-build-rs-pin-parse
 description: FIXED on 261021/cuda (uncommitted) - rstsr-cuda build.rs parses cudarc's actual `{major}0{minor}0` spelling of CUDARC_CUDA_VERSION; pin 13000 now yields both cfgs; CI tier3-cuda checks `--print cfg` per pin
metadata:
  type: project
---

Found in the 2026-11-26 review (`2026-11-26-cuda-matmul-review/REVIEW.md`, C4), fixed 2026-11-27.

- cudarc 0.19.10 spells the pin `format!("{major}0{minor}0")` and matches the whole string
  against its supported list, else panics: two-digit major, literal zero, one-digit minor,
  literal zero — `12060` is 12.6, `13000` is 13.0, `13040` is 13.4. (The old note here said
  "one-digit major, two-digit minor"; that was wrong.)
- `build.rs` now parses that shape (`parse_cudarc_version`, strip trailing zero, split at the
  first zero, lengths pinned to 2/1) and unit-tests it; because `cargo test` does not run
  `build.rs` as a target, the file is included into the crate via
  `#[cfg(test)] #[path = "../build.rs"] mod build_script` in `lib.rs` so the tests execute.
- Verified (`results/cfg_*_after_fix.txt`): pin `13000` -> both cfgs (was neither); `12060` ->
  neither (correct for 12.6, which is what hid the bug); unset (nvcc 13.2) -> both. The whole
  hardware suite passes on the `13000` pin.
- CI `tier3-cuda.yml` now has a matrix `cudarc_cuda_version: [12060, 13000]` with a step
  asserting each pin's cfg set (`--print cfg`); note `--print cfg` works on stable.
- The gating idea is right, see `cudarc-cublas-version-gating.md`.

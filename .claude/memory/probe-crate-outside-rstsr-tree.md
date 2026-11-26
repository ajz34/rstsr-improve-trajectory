---
name: probe-crate-outside-rstsr-tree
 description: How to probe rstsr on the GPU without touching its tree - a separate crate in the trajectory task dir with path deps, same toolchain and rustflags, CARGO_TARGET_DIR set to the rstsr target (reuses the cache); probe pitfalls listed
metadata:
  type: feedback
---

Used for the 2026-11-26 matmul review (`2026-11-26-cuda-matmul-review/probe/`), ~2 s incremental builds:

- Crate in the task directory with path deps on `rstsr`, `rstsr-core`, the device crate; `num`/`half` with the
  same features as the workspace (copy `Cargo.lock` from rstsr, ignored by git); `rust-toolchain.toml` = the
  workspace's; a `.cargo/config.toml` with the workspace's rustflags; run with
  `CARGO_TARGET_DIR=<rstsr>/target`. Without identical flags/toolchain everything rebuilds.
- Prefer printing `ok` / `FAIL` / `ERR` / `ref` lines over asserting: forced single drivers legitimately decline
  layouts they cannot express, and the run must not stop at the first finding.
- Pitfalls met: the CPU device's default order is **row-major** under the workspace's default features (call
  `set_default_order(order)` on a `DeviceCpuSerial` for batched references); a tensor's flags come from the
  device value it was created with (set `trace_*` before creating tensors); `CudaRuntimeFlags` is
  `#[non_exhaustive]` (no struct-update syntax outside the crate); `Bases`-like fixture structs need
  `B: DeviceRawAPI<S>`; a reference for `beta != 0` must start from the same initial `c`; f32-rounded operands
  need an f32-rounded CPU reference or a looser tolerance.
- An IDE may build the probe into its own `probe/target/` (801 MB here); it is gitignored, delete it.

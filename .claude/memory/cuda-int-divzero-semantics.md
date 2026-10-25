---
name: cuda-int-divzero-semantics
description: "CUDA integer division by zero returns -1 and remainder by zero returns 0 (observed on sm_120/NVRTC 13), while Rust `%`/`/` panic; MIN / -1 wraps to MIN."
metadata:
  type: reference
---

Measured 2026-10-25 on the RTX 5070 (sm_120, CUDA 13.x) with rstsr-cuda
kernels compiled from `int` expressions:

- `7 / 0` → `-1`, `-8 / 0` → `-1` (not a trap, not 0).
- `7 % 0` → `0` (and `-7 % 0` → `0`).
- `i32::MIN / -1` → `i32::MIN` (wraps), no fault.

The corresponding Rust expressions panic ("attempt to divide by zero" /
"attempt to calculate the remainder with a divisor of zero"), so a device
kernel cannot inherit the CPU reference's behavior by emitting plain
`a / b` or `a % b` in CUDA C. The signed cases are also C++ UB at the
language level; the values above are what the hardware/driver produce.

**How to apply:** if parity with the Rust device matters, guard zero (and
`MIN / -1`) explicitly and decide one contract for `/`, `%=` and
`floor_divide` together; never cite "CPU guard `0 % x == 0`" for the CPU
device -- the CPU device uses Rust `%` for integers and only routes floats
through `ExtReal::ext_rem`.

Related: [[cuda-kernel-nvrtc-grill]], [[rstsr-pow-signbit-decisions]]

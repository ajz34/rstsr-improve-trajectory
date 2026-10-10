# arrayapi-linalg (2026-10-10)

Work plan for the array-API **`linalg` extension** — the last remaining block of
the rstsr_faer.api conformance convergence.

## Context

- Subject: `rstsr_faer.api` (the pyo3 shim over rstsr's `DeviceFaer`) graded by
  the official `array-api-tests` suite (`@6c0b59f`, API version `2025.12`).
- Provenance: rstsr `main` `9db98e8` (element-wise `clip`), overall suite state
  **1249 / 51 / 82 of 1382**.
- The `fft` extension is **out of scope for this version** (decision
  2026-10-10): its absence is graceful — the suite self-skips the 28
  marker-guarded nodes and leaves 14 `test_has_names[fft-*]` failing (no panic;
  a user hitting `rstsr_faer.api.fft` gets a plain `AttributeError`). Those 14
  are left as-is by request.
- Therefore the **entire remaining red map is `linalg`**: 37 fails + 49 skips.

## Files

- `CHECKLIST.md` — the ordered work plan (red map, rust inventory, work items,
  ordering, verification, decisions).

## Related

- `../2026-10-07-arrayapi-convergence/` — the frozen status read-out and the
  now-drained buckets (rust-side queue, shim-side quick wins, dtype-function
  surface). Its `CHECKLIST.md` §B1 is this directory's parent decision.
- `../2026-10-04-rstsr-faer-py/GAP-REGISTER.md` — register ids (`G-004` QR,
  `G-005` slogdet/solve_symmetric, `G-023` the namespace itself).
- Skill `rstsr-faer-py-tests` — the build/run/measure harness.

Nothing here is committed to the main rstsr repos; it is the plan whose rust
items get registered as requests there.

# 2026-10-04-rstsr-cpu-pyo3 — pyo3 adapter on top of `rstsr-cpu-dlpack`

Status: **design / grilling in progress** — R1 questions open in `QUESTIONS-R1.md`.

## Goal (being sharpened)

Make numpy ↔ rstsr transfer approachable by replacing the ctypes "last mile" of
`rstsr-cpu-dlpack` (the reference shim in its `examples/` plus the `demo-ffi` host) with
a native pyo3 extension module: a Python-visible rstsr tensor implementing the DLPack
protocol, so `np.from_dlpack(t)` and `from_dlpack(numpy_array)` just work from Python,
zero-copy, with the capsule footguns owned by Rust instead of ctypes glue.

## Pinned state

- rstsr checkout: `main` @ `f179c46` (v0.9.0), clean, up to date with origin.
- Dependency: branch `261003/rstsr-cpu-dlpack` @ `d84f235` (pushed to fork `ajz34`,
  **not merged to main**) — provides `crates-interop/rstsr-cpu-dlpack`,
  the `rstsr` facade feature `dlpack`, and core `Clone for DataArc`/`TensorArc`.
- `dlpack-ffi` 1.3.0 from crates.io (the vendored-binding sibling repo is not needed
  at runtime).
- Inherited decisions: `../2026-10-03-rust-numpy-review/DECISIONS-R2.md` (pyo3 adapter
  is "a packaging step on top, not a redesign" — shape B; no rust-numpy reimplementation),
  `../2026-10-03-rstsr-cpu-dlpack/{DESIGN.md,FOLLOWUPS.md}` (§8 Python boundary; zero-copy
  adoption rejected; shared view export shipped; A1 mutability rung).

## Environment

TBD at implementation time. Expected shape (mirroring the dlpack task): stable Rust +
pyo3, maturin for the extension, conda env `torch` (Python 3.13 + NumPy 2.5.1) for the
Python e2e suite.

## Numbers

N/A — design phase. The deliverable is correctness, not speed: the e2e suite asserts
zero-copy pointer equality, flag behaviour, and deleter-exactly-once, mirroring the
dlpack prototype's 8 cases / 64 checks.

## Contents

- `initial-prompt.md` — maintainer tasking, verbatim.
- `QUESTIONS-R1.md` — round-1 design tree: inherited constraints, fact base, and the
  open questions with recommendations.

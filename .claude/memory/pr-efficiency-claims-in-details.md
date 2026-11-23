---
name: pr-efficiency-claims-in-details
description: "PR bodies: performance numbers belong in Details, never in the changelog-ready Changes part, and must name the CPU, threads and BLAS build"
metadata:
  type: feedback
---

Efficiency claims in an rstsr PR go in the *Details* part, not in the
changelog-ready *Changes* part. `Changes` is pasted verbatim into
`CHANGELOG.md` by the release manager, and benchmark numbers plus their
hardware do not belong there: a release note states the capability ("dispatch
vector-shaped `matmul` to BLAS-2 `gemv`"), the evidence lives in `Details`.

Every reported number names the stack it came from: CPU model, thread count,
BLAS library and build flavour, and the dtype — e.g. "AMD Ryzen 9 9950X3D,
16 threads, system OpenBLAS 0.3.32 (DYNAMIC_ARCH, OpenMP, `Cooperlake`
kernel), f64". A bare "8.8× faster" is not reviewable.

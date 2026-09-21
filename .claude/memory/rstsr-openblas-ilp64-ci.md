---
name: rstsr-openblas-ilp64-ci
description: CI builds rstsr-openblas ILP64 (blas_int = i64), local builds 32-bit — never cast blas_int to fixed-width ints in tests
metadata:
  type: project
---

rstsr CI (restgroup/rstsr) builds rstsr-openblas against an **ILP64
OpenBLAS** (`blas_int = i64`); local dev builds use the 32-bit default
(`blas_int = i32`).

**Why:** a test written locally with `v as i64` compiled clean locally but
failed CI clippy (`clippy::unnecessary_cast`, `-D warnings`) — hit on PR
#106 (issue_getrf_getri, fixed 74b7df7).

**How to apply:** compare/assert `blas_int` values in their native dtype
(`ipiv.raw().to_vec()`, `vec![1 as blas_int, ...]`), never via fixed-width
casts. Lint-visible local clippy runs won't catch this — the divergence
only shows on CI.

Related: [[rstsr-soundness-t1-unsafe-audit]] (R3 fix record),
[[rstsr-bench-context]] (machine facts).

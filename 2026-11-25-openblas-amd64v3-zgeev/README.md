# 2026-11-25 — `zgeev` in resolute's `amd64v3` OpenBLAS returns wrong eigenvectors

- **Date**: 2026-11-25 (Asia/Shanghai).
- **Task**: diagnose the CI-only `test_zgeev` failure of the rstsr eig branch.
  The failure reproduces with the library alone, i.e. it is a defect of the
  distro build, not of rstsr. This directory holds the evidence and a
  Launchpad-ready report.
- **Status**: diagnosis complete; the report below is drafted for filing (not
  filed).

## Provenance

| item | value |
|--|--|
| rstsr | branch `261125/blas-eig` @ `16519a0` (PR #170); `main` @ `b01b6e0` |
| failing CI | GitHub Actions `tier1-openblas`, `ubuntu-latest` = resolute (26.04) |
| defective library | `libopenblas0-pthread` / `libopenblas0-openmp` 0.3.32+ds-5, **`amd64v3`** package (`..._amd64v3.deb`) |
| correct references | same version plain `amd64` package; noble 0.3.26; local OpenBLAS 0.3.34; MKL 2026.1; AOCL 5.3.0 |
| CPU | Cooperlake; both builds report `OpenBLAS 0.3.32 NO_LAPACKE DYNAMIC_ARCH NO_AFFINITY Cooperlake MAX_THREADS=128` and select the same `Cooperlake` core |
| reproducer | `python3 repro_zgeev.py 76 <path>/libopenblas.so.0` (ctypes only, ~0.1 s) |

## The defect

`zgeev` on the `amd64v3` build returns eigenvectors that do not satisfy the
defining relation. For a matrix `A` with computed eigenvalues `λ_j` and right
eigenvectors `v_j`:

```
|| A v_j - λ_j v_j || / || v_j ||     ~ 2e-1 .. 6e0      (expected ~1e-14)
|| A^H v_j - conj(λ_j) v_j || / || v_j ||  ~ 2e-1 .. 6e0  (expected ~1e-14)
```

The eigenvalues are correct; only the two eigenvector blocks are wrong, and
both `jobvl='V'` and `jobvr='V'` (requested independently) show it.

Size dependence with the same matrix family: `n <= 75` is correct
(`n = 73, 74, 75` give ~2e-14), `n >= 76` is wrong. The matrix comes from a fixed
generator in `repro_zgeev.py`, and numpy-generated matrices reproduce it too
(seeds 0-4 at `n = 100`, all wrong).

## Evidence

`repro_zgeev.py 80 <library>` on this machine:

| library | left residual | right residual |
|--|--|--|
| resolute `amd64v3` 0.3.32+ds-5, pthread | 3.6e-01 | 2.0e-01 |
| resolute `amd64v3` 0.3.32+ds-5, openmp | 3.6e-01 | 2.0e-01 |
| resolute plain `amd64` 0.3.32+ds-5, pthread | 2.1e-14 | 2.5e-14 |
| noble 0.3.26 pthread | 2.0e-14 | 2.7e-14 |

The same library also fails rstsr's LAPACK driver tests: the 1024×1024 complex
solve pins `|vl|` at `-11.30` where a correct build gives `-5.4564`, and a
129×129 case reports a left-relation residual of 2.0e-01. Both `amd64v3`
variants fail identically, and `OPENBLAS_CORETYPE` does not change the outcome
(all of `ZEN`, `COOPERLAKE`, `SKYLAKEX`, `ICELAKE`, `HASWELL`, `SANDYBRIDGE`,
`NEHALEM`, `PRESCOTT`, `ATHLON` fail), so it is not a single bad BLAS kernel
but the generic LAPACK code of that build — consistent with a build-flag
difference of the `amd64v3` rebuild.

Unaffected on the same library: `dgeev` (the 1024×1024 real solve and all
layout cases pass) and `zheev` (Hermitian, residual 4e-14). So the defect is
limited to the non-symmetric complex eigenvector path.

## Draft Launchpad report

**Title**: `libopenblas 0.3.32+ds-5 (amd64v3): zgeev returns incorrect eigenvectors`

**Package**: `openblas` — `libopenblas0-pthread`, `libopenblas0-openmp`
`0.3.32+ds-5` (also `libopenblas-openmp-dev`, `-pthread-dev`), architecture
package `amd64v3`, Ubuntu resolute (26.04).

**Summary**: on the `amd64v3` rebuild, `zgeev` returns eigenvectors that do
not satisfy `A v = λ v` (residuals of order 1e-1 instead of 1e-14), while the
`amd64` package of the same version is correct. Eigenvalues are correct;
`dgeev` and `zheev` are correct.

**Steps to reproduce**:

1. Download/install the `amd64v3` package, e.g.
   `http://archive.ubuntu.com/ubuntu/pool/universe/o/openblas/libopenblas0-pthread_0.3.32+ds-5_amd64v3.deb`
   and unpack it (`dpkg-deb -x`), or install it on a machine whose apt prefers
   the `amd64v3` component (GitHub's `ubuntu-latest` runner does).
2. Save `repro_zgeev.py` and run
   `python3 repro_zgeev.py 76 /path/to/openblas-pthread/libopenblas.so.0`.
3. Observed: `right relation residual: 2.2e-01` (and a similar left residual),
   exit status 1. Expected with the `amd64` package of the same version:
   `2e-14`, exit status 0.

**Expected**: eigenvectors of a well-conditioned matrix satisfy the eigen
relation to about machine precision, as with the `amd64` package, noble's
0.3.26 and upstream OpenBLAS 0.3.34.

**Impact**: any program linked against the distro `libopenblas.so.0` that
solves non-symmetric complex eigenproblems gets silently wrong eigenvectors
for `n >= 76`. GitHub Actions' `ubuntu-latest` runners install this package by
default, which is how it was found.

**Workaround**: install the plain `amd64` package of the same version
(`..._amd64.deb`), or noble's 0.3.26; both are correct.

**Notes**: the two builds report the same OpenBLAS configuration string and
select the same `Cooperlake` core, so the difference is in how the package was
compiled. The failure threshold at `n = 76` may help locate the affected code
path in the LAPACK translation unit that the rebuild miscompiles.

#!/usr/bin/env python3
"""numpy anchors for the P0 GEMV table (cross-language sanity check only).

Times ``a @ xq`` (matvec) and ``xp @ a`` (vecmat) on the same cells as the Rust
harness, for ``numpy.matmul``. numpy ships its own BLAS; the numbers are an
anchor, not the ceiling (the ceiling is the raw cblas gemv in the Rust harness).

Run with the conda env that has numpy 2.5 (``torch`` on this machine):
    python3 numpy_ref/anchors.py
"""

import time

import numpy as np

CELLS = [
    (1024, 1024, "1024x1024"),
    (4096, 4096, "4096x4096"),
    (8192, 8192, "8192x8192"),
    (65536, 64, "65536x64"),
    (256, 452676, "256x452676"),
    (452676, 256, "452676x256"),
]

REPEATS = 5


def best(fn, repeats=REPEATS):
    fn()  # warmup
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def main():
    print("numpy", np.__version__)
    print("blas:", np.__config__.show(mode="dicts"))
    print()

    for p, q, name in CELLS:
        a = np.ones((p, q), dtype=np.float64)
        xp = np.ones(p, dtype=np.float64)
        xq = np.ones(q, dtype=np.float64)
        t_matvec = best(lambda: a @ xq)
        t_vecmat = best(lambda: xp @ a)
        flops = 2.0 * p * q
        print(f"{name:>12} matvec {t_matvec * 1e3:9.3f} ms ({flops / t_matvec / 1e9:7.1f} GF/s)"
              f"   vecmat {t_vecmat * 1e3:9.3f} ms ({flops / t_vecmat / 1e9:7.1f} GF/s)")
        del a, xp, xq


if __name__ == "__main__":
    main()

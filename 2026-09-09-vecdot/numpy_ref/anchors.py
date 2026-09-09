#!/usr/bin/env python3
"""numpy anchors for T3' (plan D10) — context numbers against rstsr.

Run inside the conda `torch` env (numpy 2.5.1):
    conda activate torch
    OMP_NUM_THREADS=1 python numpy_ref/anchors.py     # single-thread anchors
    python numpy_ref/anchors.py blis                  # + BLAS-threaded np.dot

Anchors:
- einsum('ij,ij->i')  batched row-dot (rstsr batched_am1)
- einsum('ij,ij->j')  column-accumulation (rstsr batched_axis0)
- einsum('ij,ji->i')  strided-b row-dot (rstsr batched_strided)
- einsum('i,i->')     1-D dot; np.dot = BLAS ddot (threaded per build)
- np.vdot / np.dot on complex as c64 context

Same deterministic fixture pattern as the Rust harness.
"""

import os
import statistics
import sys
import time

import numpy as np

REPEAT = 5


def gen_vec(n, salt):
    i = np.arange(n, dtype=np.uint64)
    s = np.uint64((salt * 0xD1B54A32D192ED03) & 0xFFFFFFFFFFFFFFFF)
    x = (i * np.uint64(0x9E3779B97F4A7C15)) ^ s
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = x ^ (x >> np.uint64(27))
    return ((x % np.uint64(2003)).astype(np.float64) / 2003.0) - 0.5


def gen_mat(m, n, salt):
    return gen_vec(m * n, salt).reshape(m, n)


def gen_mat_c(m, n, salt):
    return (gen_vec(m * n, salt) + 1j * gen_vec(m * n, salt + 77)).reshape(m, n)


def timeit(fn, repeat=REPEAT):
    fn()  # warm up
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts), statistics.median(ts)


def report(name, best, median, bytes_moved=None, flops=None):
    extra = ""
    if bytes_moved is not None:
        extra += f"  best_GBs={bytes_moved / 1e9 / best:8.1f}"
    if flops is not None:
        extra += f"  best_GFLOPs={flops / 1e9 / best:7.2f}"
    print(f"{name:55s} best={best*1e3:10.3f} ms  median={median*1e3:10.3f} ms{extra}")


def main():
    blas_threaded = len(sys.argv) > 1 and sys.argv[1] == "blis"
    print(f"numpy {np.__version__} (conda env: torch); anchors "
          f"{'WITH BLAS threading for np.dot' if blas_threaded else 'single-threaded (OMP_NUM_THREADS=1)'}")
    print(f"repeat={REPEAT}, best/median wall time\n")

    # ---- batched vecdot cases (f64) ----
    for label, m, k in [("batched_am1_4096x512", 4096, 512), ("batched_am1_8192x256", 8192, 256),
                        ("odd_1000x777", 1000, 777)]:
        a = gen_mat(m, k, 1)
        b = gen_mat(m, k, 2)
        report(f"einsum ij,ij->i  {label}", *timeit(lambda: np.einsum("ij,ij->i", a, b)),
               bytes_moved=2 * m * k * 8, flops=2 * m * k)
        report(f"einsum ij,ij->j  {label}", *timeit(lambda: np.einsum("ij,ij->j", a, b)),
               bytes_moved=2 * m * k * 8, flops=2 * m * k)

    # strided: b stored (k, m) contiguous, transposed view
    m, k = 4096, 512
    a = gen_mat(m, k, 1)
    bst = gen_mat(k, m, 2)
    report("einsum ij,ji->i  batched_strided_4096x512", *timeit(lambda: np.einsum("ij,ji->i", a, bst)),
           bytes_moved=2 * m * k * 8, flops=2 * m * k)

    # complex batched (c64)
    ac = gen_mat_c(m, k, 1)
    bc = gen_mat_c(m, k, 2)
    report("einsum ij,ij->i  batched_am1_4096x512_c64", *timeit(lambda: np.einsum("ij,ij->i", ac, bc)),
           bytes_moved=2 * m * k * 16)

    # ---- 1-D dot ----
    for label, n in [("small", 1_000), ("medium", 100_000), ("large", 10_000_000)]:
        av = gen_vec(n, 1)
        bv = gen_vec(n, 2)
        report(f"einsum i,i->     dot1d_{label}_{n}", *timeit(lambda: np.einsum("i,i->", av, bv)),
               bytes_moved=2 * n * 8, flops=2 * n)
        report(f"np.dot (BLAS)    dot1d_{label}_{n}", *timeit(lambda: np.dot(av, bv)),
               bytes_moved=2 * n * 8, flops=2 * n)

    # complex 1-D spot
    avc = gen_vec(100_000, 1) + 1j * gen_vec(100_000, 31)
    bvc = gen_vec(100_000, 2) + 1j * gen_vec(100_000, 32)
    report("np.vdot          dot1d_medium_100000_c64", *timeit(lambda: np.vdot(avc, bvc)),
           bytes_moved=2 * 100_000 * 16)

    # ---- inner-dot-sized % (context for the rayon-fold kernel) ----
    for label, n in [("small", 64), ("medium", 10_000), ("large", 1_000_000)]:
        av = gen_vec(n, 1)
        bv = gen_vec(n, 2)
        report(f"einsum i,i->     innerdot_{label}_{n}", *timeit(lambda: np.einsum("i,i->", av, bv)),
               bytes_moved=2 * n * 8, flops=2 * n)


if __name__ == "__main__":
    main()

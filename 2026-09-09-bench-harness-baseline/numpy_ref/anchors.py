#!/usr/bin/env python3
"""numpy anchors for T0 (plan D10) — context numbers against rstsr.

Run inside the conda `torch` env (numpy 2.5.1):
    conda activate torch
    python numpy_ref/anchors.py

numpy reductions/ufuncs here are SINGLE-THREADED (no BLAS in these ops), so
they anchor mainly against rstsr's DeviceCpuSerial column.

Per op: warm up, repeat REPEAT times, report best and median wall time and
derived GB/s where the op is streaming. Uses the same deterministic fixture
pattern as the Rust harness (gen_value hash) — values do not matter for
timing but keep the runs comparable.
"""

import statistics
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


def timeit(fn, repeat=REPEAT):
    """returns (best_s, median_s)"""
    fn()  # warm up
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts), statistics.median(ts)


def report(name, best, median, bytes_moved=None):
    if bytes_moved is not None:
        gb = bytes_moved / 1e9
        print(f"{name:55s} best={best*1e3:10.3f} ms  median={median*1e3:10.3f} ms  best_GBs={gb/best:8.1f}")
    else:
        print(f"{name:55s} best={best*1e3:10.3f} ms  median={median*1e3:10.3f} ms")


def main():
    print(f"numpy {np.__version__} (conda env: torch); single-threaded anchors")
    print(f"repeat={REPEAT}, best/median wall time\n")

    mat_sizes = [("small", 64, 64), ("medium", 512, 512), ("large", 2048, 2048), ("odd", 1000, 777)]

    for label, m, n in mat_sizes:
        a = gen_mat(m, n, 1)
        report(f"transpose_copy {label}_{m}x{n} (a.T.copy())", *timeit(lambda: a.T.copy()),
               bytes_moved=2 * m * n * 8)
        report(f"sum_axis0     {label}_{m}x{n} (a.sum(0))", *timeit(lambda: a.sum(axis=0)),
               bytes_moved=m * n * 8)
        report(f"sum_axislast  {label}_{m}x{n} (a.sum(1))", *timeit(lambda: a.sum(axis=1)),
               bytes_moved=m * n * 8)
        report(f"sum_all       {label}_{m}x{n} (a.sum())", *timeit(lambda: a.sum()),
               bytes_moved=m * n * 8)

    print()
    for label, n in [("small", 1_000), ("medium", 100_000), ("large", 10_000_000)]:
        a = gen_vec(n, 1)
        b = gen_vec(n, 2)
        report(f"vecdot np.dot       {label}_{n}", *timeit(lambda: np.dot(a, b)),
               bytes_moved=2 * n * 8)
        report(f"vecdot np.einsum    {label}_{n}", *timeit(lambda: np.einsum("i,i->", a, b)),
               bytes_moved=2 * n * 8)

    print()
    m = 2048
    n = 2048
    a = gen_mat(m, n, 1)
    b = gen_mat(m, n, 2)
    brow = gen_vec(n, 3).reshape(1, n)
    bt = gen_mat(m, n, 4)  # b.t() below is the strided operand
    report("add contig     large_2048x2048 (a+b)", *timeit(lambda: a + b),
           bytes_moved=3 * m * n * 8)
    report("add broadcast  large_2048x2048 (a+brow)", *timeit(lambda: a + brow),
           bytes_moved=3 * m * n * 8)
    report("add strided    large_2048x2048 (a+bt.T)", *timeit(lambda: a + bt.T),
           bytes_moved=3 * m * n * 8)

    print()
    report("zeros          large_2048x2048 (np.zeros)", *timeit(lambda: np.zeros((2048, 2048))),
           bytes_moved=2048 * 2048 * 8)
    report("full           large_2048x2048 (np.full)", *timeit(lambda: np.full((2048, 2048), 3.25)),
           bytes_moved=2048 * 2048 * 8)

    print()
    for label, n in [("medium", 1_000_000), ("large", 10_000_000)]:
        a = gen_vec(n, 1)
        report(f"argmax         {label}_{n}", *timeit(lambda: np.argmax(a)),
               bytes_moved=n * 8)

    print()
    # numpy triad (the anchor ceiling for this environment)
    for label, n in [("large", 10_000_000), ("medium", 1_000_000)]:
        a = gen_vec(n, 1)
        b = gen_vec(n, 2)
        c = np.zeros(n)
        def triad():
            np.add(a, 3.0 * b, out=c)
        report(f"triad (a+3b->c) {label}_{n}", *timeit(triad),
               bytes_moved=3 * n * 8)
        def copy_():
            c[:] = a
        report(f"memcpy (c[:] = a) {label}_{n}", *timeit(copy_),
               bytes_moved=2 * n * 8)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""numpy anchors for T2' value reductions (plan D10) — context numbers.

Run inside the conda `torch` env (numpy 2.5.1):
    conda run -n torch python numpy_ref/anchors.py

numpy reductions here are SINGLE-THREADED, so they anchor against rstsr's
DeviceCpuSerial column. Same deterministic fixture pattern as the Rust
harness (gen_value hash).
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
    fn()
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

    for (label, m, n) in [("medium", 512, 512), ("large", 2048, 2048), ("odd", 1000, 777)]:
        a = gen_mat(m, n, 1)
        b = f"{label}_{m}x{n} f64"
        report(f"{b}: sum axis0", *timeit(lambda: a.sum(axis=0)), m * n * 8)
        report(f"{b}: sum axis1", *timeit(lambda: a.sum(axis=1)), m * n * 8)
        report(f"{b}: sum all", *timeit(lambda: a.sum()), m * n * 8)
        report(f"{b}: mean axis1", *timeit(lambda: a.mean(axis=1)), m * n * 8)
        report(f"{b}: var axis1", *timeit(lambda: a.var(axis=1)), m * n * 8)
        report(f"{b}: min axis0", *timeit(lambda: a.min(axis=0)), m * n * 8)
        print()

    n = 10_000_000
    v = gen_vec(n, 1)
    report("large_1e7 f64: sum", *timeit(lambda: v.sum()), n * 8)
    report("large_1e7 f64: mean", *timeit(lambda: v.mean()), n * 8)
    report("large_1e7 f64: var", *timeit(lambda: v.var()), n * 8)
    report("large_1e7 f64: l2 norm", *timeit(lambda: np.linalg.norm(v)), n * 8)
    report("large_1e7 f64: min", *timeit(lambda: v.min()), n * 8)
    report("large_1e7 f64: max", *timeit(lambda: v.max()), n * 8)

    # broadcast reduction semantics reference (the upstream-bug case):
    v = np.arange(1, 7).astype(float).reshape(1, 6)
    b = np.broadcast_to(v, (5, 6))
    print()
    print("broadcast semantics ref: sum(axis=0) of [5,6] broadcast of [1,6]:")
    print("  numpy:", b.sum(axis=0), " (rstsr 386948be gives 6x values — see correctness gate)")


if __name__ == "__main__":
    main()

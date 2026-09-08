#!/usr/bin/env python3
"""Optional numpy context anchors for T6 argmax/argmin.

Requires the conda `torch` env (numpy 2.5.1). Context only — the formal
reference columns are the ndarray-stats anchors in benches/anchors_ndarray.rs.
T0's standing context number: numpy argmax 1e7 f64 = 1.32 ms (L3-assisted).
"""

import time

import numpy as np


def gen(n: int, salt: int) -> np.ndarray:
    """Same deterministic pattern as the Rust harness (gen_value, [-0.5, 0.5))."""
    i = np.arange(n, dtype=np.uint64)
    x = (i * np.uint64(0x9E3779B97F4A7C15)) ^ (np.uint64(salt) * np.uint64(0xD1B54A32D192ED03))
    x = x ^ (x >> np.uint64(30))
    x = x * np.uint64(0xBF58476D1CE4E5B9)
    x = x ^ (x >> np.uint64(27))
    return ((x % np.uint64(2003)).astype(np.float64) / 2003.0 - 0.5)


def bench(fn, iters: int) -> float:
    fn()  # warmup
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1e3


def main() -> None:
    print("numpy argmax/argmin anchors (single-thread context, 9950X3D)")
    for n, iters in [(1_000_000, 50), (10_000_000, 20)]:
        a = gen(n, 1)
        t_max = bench(lambda: a.argmax(), iters)
        t_min = bench(lambda: a.argmin(), iters)
        print(f"  n={n}: argmax {t_max:.3f} ms ({n * 8 / t_max / 1e6:.1f} GB/s), "
              f"argmin {t_min:.3f} ms ({n * 8 / t_min / 1e6:.1f} GB/s)")
    a32 = gen(10_000_000, 1).astype(np.float32)
    t_max = bench(lambda: a32.argmax(), 20)
    print(f"  n=1e7 f32: argmax {t_max:.3f} ms ({10_000_000 * 4 / t_max / 1e6:.1f} GB/s)")


if __name__ == "__main__":
    main()

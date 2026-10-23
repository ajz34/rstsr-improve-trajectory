"""CuPy baseline for the elementwise bench (same scenarios/shapes as
crates-device/rstsr-cuda/benches/bench_elementwise.rs).

Run: conda run -n ruaa-dev python bench_cupy.py
"""

import time

import cupy as cp


def bench(name, n_iter, f):
    for _ in range(3):
        f()
    cp.cuda.get_current_stream().synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        f()
    cp.cuda.get_current_stream().synchronize()
    dt = (time.perf_counter() - t0) / n_iter
    print(f"{name}: {dt*1e3:.3f} ms/op")
    return dt


def main():
    n_iter = 200
    n = 2_000_000

    print("=== fill f64 [16 MiB] ===")
    bench("cupy fill", n_iter, lambda: (cp.full(n, 2.5, dtype=cp.float64)))

    print("=== add f64 contig [16 MiB] ===")
    a = cp.arange(n, dtype=cp.float64)
    b = cp.arange(n, dtype=cp.float64)
    bench("cupy add", n_iter, lambda: a + b)

    print("=== add f64 transposed-B [4x4 MiB] ===")
    a2 = cp.arange(1_000_000, dtype=cp.float64).reshape(1000, 1000)
    b2 = cp.arange(1_000_000, dtype=cp.float64).reshape(1000, 1000)
    b2t = b2.T
    bench("cupy add-strided", n_iter, lambda: a2 + b2t)

    print("=== sin f64 [16 MiB] ===")
    bench("cupy sin", n_iter, lambda: cp.sin(a))

    print("=== to_contig transposed [4 MiB] ===")
    a2t = a2.T
    bench("cupy to_contig(T)", n_iter, lambda: a2t.copy())

    print("=== arange f64 [2e5] ===")
    bench("cupy arange", n_iter, lambda: cp.arange(0, 2e5, 1, dtype=cp.float64))


if __name__ == "__main__":
    main()

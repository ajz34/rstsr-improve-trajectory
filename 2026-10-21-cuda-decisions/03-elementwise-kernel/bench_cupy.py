"""cupy baseline for the elementwise benchmark cells (same scenarios as
benches/bench_elementwise.rs). Median-of-R wall timing, sync per repeat.

Usage: bench_cupy.py [n_iter] [repeat]
Cell names must match the Rust harness for side-by-side comparison.
"""

import sys

import cupy as cp

N_ITER = int(sys.argv[1]) if len(sys.argv) > 1 else 200
REPEAT = int(sys.argv[2]) if len(sys.argv) > 2 else 5


def bench(name, f, n_iter=N_ITER, repeat=REPEAT):
    for _ in range(30):
        f()
    cp.cuda.get_current_stream().synchronize()
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(n_iter):
            f()
        cp.cuda.get_current_stream().synchronize()
        samples.append((time.perf_counter() - t0) / n_iter)
    samples.sort()
    med = samples[len(samples) // 2]
    print(f"{name},{med * 1e3:.6f}", flush=True)
    return med


import time  # noqa: E402  (after bench def for ordering clarity)

n = 2048  # 2-D operand: 32 MiB f64

print("cell,median_ms")

# sustained warm-up to raise clocks from idle (matches the Rust harness)
_w = cp.arange(2048 * 2048, dtype=cp.float64).reshape(2048, 2048) if "cupy" in sys.modules else None
if _w is not None:
    for _ in range(40):
        (_w + _w).get()
        cp.cuda.get_current_stream().synchronize()

# fill
dt_fill = bench("fill", lambda: cp.full((n, n), 2.5))

# add contiguous
a = cp.arange(n * n, dtype=cp.float64).reshape(n, n)
b = cp.arange(n * n, dtype=cp.float64).reshape(n, n)
dt_add = bench("add", lambda: a + b)

# add B.T
bt = b.T
dt_add_t = bench("add_t", lambda: a + bt)

# add A.T (same buffer both sides)
at = a.T
bench("add_t_self", lambda: a + at)

# 3-D swapaxes(0,1) [512^3]
m = 512
a3 = cp.arange(m**3, dtype=cp.float64).reshape(m, m, m)
a3t = a3.swapaxes(0, 1)
bench("add_swap3d", lambda: a3 + a3t)

# broadcast B[:,None,:]
brow = cp.arange(m * m, dtype=cp.float64).reshape(m, 1, m)
bench("add_bcast_r", lambda: a3 + brow)

# broadcast B[:,:,None]
bcol = cp.arange(m * m, dtype=cp.float64).reshape(m, m, 1)
bench("add_bcast_c", lambda: a3 + bcol)
del a3, a3t, brow, bcol  # free the ~4 GiB arena before 2-D cells
cp.get_default_memory_pool().free_all_blocks()

# outer a[None,:] + b[:,None] -> [2048,2048]
va = cp.arange(n, dtype=cp.float64)
vb = cp.arange(n, dtype=cp.float64)
bench("add_outer", lambda: va[None, :] + vb[:, None])

# sliced windows
aw = a[:, 10:1010]
bw = b[:, 50:1050]
bench("add_slice", lambda: aw + bw)

# stepped B[:, ::2] against A[:, :1024]
ash = a[:, : n // 2]
bs = b[:, ::2]
bench("add_step", lambda: ash + bs)

# flipped B
fl = b[:, ::-1]
flh = fl[:, : n // 2]
bench("add_flip", lambda: ash + flh)

# where
mask = cp.arange(n * n, dtype=cp.int32).reshape(n, n) >= (n * n // 2)
bench("where_tt", lambda: cp.where(mask, a, b))
bench("where_ts", lambda: cp.where(mask, a, cp.float64(0.5)))

# clip scalar bounds
bench("clip", lambda: cp.clip(a, 1000.0, 3.0e6))

# remainder
dt_rem = bench("rem", lambda: cp.mod(a, b))

# sin
dt_sin = bench("sin", lambda: cp.sin(a))

# pow
bench("pow", lambda: cp.power(a, 2.5))

# strided-output assign: C.T = A
dst = cp.arange(n * n, dtype=cp.float64).reshape(n, n)
dstv = dst.T
bench("copy_T", lambda: dstv.__setitem__(Ellipsis, a))

# to_contig of transposed view
bench("to_contig", lambda: cp.ascontiguousarray(a.T))

_ = (dt_fill, dt_add, dt_add_t, dt_rem, dt_sin)
cp.cuda.get_current_stream().synchronize()

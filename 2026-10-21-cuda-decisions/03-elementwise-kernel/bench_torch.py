"""pytorch baseline for the elementwise benchmark cells (same scenarios as
benches/bench_elementwise.rs). Median-of-R wall timing, sync per repeat.

Usage: bench_torch.py [n_iter] [repeat]
Cell names must match the Rust harness for side-by-side comparison.
"""

import sys
import time

import torch

N_ITER = int(sys.argv[1]) if len(sys.argv) > 1 else 200
REPEAT = int(sys.argv[2]) if len(sys.argv) > 2 else 5

dev = torch.device("cuda:0")


def bench(name, f, n_iter=N_ITER, repeat=REPEAT):
    for _ in range(30):
        f()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(n_iter):
            f()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) / n_iter)
    samples.sort()
    med = samples[len(samples) // 2]
    print(f"{name},{med * 1e3:.6f}", flush=True)
    return med


n = 2048  # 2-D operand: 32 MiB f64

print("cell,median_ms")

# sustained warm-up to raise clocks from idle (matches the Rust harness)
_w = torch.arange(2048 * 2048, dtype=torch.float64, device=dev).reshape(2048, 2048)
for _ in range(40):
    (_w + _w)
    torch.cuda.synchronize()

# fill
dt_fill = bench("fill", lambda: torch.full((n, n), 2.5, device=dev))

# add contiguous
a = torch.arange(n * n, dtype=torch.float64, device=dev).reshape(n, n)
b = torch.arange(n * n, dtype=torch.float64, device=dev).reshape(n, n)
dt_add = bench("add", lambda: a + b)

# add B.T
bt = b.T
dt_add_t = bench("add_t", lambda: a + bt)

# add A.T
at = a.T
bench("add_t_self", lambda: a + at)

# 3-D swapaxes [512^3]
m = 512
a3 = torch.arange(m**3, dtype=torch.float64, device=dev).reshape(m, m, m)
a3t = a3.transpose(0, 1)
bench("add_swap3d", lambda: a3 + a3t)

# broadcast
brow = torch.arange(m * m, dtype=torch.float64, device=dev).reshape(m, 1, m)
bench("add_bcast_r", lambda: a3 + brow)
bcol = torch.arange(m * m, dtype=torch.float64, device=dev).reshape(m, m, 1)
bench("add_bcast_c", lambda: a3 + bcol)
del a3, a3t, brow, bcol  # free the ~4 GiB arena before 2-D cells
torch.cuda.empty_cache()

# outer
va = torch.arange(n, dtype=torch.float64, device=dev)
vb = torch.arange(n, dtype=torch.float64, device=dev)
bench("add_outer", lambda: va[None, :] + vb[:, None])

# sliced windows
aw = a[:, 10:1010]
bw = b[:, 50:1050]
bench("add_slice", lambda: aw + bw)

# stepped
ash = a[:, : n // 2]
bs = b[:, ::2]
bench("add_step", lambda: ash + bs)

# flipped
fl = b.flip(1)
flh = fl[:, : n // 2]
bench("add_flip", lambda: ash + flh)

# where
mask = torch.arange(n * n, dtype=torch.int32, device=dev).reshape(n, n) >= (n * n // 2)
bench("where_tt", lambda: torch.where(mask, a, b))
bench("where_ts", lambda: torch.where(mask, a, torch.tensor(0.5, dtype=torch.float64, device=dev)))

# clip
bench("clip", lambda: torch.clamp(a, 1000.0, 3.0e6))

# remainder
dt_rem = bench("rem", lambda: torch.remainder(a, b))

# sin
dt_sin = bench("sin", lambda: torch.sin(a))

# pow
bench("pow", lambda: torch.pow(a, 2.5))

# strided-output assign
dst = torch.arange(n * n, dtype=torch.float64, device=dev).reshape(n, n)
dstv = dst.T
bench("copy_T", lambda: dstv.copy_(a))

# contiguous copy of transposed
bench("to_contig", lambda: a.T.contiguous())

_ = (dt_fill, dt_add, dt_add_t, dt_rem, dt_sin)
torch.cuda.synchronize()

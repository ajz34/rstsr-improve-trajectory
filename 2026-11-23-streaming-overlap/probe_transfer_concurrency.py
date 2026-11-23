#!/usr/bin/env python3
"""Do H2D and D2H copies overlap with each other on this GPU?

Small companion to bench_streaming.py: the pipeline model depends on whether
the host<->device copy engine is one serial resource or two independent
directions. Measures, over the same bytes each way:

  h2d only        all chunks up on stream A
  d2h only        all chunks down on stream B
  concurrent      both directions alternating on A and B

concurrent == h2d + d2h means one serial engine; concurrent == max(h2d, d2h)
means fully independent directions. Anything between is partial overlap.

Usage: probe_transfer_concurrency.py [--mib 64] [--chunks 8]
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import cupy as cp

DTYPE = np.float64


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mib", type=float, default=64.0, help="bytes per chunk (MiB)")
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()

    n = int(args.mib * 1024 * 1024) // np.dtype(DTYPE).itemsize
    nbytes = n * np.dtype(DTYPE).itemsize
    k = args.chunks

    with cp.cuda.Device(0):
        pool = cp.cuda.PinnedMemoryPool()
        cp.cuda.set_pinned_memory_allocator(pool.malloc)
        pinned = []
        for _ in range(2 * k):
            mem = cp.cuda.alloc_pinned_memory(nbytes)
            arr = np.frombuffer(mem, dtype=DTYPE, count=n)
            arr[:] = 1.0
            pinned.append((mem, arr))
        dev = [cp.empty(n, dtype=DTYPE) for _ in range(2 * k)]
        sa = cp.cuda.Stream(non_blocking=True)
        sb = cp.cuda.Stream(non_blocking=True)

        def sync():
            sa.synchronize()
            sb.synchronize()
            cp.cuda.get_current_stream().synchronize()

        def h2d_only():
            for i in range(k):
                dev[i].set(pinned[i][1], stream=sa)

        def d2h_only():
            for i in range(k):
                dev[k + i].get(out=pinned[k + i][1], stream=sb, blocking=False)

        def concurrent():
            for i in range(k):
                dev[i].set(pinned[i][1], stream=sa)
                dev[k + i].get(out=pinned[k + i][1], stream=sb, blocking=False)

        def measure(fn):
            for _ in range(2):
                fn()
                sync()
            ts = []
            for _ in range(args.reps):
                t0 = time.perf_counter()
                fn()
                sync()
                ts.append(time.perf_counter() - t0)
            ts.sort()
            return ts[len(ts) // 2]

        t_up = measure(h2d_only)
        t_down = measure(d2h_only)
        t_both = measure(concurrent)

    total = k * nbytes
    print(f"{k} chunks x {args.mib:g} MiB, f64, median of {args.reps}")
    print(f"  h2d only    {t_up*1e3:7.2f} ms   {total/t_up/1e9:5.2f} GB/s")
    print(f"  d2h only    {t_down*1e3:7.2f} ms   {total/t_down/1e9:5.2f} GB/s")
    print(f"  concurrent  {t_both*1e3:7.2f} ms   {2*total/t_both/1e9:5.2f} GB/s "
          f"aggregate")
    print(f"  sum = {(t_up+t_down)*1e3:7.2f} ms,  max = {max(t_up,t_down)*1e3:7.2f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

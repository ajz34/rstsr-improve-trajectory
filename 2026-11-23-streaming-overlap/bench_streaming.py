#!/usr/bin/env python3
"""H2D -> compute -> D2H streaming-overlap benchmark (CuPy reference).

Question 1: does separating host<->device transfers from device computation
cost the full sum of the phases?
Question 2: can a per-chunk multi-stream pipeline hide the transfer time behind
the device computation, so the wall time approaches max(h2d, compute, d2h)?

Modes
  serial_all         all H2D, then all compute, then all D2H (the naive flow)
  serial_chunk       per chunk: H2D, compute, D2H, everything blocking
  pipeline           per chunk: async H2D, compute, async D2H on a rotating
                     stream, over pinned host memory (the streaming flow)
  pipeline_pageable  same, but the host buffers are ordinary pageable numpy
  pipeline_blocking  same, but the D2H call blocks
  pipeline_1stream   same, but every chunk uses one stream

The last three are negative controls: each removes one ingredient and should
collapse back to the serial time.

Total runtime is ~4 s (well under the 20 s budget); --budget-guard aborts if the
measured configuration would exceed it.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

import cupy as cp

DTYPE = np.float64
_C_KERNELS: dict[int, cp.ElementwiseKernel] = {}


def get_kernel(n_fma: int) -> cp.ElementwiseKernel:
    """Dependent FMA chain: y = fma(y, c, 0.5) repeated ``n_fma`` times."""
    kern = _C_KERNELS.get(n_fma)
    if kern is None:
        src = (
            "y = x;\n"
            f"for (int i = 0; i < {n_fma}; i++) {{ y = y * c + 0.5; }}"
        )
        kern = cp.ElementwiseKernel(
            "float64 x, float64 c", "float64 y", src, f"fma_chain_{n_fma}"
        )
        _C_KERNELS[n_fma] = kern
    return kern


class Buffers:
    """One set of K chunks: pinned-in, pinned-out, device-in, device-out."""

    def __init__(self, n_elem: int, n_chunk: int, pageable_host: bool = False):
        self.n_elem = n_elem
        self.n_chunk = n_chunk
        self.nbytes = n_elem * np.dtype(DTYPE).itemsize
        self._pinned = []
        self.h_in, self.h_out, self.d_in, self.d_out = [], [], [], []
        rng = np.random.default_rng(0)
        base = rng.random(n_elem) + 0.5
        for _ in range(n_chunk):
            hin = self._host(pageable_host)
            hout = self._host(pageable_host)
            hin[:] = base
            hout[:] = 0.0
            self.h_in.append(hin)
            self.h_out.append(hout)
            self.d_in.append(cp.empty(n_elem, dtype=DTYPE))
            self.d_out.append(cp.empty(n_elem, dtype=DTYPE))

    def _host(self, pageable: bool):
        if pageable:
            return np.zeros(self.n_elem, dtype=DTYPE)
        mem = cp.cuda.alloc_pinned_memory(self.nbytes)
        self._pinned.append(mem)  # keep the allocation alive
        return np.frombuffer(mem, dtype=DTYPE, count=self.n_elem)


def bus(gb_per_s: float) -> str:
    return f"{gb_per_s:5.2f} GB/s"


class Harness:
    def __init__(self, args):
        self.args = args
        self.n_elem = int(args.chunk_mib * 1024 * 1024) // np.dtype(DTYPE).itemsize
        self.device = cp.cuda.Device(args.device)
        with self.device:
            self.streams = [
                cp.cuda.Stream(non_blocking=True) for _ in range(args.streams)
            ]
        self.buf = Buffers(self.n_elem, args.chunks)
        self.c0 = np.float64(1.0000001)
        self.n_fma = args.fma

    # -- timing -----------------------------------------------------------
    def sync(self) -> None:
        cp.cuda.get_current_stream().synchronize()
        for s in self.streams:
            s.synchronize()

    def measure(self, fn, reps: int, warm: int) -> float:
        for _ in range(warm):
            fn()
            self.sync()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            self.sync()
            ts.append(time.perf_counter() - t0)
        ts.sort()
        return ts[len(ts) // 2]

    # -- phases (used for the isolated per-phase totals) -------------------
    def phase_h2d(self) -> None:
        for i in range(self.buf.n_chunk):
            self.buf.d_in[i].set(self.buf.h_in[i])

    def phase_compute(self) -> None:
        kern = get_kernel(self.n_fma)
        for i in range(self.buf.n_chunk):
            kern(self.buf.d_in[i], self.c0, self.buf.d_out[i])

    def phase_d2h(self) -> None:
        for i in range(self.buf.n_chunk):
            self.buf.d_out[i].get(out=self.buf.h_out[i], blocking=True)

    # -- modes -------------------------------------------------------------
    def serial_all(self) -> None:
        self.phase_h2d()
        self.phase_compute()
        self.phase_d2h()

    def serial_chunk(self) -> None:
        kern = get_kernel(self.n_fma)
        for i in range(self.buf.n_chunk):
            self.buf.d_in[i].set(self.buf.h_in[i])
            kern(self.buf.d_in[i], self.c0, self.buf.d_out[i])
            self.buf.d_out[i].get(out=self.buf.h_out[i], blocking=True)

    def _pipeline(self, streams, blocking_d2h: bool) -> None:
        kern = get_kernel(self.n_fma)
        buf = self.buf
        for i in range(buf.n_chunk):
            s = streams[i % len(streams)] if streams else cp.cuda.get_current_stream()
            buf.d_in[i].set(buf.h_in[i], stream=s)
            kern(buf.d_in[i], self.c0, buf.d_out[i], stream=s)
            buf.d_out[i].get(
                out=buf.h_out[i], stream=s, blocking=blocking_d2h
            )
        for s in streams or ():
            s.synchronize()

    def pipeline(self) -> None:
        self._pipeline(self.streams, blocking_d2h=False)

    def pipeline_blocking(self) -> None:
        self._pipeline(self.streams, blocking_d2h=True)

    def pipeline_1stream(self) -> None:
        self._pipeline([self.streams[0]], blocking_d2h=False)

    # -- calibration -------------------------------------------------------
    def autotune_fma(self, transfer_s: float, margin: float = 1.15,
                     budget_s: float = 5.0) -> int:
        """Pick the FMA-chain length making the compute phase ~margin x transfer.

        The GPU clock drifts between runs (2x on this laptop part), so the
        scaling is verified by measurement and corrected at most twice.
        """
        t0 = time.perf_counter()
        target = margin * transfer_s  # desired compute total over all chunks
        self.n_fma = 32
        for _ in range(3):
            t = self.measure(self.phase_compute, reps=3, warm=1)
            if 1.02 * transfer_s <= t <= 1.35 * transfer_s:
                break
            self.n_fma = max(1, int(round(self.n_fma * target / max(t, 1e-9))))
            if time.perf_counter() - t0 > budget_s:
                break
        return self.n_fma

    def compute_gflops(self, per_chunk_s: float) -> float:
        flop = 2.0 * self.n_elem * self.n_fma
        return flop / per_chunk_s / 1e9


def timeline(harness: Harness) -> list[tuple[int, int, float, float, float, float]]:
    """Replay the pipeline once with events.

    Returns per chunk: (chunk, stream, t_h2d_start, t_h2d_end, t_comp_end,
    t_d2h_end) in ms relative to the first enqueue. A bar spans the window in
    which the stream considers the phase to be running; for H2D that window
    includes waiting for the copy engine, so a deep queue lengthens it.
    """
    buf, kern = harness.buf, get_kernel(harness.n_fma)
    t0 = cp.cuda.Event()
    ev = [[cp.cuda.Event() for _ in range(4)] for _ in range(buf.n_chunk)]
    t0.record(cp.cuda.get_current_stream())
    for i in range(buf.n_chunk):
        s = harness.streams[i % len(harness.streams)]
        ev[i][0].record(s)
        buf.d_in[i].set(buf.h_in[i], stream=s)
        ev[i][1].record(s)
        kern(buf.d_in[i], harness.c0, buf.d_out[i], stream=s)
        ev[i][2].record(s)
        buf.d_out[i].get(out=buf.h_out[i], stream=s, blocking=False)
        ev[i][3].record(s)
    for s in harness.streams:
        s.synchronize()
    rows = []
    for i in range(buf.n_chunk):
        t = [cp.cuda.runtime.eventElapsedTime(t0.ptr, ev[i][j].ptr)
             for j in range(4)]
        rows.append((i, i % len(harness.streams), t[0], t[1], t[2], t[3]))
    return rows


def ascii_gantt(rows, n_stream: int, width: int = 96) -> str:
    span = max(r[5] for r in rows)
    col_ms = span / width
    grid = [[" "] * width for _ in range(n_stream)]
    for _, s, h0, h1, c1, d1 in rows:
        for a, b, ch in ((h0, h1, "h"), (h1, c1, "c"), (c1, d1, "d")):
            for col in range(int(a / col_ms), min(int(np.ceil(b / col_ms)), width)):
                grid[s][col] = ch
    lines = [f"  0 ms{' ' * max(1, width - 12)}{span:6.1f} ms"]
    for s, row in enumerate(grid):
        lines.append(f"  s{s} |{''.join(row)}|")
    lines.append(f"      h = H2D   c = compute   d = D2H   "
                 f"(1 char ~ {col_ms:.2f} ms)")
    return "\n".join(lines)


def round_robin(harness: "Harness", probes, reps: int, warm: int):
    """Run every probe once per round and take a per-probe median.

    The GPU clock on this laptop part moves ~2x within seconds, so timing each
    mode in its own back-to-back block mixes the modal state with the mode;
    interleaving keeps every probe's samples spread over the same states.
    """
    for _ in range(warm):
        for _, fn in probes:
            fn()
            harness.sync()
    samples = {name: [] for name, _ in probes}
    for _ in range(reps):
        for name, fn in probes:
            t0 = time.perf_counter()
            fn()
            harness.sync()
            samples[name].append(time.perf_counter() - t0)
    return {k: float(np.median(v)) for k, v in samples.items()}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="H2D/compute/D2H streaming-overlap benchmark (CuPy)"
    )
    ap.add_argument("--chunk-mib", type=float, default=8.0,
                    help="payload size per chunk, MiB (default 8)")
    ap.add_argument("--chunks", type=int, default=32,
                    help="number of chunks (default 32)")
    ap.add_argument("--streams", type=int, default=4,
                    help="streams used by the pipeline mode (default 4)")
    ap.add_argument("--reps", type=int, default=7, help="timed rounds (default 7)")
    ap.add_argument("--warmup", type=int, default=1, help="warmup rounds (default 1)")
    ap.add_argument("--fma", type=int, default=0,
                    help="FMA chain length; 0 = autotune to balance the phases")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--no-timeline", action="store_true")
    ap.add_argument("--budget-guard", type=float, default=20.0,
                    help="abort before running if the estimate exceeds this (s)")
    args = ap.parse_args()

    t_start = time.perf_counter()
    with cp.cuda.Device(args.device):
        cp.cuda.Device(args.device).use()
        pool = cp.cuda.PinnedMemoryPool()
        cp.cuda.set_pinned_memory_allocator(pool.malloc)

        harness = Harness(args)
        buf = harness.buf
        total_mib = args.chunk_mib * args.chunks
        print(
            f"device   : {cp.cuda.runtime.getDeviceProperties(args.device)['name'].decode()}"
        )
        print(
            f"config   : {args.chunks} chunks x {args.chunk_mib:g} MiB "
            f"= {total_mib:g} MiB per direction, {args.streams} streams, "
            f"{args.reps} timed rounds"
        )

        # Sustained load first: this GPU idles at 630 MHz and the f64 rate moves
        # ~2x with the clock, so nothing is measured before the clocks settle.
        burn = cp.ones(4 * 1024 * 1024, dtype=DTYPE)
        for _ in range(400):
            burn += 1.0
        burn.get()
        harness.sync()

        # --- balance the phases: compute ~= 1.15 x (h2d + d2h) -------------
        harness.n_fma = 8
        t_h2d0 = harness.measure(harness.phase_h2d, reps=3, warm=1)
        t_d2h0 = harness.measure(harness.phase_d2h, reps=3, warm=1)
        if args.fma > 0:
            harness.n_fma = args.fma
        else:
            harness.autotune_fma(t_h2d0 + t_d2h0, budget_s=5.0)

        # --- all probes, interleaved ---------------------------------------
        pageable = Buffers(buf.n_elem, buf.n_chunk, pageable_host=True)

        def with_pageable(fn):
            def run():
                saved = harness.buf
                harness.buf = pageable
                try:
                    fn()
                finally:
                    harness.buf = saved
            return run

        probes = [
            ("h2d", harness.phase_h2d),
            ("compute", harness.phase_compute),
            ("d2h", harness.phase_d2h),
            ("serial_all", harness.serial_all),
            ("serial_chunk", harness.serial_chunk),
            ("pipeline", harness.pipeline),
            ("pipeline_pageable", with_pageable(harness.pipeline)),
            ("pipeline_blocking", harness.pipeline_blocking),
            ("pipeline_1stream", harness.pipeline_1stream),
        ]
        est = (args.reps + args.warmup) * 0.6 * sum(
            (3 if n in ("h2d", "d2h") else 40 if n == "compute" else 80)
            for n, _ in probes
        ) / 1e3
        if est > args.budget_guard:
            print(f"! estimated runtime {est:.1f} s exceeds the "
                  f"{args.budget_guard:.0f} s guard; reduce --reps/--chunks",
                  file=sys.stderr)
            return 2
        res = round_robin(harness, probes, args.reps, args.warmup)

        t_h2d, t_comp, t_d2h = res["h2d"], res["compute"], res["d2h"]
        pred_serial = t_h2d + t_comp + t_d2h
        pred_ideal = max(t_h2d, t_comp, t_d2h)
        pred_bound = max(t_comp, t_h2d + t_d2h)  # one shared copy engine

        # --- the pipeline must move every chunk intact ---------------------
        harness.serial_all()
        ref = [h.copy() for h in buf.h_out]
        harness.pipeline()
        ok = all(np.array_equal(a, b) for a, b in zip(ref, buf.h_out))
        assert ok, "pipeline output differs from the serial-all output"

        # --- report ---------------------------------------------------------
        nbytes_total = total_mib * 1024 * 1024
        print()
        print("phase totals (isolated, interleaved median)")
        print(f"  h2d      {t_h2d*1e3:8.2f} ms   {bus(nbytes_total / t_h2d / 1e9)}")
        print(f"  compute  {t_comp*1e3:8.2f} ms   "
              f"{harness.compute_gflops(t_comp / args.chunks):6.1f} GFLOP/s "
              f"(f64 FMA chain, {harness.n_fma} FMA/elem)")
        print(f"  d2h      {t_d2h*1e3:8.2f} ms   {bus(nbytes_total / t_d2h / 1e9)}")
        print()
        print(f"predicted serial   {pred_serial*1e3:8.2f} ms  (h2d + compute + d2h)")
        print(f"predicted pipeline {pred_bound*1e3:8.2f} ms  "
              f"(max of compute and h2d+d2h)")
        print(f"                   {pred_ideal*1e3:8.2f} ms  "
              f"(if the two copy directions overlapped fully)")
        print()
        hdr = (f"{'mode':<20s}{'ms':>9s}{'vs serial_all':>15s}"
               f"{'vs serial_chunk':>17s}{'pipeline eff':>14s}")
        print(hdr)
        print("-" * len(hdr))
        base_all = res["serial_all"]
        base_chunk = res["serial_chunk"]
        for name, _ in probes:
            t = res[name]
            eff = pred_bound / t if name.startswith("pipeline") else float("nan")
            eff_s = f"{eff*100:6.1f}%" if name.startswith("pipeline") else "-"
            print(f"{name:<20s}{t*1e3:9.2f}{base_all/t:14.2f}x"
                  f"{base_chunk/t:16.2f}x{eff_s:>14s}")
        print()
        hidden = 1.0 - (res["pipeline"] - t_comp) / (t_h2d + t_d2h)
        print(f"transfer time hidden by the pipeline: {hidden*100:5.1f}% of "
              f"h2d+d2h")
        print(f"speedup, naive separation -> pipeline: "
              f"{base_all/res['pipeline']:.2f}x")

        if not args.no_timeline:
            rows = timeline(harness)
            print()
            print("pipeline timeline (one replay, GPU timestamps)")
            print(ascii_gantt(rows, len(harness.streams)))

        print()
        print(f"result match (serial vs pipeline): {'OK' if ok else 'MISMATCH'}")
        print(f"wall time: {time.perf_counter() - t_start:.2f} s")

        harness.buf = buf
        del pageable, harness, buf
        cp.get_default_memory_pool().free_all_blocks()
        cp.cuda.set_pinned_memory_allocator(None)
        pool.free_all_blocks()
    return 0


if __name__ == "__main__":
    sys.exit(main())

//! perf profiling entry points for T2' (plan §3.8 / D11).
//!
//! Each subcommand runs ONE reduction for a FIXED number of iterations with
//! black_box on inputs and outputs, so `perf stat -d` can wrap the whole
//! process. Serial device (DeviceCpuSerial) — single-threaded profiles, run
//! once per RUSTFLAGS config (portable + native).
//!
//! Usage: profile_ops <sum_axis0|sum_axis1|sum_all|min_axis0>
//!
//! Iteration counts target ~1 s per run at 9950X3D speeds in both configs;
//! the binary prints wall time + nominal bytes moved so achieved GB/s can be
//! derived from perf's task-clock. Page-faults/iter from perf stat is the
//! allocation-rider sanity row (T7: expected ~0 for these small outputs).

use std::hint::black_box;
use std::time::Instant;

use reductions::gen_mat_f64;
use rstsr_core::prelude::*;

fn main() {
    let op = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: profile_ops <sum_axis0|sum_axis1|sum_all|min_axis0>");
        std::process::exit(2);
    });
    eprintln!("[profile_ops] op={op} device=DeviceCpuSerial config-agnostic");

    let device = DeviceCpuSerial::default();
    let (m, n) = (2048usize, 2048usize);

    let (elapsed, iters, bytes) = match op.as_str() {
        "sum_axis0" => {
            const ITERS: usize = 600;
            let a = gen_mat_f64!(m, n, 1, &device);
            let _ = a.sum_axes(0); // warmup
            let t0 = Instant::now();
            for _ in 0..ITERS {
                black_box(black_box(&a).sum_axes(0));
            }
            (t0.elapsed(), ITERS, ITERS as f64 * (m * n) as f64 * 8.0)
        },
        "sum_axis1" => {
            const ITERS: usize = 1500;
            let a = gen_mat_f64!(m, n, 1, &device);
            let _ = a.sum_axes(-1); // warmup
            let t0 = Instant::now();
            for _ in 0..ITERS {
                black_box(black_box(&a).sum_axes(-1));
            }
            (t0.elapsed(), ITERS, ITERS as f64 * (m * n) as f64 * 8.0)
        },
        "sum_all" => {
            const ITERS: usize = 1500;
            let a = gen_mat_f64!(m, n, 1, &device);
            let _ = a.sum_all(); // warmup
            let t0 = Instant::now();
            for _ in 0..ITERS {
                black_box(black_box(&a).sum_all());
            }
            (t0.elapsed(), ITERS, ITERS as f64 * (m * n) as f64 * 8.0)
        },
        "min_axis0" => {
            const ITERS: usize = 600;
            let a = gen_mat_f64!(m, n, 1, &device);
            let _ = a.min_axes(0); // warmup
            let t0 = Instant::now();
            for _ in 0..ITERS {
                black_box(black_box(&a).min_axes(0));
            }
            (t0.elapsed(), ITERS, ITERS as f64 * (m * n) as f64 * 8.0)
        },
        "min_all_1e7" => {
            const ITERS: usize = 300;
            let nn = 10_000_000usize;
            let v = reductions::gen_vec_f64(nn, 1);
            let a = rt::asarray((v, [nn], &device));
            let _ = a.min_all(); // warmup
            let t0 = Instant::now();
            for _ in 0..ITERS {
                black_box(black_box(&a).min_all());
            }
            (t0.elapsed(), ITERS, ITERS as f64 * nn as f64 * 8.0)
        },
        other => {
            eprintln!("unknown op: {other}");
            std::process::exit(2);
        },
    };

    // printed AFTER the timed section; perf attributes it to the tail
    eprintln!(
        "[profile_ops] done: iters={iters} elapsed={:.4?} s, bytes_moved={:.3e}, nominal GB/s={:.1}",
        elapsed.as_secs_f64(),
        bytes,
        bytes / elapsed.as_secs_f64() / 1e9
    );
}

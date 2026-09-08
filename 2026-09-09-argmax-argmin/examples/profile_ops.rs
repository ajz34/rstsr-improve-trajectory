//! perf profiling entry points (plan §3.8 / D11) for T6.
//!
//! Each subcommand runs ONE op for a FIXED number of iterations with
//! black_box on inputs and outputs, so `perf stat -d` can wrap the whole
//! process. Serial device (DeviceCpuSerial) — single-threaded profiles.
//!
//! Usage: profile_ops <op>
//!   ops: argmax | argmin | argmax_1e6
//!
//! Iteration counts are fixed targeting ~1-3 s per run at 9950X3D speeds; the
//! binary prints wall time + nominal bytes read so achieved GB/s can be
//! derived from perf's task-clock as a cross-check.

use std::hint::black_box;
use std::time::Instant;

use bench_argmax_argmin::gen_vec_f64;
use rstsr_core::prelude::*;

fn main() {
    let op = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: profile_ops <argmax|argmin|argmax_1e6>");
        std::process::exit(2);
    });
    eprintln!("[profile_ops] op={op} device=DeviceCpuSerial");

    let device = DeviceCpuSerial::default();

    let (elapsed, bytes) = match op.as_str() {
        "argmax" => {
            const ITERS: usize = 150;
            let nn = 10_000_000usize;
            let a = rt::asarray((gen_vec_f64(nn, 1), [nn], &device));
            let _ = rt::argmax(&a); // warmup
            let t0 = Instant::now();
            for _ in 0..ITERS {
                black_box(rt::argmax(black_box(&a)));
            }
            (t0.elapsed(), ITERS as f64 * nn as f64 * 8.0)
        },
        "argmin" => {
            const ITERS: usize = 150;
            let nn = 10_000_000usize;
            let a = rt::asarray((gen_vec_f64(nn, 1), [nn], &device));
            let _ = rt::argmin(&a); // warmup
            let t0 = Instant::now();
            for _ in 0..ITERS {
                black_box(rt::argmin(black_box(&a)));
            }
            (t0.elapsed(), ITERS as f64 * nn as f64 * 8.0)
        },
        "argmax_1e6" => {
            const ITERS: usize = 800;
            let nn = 1_000_000usize;
            let a = rt::asarray((gen_vec_f64(nn, 1), [nn], &device));
            let _ = rt::argmax(&a); // warmup
            let t0 = Instant::now();
            for _ in 0..ITERS {
                black_box(rt::argmax(black_box(&a)));
            }
            (t0.elapsed(), ITERS as f64 * nn as f64 * 8.0)
        },
        other => {
            eprintln!("unknown op: {other}");
            std::process::exit(2);
        },
    };

    // printed AFTER the timed section; perf attributes it to the tail
    eprintln!(
        "[profile_ops] done: elapsed={:.4?} s, bytes_read={:.3e}, nominal GB/s={:.1}",
        elapsed.as_secs_f64(),
        bytes,
        bytes / elapsed.as_secs_f64() / 1e9
    );
}

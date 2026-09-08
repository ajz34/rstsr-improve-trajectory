//! perf profiling entry points (plan §3.8 / D11).
//!
//! Each subcommand runs ONE op for a FIXED number of iterations with
//! black_box on inputs and outputs, so `perf stat -d` can wrap the whole
//! process. Serial device (DeviceCpuSerial) — single-threaded profiles.
//!
//! Usage: profile_ops <op>
//!   ops: triad | transpose | sum_axis0 | sum_axis1 | vecdot | add_contig
//!        | argmax | zeros
//!
//! Iteration counts are fixed per op (see ITERS) targeting ~1-3 s per run at
//! 9950X3D speeds; the binary prints wall time + nominal bytes moved so
//! achieved GB/s can be derived from perf's task-clock as a cross-check.

use std::hint::black_box;
use std::time::Instant;

use bench_harness_baseline::gen_vec_f64;
use bench_harness_baseline::gen_mat_f64;
use rstsr_core::prelude::*;

fn main() {
    let op = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: profile_ops <triad|transpose|sum_axis0|sum_axis1|vecdot|add_contig|argmax|zeros>");
        std::process::exit(2);
    });
    eprintln!("[profile_ops] op={op} device=DeviceCpuSerial");

    let device = DeviceCpuSerial::default();
    let (m, n) = (2048usize, 2048usize);

    let (elapsed, bytes) = match op.as_str() {
        "triad" => {
            const ITERS: usize = 300;
            let nn = 10_000_000usize;
            let a = gen_vec_f64(nn, 1);
            let b = gen_vec_f64(nn, 2);
            let mut c = vec![0.0_f64; nn];
            // warmup
            for i in 0..nn {
                c[i] = a[i] + 3.0 * b[i];
            }
            let t0 = Instant::now();
            for _ in 0..ITERS {
                let a = black_box(&a);
                let b = black_box(&b);
                let c = black_box(&mut c);
                for (c_i, (a_i, b_i)) in c.iter_mut().zip(a.iter().zip(b.iter())) {
                    *c_i = a_i + 3.0 * b_i;
                }
                black_box(&c);
            }
            (t0.elapsed(), ITERS as f64 * 3.0 * nn as f64 * 8.0)
        },
        "transpose" => {
            const ITERS: usize = 40;
            let a = gen_mat_f64!(m, n, 1, &device);
            let _ = a.t().to_contig(RowMajor); // warmup
            let t0 = Instant::now();
            for _ in 0..ITERS {
                black_box(black_box(&a).t().to_contig(RowMajor));
            }
            (t0.elapsed(), ITERS as f64 * 2.0 * (m * n) as f64 * 8.0)
        },
        "sum_axis0" => {
            const ITERS: usize = 80;
            let a = gen_mat_f64!(m, n, 1, &device);
            let _ = a.sum_axes(0);
            let t0 = Instant::now();
            for _ in 0..ITERS {
                black_box(a.sum_axes(0));
            }
            (t0.elapsed(), ITERS as f64 * (m * n) as f64 * 8.0)
        },
        "sum_axis1" => {
            const ITERS: usize = 300;
            let a = gen_mat_f64!(m, n, 1, &device);
            let _ = a.sum_axes(-1);
            let t0 = Instant::now();
            for _ in 0..ITERS {
                black_box(a.sum_axes(-1));
            }
            (t0.elapsed(), ITERS as f64 * (m * n) as f64 * 8.0)
        },
        "vecdot" => {
            const ITERS: usize = 100;
            let nn = 10_000_000usize;
            let a = rt::asarray((gen_vec_f64(nn, 1), [nn], &device));
            let b = rt::asarray((gen_vec_f64(nn, 2), [nn], &device));
            let _ = rt::vecdot(&a, &b, None);
            let t0 = Instant::now();
            for _ in 0..ITERS {
                black_box(rt::vecdot(black_box(&a), black_box(&b), None));
            }
            (t0.elapsed(), ITERS as f64 * 2.0 * nn as f64 * 8.0)
        },
        "add_contig" => {
            const ITERS: usize = 100;
            let a = gen_mat_f64!(m, n, 1, &device);
            let b = gen_mat_f64!(m, n, 2, &device);
            let _ = &a + &b;
            let t0 = Instant::now();
            for _ in 0..ITERS {
                black_box(black_box(&a) + black_box(&b));
            }
            (t0.elapsed(), ITERS as f64 * 3.0 * (m * n) as f64 * 8.0)
        },
        "argmax" => {
            const ITERS: usize = 150;
            let nn = 10_000_000usize;
            let a = rt::asarray((gen_vec_f64(nn, 1), [nn], &device));
            let _ = rt::argmax(&a);
            let t0 = Instant::now();
            for _ in 0..ITERS {
                black_box(rt::argmax(black_box(&a)));
            }
            (t0.elapsed(), ITERS as f64 * nn as f64 * 8.0)
        },
        "zeros" => {
            const ITERS: usize = 400;
            let _z: Tensor<f64, _> = rt::zeros(([m, n], &device));
            let _ = _z;
            let t0 = Instant::now();
            for _ in 0..ITERS {
                let t: Tensor<f64, _> = rt::zeros(([m, n], &device));
                black_box(t);
            }
            (t0.elapsed(), ITERS as f64 * (m * n) as f64 * 8.0)
        },
        other => {
            eprintln!("unknown op: {other}");
            std::process::exit(2);
        },
    };

    // printed AFTER the timed section; perf attributes it to the tail
    eprintln!(
        "[profile_ops] done: elapsed={:.4?} s, bytes_moved={:.3e}, nominal GB/s={:.1}",
        elapsed.as_secs_f64(),
        bytes,
        bytes / elapsed.as_secs_f64() / 1e9
    );
}

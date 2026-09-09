//! perf profiling entry points (plan §3.8 / D11) for T3'.
//!
//! Each subcommand runs ONE op for a FIXED number of iterations with
//! black_box on inputs and outputs, so `perf stat -d` can wrap the whole
//! process. `innerdot_faer`/`batched_faer` run on DeviceFaer (rayon, 16
//! threads via RAYON_NUM_THREADS); the rest are DeviceCpuSerial.
//!
//! Usage: profile_ops <op>
//!   ops: batched_am1 | batched_axis0 | batched_strided | dot1d
//!        | innerdot_serial | innerdot_faer | batched_faer
//!
//! Iteration counts target ~1-3 s per run at 9950X3D speeds; the binary
//! prints wall time + nominal bytes moved so achieved GB/s can be derived
//! from perf's task-clock.

use std::hint::black_box;
use std::time::Instant;

use exp_vecdot::{assert_faer_threads, gen_vec_f64};
use rstsr_core::prelude::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let op = args.get(1).cloned().unwrap_or_else(|| {
        eprintln!("usage: profile_ops <batched_am1|batched_axis0|batched_strided|dot1d|innerdot_serial|innerdot_faer|batched_faer>");
        std::process::exit(2);
    });

    let (m, k) = (4096usize, 512usize);

    match op.as_str() {
        // serial row-dot (branch-1): T0's 466 µs case
        "batched_am1" => {
            const ITERS: usize = 2000;
            let dev = DeviceCpuSerial::default();
            eprintln!("[profile_ops] batched_am1 (4096,512) ax-1 serial, {ITERS} iters");
            let a = rt::asarray((gen_vec_f64(m * k, 1), [m, k], &dev));
            let b = rt::asarray((gen_vec_f64(m * k, 2), [m, k], &dev));
            let _ = rt::vecdot(&a, &b, None);
            let t0 = Instant::now();
            for _ in 0..ITERS {
                black_box(rt::vecdot(black_box(&a), black_box(&b), None));
            }
            report(t0.elapsed().as_secs_f64(), ITERS, 2 * m * k * 8);
        },
        // serial column-accumulation (branch-2 RMW)
        "batched_axis0" => {
            const ITERS: usize = 400;
            let dev = DeviceCpuSerial::default();
            eprintln!("[profile_ops] batched_axis0 (512,4096) ax0 serial, {ITERS} iters");
            let a = rt::asarray((gen_vec_f64(m * k, 1), [k, m], &dev));
            let b = rt::asarray((gen_vec_f64(m * k, 2), [k, m], &dev));
            let _ = rt::vecdot(&a, &b, 0);
            let t0 = Instant::now();
            for _ in 0..ITERS {
                black_box(rt::vecdot(black_box(&a), black_box(&b), 0));
            }
            report(t0.elapsed().as_secs_f64(), ITERS, 2 * m * k * 8);
        },
        // serial strided (general branch)
        "batched_strided" => {
            const ITERS: usize = 300;
            let dev = DeviceCpuSerial::default();
            eprintln!("[profile_ops] batched_strided (4096,512).(512,4096).t() serial, {ITERS} iters");
            let a = rt::asarray((gen_vec_f64(m * k, 1), [m, k], &dev));
            let bst = rt::asarray((gen_vec_f64(m * k, 2), [k, m], &dev));
            let bv = bst.t();
            let _ = rt::vecdot(&a, &bv, None);
            let t0 = Instant::now();
            for _ in 0..ITERS {
                black_box(rt::vecdot(black_box(&a), black_box(&bv), None));
            }
            report(t0.elapsed().as_secs_f64(), ITERS, 2 * m * k * 8);
        },
        // serial 1-D dot 1e7 (gate)
        "dot1d" => {
            const ITERS: usize = 500;
            let dev = DeviceCpuSerial::default();
            eprintln!("[profile_ops] dot1d 1e7 serial, {ITERS} iters");
            let nn = 10_000_000usize;
            let a = rt::asarray((gen_vec_f64(nn, 1), [nn], &dev));
            let b = rt::asarray((gen_vec_f64(nn, 2), [nn], &dev));
            let _ = rt::vecdot(&a, &b, None);
            let t0 = Instant::now();
            for _ in 0..ITERS {
                black_box(rt::vecdot(black_box(&a), black_box(&b), None));
            }
            report(t0.elapsed().as_secs_f64(), ITERS, 2 * nn * 8);
        },
        // serial inner_dot (1-D % 1-D) 1e7
        "innerdot_serial" => {
            const ITERS: usize = 300;
            let dev = DeviceCpuSerial::default();
            eprintln!("[profile_ops] innerdot 1e7 serial (1-D % 1-D), {ITERS} iters");
            let nn = 10_000_000usize;
            let a = rt::asarray((gen_vec_f64(nn, 1), [nn], &dev));
            let b = rt::asarray((gen_vec_f64(nn, 2), [nn], &dev));
            let _ = &a % &b;
            let t0 = Instant::now();
            for _ in 0..ITERS {
                black_box(black_box(&a) % black_box(&b));
            }
            report(t0.elapsed().as_secs_f64(), ITERS, 2 * nn * 8);
        },
        // faer16 inner_dot (inner_dot_naive_cpu_rayon) 1e7
        "innerdot_faer" => {
            const ITERS: usize = 1000;
            let dev = DeviceFaer::new(0);
            assert_faer_threads(&dev, 16);
            eprintln!("[profile_ops] innerdot 1e7 faer16 (inner_dot_naive_cpu_rayon), {ITERS} iters");
            let nn = 10_000_000usize;
            let a = rt::asarray((gen_vec_f64(nn, 1), [nn], &dev));
            let b = rt::asarray((gen_vec_f64(nn, 2), [nn], &dev));
            let _ = &a % &b;
            let t0 = Instant::now();
            for _ in 0..ITERS {
                black_box(black_box(&a) % black_box(&b));
            }
            report(t0.elapsed().as_secs_f64(), ITERS, 2 * nn * 8);
        },
        // faer16 batched row-dot (the 148 µs case — must not regress)
        "batched_faer" => {
            const ITERS: usize = 5000;
            let dev = DeviceFaer::new(0);
            assert_faer_threads(&dev, 16);
            eprintln!("[profile_ops] batched_am1 (4096,512) ax-1 faer16, {ITERS} iters");
            let a = rt::asarray((gen_vec_f64(m * k, 1), [m, k], &dev));
            let b = rt::asarray((gen_vec_f64(m * k, 2), [m, k], &dev));
            let _ = rt::vecdot(&a, &b, None);
            let t0 = Instant::now();
            for _ in 0..ITERS {
                black_box(rt::vecdot(black_box(&a), black_box(&b), None));
            }
            report(t0.elapsed().as_secs_f64(), ITERS, 2 * m * k * 8);
        },
        other => {
            eprintln!("unknown op: {other}");
            std::process::exit(2);
        },
    }
}

fn report(elapsed: f64, iters: usize, bytes_per_iter: usize) {
    let per_iter = elapsed / iters as f64;
    let us = per_iter * 1e6;
    let gbps = bytes_per_iter as f64 / per_iter / 1e9;
    println!("profile_ops: {iters} iters, {elapsed:.3} s total, {us:.1} µs/iter, {gbps:.1} GB/s nominal");
}

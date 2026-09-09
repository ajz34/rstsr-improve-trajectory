//! Fixed-iteration op runner for `perf stat` (T1' perf stage).
//!
//! Modes (argv[1]):
//! - `ta_serial`: variant A (allocating idiom) 2048x2048 f64, DeviceCpuSerial
//! - `tb_serial`: variant B (`c.assign(&a.t())` reuse) 2048x2048 f64, serial
//! - `tb_faer16`: variant B 2048x2048 f64, DeviceFaer (RAYON_NUM_THREADS=16)
//!
//! The B variants are the perf baseline the campaign beats (T7: judge
//! kernels on reuse denominators); `ta_serial` reproduces the T0/T7 A-row
//! for context. Iteration counts are chosen for ~0.5-1 s of work so perf
//! counters average over enough iterations without magnifying startup.

use std::hint::black_box;
use std::time::Instant;

use transpose_assign::{assert_faer_threads, faer_device, gen_mat_f64, serial_device, warm_output_mat};
use rstsr_core::prelude::*;

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "tb_serial".to_string());
    let (m, n) = (2048usize, 2048usize);
    let bytes = (2 * m * n * std::mem::size_of::<f64>()) as f64; // read + write

    match mode.as_str() {
        "ta_serial" => {
            let dev = serial_device();
            let a = gen_mat_f64!(m, n, 1, &dev);
            let iters = 30;
            let mut sink = 0.0f64;
            let t0 = Instant::now();
            for _ in 0..iters {
                let at = black_box(&a).t();
                let out = at.to_contig(RowMajor).into_owned();
                sink += out[[0, 0]] + out[[n - 1, m - 1]];
                black_box(out);
            }
            let dt = t0.elapsed();
            black_box(sink);
            report("ta_serial", iters, dt, bytes);
        },
        "tb_serial" => {
            let dev = serial_device();
            let a = gen_mat_f64!(m, n, 1, &dev);
            let mut c = warm_output_mat!(n, m, &dev);
            let iters = 40;
            let t0 = Instant::now();
            for _ in 0..iters {
                c.assign(&black_box(&a).t());
                black_box(&c);
            }
            let dt = t0.elapsed();
            report("tb_serial", iters, dt, bytes);
        },
        "tb_faer16" => {
            let dev = faer_device();
            assert_faer_threads(&dev, 16);
            let a = gen_mat_f64!(m, n, 1, &dev);
            let mut c = warm_output_mat!(n, m, &dev);
            let iters = 400;
            let t0 = Instant::now();
            for _ in 0..iters {
                c.assign(&black_box(&a).t());
                black_box(&c);
            }
            let dt = t0.elapsed();
            report("tb_faer16", iters, dt, bytes);
        },
        other => {
            eprintln!("unknown mode: {other} (use ta_serial | tb_serial | tb_faer16)");
            std::process::exit(2);
        },
    }
}

fn report(tag: &str, iters: usize, dt: std::time::Duration, bytes: f64) {
    let per_iter = dt.as_secs_f64() / iters as f64;
    println!(
        "{tag}: {iters} iters in {:.3} s -> {:.3} ms/iter, {:.1} GB/s (2x tensor bytes)",
        dt.as_secs_f64(),
        per_iter * 1e3,
        bytes / per_iter / 1e9
    );
}

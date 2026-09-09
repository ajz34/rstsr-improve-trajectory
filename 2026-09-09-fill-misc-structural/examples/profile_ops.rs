//! T5 fixed-iteration op runner for `perf stat` (pattern from T0/T7).
//!
//! Modes (all 2048x2048 f64, DeviceCpuSerial unless noted; --iters N):
//! - full_a       : `rt::full` allocating — expect the T7 fault rider
//!                  (~8193 minor faults/iter, sys > user) + `vec![fill; len]`.
//! - ones_a       : same class as full_a (ones_impl).
//! - zeros_c      : calloc-lazy control (2-3 µs, ~1 fault).
//! - fill_b       : `c.fill(v)` reuse — kernel-only (`fill_promote_cpu_serial`).
//! - fill_b_faer  : same on DeviceFaer (parallel twin).
//! - fill_bound   : raw slice broadcast loop (write-only floor).
//! - fill_bt      : device-level diag-layout fill (eye path, strided branch).
//!
//! Usage: cargo run --release --example profile_ops -- <mode> [iters]

use std::hint::black_box;

use fill_misc_structural::{assert_faer_threads, faer_device, serial_device, FILL_VALUE_F64};
use rstsr_core::operators::assignment::OpAssignAPI;
use rstsr_core::prelude::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).cloned().unwrap_or_else(|| "full_a".into());
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(200);

    let (m, n) = (2048usize, 2048usize);
    match mode.as_str() {
        "fill_b_faer" => {
            let dev = faer_device();
            assert_faer_threads(&dev, 16);
            let mut c: Tensor<f64, _> = rt::zeros(([m, n], &dev));
            c.fill(FILL_VALUE_F64);
            for _ in 0..iters {
                c.fill(FILL_VALUE_F64);
                black_box(&c);
            }
        }
        "fill_b" | "fill_bound" | "fill_bt" | "zeros_c" | "zeros_med" | "zeros_odd" | "full_med" | "full_odd" => {
            let dev = serial_device();
            match mode.as_str() {
                "fill_b" => {
                    let mut c: Tensor<f64, _> = rt::zeros(([m, n], &dev));
                    c.fill(FILL_VALUE_F64);
                    for _ in 0..iters {
                        c.fill(FILL_VALUE_F64);
                        black_box(&c);
                    }
                }
                "fill_bound" => {
                    let mut v = vec![FILL_VALUE_F64; m * n];
                    for _ in 0..iters {
                        v.iter_mut().for_each(|x| *x = FILL_VALUE_F64);
                        black_box(&v);
                    }
                }
                "fill_bt" => {
                    let k = 2048usize;
                    let mut v = vec![0.0f64; k * k + 1];
                    let layout = Layout::new([k], [(k + 1) as isize], 0).unwrap();
                    for _ in 0..iters {
                        dev.fill(&mut v, &layout, FILL_VALUE_F64).unwrap();
                        black_box(&v);
                    }
                }
                "zeros_c" => {
                    for _ in 0..iters {
                        let t: Tensor<f64, _> = rt::zeros(([m, n], &dev));
                        black_box(t);
                    }
                }
                "zeros_med" => {
                    for _ in 0..iters {
                        let t: Tensor<f64, _> = rt::zeros(([512, 512], &dev));
                        black_box(t);
                    }
                }
                "zeros_odd" => {
                    for _ in 0..iters {
                        let t: Tensor<f64, _> = rt::zeros(([1000, 777], &dev));
                        black_box(t);
                    }
                }
                "full_med" => {
                    for _ in 0..iters {
                        let t: Tensor<f64, _> = rt::full(([512, 512], FILL_VALUE_F64, &dev));
                        black_box(t);
                    }
                }
                "full_odd" => {
                    for _ in 0..iters {
                        let t: Tensor<f64, _> = rt::full(([1000, 777], FILL_VALUE_F64, &dev));
                        black_box(t);
                    }
                }
                _ => unreachable!(),
            }
        }
        "full_a" | "ones_a" => {
            let dev = serial_device();
            for _ in 0..iters {
                match mode.as_str() {
                    "full_a" => {
                        let t: Tensor<f64, _> = rt::full(([m, n], FILL_VALUE_F64, &dev));
                        black_box(t);
                    }
                    _ => {
                        let t: Tensor<f64, _> = rt::ones(([m, n], &dev));
                        black_box(t);
                    }
                }
            }
        }
        _ => {
            eprintln!("unknown mode {mode}");
            std::process::exit(2);
        }
    }
    eprintln!("profile_ops: mode={mode} iters={iters}");
}

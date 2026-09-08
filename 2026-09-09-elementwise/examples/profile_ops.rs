//! Fixed-iteration op runner for `perf stat -d` (serial device).
//!
//! Cases:
//! - `add_b`          — reuse variant B, add contig  2048x2048 (300 iters)
//! - `add_strided_b`  — reuse variant B, add strided 2048x2048 (100 iters)
//! - `add_a`          — allocating variant A, add contig 2048x2048 (100 iters)
//!
//! Derived metrics come from perf task-clock / instruction counts divided by
//! the fixed iteration counts printed at the end.

use std::hint::black_box;
use std::mem::MaybeUninit;

use elementwise::{gen_mat_f64, serial_device};
use rstsr_core::prelude::*;
use rstsr_core::tensor::operators::op_with_func::op_mutc_refa_refb_func;

fn main() {
    let dev = serial_device();
    let case = std::env::args().nth(1).unwrap_or_else(|| "add_b".to_string());
    let (m, n) = (2048usize, 2048usize);

    match case.as_str() {
        "add_b" => {
            let iters = 300usize;
            let a = gen_mat_f64!(m, n, 1, &dev);
            let b = gen_mat_f64!(m, n, 2, &dev);
            let mut c: Tensor<f64, _> = rt::zeros(([m, n], &dev));
            c.fill(0.0);
            let mut f = |cv: &mut MaybeUninit<f64>, x: &f64, y: &f64| { cv.write(x + y); };
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                op_mutc_refa_refb_func(&mut c, black_box(&a), black_box(&b), &mut f).unwrap();
                black_box(&c);
            }
            let sum: f64 = c.raw().as_slice().iter().take(1024).sum();
            println!("case={case} iters={iters} wall_ms={:.3} guard={sum}", t0.elapsed().as_secs_f64() * 1e3);
        },
        "add_strided_b" => {
            let iters = 100usize;
            let a = gen_mat_f64!(m, n, 1, &dev);
            let bt = gen_mat_f64!(m, n, 4, &dev);
            let mut c: Tensor<f64, _> = rt::zeros(([m, n], &dev));
            c.fill(0.0);
            let mut f = |cv: &mut MaybeUninit<f64>, x: &f64, y: &f64| { cv.write(x + y); };
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                op_mutc_refa_refb_func(&mut c, black_box(&a), black_box(&bt.t()), &mut f).unwrap();
                black_box(&c);
            }
            let sum: f64 = c.raw().as_slice().iter().take(1024).sum();
            println!("case={case} iters={iters} wall_ms={:.3} guard={sum}", t0.elapsed().as_secs_f64() * 1e3);
        },
        "add_a" => {
            let iters = 100usize;
            let a = gen_mat_f64!(m, n, 1, &dev);
            let b = gen_mat_f64!(m, n, 2, &dev);
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                let c = black_box(&a) + black_box(&b);
                black_box(c);
            }
            println!("case={case} iters={iters} wall_ms={:.3} guard=0", t0.elapsed().as_secs_f64() * 1e3);
        },
        other => panic!("unknown case {other}"),
    }
}

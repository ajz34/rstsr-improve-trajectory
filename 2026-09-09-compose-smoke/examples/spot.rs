//! T8 compose-smoke spot benches: fixed-iteration medians for the six
//! headline cases, on the COMBINED (5-patch) tree. Serial device primary;
//! the `%` case runs on the default DeviceFaer (RAYON_NUM_THREADS=16).
//!
//! Usage: `cargo run --release -p compose-smoke --example spot`
//! Run once per RUSTFLAGS config (portable / native builds differ).
//!
//! Individual-candidate reference medians are cited in README.md
//! (spot table); verdict bands: within +-5% holds, 5-15% drifts,
//! >15% regresses (re-run 3x before claiming a regression).

use std::hint::black_box;
use std::time::Instant;

use compose_smoke::{assert_faer_threads, faer_device, gen_vec_f64, serial_device};
use rstsr_core::prelude::*;
use rstsr_core::tensor::operators::op_with_func::op_mutc_refa_refb_func;
use std::mem::MaybeUninit;

/// Median wall time (ms) of `iters` runs after `warmup` untimed runs.
fn median_ms<F: FnMut()>(warmup: usize, iters: usize, mut f: F) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let mut ts: Vec<f64> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        ts.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ts[iters / 2]
}

fn main() {
    println!("=== T8 compose-smoke spot benches (combined tree) ===");
    println!("(times are medians of fixed iterations; black_box on inputs+outputs)");

    // 1. argmax 1e7 f64, serial [T6]
    {
        let dev = serial_device();
        let n = 10_000_000usize;
        let data = gen_vec_f64(n, 1);
        let a = rt::asarray((data, [n], &dev));
        let t = median_ms(5, 40, || {
            black_box(rt::argmax(black_box(&a)));
        });
        println!("RESULT argmax_1e7_serial        {:>10.3} ms", t);
    }

    // 2. strided add 2048^2 REUSE variant, serial [T4']
    {
        let dev = serial_device();
        let (m, n) = (2048usize, 2048usize);
        let a = rt::asarray((gen_vec_f64(m * n, 1), [m, n], &dev));
        let bt = rt::asarray((gen_vec_f64(m * n, 4), [n, m], &dev)); // bt.t() is [m, n]
        let mut c: Tensor<f64, _> = rt::zeros(([m, n], &dev));
        c.fill(0.0);
        let t = median_ms(3, 30, || {
            let mut f = |cv: &mut MaybeUninit<f64>, x: &f64, y: &f64| { cv.write(x + y); };
            op_mutc_refa_refb_func(&mut c, black_box(&a), black_box(&bt.t()), &mut f).unwrap();
            black_box(&c);
        });
        println!("RESULT add_strided_b_2048_serial {:>10.3} ms", t);
    }

    // 3. transpose copy 2048^2 REUSE (c.assign(&a.t())), serial [T1']
    {
        let dev = serial_device();
        let (m, n) = (2048usize, 2048usize);
        let a = rt::asarray((gen_vec_f64(m * n, 1), [m, n], &dev));
        let mut c: Tensor<f64, _> = rt::zeros(([n, m], &dev));
        c.fill(0.0);
        let t = median_ms(3, 30, || {
            c.assign(black_box(&a.t()));
            black_box(&c);
        });
        println!("RESULT transpose_b_2048_serial  {:>10.3} ms", t);
    }

    // 4. sum_axis0 2048^2, serial [T2']
    {
        let dev = serial_device();
        let (m, n) = (2048usize, 2048usize);
        let a = rt::asarray((gen_vec_f64(m * n, 1), [m, n], &dev));
        let t = median_ms(5, 40, || {
            black_box(a.sum_axes(0));
        });
        println!("RESULT sum_axis0_2048_serial    {:>10.3} ms", t);
    }

    // 5. batched vecdot, serial [T3']
    {
        let dev = serial_device();
        // axis 0: (512, 4096)
        {
            let (m, k) = (512usize, 4096usize);
            let ta = rt::asarray((gen_vec_f64(m * k, 11), [m, k], &dev));
            let tb = rt::asarray((gen_vec_f64(m * k, 12), [m, k], &dev));
            let t = median_ms(10, 200, || {
                black_box(rt::vecdot(black_box(&ta), black_box(&tb), 0));
            });
            println!("RESULT vecdot_axis0_512x4096_ser {:>10.4} ms", t);
        }
        // axis -1: (4096, 512)
        {
            let (m, k) = (4096usize, 512usize);
            let ta = rt::asarray((gen_vec_f64(m * k, 11), [m, k], &dev));
            let tb = rt::asarray((gen_vec_f64(m * k, 12), [m, k], &dev));
            let t = median_ms(10, 200, || {
                black_box(rt::vecdot(black_box(&ta), black_box(&tb), None));
            });
            println!("RESULT vecdot_am1_4096x512_ser  {:>10.4} ms", t);
        }
    }

    // 6. `%` 1-D 1e4, default device faer16 [T3']
    {
        let dev = faer_device();
        assert_faer_threads(&dev, 16);
        let n = 10_000usize;
        let a = rt::asarray((gen_vec_f64(n, 1), [n], &dev));
        let b = rt::asarray((gen_vec_f64(n, 2), [n], &dev));
        let t_ms = median_ms(50, 2000, || {
            black_box(black_box(&a) % black_box(&b));
        });
        println!("RESULT innerdot_1e4_faer16      {:>10.3} us", t_ms * 1e3);
    }

    println!("done");
}

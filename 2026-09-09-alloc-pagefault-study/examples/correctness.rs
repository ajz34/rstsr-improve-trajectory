//! Correctness gate (plan §3.2) for the reuse/emulated variants of this
//! study. Every variant benched in benches/reuse.rs must agree with BOTH
//! (a) the allocating rstsr call and (b) a naive scalar reference — including
//! odd size 1000x777, a zero-stride broadcast input, and BOTH devices.
//!
//! Exits non-zero on any mismatch.

use std::hint::black_box;
use std::mem::MaybeUninit;

use alloc_pagefault_study::{assert_faer_threads, faer_device, gen_vec_f64, serial_device};
use rstsr_core::prelude::*;
use rstsr_core::tensor::operators::op_with_func::op_mutc_refa_refb_func;

fn ref_tol(n: usize) -> f64 {
    // generous but bug-catching: summation-order noise vs wrong-kernel garbage
    1e-7 * (1.0 + (n as f64).sqrt())
}

fn check(name: &str, cond: bool) {
    if cond {
        println!("  [ok]   {name}");
    } else {
        eprintln!("  [FAIL] {name}");
        std::process::exit(1);
    }
}

fn allclose(a: &[f64], b: &[f64], tol: f64) -> bool {
    a.len() == b.len()
        && a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= tol * (1.0 + x.abs().max(y.abs())))
}

/// The whole gate, instantiated per device via macro (concrete ops only; no
/// trait-bound gymnastics — same trick as the T0 harness).
macro_rules! check_device {
    ($dev_name:expr, $_device:expr) => {{
        let device = &$_device;
        println!("-- device: {}", $dev_name);

        for (m, n) in [(64usize, 64usize), (1000usize, 777usize)] {
            let a_data = gen_vec_f64(m * n, 1);
            let b_data = gen_vec_f64(m * n, 2);
            let bt_data = gen_vec_f64(n * m, 4); // [n, m] so bt.t() is [m, n]
            let a = rt::asarray((a_data.clone(), [m, n], device));
            let b = rt::asarray((b_data.clone(), [m, n], device));
            let bt = rt::asarray((bt_data.clone(), [n, m], device));

            // --- add: allocating reference --------------------------------
            let want: Vec<f64> = (&a + &b).reshape(-1).to_vec();

            // variant B_reuse_mutc: single pass into existing storage via the
            // public rstsr-core fn op_mutc_refa_refb_func (c <- a + b)
            let mut c: Tensor<f64, _> = rt::zeros(([m, n], device));
            {
                let mut f = |cv: &mut MaybeUninit<f64>, x: &f64, y: &f64| { cv.write(x + y); };
                op_mutc_refa_refb_func(&mut c, &a, &b, &mut f).unwrap();
            }
            check(&format!("add reuse-mutc {m}x{n}"), allclose(&c.reshape(-1).to_vec(), &want, ref_tol(m * n)));

            // variant B_reuse_2pass: assign + add_assign (tensor methods)
            let mut c: Tensor<f64, _> = rt::zeros(([m, n], device));
            c.assign(&a);
            c += &b;
            check(&format!("add reuse-2pass {m}x{n}"), allclose(&c.reshape(-1).to_vec(), &want, ref_tol(m * n)));

            // variant B_bound: emulated single-pass raw loop
            let mut c2 = vec![0.0_f64; m * n];
            for ((cv, av), bv) in c2.iter_mut().zip(a_data.iter()).zip(b_data.iter()) {
                *cv = av + bv;
            }
            check(&format!("add bound {m}x{n}"), allclose(&c2, &want, ref_tol(m * n)));

            // --- strided add (bt.t() operand) ------------------------------
            let wants: Vec<f64> = (&a + &bt.t()).reshape(-1).to_vec();
            let mut c: Tensor<f64, _> = rt::zeros(([m, n], device));
            {
                let mut f = |cv: &mut MaybeUninit<f64>, x: &f64, y: &f64| { cv.write(x + y); };
                op_mutc_refa_refb_func(&mut c, &a, bt.t(), &mut f).unwrap();
            }
            check(&format!("strided add reuse-mutc {m}x{n}"), allclose(&c.reshape(-1).to_vec(), &wants, ref_tol(m * n)));

            let mut c: Tensor<f64, _> = rt::zeros(([m, n], device));
            c.assign(&a);
            c += &bt.t();
            check(&format!("strided add reuse-2pass {m}x{n}"), allclose(&c.reshape(-1).to_vec(), &wants, ref_tol(m * n)));

            let mut c3 = vec![0.0_f64; m * n];
            for i in 0..m {
                for j in 0..n {
                    c3[i * n + j] = a_data[i * n + j] + bt_data[j * m + i];
                }
            }
            check(&format!("strided add bound {m}x{n}"), allclose(&c3, &wants, ref_tol(m * n)));

            // --- transpose copy ---------------------------------------------
            let wantt: Vec<f64> = a.t().to_contig(RowMajor).reshape(-1).to_vec();
            let mut c: Tensor<f64, _> = rt::zeros(([n, m], device));
            c.assign(&a.t());
            check(&format!("transpose reuse-assign {m}x{n}"), allclose(&c.reshape(-1).to_vec(), &wantt, ref_tol(m * n)));

            let mut c4 = vec![0.0_f64; n * m];
            for i in 0..m {
                for j in 0..n {
                    c4[j * m + i] = a_data[i * n + j];
                }
            }
            check(&format!("transpose bound {m}x{n}"), allclose(&c4, &wantt, ref_tol(m * n)));

            // --- sum_axis0 ---------------------------------------------------
            let wsum: Vec<f64> = a.sum_axes(0).reshape(-1).to_vec();
            let mut out = vec![0.0_f64; n];
            for i in 0..m {
                for (o, r) in out.iter_mut().zip(a_data[i * n..(i + 1) * n].iter()) {
                    *o += r;
                }
            }
            check(&format!("sum_axis0 bound {m}x{n}"), allclose(&out, &wsum, ref_tol(m * n)));
        }

        // --- zero-stride broadcast (explicit broadcast_to view) -------------
        {
            let (m, n) = (64usize, 64usize);
            let a_data = gen_vec_f64(m * n, 1);
            let row_data = gen_vec_f64(n, 3);
            let a = rt::asarray((a_data.clone(), [m, n], device));
            let row = rt::asarray((row_data.clone(), [1, n], device));
            let brow = row.broadcast_to(vec![m, n]);
            let want: Vec<f64> = (&a + &brow).reshape(-1).to_vec();
            let mut c: Tensor<f64, _> = rt::zeros(([m, n], device));
            {
                let mut f = |cv: &mut MaybeUninit<f64>, x: &f64, y: &f64| { cv.write(x + y); };
                op_mutc_refa_refb_func(&mut c, &a, &row, &mut f).unwrap();
            }
            check(&format!("broadcast add reuse-mutc"), allclose(&c.reshape(-1).to_vec(), &want, ref_tol(m * n)));
            let mut c: Tensor<f64, _> = rt::zeros(([m, n], device));
            c.assign(&a);
            c += &brow;
            check(&format!("broadcast add reuse-2pass"), allclose(&c.reshape(-1).to_vec(), &want, ref_tol(m * n)));
        }

        // --- batched vecdot --------------------------------------------------
        {
            let (rows, cols) = (4096usize, 512usize);
            let a_data = gen_vec_f64(rows * cols, 1);
            let b_data = gen_vec_f64(rows * cols, 2);
            let a = rt::asarray((a_data.clone(), [rows, cols], device));
            let b = rt::asarray((b_data.clone(), [rows, cols], device));
            let want: Vec<f64> = rt::vecdot(&a, &b, None).reshape(-1).to_vec();
            let mut out = vec![0.0_f64; rows];
            for k in 0..rows {
                let mut acc = 0.0;
                for j in 0..cols {
                    acc += a_data[k * cols + j] * b_data[k * cols + j];
                }
                out[k] = acc;
            }
            check(&format!("batched vecdot bound"), allclose(&out, &want, ref_tol(rows * cols)));
        }

        // --- full/zeros -------------------------------------------------------
        {
            let (m, n) = (128usize, 128usize);
            let f: Tensor<f64, _> = rt::full(([m, n], 3.5, device));
            let mut c: Tensor<f64, _> = rt::zeros(([m, n], device));
            c.fill(3.5);
            check(&format!("full reuse-fill"), allclose(&c.reshape(-1).to_vec(), &f.reshape(-1).to_vec(), ref_tol(m * n)));
            let z: Tensor<f64, _> = rt::zeros(([m, n], device));
            check(&format!("zeros reads as zeros (calloc-lazy, untouched)"), {
                z.reshape(-1).to_vec().iter().all(|&x| x == 0.0)
            });
        }

        // anti-optimization sanity
        {
            let a = rt::asarray((gen_vec_f64(64, 1), [8usize, 8], device));
            let b = rt::asarray((gen_vec_f64(64, 2), [8usize, 8], device));
            let sum: f64 = (&a + &b).reshape(-1).to_vec().iter().sum();
            check(&format!("sanity black_box"), black_box(sum).is_finite());
        }
    }};
}

fn main() {
    println!("correctness gate: alloc-pagefault-study");

    let dev_serial = serial_device();
    check_device!("DeviceCpuSerial", dev_serial);

    let dev_faer = faer_device();
    assert_faer_threads(&dev_faer, 16);
    check_device!("DeviceFaer(16)", dev_faer);

    println!("all correctness checks passed");
}

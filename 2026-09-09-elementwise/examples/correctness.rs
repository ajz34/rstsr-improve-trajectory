//! Correctness gate (plan §3.2): every T4' case against a naive scalar
//! reference (+ ndarray cross-check for add), covering:
//! - odd size 1000x777 and a small 37x53,
//! - explicit zero-stride broadcast view (`brow.broadcast_to([m, n])`) AND
//!   the auto-broadcast `[1, n]` form,
//! - transpose-view strided operand (`bt.t()`),
//! - mul and scale spots,
//! - the reuse variant (op_mutc_refa_refb_func into pre-allocated c),
//! - f64 AND f32,
//! - DeviceCpuSerial AND DeviceFaer (RAYON_NUM_THREADS=16 asserted).
//!
//! Device cases are expanded through a macro (not a generic fn) so rstsr's
//! trait bounds are inherited at the call site (T0 harness lesson).
//! Exits non-zero on any mismatch; run under BOTH RUSTFLAGS configs.

use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicUsize, Ordering};

use num::Complex;

use elementwise::{assert_faer_threads, faer_device, gen_mat_f32, gen_mat_f64, serial_device};
use rstsr_core::prelude::*;
use rstsr_core::tensor::operators::op_with_func::op_mutc_refa_refb_func;

static FAILURES: AtomicUsize = AtomicUsize::new(0);

fn check(cond: bool, what: &str) {
    if cond {
        println!("[PASS] {what}");
    } else {
        println!("[FAIL] {what}");
        FAILURES.fetch_add(1, Ordering::SeqCst);
    }
}

macro_rules! run_device {
    ($dev:expr, $devtag:expr) => {{
        let dev = $dev;
        let devtag: &str = $devtag;

        for (m, n) in [(37usize, 53usize), (1000usize, 777usize)] {
            // ---- f64: contig add / mul, alloc + reuse ----------------------
            let a = gen_mat_f64!(m, n, 1, dev);
            let b = gen_mat_f64!(m, n, 2, dev);
            let (av, bv) = (elementwise::gen_vec_f64(m * n, 1), elementwise::gen_vec_f64(m * n, 2));
            let want_add: Vec<f64> = av.iter().zip(bv.iter()).map(|(x, y)| x + y).collect();
            let want_mul: Vec<f64> = av.iter().zip(bv.iter()).map(|(x, y)| x * y).collect();

            let c = &a + &b;
            check(all_close64(c.raw().as_slice(), &want_add), &format!("add contig {m}x{n} alloc {devtag} f64"));

            let mut cr: Tensor<f64, _> = rt::zeros(([m, n], dev));
            cr.fill(0.0);
            let mut f = |cv: &mut MaybeUninit<f64>, x: &f64, y: &f64| { cv.write(x + y); };
            op_mutc_refa_refb_func(&mut cr, &a, &b, &mut f).unwrap();
            check(all_close64(cr.raw().as_slice(), &want_add), &format!("add contig {m}x{n} reuse {devtag} f64"));

            let c = &a * &b;
            check(all_close64(c.raw().as_slice(), &want_mul), &format!("mul contig {m}x{n} alloc {devtag} f64"));

            let mut cr: Tensor<f64, _> = rt::zeros(([m, n], dev));
            cr.fill(0.0);
            let mut f = |cv: &mut MaybeUninit<f64>, x: &f64, y: &f64| { cv.write(x * y); };
            op_mutc_refa_refb_func(&mut cr, &a, &b, &mut f).unwrap();
            check(all_close64(cr.raw().as_slice(), &want_mul), &format!("mul contig {m}x{n} reuse {devtag} f64"));

            // scale: a * 2.0, plus the reuse-as-broadcast-row form
            let c = &a * 2.0;
            let want_scale: Vec<f64> = av.iter().map(|x| x * 2.0).collect();
            check(all_close64(c.raw().as_slice(), &want_scale), &format!("scale contig {m}x{n} alloc {devtag} f64"));

            let row2: Vec<f64> = vec![2.0; n];
            let row2 = rt::asarray((row2, [1, n], dev));
            let mut cr: Tensor<f64, _> = rt::zeros(([m, n], dev));
            cr.fill(0.0);
            let mut f = |cv: &mut MaybeUninit<f64>, x: &f64, y: &f64| { cv.write(x * y); };
            op_mutc_refa_refb_func(&mut cr, &a, &row2, &mut f).unwrap();
            check(all_close64(cr.raw().as_slice(), &want_scale), &format!("scale contig {m}x{n} reuse {devtag} f64"));

            // ---- f64: broadcast, explicit zero-stride view AND auto form ----
            let rowv: Vec<f64> = elementwise::gen_vec_f64(n, 3);
            let brow = rt::asarray((rowv.clone(), [1, n], dev));
            let want_bc: Vec<f64> = (0..m * n).map(|k| av[k] + rowv[k % n]).collect();

            let c = &a + &brow; // auto-broadcast [1, n]
            check(all_close64(c.raw().as_slice(), &want_bc), &format!("add bcast auto [1,{n}] {m}x{n} alloc {devtag} f64"));

            let brow_bc = brow.broadcast_to(vec![m, n]); // explicit zero-stride view (IxD)
            let c = &a + &brow_bc;
            check(
                all_close64(c.raw().as_slice(), &want_bc),
                &format!("add bcast explicit zero-stride view {m}x{n} alloc {devtag} f64"),
            );

            let mut cr: Tensor<f64, _> = rt::zeros(([m, n], dev));
            cr.fill(0.0);
            let mut f = |cv: &mut MaybeUninit<f64>, x: &f64, y: &f64| { cv.write(x + y); };
            op_mutc_refa_refb_func(&mut cr, &a, &brow_bc, &mut f).unwrap();
            check(
                all_close64(cr.raw().as_slice(), &want_bc),
                &format!("add bcast zero-stride {m}x{n} reuse {devtag} f64"),
            );

            // ---- f64: strided (transpose view) ------------------------------
            let btv = elementwise::gen_vec_f64(m * n, 4);
            let bt = rt::asarray((btv.clone(), [n, m], dev)); // [n, m]: bt.t() is [m, n] (T0 note)
            let want_st: Vec<f64> = (0..m * n).map(|k| av[k] + btv[(k % n) * m + k / n]).collect();
            let c = &a + &bt.t();
            check(all_close64(c.raw().as_slice(), &want_st), &format!("add strided {m}x{n} alloc {devtag} f64"));

            let mut cr: Tensor<f64, _> = rt::zeros(([m, n], dev));
            cr.fill(0.0);
            let mut f = |cv: &mut MaybeUninit<f64>, x: &f64, y: &f64| { cv.write(x + y); };
            op_mutc_refa_refb_func(&mut cr, &a, &bt.t(), &mut f).unwrap();
            check(all_close64(cr.raw().as_slice(), &want_st), &format!("add strided {m}x{n} reuse {devtag} f64"));

            // ---- negative-stride flip views (isize-offset contract) ---------
            let a_flip = a.flip(0); // (i, j) -> av[(m-1-i)*n + j]
            let want_f0: Vec<f64> = (0..m * n)
                .map(|k| {
                    let (i, j) = (k / n, k % n);
                    av[(m - 1 - i) * n + j] + bv[k]
                })
                .collect();
            let c = &a_flip + &b;
            check(all_close64(c.raw().as_slice(), &want_f0), &format!("add flip(0) operand {m}x{n} alloc {devtag} f64"));

            let b_flip = b.flip(1); // (i, j) -> bv[i*n + (n-1-j)]
            let want_ff: Vec<f64> = (0..m * n)
                .map(|k| {
                    let (i, j) = (k / n, k % n);
                    av[(m - 1 - i) * n + j] + bv[i * n + (n - 1 - j)]
                })
                .collect();
            let c = &a_flip + &b_flip;
            check(all_close64(c.raw().as_slice(), &want_ff), &format!("add flip(0)+flip(1) {m}x{n} alloc {devtag} f64"));

            // reuse with flip operand + transpose view (blocked-path fixture)
            let mut cr: Tensor<f64, _> = rt::zeros(([m, n], dev));
            cr.fill(0.0);
            let mut f = |cv: &mut MaybeUninit<f64>, x: &f64, y: &f64| { cv.write(x + y); };
            op_mutc_refa_refb_func(&mut cr, &a_flip, &bt.t(), &mut f).unwrap();
            let want_ft: Vec<f64> = (0..m * n)
                .map(|k| {
                    let (i, j) = (k / n, k % n);
                    av[(m - 1 - i) * n + j] + btv[j * m + i]
                })
                .collect();
            check(all_close64(cr.raw().as_slice(), &want_ft), &format!("add flip(0)+t() {m}x{n} reuse {devtag} f64"));

            // ---- i32 / Complex<f64> spots (generic-path dtype coverage) -----
            let avi: Vec<i32> = (0..m * n).map(|k| (k % 7) as i32 - 3).collect();
            let bvi: Vec<i32> = (0..m * n).map(|k| (k % 5) as i32 - 2).collect();
            let ai = rt::asarray((avi.clone(), [m, n], dev));
            let bi = rt::asarray((bvi.clone(), [m, n], dev));
            let want_addi: Vec<i32> = avi.iter().zip(bvi.iter()).map(|(x, y)| x + y).collect();
            let want_muli: Vec<i32> = avi.iter().zip(bvi.iter()).map(|(x, y)| x * y).collect();
            let ci = &ai + &bi;
            check(eq_slices(ci.raw().as_slice(), &want_addi), &format!("add contig {m}x{n} alloc {devtag} i32"));
            let mut cri: Tensor<i32, _> = rt::zeros(([m, n], dev));
            cri.fill(0);
            let mut fi = |cv: &mut MaybeUninit<i32>, x: &i32, y: &i32| { cv.write(x + y); };
            op_mutc_refa_refb_func(&mut cri, &ai, &bi, &mut fi).unwrap();
            check(eq_slices(cri.raw().as_slice(), &want_addi), &format!("add contig {m}x{n} reuse {devtag} i32"));
            let ci = &ai * &bi;
            check(eq_slices(ci.raw().as_slice(), &want_muli), &format!("mul contig {m}x{n} alloc {devtag} i32"));

            let avc: Vec<Complex<f64>> =
                (0..m * n).map(|k| Complex::new(elementwise::gen_value(k, 1), elementwise::gen_value(k, 3))).collect();
            let bvc: Vec<Complex<f64>> =
                (0..m * n).map(|k| Complex::new(elementwise::gen_value(k, 2), elementwise::gen_value(k, 4))).collect();
            let ac = rt::asarray((avc.clone(), [m, n], dev));
            let bc = rt::asarray((bvc.clone(), [m, n], dev));
            let want_addc: Vec<Complex<f64>> = avc.iter().zip(bvc.iter()).map(|(x, y)| x + y).collect();
            let want_mulc: Vec<Complex<f64>> = avc.iter().zip(bvc.iter()).map(|(x, y)| x * y).collect();
            let cc = &ac + &bc;
            check(eq_slices(cc.raw().as_slice(), &want_addc), &format!("add contig {m}x{n} alloc {devtag} complex64"));
            let mut crc: Tensor<Complex<f64>, _> = rt::zeros(([m, n], dev));
            crc.fill(Complex::new(0.0, 0.0));
            let mut fc = |cv: &mut MaybeUninit<Complex<f64>>, x: &Complex<f64>, y: &Complex<f64>| { cv.write(x + y); };
            op_mutc_refa_refb_func(&mut crc, &ac, &bc, &mut fc).unwrap();
            check(eq_slices(crc.raw().as_slice(), &want_addc), &format!("add contig {m}x{n} reuse {devtag} complex64"));
            let cc = &ac * &bc;
            check(eq_slices(cc.raw().as_slice(), &want_mulc), &format!("mul contig {m}x{n} alloc {devtag} complex64"));

            // ---- f32 spots ---------------------------------------------------
            let a32 = gen_mat_f32!(m, n, 1, dev);
            let b32 = gen_mat_f32!(m, n, 2, dev);
            let (av32, bv32) = (elementwise::gen_vec_f32(m * n, 1), elementwise::gen_vec_f32(m * n, 2));
            let want_add32: Vec<f32> = av32.iter().zip(bv32.iter()).map(|(x, y)| x + y).collect();
            let want_mul32: Vec<f32> = av32.iter().zip(bv32.iter()).map(|(x, y)| x * y).collect();

            let c = &a32 + &b32;
            check(all_close32(c.raw().as_slice(), &want_add32), &format!("add contig {m}x{n} alloc {devtag} f32"));

            let mut cr32: Tensor<f32, _> = rt::zeros(([m, n], dev));
            cr32.fill(0.0);
            let mut f32_ = |cv: &mut MaybeUninit<f32>, x: &f32, y: &f32| { cv.write(x + y); };
            op_mutc_refa_refb_func(&mut cr32, &a32, &b32, &mut f32_).unwrap();
            check(all_close32(cr32.raw().as_slice(), &want_add32), &format!("add contig {m}x{n} reuse {devtag} f32"));

            let c = &a32 * &b32;
            check(all_close32(c.raw().as_slice(), &want_mul32), &format!("mul contig {m}x{n} alloc {devtag} f32"));
        }

        // ---- ndarray cross-check for add (f64, odd size) --------------------
        use ndarray::Array2;
        let (m, n) = (1000usize, 777usize);
        let av = elementwise::gen_vec_f64(m * n, 1);
        let bv = elementwise::gen_vec_f64(m * n, 2);
        let an = Array2::from_shape_vec((m, n), av.clone()).unwrap();
        let bn = Array2::from_shape_vec((m, n), bv.clone()).unwrap();
        let cn = an + bn;
        let a = gen_mat_f64!(m, n, 1, dev);
        let b = gen_mat_f64!(m, n, 2, dev);
        let c = &a + &b;
        check(
            all_close64(c.raw().as_slice(), cn.as_slice().unwrap()),
            &format!("add contig {m}x{n} vs ndarray {devtag} f64"),
        );
    }};
}

fn all_close64(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= 1e-12 * (1.0 + x.abs().max(y.abs())))
}

fn all_close32(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= 1e-4 * (1.0 + x.abs().max(y.abs())))
}

fn eq_slices<T: PartialEq>(a: &[T], b: &[T]) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| x == y)
}

fn main() {
    let dev_serial = serial_device();
    let dev_faer = faer_device();
    assert_faer_threads(&dev_faer, 16);

    run_device!(&dev_serial, "serial");
    run_device!(&dev_faer, "faer16");

    let failures = FAILURES.load(Ordering::SeqCst);
    if failures > 0 {
        panic!("{failures} correctness failures");
    }
    println!("\nALL CORRECTNESS CHECKS PASSED");
}

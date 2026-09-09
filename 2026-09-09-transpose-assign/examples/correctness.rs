//! Correctness gate (plan §3.2) for T1' transpose copy + generic strided
//! assign. Covers, against naive scalar references (+ ndarray cross-check):
//! - the A idiom `a.t().to_contig(RowMajor).into_owned()` (both orientations
//!   1000x777 / 777x1000, small 37x53, degenerate 1x7 / 7x1),
//! - `to_contig(ColMajor)` (the f-contig copy path),
//! - the B reuse form `c.assign(&a.t())`,
//! - negative strides: `flip(0)` / `flip(1)` transposed views (tensor level
//!   AND direct-kernel level — the dormant kernel's isize slow-axis stride),
//! - sliced non-contiguous views (fast-axis stride 2 — must NOT take a
//!   stride-1 fast path),
//! - zero-stride broadcast views, plain and transposed,
//! - a 3-D reverse-axes spot (ndim > 2 must stay on the generic path),
//! - i32 + f32 dtype spots (dtype-generic path),
//! - contiguous assign sanity,
//! - DIRECT kernel calls: assign_arbitary_uninit_{serial,rayon},
//!   orderchange_out_{c2r,r2c}_ix2_{serial,rayon} on raw slices + layouts.
//!
//! Device cases are expanded through a macro (not a generic fn) so rstsr's
//! trait bounds are inherited at the call site. Exits non-zero on any
//! mismatch; run under BOTH RUSTFLAGS configs.

use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicUsize, Ordering};

use transpose_assign::{assert_faer_threads, faer_device, gen_mat_f32, gen_mat_f64, gen_vec_f64, serial_device};
use rstsr_core::prelude::*;
use rstsr_core::prelude_dev::*;

static FAILURES: AtomicUsize = AtomicUsize::new(0);

fn check(cond: bool, what: &str) {
    if cond {
        println!("[PASS] {what}");
    } else {
        println!("[FAIL] {what}");
        FAILURES.fetch_add(1, Ordering::SeqCst);
    }
}

/// Expected row-major content of the c-contiguous [n, m] transpose of a
/// [m, n] row-major `av`.
fn want_transpose(av: &[f64], m: usize, n: usize) -> Vec<f64> {
    (0..n * m).map(|k| av[(k % m) * n + k / m]).collect()
}

macro_rules! run_device {
    ($dev:expr, $devtag:expr) => {{
        let dev = $dev;
        let devtag: &str = $devtag;

        for (m, n) in [(37usize, 53usize), (1000usize, 777usize), (777usize, 1000usize), (1usize, 7usize), (7usize, 1usize)] {
            let a = gen_mat_f64!(m, n, 1, dev);
            let av = gen_vec_f64(m * n, 1);
            let want = want_transpose(&av, m, n);

            // ---- A idiom: t().to_contig(RowMajor).into_owned() ------------
            let out = a.t().to_contig(RowMajor).into_owned();
            check(out.raw().as_slice() == want, &format!("transpose A idiom {m}x{n} {devtag} f64"));

            // ---- A idiom, ColMajor order: f-contig [n, m] == av flattened --
            let outf = a.t().to_contig(ColMajor).into_owned();
            check(outf.raw().as_slice() == av, &format!("transpose A idiom ColMajor {m}x{n} {devtag} f64"));

            // ---- B reuse: c.assign(&a.t()) --------------------------------
            let mut c: Tensor<f64, _> = rt::zeros(([n, m], dev));
            c.fill(0.0);
            c.assign(&a.t());
            check(c.raw().as_slice() == want, &format!("transpose B assign-reuse {m}x{n} {devtag} f64"));

            // ---- flip views (negative strides) -----------------------------
            let want_f0: Vec<f64> = (0..n * m).map(|k| av[(m - 1 - k % m) * n + k / m]).collect();
            let outf0 = a.flip(0).t().to_contig(RowMajor).into_owned();
            check(outf0.raw().as_slice() == want_f0, &format!("transpose flip(0).t() A {m}x{n} {devtag} f64"));
            let mut cf: Tensor<f64, _> = rt::zeros(([n, m], dev));
            cf.fill(0.0);
            cf.assign(&a.flip(0).t());
            check(cf.raw().as_slice() == want_f0, &format!("transpose flip(0).t() B reuse {m}x{n} {devtag} f64"));
            let want_f1: Vec<f64> = (0..n * m).map(|k| av[(k % m) * n + (n - 1 - k / m)]).collect();
            let outf1 = a.flip(1).t().to_contig(RowMajor).into_owned();
            check(outf1.raw().as_slice() == want_f1, &format!("transpose flip(1).t() A {m}x{n} {devtag} f64"));

            // ---- sliced view (fast-axis stride 2; generic path) ------------
            let n2 = (n + 1) / 2;
            let a_sliced = a.i((.., slice!(0, n, 2))); // [m, n2] stride [n, 2]
            let want_sliced: Vec<f64> = (0..m * n2).map(|k| av[(k / n2) * n + (k % n2) * 2]).collect();
            let outs = a_sliced.to_contig(RowMajor).into_owned();
            check(outs.raw().as_slice() == want_sliced, &format!("sliced view to_contig {m}x{n2} {devtag} f64"));
            // ... and its transpose (stride [2, n] view through assign)
            let mut cs: Tensor<f64, _> = rt::zeros(([n2, m], dev));
            cs.fill(0.0);
            cs.assign(&a_sliced.t());
            let want_sliced_t: Vec<f64> =
                (0..n2 * m).map(|k| av[(k % m) * n + (k / m) * 2]).collect();
            check(cs.raw().as_slice() == want_sliced_t, &format!("sliced.t() assign {n2}x{m} {devtag} f64"));

            // ---- r2c orientation: c-contig input -> f-contig output --------
            // (input fast on axis 1, output fast on axis 0: the r2c guard).
            // f-contig [m, n] storage order: out[i + j*m] = av[i*n + j]
            let want_r2c: Vec<f64> = (0..m * n).map(|p| av[(p % m) * n + p / m]).collect();
            let outf2 = a.to_contig(ColMajor).into_owned();
            check(outf2.raw().as_slice() == want_r2c, &format!("to_contig(ColMajor) r2c {m}x{n} {devtag} f64"));
            // reuse form: pre-allocated f-contig output
            let mut cf: Tensor<f64, _> = rt::zeros(([m, n], dev));
            cf = cf.to_contig(ColMajor).into_owned();
            cf.assign(&a);
            check(cf.raw().as_slice() == want_r2c, &format!("assign into f-contig r2c {m}x{n} {devtag} f64"));

            // ---- contig assign sanity (slice-copy path) --------------------
            let mut cc: Tensor<f64, _> = rt::zeros(([m, n], dev));
            cc.fill(0.0);
            cc.assign(&a);
            check(cc.raw().as_slice() == av, &format!("contig assign {m}x{n} {devtag} f64"));
        }

        // ---- zero-stride broadcast, plain + transposed ---------------------
        {
            let (m, n) = (37usize, 53usize);
            let rowv = gen_vec_f64(n, 3);
            let brow = rt::asarray((rowv.clone(), [1usize, n], dev));
            let brow_bc = brow.broadcast_to(vec![m, n]); // zero-stride view (IxD)

            let mut c1: Tensor<f64, _> = rt::zeros(([m, n], dev));
            c1.fill(0.0);
            c1.assign(&brow_bc);
            let want_bc: Vec<f64> = (0..m * n).map(|k| rowv[k % n]).collect();
            check(c1.raw().as_slice() == want_bc, &format!("bcast assign [1,{n}]->{m}x{n} {devtag} f64"));

            // broadcast transposed: [n, m] view, stride [1, 0]
            // out[j][i] = brow_bc[i][j] = rowv[j]; k = j*m + i -> j = k/m
            let brow_bt = brow_bc.t();
            let out_bt = brow_bt.to_contig(RowMajor).into_owned();
            let want_bt: Vec<f64> = (0..n * m).map(|k| rowv[k / m]).collect();
            check(out_bt.raw().as_slice() == want_bt, &format!("bcast.t() to_contig {n}x{m} {devtag} f64"));
            let mut c2: Tensor<f64, _> = rt::zeros(([n, m], dev));
            c2.fill(0.0);
            c2.assign(&brow_bt);
            check(c2.raw().as_slice() == want_bt, &format!("bcast.t() assign {n}x{m} {devtag} f64"));

            // ---- 3-D reverse-axes spot (generic ndim path) -----------------
            let a3v = gen_vec_f64(2 * 3 * 4, 5);
            let a3 = rt::asarray((a3v.clone(), [2usize, 3, 4], dev));
            let out3 = a3.t().to_contig(RowMajor).into_owned(); // [4, 3, 2]
            let want3: Vec<f64> = (0..24).map(|k| a3v[((k % 2) * 3 + (k / 2) % 3) * 4 + k / 6]).collect();
            check(out3.raw().as_slice() == want3, &format!("3-D t().to_contig 2x3x4 {devtag} f64"));

            // ---- i32 dtype spot --------------------------------------------
            let avi: Vec<i32> = (0..m * n).map(|k| (k % 7) as i32 - 3).collect();
            let ai = rt::asarray((avi.clone(), [m, n], dev));
            let outi = ai.t().to_contig(RowMajor).into_owned();
            let wanti: Vec<i32> = (0..n * m).map(|k| avi[(k % m) * n + k / m]).collect();
            check(outi.raw().as_slice() == wanti, &format!("transpose A idiom {m}x{n} {devtag} i32"));
        }

        // ---- f32 spots ------------------------------------------------------
        for (m, n) in [(1000usize, 777usize)] {
            let a32 = gen_mat_f32!(m, n, 1, dev);
            let av32: Vec<f32> = gen_vec_f64(m * n, 1).iter().map(|x| *x as f32).collect();
            let want32: Vec<f32> = (0..n * m).map(|k| av32[(k % m) * n + k / m]).collect();
            let out32 = a32.t().to_contig(RowMajor).into_owned();
            check(out32.raw().as_slice() == want32, &format!("transpose A idiom {m}x{n} {devtag} f32"));
            let mut c32: Tensor<f32, _> = rt::zeros(([n, m], dev));
            c32.fill(0.0);
            c32.assign(&a32.t());
            check(c32.raw().as_slice() == want32, &format!("transpose B assign-reuse {m}x{n} {devtag} f32"));
        }

        // ---- ndarray cross-check (odd size, f64) ----------------------------
        // NOTE: ndarray's `.t().to_owned()` PRESERVES the f-order layout
        // (verified: is_standard_layout()==false, as_slice()==None) — content
        // is the logical transpose. Compare via logical-order iteration.
        {
            use ndarray::Array2;
            let (m, n) = (1000usize, 777usize);
            let av = gen_vec_f64(m * n, 1);
            let an = Array2::from_shape_vec((m, n), av.clone()).unwrap();
            let wantn: Vec<f64> = an.t().iter().cloned().collect();
            let a = gen_mat_f64!(m, n, 1, dev);
            let out = a.t().to_contig(RowMajor).into_owned();
            check(
                out.raw().as_slice() == wantn.as_slice(),
                &format!("transpose A idiom {m}x{n} vs ndarray {devtag} f64"),
            );
        }
    }};
}

/// Direct raw-kernel checks (device-free; explicit pool for rayon twins).
fn run_direct_kernels() {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(16).build().unwrap();

    for (m, n) in [(1000usize, 777usize), (777usize, 1000usize), (1usize, 7usize), (7usize, 1usize)] {
        let av = gen_vec_f64(m * n, 1);
        let want = want_transpose(&av, m, n);
        // tensor-path layout pair: la = a.t() = [n, m] stride [1, n];
        // lc = c-contig [n, m] stride [m, 1]
        let lc: Layout<Ix2> = Layout::new([n, m], [m as isize, 1], 0).unwrap();
        let la: Layout<Ix2> = Layout::new([n, m], [1, n as isize], 0).unwrap();
        let size = n * m;

        // generic assign kernel, serial
        let mut c_mu: Vec<MaybeUninit<f64>> = vec![MaybeUninit::uninit(); size];
        assign_arbitary_uninit_cpu_serial(&mut c_mu, &lc, &av, &la, RowMajor).unwrap();
        let got: Vec<f64> = c_mu.iter().map(|x| unsafe { x.assume_init() }).collect();
        check(got == want, &format!("direct assign_arbitary_uninit_cpu_serial {m}x{n}"));

        // generic assign kernel, rayon
        let mut c_mu: Vec<MaybeUninit<f64>> = vec![MaybeUninit::uninit(); size];
        assign_arbitary_uninit_cpu_rayon(&mut c_mu, &lc, &av, &la, RowMajor, Some(&pool)).unwrap();
        let got: Vec<f64> = c_mu.iter().map(|x| unsafe { x.assume_init() }).collect();
        check(got == want, &format!("direct assign_arbitary_uninit_cpu_rayon {m}x{n}"));

        // dormant blocked kernel, c2r orientation, serial
        let mut c: Vec<f64> = vec![0.0; size];
        orderchange_out_c2r_ix2_cpu_serial(&mut c, &lc, &av, &la).unwrap();
        check(c == want, &format!("direct orderchange_out_c2r_ix2_cpu_serial {m}x{n}"));

        // dormant blocked kernel, r2c orientation (role-swapped pair), serial
        // output col-major [m, n] -> stride [1, m] (ldc = row count)
        let lc2: Layout<Ix2> = Layout::new([m, n], [1, m as isize], 0).unwrap();
        let la2: Layout<Ix2> = Layout::new([m, n], [n as isize, 1], 0).unwrap();
        let mut c: Vec<f64> = vec![0.0; size];
        orderchange_out_r2c_ix2_cpu_serial(&mut c, &lc2, &av, &la2).unwrap();
        check(c == want, &format!("direct orderchange_out_r2c_ix2_cpu_serial {m}x{n}"));

        // dormant blocked kernel, c2r, rayon (parallel block writes)
        let mut c: Vec<f64> = vec![0.0; size];
        orderchange_out_c2r_ix2_cpu_rayon(&mut c, &lc, &av, &la, Some(&pool)).unwrap();
        check(c == want, &format!("direct orderchange_out_c2r_ix2_cpu_rayon {m}x{n}"));

        // negative SLOW-axis stride (flip(0).t()): la stride [1, -n],
        // offset (m-1)*n — accepted by the kernel's guard (isize math)
        if m > 1 && n > 1 {
            let lan: Layout<Ix2> = Layout::new([n, m], [1, -(n as isize)], (m - 1) * n).unwrap();
            let want_f0: Vec<f64> = (0..n * m).map(|k| av[(m - 1 - k % m) * n + k / m]).collect();
            let mut c: Vec<f64> = vec![0.0; size];
            orderchange_out_c2r_ix2_cpu_serial(&mut c, &lc, &av, &lan).unwrap();
            check(c == want_f0, &format!("direct orderchange c2r negative slow stride {m}x{n}"));
        }
    }
}

fn main() {
    let dev_serial = serial_device();
    let dev_faer = faer_device();
    assert_faer_threads(&dev_faer, 16);

    run_device!(&dev_serial, "serial");
    run_device!(&dev_faer, "faer16");
    run_direct_kernels();

    let failures = FAILURES.load(Ordering::SeqCst);
    if failures > 0 {
        panic!("{failures} correctness failures");
    }
    println!("\nALL CORRECTNESS CHECKS PASSED");
}

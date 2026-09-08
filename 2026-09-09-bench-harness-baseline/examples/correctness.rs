//! Correctness gate for T0 (run BEFORE trusting any perf number).
//!
//! Compares every benched rstsr op against a naive scalar reference on
//! deterministic fixtures, including:
//! - odd size 1000x777 (lane-tail / block-edge behavior),
//! - zero-stride broadcast input (explicit `broadcast_to` view AND the
//!   auto-broadcast `[1, n]` form),
//! - transpose-view (strided) operands,
//! - f64 primary + f32 spot checks,
//! - BOTH devices: DeviceCpuSerial and DeviceFaer (16-thread rayon pool,
//!   RAYON_NUM_THREADS must be set to 16 by the runner).
//!
//! Exits non-zero on any mismatch. Run under both RUSTFLAGS configs.

use bench_harness_baseline::{assert_faer_threads, faer_device, gen_vec_f64, serial_device};
use rstsr_core::prelude::*;

fn ref_tol_f64(n: usize) -> f64 {
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

fn allclose_f64(a: &[f64], b: &[f64], tol: f64) -> bool {
    a.len() == b.len()
        && a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= tol * (1.0 + x.abs().max(y.abs())))
}

fn allclose_f32(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b.iter()).all(|(x, y)| {
            let t = 1e-3 * (1.0 + x.abs().max(y.abs()));
            (x - y).abs() <= t
        })
}

/// The whole gate, instantiated per device via macro (concrete ops only; no
/// trait-bound gymnastics). `_device` is any rstsr device expression.
macro_rules! check_device {
    ($dev_name:expr, $_device:expr) => {{
        let device = &$_device;
        println!("-- device: {}", $dev_name);

        // --------------------------------------------------------------
        // transpose copy
        // --------------------------------------------------------------
        for (m, n) in [(64usize, 64usize), (512, 512), (256, 256), (1000, 777)] {
            let a_data = gen_vec_f64(m * n, 1);
            let a = rt::asarray((a_data.clone(), [m, n], device));
            // a.t() has shape [n, m]; its RowMajor-contiguous copy flattens as
            // out[j*m + i] == A[i][j]
            let out = a.t().to_contig(RowMajor).reshape(-1).to_vec();
            let mut want = vec![0.0; m * n];
            for i in 0..m {
                for j in 0..n {
                    want[j * m + i] = a_data[i * n + j];
                }
            }
            check(&format!("transpose copy {m}x{n}"), allclose_f64(&out, &want, ref_tol_f64(m * n)));
        }

        // f32 spot (odd size)
        {
            let (m, n) = (256, 383);
            let a_data: Vec<f32> = gen_vec_f64(m * n, 7).into_iter().map(|x| x as f32).collect();
            let a = rt::asarray((a_data.clone(), [m, n], device));
            let out = a.t().to_contig(RowMajor).reshape(-1).to_vec();
            let mut want = vec![0.0f32; m * n];
            for i in 0..m {
                for j in 0..n {
                    want[j * m + i] = a_data[i * n + j];
                }
            }
            check(&format!("transpose copy f32 {m}x{n}"), allclose_f32(&out, &want));
        }

        // --------------------------------------------------------------
        // reductions
        // --------------------------------------------------------------
        for (m, n) in [(64usize, 64usize), (512, 512), (1000, 777)] {
            let a_data = gen_vec_f64(m * n, 1);
            let a = rt::asarray((a_data.clone(), [m, n], device));

            let s0 = a.sum_axes(0).to_vec();
            let want0: Vec<f64> = (0..n).map(|j| (0..m).map(|i| a_data[i * n + j]).sum()).collect();
            check(&format!("sum_axis0 {m}x{n}"), allclose_f64(&s0, &want0, ref_tol_f64(m * n)));

            let s1 = a.sum_axes(-1).to_vec();
            let want1: Vec<f64> = (0..m).map(|i| (0..n).map(|j| a_data[i * n + j]).sum()).collect();
            check(&format!("sum_axislast {m}x{n}"), allclose_f64(&s1, &want1, ref_tol_f64(m * n)));

            let sall: f64 = a.sum_all();
            let wantall: f64 = a_data.iter().sum();
            check(
                &format!("sum_all {m}x{n}"),
                (sall - wantall).abs() <= ref_tol_f64(m * n) * (1.0 + wantall.abs()),
            );
        }

        // --------------------------------------------------------------
        // vecdot: 1-D and batched (contract last axis)
        // --------------------------------------------------------------
        for n in [1000usize, 100_003, 1_000_000] {
            let a = rt::asarray((gen_vec_f64(n, 1), [n], device));
            let b = rt::asarray((gen_vec_f64(n, 2), [n], device));
            let got: f64 = rt::vecdot(&a, &b, None).reshape(-1).to_vec()[0];
            let want: f64 =
                gen_vec_f64(n, 1).iter().zip(gen_vec_f64(n, 2).iter()).map(|(x, y)| x * y).sum();
            check(&format!("vecdot 1-D n={n}"), (got - want).abs() <= ref_tol_f64(n) * (1.0 + want.abs()));
        }

        {
            let (k, c) = (409usize, 512usize); // small batch to keep the gate quick
            let a_data = gen_vec_f64(k * c, 1);
            let b_data = gen_vec_f64(k * c, 2);
            let a = rt::asarray((a_data.clone(), [k, c], device));
            let b = rt::asarray((b_data.clone(), [k, c], device));
            let got = rt::vecdot(&a, &b, None).to_vec();
            let want: Vec<f64> =
                (0..k).map(|i| (0..c).map(|j| a_data[i * c + j] * b_data[i * c + j]).sum()).collect();
            check(&format!("vecdot batched {k}x{c}"), allclose_f64(&got, &want, ref_tol_f64(k * c)));
        }

        // --------------------------------------------------------------
        // elementwise add: contiguous / broadcast (zero-stride) / strided
        // --------------------------------------------------------------
        for (m, n) in [(64usize, 64usize), (1000, 777)] {
            let a_data = gen_vec_f64(m * n, 1);
            let b_data = gen_vec_f64(m * n, 2);
            let a = rt::asarray((a_data.clone(), [m, n], device));
            let b = rt::asarray((b_data.clone(), [m, n], device));

            // contiguous
            let got = (&a + &b).reshape(-1).to_vec();
            let want: Vec<f64> = a_data.iter().zip(b_data.iter()).map(|(x, y)| x + y).collect();
            check(&format!("add contig {m}x{n}"), allclose_f64(&got, &want, 1e-12));

            // broadcast via auto-broadcast of a [1, n] row vector
            let row_data = gen_vec_f64(n, 3);
            let row = rt::asarray((row_data.clone(), [1, n], device));
            let got = (&a + &row).reshape(-1).to_vec();
            let want: Vec<f64> = (0..m * n).map(|idx| a_data[idx] + row_data[idx % n]).collect();
            check(&format!("add broadcast row [1,{n}] {m}x{n}"), allclose_f64(&got, &want, 1e-12));

            // broadcast via EXPLICIT zero-stride view
            let row_view = row.broadcast_to(vec![m, n]);
            let got = (&a + &row_view).reshape(-1).to_vec();
            check(&format!("add explicit zero-stride broadcast {m}x{n}"), allclose_f64(&got, &want, 1e-12));

            // strided: b = transpose view of an [n, m] tensor, so b.t() is [m, n]
            // (a + b.t() only broadcasts for this pairing when m != n)
            let bt_data = gen_vec_f64(m * n, 4);
            let bt = rt::asarray((bt_data.clone(), [n, m], device));
            let got = (&a + &bt.t()).reshape(-1).to_vec();
            let mut want = vec![0.0_f64; m * n];
            for i in 0..m {
                for j in 0..n {
                    want[i * n + j] = a_data[i * n + j] + bt_data[j * m + i];
                }
            }
            check(&format!("add strided (a + b.t) {m}x{n}"), allclose_f64(&got, &want, 1e-12));
        }

        // f32 add spot incl. odd size
        {
            let (m, n) = (255, 769);
            let a_data: Vec<f32> = gen_vec_f64(m * n, 5).into_iter().map(|x| x as f32).collect();
            let b_data: Vec<f32> = gen_vec_f64(m * n, 6).into_iter().map(|x| x as f32).collect();
            let a = rt::asarray((a_data.clone(), [m, n], device));
            let b = rt::asarray((b_data.clone(), [m, n], device));
            let got = (&a + &b).reshape(-1).to_vec();
            let want: Vec<f32> = a_data.iter().zip(b_data.iter()).map(|(x, y)| x + y).collect();
            check(&format!("add contig f32 {m}x{n}"), allclose_f32(&got, &want));
        }

        // --------------------------------------------------------------
        // fill: zeros / full
        // --------------------------------------------------------------
        {
            let z: Tensor<f64, _> = rt::zeros(([256, 383], device));
            check("zeros 256x383", z.reshape(-1).to_vec().iter().all(|&x| x == 0.0));
            let f: Tensor<f64, _> = rt::full(([256, 383], 3.25_f64, device));
            check("full 256x383", f.reshape(-1).to_vec().iter().all(|&x| x == 3.25));
        }

        // --------------------------------------------------------------
        // argmax (first max wins, flat row-major index)
        // --------------------------------------------------------------
        for n in [1_000_000usize, 1_000_003] {
            let a = rt::asarray((gen_vec_f64(n, 1), [n], device));
            let got: usize = rt::argmax(&a);
            let data = gen_vec_f64(n, 1);
            let want = data
                .iter()
                .enumerate()
                .fold((0usize, f64::NEG_INFINITY), |(bi, bv), (i, &x)| if x > bv { (i, x) } else { (bi, bv) });
            check(&format!("argmax n={n}"), got == want.0);
        }
    }};
}

fn main() {
    check_device!("DeviceCpuSerial", serial_device());
    let dev = faer_device();
    assert_faer_threads(&dev, 16);
    check_device!("DeviceFaer (rayon, 16 threads)", dev);
    println!("correctness gate: ALL PASSED");
}

//! Correctness gate for T2' value reductions (run BEFORE trusting any perf
//! number, in BOTH RUSTFLAGS configs).
//!
//! Compares every reduction family (sum/mean/var/l2_norm/min/max) against
//! naive scalar references, over:
//! - tiny 8x6, odd 1000x777, 3-D 4x5x7 spots,
//! - zero-stride broadcast input (explicit `broadcast_to` view AND the
//!   auto-broadcast [1, n] form),
//! - f64 primary + f32 spots,
//! - NaN semantics documentation fixtures,
//! - BOTH devices: DeviceCpuSerial and DeviceFaer (16 threads).
//!
//! Establishes the CURRENT semantics at 386948be (phase-1 contract: any
//! candidate kernel must preserve these exactly):
//! - sum/mean/var/norm: IEEE — NaN in the reduced slice poisons the output.
//! - min/max: `f64::min`/`f64::max` semantics (ExtReal) — NaN operands are
//!   SKIPPED, not propagated; an all-NaN slice yields the seed value
//!   (`f64::MAX` for min, `f64::MIN` for max).
//! - axis conventions: `sum_axes(0)` on [m, n] -> shape [n] (numpy
//!   convention); `sum_axes(-1)` -> shape [m]. Output layout is row-major
//!   (IxD).
//!
//! Exits non-zero on any mismatch.

use reductions::{assert_faer_threads, faer_device, gen_vec_f32, gen_vec_f64, serial_device};
use rstsr_core::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

static N_CHECKS: AtomicUsize = AtomicUsize::new(0);

fn ref_tol_f64(n: usize) -> f64 {
    // generous but bug-catching: summation-order noise (8-lane unroll) vs
    // wrong-kernel garbage
    1e-9 * (1.0 + (n as f64).sqrt())
}

fn check(name: &str, cond: bool) {
    if cond {
        let n = N_CHECKS.fetch_add(1, Ordering::Relaxed) + 1;
        println!("  [ok {n:3}] {name}");
    } else {
        eprintln!("  [FAIL] {name}");
        std::process::exit(1);
    }
}

fn allclose(a: &[f64], b: &[f64], tol: f64) -> bool {
    a.len() == b.len()
        && a.iter().zip(b.iter()).all(|(x, y)| {
            if x.is_nan() && y.is_nan() {
                true
            } else {
                (x - y).abs() <= tol * (1.0 + x.abs().max(y.abs()))
            }
        })
}

/// Column sums / means / vars / norms / min / max of a row-major [m, n].
struct ColRefs {
    sum: Vec<f64>,
    mean: Vec<f64>,
    var: Vec<f64>,
    norm: Vec<f64>,
    min: Vec<f64>,
    max: Vec<f64>,
    row_sum: Vec<f64>,
    row_mean: Vec<f64>,
    row_var: Vec<f64>,
    row_norm: Vec<f64>,
    row_min: Vec<f64>,
    row_max: Vec<f64>,
}

fn col_refs(a_data: &[f64], m: usize, n: usize) -> ColRefs {
    let mut c = ColRefs {
        sum: vec![0.0; n],
        mean: vec![0.0; n],
        var: vec![0.0; n],
        norm: vec![0.0; n],
        min: vec![f64::INFINITY; n],
        max: vec![f64::NEG_INFINITY; n],
        row_sum: vec![0.0; m],
        row_mean: vec![0.0; m],
        row_var: vec![0.0; m],
        row_norm: vec![0.0; m],
        row_min: vec![f64::INFINITY; m],
        row_max: vec![f64::NEG_INFINITY; m],
    };
    for i in 0..m {
        for j in 0..n {
            let x = a_data[i * n + j];
            c.sum[j] += x;
            c.norm[j] += x * x;
            c.min[j] = c.min[j].min(x);
            c.max[j] = c.max[j].max(x);
            c.row_sum[i] += x;
            c.row_norm[i] += x * x;
            c.row_min[i] = c.row_min[i].min(x);
            c.row_max[i] = c.row_max[i].max(x);
        }
    }
    for j in 0..n {
        c.mean[j] = c.sum[j] / m as f64;
        c.var[j] = c.norm[j] / m as f64 - c.mean[j] * c.mean[j];
        c.norm[j] = c.norm[j].sqrt();
    }
    for i in 0..m {
        c.row_mean[i] = c.row_sum[i] / n as f64;
        c.row_var[i] = c.row_norm[i] / n as f64 - c.row_mean[i] * c.row_mean[i];
        c.row_norm[i] = c.row_norm[i].sqrt();
    }
    c
}

/// The whole gate, instantiated per device via macro (concrete ops only).
macro_rules! check_device {
    ($dev_name:expr, $_device:expr) => {{
        let device = &$_device;
        println!("-- device: {}", $dev_name);

        // --------------------------------------------------------------
        // 2-D families: tiny + odd sizes, all ops, both axes
        // --------------------------------------------------------------
        for (m, n) in [(8usize, 6usize), (1000, 777)] {
            let a_data = gen_vec_f64(m * n, 1);
            let a = rt::asarray((a_data.clone(), [m, n], device));
            let r = col_refs(&a_data, m, n);

            // axis conventions (documented): sum_axes(0) on [m,n] -> [n]
            check(
                &format!("shape sum_axes(0) {m}x{n} == [{n}]"),
                a.sum_axes(0).shape().to_vec() == vec![n],
            );
            check(
                &format!("shape sum_axes(-1) {m}x{n} == [{m}]"),
                a.sum_axes(-1).shape().to_vec() == vec![m],
            );

            let got = a.sum_axes(0).to_vec();
            check(&format!("sum_axis0 {m}x{n}"), allclose(&got, &r.sum, ref_tol_f64(m * n)));
            let got = a.sum_axes(-1).to_vec();
            check(&format!("sum_axis1 {m}x{n}"), allclose(&got, &r.row_sum, ref_tol_f64(m * n)));
            let got = a.mean_axes(0).to_vec();
            check(&format!("mean_axis0 {m}x{n}"), allclose(&got, &r.mean, ref_tol_f64(m * n)));
            let got = a.mean_axes(-1).to_vec();
            check(&format!("mean_axis1 {m}x{n}"), allclose(&got, &r.row_mean, ref_tol_f64(m * n)));
            let got = a.var_axes(0).to_vec();
            check(&format!("var_axis0 {m}x{n}"), allclose(&got, &r.var, ref_tol_f64(m * n)));
            let got = a.var_axes(-1).to_vec();
            check(&format!("var_axis1 {m}x{n}"), allclose(&got, &r.row_var, ref_tol_f64(m * n)));
            let got = a.l2_norm_axes(0).to_vec();
            check(&format!("norm_axis0 {m}x{n}"), allclose(&got, &r.norm, ref_tol_f64(m * n)));
            let got = a.l2_norm_axes(-1).to_vec();
            check(&format!("norm_axis1 {m}x{n}"), allclose(&got, &r.row_norm, ref_tol_f64(m * n)));
            let got = a.min_axes(0).to_vec();
            check(&format!("min_axis0 {m}x{n}"), allclose(&got, &r.min, 0.0));
            let got = a.min_axes(-1).to_vec();
            check(&format!("min_axis1 {m}x{n}"), allclose(&got, &r.row_min, 0.0));
            let got = a.max_axes(0).to_vec();
            check(&format!("max_axis0 {m}x{n}"), allclose(&got, &r.max, 0.0));
            let got = a.max_axes(-1).to_vec();
            check(&format!("max_axis1 {m}x{n}"), allclose(&got, &r.row_max, 0.0));

            // full reductions (scalar)
            let got: f64 = a.sum_all();
            let want: f64 = a_data.iter().sum();
            check(&format!("sum_all {m}x{n}"), (got - want).abs() <= ref_tol_f64(m * n) * (1.0 + want.abs()));
            let got: f64 = a.mean_all();
            check(&format!("mean_all {m}x{n}"), (got - want / (m * n) as f64).abs() <= ref_tol_f64(m * n));
            let got: f64 = a.var_all();
            let mu = want / (m * n) as f64;
            let want_var: f64 = a_data.iter().map(|&x| (x - mu) * (x - mu)).sum::<f64>() / (m * n) as f64;
            check(&format!("var_all {m}x{n}"), (got - want_var).abs() <= ref_tol_f64(m * n) * (1.0 + want_var));
            let got: f64 = a.l2_norm_all();
            let want_norm: f64 = a_data.iter().map(|x| x * x).sum::<f64>().sqrt();
            check(&format!("norm_all {m}x{n}"), (got - want_norm).abs() <= ref_tol_f64(m * n) * (1.0 + want_norm));
            let got: f64 = a.min_all();
            let want_min = a_data.iter().copied().fold(f64::INFINITY, f64::min);
            check(&format!("min_all {m}x{n}"), got == want_min);
            let got: f64 = a.max_all();
            let want_max = a_data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            check(&format!("max_all {m}x{n}"), got == want_max);
        }

        // --------------------------------------------------------------
        // 3-D spot: [4, 5, 7], sum over axis 0 and axis 2
        // --------------------------------------------------------------
        {
            let (p, q, r_) = (4usize, 5usize, 7usize);
            let a_data = gen_vec_f64(p * q * r_, 3);
            let a = rt::asarray((a_data.clone(), [p, q, r_], device));

            let got = a.sum_axes(0).reshape(-1).to_vec();
            let want: Vec<f64> = (0..q * r_)
                .map(|idx| (0..p).map(|k| a_data[k * q * r_ + idx]).sum())
                .collect();
            check(&format!("sum_axis0 3-D {p}x{q}x{r_}"), allclose(&got, &want, ref_tol_f64(p * q * r_)));

            let got = a.sum_axes(-1).reshape(-1).to_vec();
            let want: Vec<f64> = (0..p * q)
                .map(|idx| (0..r_).map(|l| a_data[idx * r_ + l]).sum())
                .collect();
            check(&format!("sum_axis2 3-D {p}x{q}x{r_}"), allclose(&got, &want, ref_tol_f64(p * q * r_)));

            check("shape sum_axes(0) 3-D == [q, r_]", a.sum_axes(0).shape().to_vec() == vec![q, r_]);

            let got = a.min_axes(0).reshape(-1).to_vec();
            let want: Vec<f64> = (0..q * r_)
                .map(|idx| (0..p).map(|k| a_data[k * q * r_ + idx]).fold(f64::INFINITY, f64::min))
                .collect();
            check(&format!("min_axis0 3-D {p}x{q}x{r_}"), allclose(&got, &want, 0.0));
        }

        // --------------------------------------------------------------
        // zero-stride broadcast inputs: sum over the broadcast axis
        // duplicates values (sum multiplies; min/max/idempotent ops skip)
        // --------------------------------------------------------------
        {
            let (m, n) = (200usize, 777usize);
            let row_data = gen_vec_f64(n, 5);
            let row = rt::asarray((row_data.clone(), [1, n], device));

            // auto-broadcast [1, n] + [m, n]
            let a_data = gen_vec_f64(m * n, 6);
            let a = rt::asarray((a_data.clone(), [m, n], device));
            let got = (&row + &a).sum_axes(0).to_vec();
            let want: Vec<f64> = (0..n)
                .map(|j| row_data[j] * m as f64 + (0..m).map(|i| a_data[i * n + j]).sum::<f64>())
                .collect();
            check(&format!("broadcast-row add+sum_axis0 {m}x{n}"), allclose(&got, &want, ref_tol_f64(m * n)));

            // EXPLICIT zero-stride view reduced over the broadcast axis.
            //
            // CURRENT-BEHAVIOR LOCK (upstream bug at 386948be, preserved):
            let row_view = row.broadcast_to(vec![m, n]);
            // reducing over a stride-0 SUMMED axis multiplies by the WRONG
            // count. The multiplier `size_s0`
            // (rstsr-native-impl/src/cpu_serial/reduction.rs:226,
            // `as0.iter().map(|&i| lm.shape()[i])`) indexes the REMAINING
            // layout `lm` with the SUMMED layout's broadcast-axis indices;
            // for a [1,n]->[m,n] view summed over axis 0 it yields `n`
            // (the row length) instead of `m` (numpy: m×v; rstsr: n×v).
            // mean inherits it (n/m×v). min/max are insensitive (idempotent).
            // A T2' candidate must reproduce this behavior bit-for-bit; the
            // fix itself is a separate correctness decision, not a perf edit.
            let got = row_view.sum_axes(0).to_vec();
            let want_current_behavior: Vec<f64> = row_data.iter().map(|&x| x * n as f64).collect();
            check(
                &format!("explicit zero-stride sum_axis0 {m}x{n} (CURRENT-BEHAVIOR lock, upstream bug)"),
                allclose(&got, &want_current_behavior, ref_tol_f64(m * n)),
            );
            let got = row_view.mean_axes(0).to_vec();
            let want_mean_current: Vec<f64> = row_data.iter().map(|&x| x * n as f64 / m as f64).collect();
            check(
                &format!("explicit zero-stride mean_axis0 {m}x{n} (CURRENT-BEHAVIOR lock, upstream bug)"),
                allclose(&got, &want_mean_current, ref_tol_f64(m * n)),
            );

            let got = row_view.min_axes(0).to_vec();
            let want: Vec<f64> = row_data.iter().copied().collect();
            check(&format!("explicit zero-stride min_axis0 {m}x{n}"), allclose(&got, &want, 0.0));

            // reduce over a NON-broadcast axis of the broadcast view
            let got = row_view.sum_axes(-1).to_vec();
            let want: Vec<f64> = (0..m).map(|_| row_data.iter().sum::<f64>()).collect();
            check(&format!("explicit zero-stride sum_axis1 {m}x{n}"), allclose(&got, &want, ref_tol_f64(m * n)));
        }

        // --------------------------------------------------------------
        // transposed (strided) input: both reduction directions
        // --------------------------------------------------------------
        {
            let (m, n) = (256usize, 384usize);
            let a_data = gen_vec_f64(m * n, 8);
            let a = rt::asarray((a_data.clone(), [m, n], device));
            let at = a.t(); // shape [n, m], strides [1, n]
            let r = col_refs(&a_data, m, n);
            let got = at.sum_axes(0).to_vec(); // = column sums of a.t() = row sums of a
            check("sum_axis0 of t-view", allclose(&got, &r.row_sum, ref_tol_f64(m * n)));
            let got = at.sum_axes(-1).to_vec();
            check("sum_axis1 of t-view", allclose(&got, &r.sum, ref_tol_f64(m * n)));
        }

        // --------------------------------------------------------------
        // NaN semantics (current behavior at 386948be — must be preserved)
        // --------------------------------------------------------------
        {
            let mut a_data = gen_vec_f64(64, 9);
            a_data[3] = f64::NAN; // one NaN in row 0
            let a = rt::asarray((a_data.clone(), [8, 8], device));

            let s1 = a.sum_axes(-1).to_vec();
            check(
                "NaN poisons sum of its row (axis1)",
                s1[0].is_nan() && s1[1..].iter().all(|x| !x.is_nan()),
            );
            let s0 = a.sum_axes(0).to_vec();
            check(
                "NaN poisons sum of its column (axis0)",
                s0[3].is_nan() && s0.iter().enumerate().all(|(j, x)| j == 3 || !x.is_nan()),
            );
            let v1 = a.var_axes(-1).to_vec();
            check("NaN poisons var of its row", v1[0].is_nan());
            let n1 = a.l2_norm_axes(-1).to_vec();
            check("NaN poisons norm of its row", n1[0].is_nan());

            // min/max SKIP NaN (f64::min/max semantics), all-NaN -> seed
            let m1 = a.min_axes(-1).to_vec();
            let want_min: Vec<f64> = (0..8usize)
                .map(|i| (0..8usize).map(|j| a_data[i * 8 + j]).filter(|x| !x.is_nan()).fold(f64::INFINITY, f64::min))
                .collect();
            check("NaN skipped by min_axis1", allclose(&m1, &want_min, 0.0));
            let mx1 = a.max_axes(-1).to_vec();
            let want_max: Vec<f64> = (0..8usize)
                .map(|i| (0..8usize).map(|j| a_data[i * 8 + j]).filter(|x| !x.is_nan()).fold(f64::NEG_INFINITY, f64::max))
                .collect();
            check("NaN skipped by max_axis1", allclose(&mx1, &want_max, 0.0));

            let allnan_data = vec![f64::NAN; 16];
            let an = rt::asarray((allnan_data.clone(), [4, 4], device));
            let m0 = an.min_axes(0).to_vec();
            let mx0 = an.max_axes(0).to_vec();
            check(
                "all-NaN axis0: min -> +MAX seed, max -> -MIN seed",
                m0.iter().all(|&x| x == f64::MAX) && mx0.iter().all(|&x| x == f64::MIN),
            );
            let sall: f64 = an.sum_all();
            check("all-NaN sum_all -> NaN", sall.is_nan());
        }

        // --------------------------------------------------------------
        // sign-of-zero spot (NEW-behavior lock after the A2 strict-compare
        // rewrite). On ±0.0 ties the strict-compare fold keeps the
        // incumbent, so the sign of the zero result follows the first
        // strictly-winning element. The PRE-patch `f64::min` (minnum) form
        // returned the LATER operand on ties (measured on this machine,
        // results/signbit_pre_patch.txt: min_all([-0,+0]) = +0,
        // max_all([+0,-0]) = -0). Values are equal (== 0.0); only the sign
        // bit can differ — value-equal, documented in the README.
        // --------------------------------------------------------------
        {
            let a = rt::asarray((vec![-0.0f64, 0.0], [1usize, 2], device));
            let m: f64 = a.min_all();
            check(
                "signbit lock: min_all([-0,+0]) keeps incumbent -0.0 (value == 0.0)",
                m == 0.0 && m.is_sign_negative(),
            );
            let b = rt::asarray((vec![0.0f64, -0.0], [1usize, 2], device));
            let x: f64 = b.max_all();
            check(
                "signbit lock: max_all([+0,-0]) keeps incumbent +0.0 (value == 0.0)",
                x == 0.0 && !x.is_sign_negative(),
            );
            let c = rt::asarray((vec![-0.0f64, 0.0, 0.0, -0.0], [2usize, 2], device));
            let mn = c.min_axes(-1).reshape(-1).to_vec();
            check(
                "signbit lock: min_axes(-1) [[-0,+0],[+0,-0]] -> [-0,+0] (incumbent kept: -0.0 == +0.0)",
                mn[0] == 0.0 && mn[0].is_sign_negative() && mn[1] == 0.0 && !mn[1].is_sign_negative(),
            );
            let mx = c.max_axes(-1).reshape(-1).to_vec();
            check(
                "signbit lock: max_axes(-1) [[-0,+0],[+0,-0]] -> [-0,+0] (incumbent kept)",
                mx[0] == 0.0 && mx[0].is_sign_negative() && mx[1] == 0.0 && !mx[1].is_sign_negative(),
            );
            let mn0 = c.min_axes(0).reshape(-1).to_vec();
            check(
                "signbit lock: min_axes(0) [[-0,+0],[+0,-0]] -> [-0,+0] (incumbent kept)",
                mn0[0] == 0.0 && mn0[0].is_sign_negative() && mn0[1] == 0.0 && !mn0[1].is_sign_negative(),
            );
        }

        // --------------------------------------------------------------
        // i32 min/max spot (integers take the same strict-compare fold;
        // Ord::min/max equivalence)
        // --------------------------------------------------------------
        {
            let (m, n) = (64usize, 97usize);
            let a_data: Vec<i32> =
                (0..m * n).map(|i| ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) % 2003) as i32 - 1000).collect();
            let a = rt::asarray((a_data.clone(), [m, n], device));
            let got = a.min_axes(0).reshape(-1).to_vec();
            let want: Vec<i32> = (0..n)
                .map(|j| (0..m).map(|i| a_data[i * n + j]).fold(i32::MAX, i32::min))
                .collect();
            check(&format!("i32 min_axis0 {m}x{n}"), got == want);
            let got = a.max_axes(-1).reshape(-1).to_vec();
            let want: Vec<i32> = (0..m)
                .map(|i| (0..n).map(|j| a_data[i * n + j]).fold(i32::MIN, i32::max))
                .collect();
            check(&format!("i32 max_axis1 {m}x{n}"), got == want);
            let got: i32 = a.min_all();
            let want = a_data.iter().copied().fold(i32::MAX, i32::min);
            check("i32 min_all", got == want);
            let got: i32 = a.max_all();
            let want = a_data.iter().copied().fold(i32::MIN, i32::max);
            check("i32 max_all", got == want);
        }

        // --------------------------------------------------------------
        // f32 spot: odd size, both axes + all
        // --------------------------------------------------------------
        {
            let (m, n) = (255usize, 769usize);
            let a_data = gen_vec_f32(m * n, 10);
            let a = rt::asarray((a_data.clone(), [m, n], device));
            let got = a.sum_axes(0).to_vec();
            let want: Vec<f32> = (0..n).map(|j| (0..m).map(|i| a_data[i * n + j]).sum()).collect();
            check(
                &format!("sum_axis0 f32 {m}x{n}"),
                got.iter().zip(want.iter()).all(|(x, y)| ((x - y) as f64).abs() <= 1e-2 * (1.0 + (*x).abs().max((*y).abs()) as f64)),
            );
            let got = a.sum_axes(-1).to_vec();
            let want: Vec<f32> = (0..m).map(|i| (0..n).map(|j| a_data[i * n + j]).sum()).collect();
            check(
                &format!("sum_axis1 f32 {m}x{n}"),
                got.iter().zip(want.iter()).all(|(x, y)| ((x - y) as f64).abs() <= 1e-2 * (1.0 + (*x).abs().max((*y).abs()) as f64)),
            );
            let got = a.min_axes(0).to_vec();
            let want: Vec<f32> = (0..n)
                .map(|j| (0..m).map(|i| a_data[i * n + j]).fold(f32::INFINITY, f32::min))
                .collect();
            check(&format!("min_axis0 f32 {m}x{n}"), got == want);
        }
    }};
}

fn main() {
    check_device!("DeviceCpuSerial", serial_device());
    let dev = faer_device();
    assert_faer_threads(&dev, 16);
    check_device!("DeviceFaer (rayon, 16 threads)", dev);
    let n = N_CHECKS.load(Ordering::Relaxed);
    println!("correctness gate: ALL PASSED ({n} checks; per-device counts above)");
}

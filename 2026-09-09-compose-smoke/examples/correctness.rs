//! T8 compose-smoke correctness gate: ESSENTIAL fixtures from the five
//! campaign experiments, re-run against the COMBINED (5-patch) tree.
//!
//! Sections (source experiment of each fixture family in brackets):
//! - [T6 argmax-argmin] ties (all-equal, duplicated extremes), NaN-lane
//!   (front poison / middle+end skip / all-NaN), NaN-at-chunk-start,
//!   empty (Err + panic), 2-D contig + t-view flat contract, broadcast
//!   view, argmax_axes(-1/0) + t-view, f32 spots.
//! - [T4' elementwise] contig add/mul alloc+reuse, scale, auto + explicit
//!   zero-stride broadcast, strided t-view operand (alloc+reuse — the
//!   blocked-tile path), flip-view operands, flip+t() reuse, i32 and
//!   Complex<f64> generic-path spots, f32 spots; odd 1000x777 + small 37x53.
//! - [T1' transpose-assign] A idiom to_contig(RowMajor) both orientations,
//!   ColMajor (r2c), B reuse `c.assign(&a.t())`, flip-transposed views,
//!   sliced stride-2 fall-through, broadcast slow-axis (incl. transposed),
//!   3-D generic spot, degenerate 1x7/7x1, i32 + f32 spots.
//! - [T2' reductions] sum/mean/var/norm/min/max axis0+axis1 + all, 3-D
//!   spot, zero-stride broadcast locks (incl. the documented upstream
//!   stride-0 sum multiplier bug, PRESERVED), NaN poison (sum) vs NaN skip
//!   (min/max) + all-NaN seed, signbit locks (±0 ties keep incumbent),
//!   i32 min/max, f32 spots.
//! - [T3' vecdot] 1-D dot f64/f32/c64 + NaN poison, batched am1 + axis0
//!   (incl. 32 KiB-budget fall-through), 3-D ax1 fall-through, strided
//!   (transposed-view) b, f-contig a, broadcast (remaining/summed axis),
//!   inner_dot `%` contig/strided-row/c64/no-conj/NaN/n=1/n=300-seq,
//!   alpha/beta fallback via rt::matmul_from.
//!
//! Everything vs naive scalar references; BOTH devices (DeviceCpuSerial,
//! DeviceFaer default-device with RAYON_NUM_THREADS=16 asserted). Exits
//! non-zero on any mismatch. Run under portable AND native RUSTFLAGS.

use std::mem::MaybeUninit;
use std::panic::{catch_unwind, set_hook, take_hook};
use std::sync::atomic::{AtomicUsize, Ordering};

use num::Complex;

use compose_smoke::{assert_faer_threads, faer_device, gen_value, gen_vec_f32, gen_vec_f64, serial_device};
use rstsr_core::prelude::*;
use rstsr_core::prelude_dev::*;
use rstsr_core::tensor::operators::op_with_func::op_mutc_refa_refb_func;

static N_CHECKS: AtomicUsize = AtomicUsize::new(0);

fn check(name: &str, cond: bool) {
    if cond {
        let n = N_CHECKS.fetch_add(1, Ordering::Relaxed) + 1;
        println!("  [ok {n:3}] {name}");
    } else {
        eprintln!("  [FAIL] {name}");
        std::process::exit(1);
    }
}

fn all_close64(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= 1e-12 * (1.0 + x.abs().max(y.abs())))
}

fn eq_slices<T: PartialEq>(a: &[T], b: &[T]) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| x == y)
}

/// Naive reference: first-index tie-break fold over a visit sequence.
fn ref_arg(visit: &[(usize, f64)], want_max: bool) -> usize {
    assert!(!visit.is_empty());
    let mut best_idx = visit[0].0;
    let mut best_val = visit[0].1;
    for &(idx, x) in &visit[1..] {
        let better = if want_max { x > best_val } else { x < best_val };
        if better {
            best_idx = idx;
            best_val = x;
        }
    }
    best_idx
}

// ===========================================================================
// [T6 argmax-argmin]
// ===========================================================================
macro_rules! section_arg {
    ($dev:expr) => {{
        let device = &$dev;

        // odd contiguous 1-D
        {
            let n = 999_983usize;
            let data = gen_vec_f64(n, 1);
            let a = rt::asarray((data.clone(), [n], device));
            let visit: Vec<(usize, f64)> = (0..n).map(|k| (k, data[k])).collect();
            check("argmax 1-D contig odd n=999983", rt::argmax(&a) == ref_arg(&visit, true));
            check("argmin 1-D contig odd n=999983", rt::argmin(&a) == ref_arg(&visit, false));
        }

        // ties: all-equal + duplicated extremes
        for &n in &[127usize, 4099usize] {
            let a = rt::asarray((vec![2.5_f64; n], [n], device));
            check(&format!("argmax all-equal n={n} -> 0"), rt::argmax(&a) == 0);
            check(&format!("argmin all-equal n={n} -> 0"), rt::argmin(&a) == 0);
        }
        {
            let n = 4099usize;
            let mut data = gen_vec_f64(n, 2);
            for &k in &[0usize, 17, 1024, 4000, n - 1] {
                data[k] = 100.0;
            }
            let a = rt::asarray((data.clone(), [n], device));
            check("argmax duplicated max (first at 0) -> 0", rt::argmax(&a) == 0);
            let mut data2 = gen_vec_f64(n, 2);
            for &k in &[23usize, 1024, 4000] {
                data2[k] = 100.0;
            }
            let a2 = rt::asarray((data2.clone(), [n], device));
            check("argmax duplicated max (first at 23) -> 23", rt::argmax(&a2) == 23);
            let mut data3 = gen_vec_f64(n, 3);
            for &k in &[31usize, 777, 4098] {
                data3[k] = -100.0;
            }
            let a3 = rt::asarray((data3.clone(), [n], device));
            check("argmin duplicated min (first at 31) -> 31", rt::argmin(&a3) == 31);
        }

        // NaN placements
        {
            let n = 5003usize;
            let base = gen_vec_f64(n, 4);
            let mut d = base.clone();
            d[n / 2] = f64::NAN;
            let a = rt::asarray((d.clone(), [n], device));
            let visit: Vec<(usize, f64)> = (0..n).map(|k| (k, d[k])).collect();
            check("argmax NaN middle (real max wins)", rt::argmax(&a) == ref_arg(&visit, true));
            let mut d = base.clone();
            d[n - 1] = f64::NAN;
            let a = rt::asarray((d.clone(), [n], device));
            let visit: Vec<(usize, f64)> = (0..n).map(|k| (k, d[k])).collect();
            check("argmax NaN end (real max wins)", rt::argmax(&a) == ref_arg(&visit, true));
            let mut d = base.clone();
            d[0] = f64::NAN;
            let a = rt::asarray((d.clone(), [n], device));
            check("argmax NaN front -> 0 (NaN poisons)", rt::argmax(&a) == 0);

            // NaN at 1..8 with true max later in the same 8-lane scan
            let m = 10_007usize;
            for k in 1..8usize {
                let mut dd = gen_vec_f64(m, 46);
                let max_pos = k + 8;
                dd[k] = f64::NAN;
                dd[max_pos] = 1000.0;
                let a4 = rt::asarray((dd.clone(), [m], device));
                check(
                    &format!("argmax NaN at {k}, max at {max_pos} (same lane) -> {max_pos}"),
                    rt::argmax(&a4) == max_pos,
                );
            }

            // NaN at chunk-start offsets (deterministic combine)
            let m2 = 10_003usize;
            let mut dd = gen_vec_f64(m2, 44);
            let max_idx = m2 / 2;
            dd[max_idx] = 1000.0;
            for &k in &[156usize, 312, 625, 1250, 2500, 5000] {
                dd[k] = f64::NAN;
            }
            let a2 = rt::asarray((dd.clone(), [m2], device));
            check(
                "argmax NaN at chunk-start offsets (real max wins)",
                rt::argmax(&a2) == max_idx,
            );
            let mut dd2 = gen_vec_f64(m2, 45);
            let min_idx = m2 / 2;
            dd2[min_idx] = -1000.0;
            for &k in &[156usize, 312, 625, 1250, 2500, 5000] {
                dd2[k] = f64::NAN;
            }
            let a3 = rt::asarray((dd2, [m2], device));
            check(
                "argmin NaN at chunk-start offsets (real min wins)",
                rt::argmin(&a3) == min_idx,
            );

            // all-NaN
            let a = rt::asarray((vec![f64::NAN; n], [n], device));
            check("argmax all-NaN -> 0 (no error)", rt::argmax(&a) == 0);
            check("argmin all-NaN -> 0 (no error)", rt::argmin(&a) == 0);

            // argmin NaN front
            let mut d = base.clone();
            d[0] = f64::NAN;
            let a = rt::asarray((d, [n], device));
            check("argmin NaN front -> 0 (NaN poisons)", rt::argmin(&a) == 0);
        }

        // empty: Err + panic
        {
            let empty = rt::asarray((Vec::<f64>::new(), [0usize], device));
            let e_max = rt::argmax_f(&empty);
            let e_min = rt::argmin_f(&empty);
            check("argmax_f/argmin_f empty -> Err", e_max.is_err() && e_min.is_err());
            let prev_hook = take_hook();
            set_hook(Box::new(|_| {}));
            let panicked = catch_unwind(std::panic::AssertUnwindSafe(|| rt::argmax(&empty)));
            set_hook(prev_hook);
            match panicked {
                Ok(_) => check("argmax empty -> panics (expected)", false),
                Err(payload) => {
                    let msg = payload
                        .downcast_ref::<String>()
                        .cloned()
                        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                        .unwrap_or_default();
                    check("argmax empty -> panics mentioning 'empty sequence'", msg.contains("empty sequence"));
                },
            }
        }

        // 2-D contig + t-view flat contract
        {
            let (m, n) = (37usize, 53usize);
            let data = gen_vec_f64(m * n, 5);
            let a = rt::asarray((data.clone(), [m, n], device));
            let visit: Vec<(usize, f64)> = (0..m * n).map(|k| (k, data[k])).collect();
            check("argmax 2-D contig 37x53 (flat row-major)", rt::argmax(&a) == ref_arg(&visit, true));
            let at = a.t();
            let mut visit_t: Vec<(usize, f64)> = Vec::with_capacity(n * m);
            for i in 0..n {
                for j in 0..m {
                    visit_t.push((i * m + j, data[j * n + i]));
                }
            }
            check("argmax 2-D t-view 53x37 (flat of view)", rt::argmax(&at) == ref_arg(&visit_t, true));
            check("argmin 2-D t-view 53x37 (flat of view)", rt::argmin(&at) == ref_arg(&visit_t, false));
        }

        // broadcast view
        {
            let (m, n) = (13usize, 41usize);
            let row_data = gen_vec_f64(n, 8);
            let row = rt::asarray((row_data.clone(), [1, n], device));
            let b = row.broadcast_to(vec![m, n]);
            let mut want = 0usize;
            let mut best = f64::NEG_INFINITY;
            for j in 0..n {
                if row_data[j] > best {
                    best = row_data[j];
                    want = j;
                }
            }
            check("argmax broadcast view (max in row 0)", rt::argmax(&b) == want);
            let row_eq = rt::asarray((vec![7.0_f64; n], [1, n], device));
            let b_eq = row_eq.broadcast_to(vec![m, n]);
            check("argmax broadcast all-equal -> 0", rt::argmax(&b_eq) == 0);
            check("argmin broadcast all-equal -> 0", rt::argmin(&b_eq) == 0);
        }

        // argmax_axes / argmin_axes
        {
            let (m, n) = (37usize, 53usize);
            let data = gen_vec_f64(m * n, 9);
            let a = rt::asarray((data.clone(), [m, n], device));
            let got_last_max = a.argmax_axes(-1).to_vec();
            let want_last_max: Vec<usize> = (0..m)
                .map(|i| {
                    ref_arg(
                        &data[i * n..(i + 1) * n].iter().enumerate().map(|(k, &x)| (k, x)).collect::<Vec<_>>(),
                        true,
                    )
                })
                .collect();
            check("argmax_axes(-1) 37x53", got_last_max == want_last_max);
            let got_col_max = a.argmax_axes(0).to_vec();
            let want_col_max: Vec<usize> = (0..n)
                .map(|j| ref_arg(&(0..m).map(|i| (i, data[i * n + j])).collect::<Vec<_>>(), true))
                .collect();
            check("argmax_axes(0) 37x53", got_col_max == want_col_max);
            let at = a.t();
            let got_t_last = at.argmax_axes(-1).to_vec();
            let want_t_last: Vec<usize> = (0..n)
                .map(|i| ref_arg(&(0..m).map(|j| (j, data[j * n + i])).collect::<Vec<_>>(), true))
                .collect();
            check("argmax_axes(-1) on t-view 53x37", got_t_last == want_t_last);
        }

        // f32 spots
        {
            let n = 999_983usize;
            let data = gen_vec_f32(n, 6);
            let a = rt::asarray((data.clone(), [n], device));
            let visit: Vec<(usize, f64)> = (0..n).map(|k| (k, data[k] as f64)).collect();
            check("argmax 1-D contig odd f32", rt::argmax(&a) == ref_arg(&visit, true));
            let mut d32 = gen_vec_f32(n, 6);
            d32[0] = f32::NAN;
            let a32 = rt::asarray((d32, [n], device));
            check("argmax f32 NaN front -> 0 (NaN poisons)", rt::argmax(&a32) == 0);
        }
    }};
}

// ===========================================================================
// [T4' elementwise] + [T1' transpose-assign] 2-D tensor-level fixtures
// ===========================================================================
macro_rules! section_elem_trans {
    ($dev:expr) => {{
        let device = &$dev;

        for (m, n) in [(37usize, 53usize), (1000usize, 777usize)] {
            // ---- [T4'] contig add/mul alloc + reuse ------------------------
            let av = gen_vec_f64(m * n, 1);
            let bv = gen_vec_f64(m * n, 2);
            let a = rt::asarray((av.clone(), [m, n], device));
            let b = rt::asarray((bv.clone(), [m, n], device));
            let want_add: Vec<f64> = av.iter().zip(bv.iter()).map(|(x, y)| x + y).collect();
            let want_mul: Vec<f64> = av.iter().zip(bv.iter()).map(|(x, y)| x * y).collect();

            let c = &a + &b;
            check(&format!("add contig {m}x{n} alloc f64"), all_close64(c.raw().as_slice(), &want_add));

            let mut cr: Tensor<f64, _> = rt::zeros(([m, n], device));
            cr.fill(0.0);
            let mut f = |cv: &mut MaybeUninit<f64>, x: &f64, y: &f64| { cv.write(x + y); };
            op_mutc_refa_refb_func(&mut cr, &a, &b, &mut f).unwrap();
            check(&format!("add contig {m}x{n} reuse f64"), all_close64(cr.raw().as_slice(), &want_add));

            let c = &a * &b;
            check(&format!("mul contig {m}x{n} alloc f64"), all_close64(c.raw().as_slice(), &want_mul));
            let mut cr: Tensor<f64, _> = rt::zeros(([m, n], device));
            cr.fill(0.0);
            let mut f = |cv: &mut MaybeUninit<f64>, x: &f64, y: &f64| { cv.write(x * y); };
            op_mutc_refa_refb_func(&mut cr, &a, &b, &mut f).unwrap();
            check(&format!("mul contig {m}x{n} reuse f64"), all_close64(cr.raw().as_slice(), &want_mul));

            // ---- [T4'] broadcast: auto + explicit zero-stride view ---------
            let rowv: Vec<f64> = gen_vec_f64(n, 3);
            let brow = rt::asarray((rowv.clone(), [1, n], device));
            let want_bc: Vec<f64> = (0..m * n).map(|k| av[k] + rowv[k % n]).collect();
            let c = &a + &brow;
            check(&format!("add bcast auto [1,{n}] {m}x{n} alloc"), all_close64(c.raw().as_slice(), &want_bc));
            let brow_bc = brow.broadcast_to(vec![m, n]);
            let c = &a + &brow_bc;
            check(&format!("add bcast zero-stride view {m}x{n} alloc"), all_close64(c.raw().as_slice(), &want_bc));
            let mut cr: Tensor<f64, _> = rt::zeros(([m, n], device));
            cr.fill(0.0);
            let mut f = |cv: &mut MaybeUninit<f64>, x: &f64, y: &f64| { cv.write(x + y); };
            op_mutc_refa_refb_func(&mut cr, &a, &brow_bc, &mut f).unwrap();
            check(&format!("add bcast zero-stride {m}x{n} reuse"), all_close64(cr.raw().as_slice(), &want_bc));

            // ---- [T4'] strided (transpose view) operand: blocked-tile path --
            let btv = gen_vec_f64(m * n, 4);
            let bt = rt::asarray((btv.clone(), [n, m], device));
            let want_st: Vec<f64> = (0..m * n).map(|k| av[k] + btv[(k % n) * m + k / n]).collect();
            let c = &a + &bt.t();
            check(&format!("add strided {m}x{n} alloc"), all_close64(c.raw().as_slice(), &want_st));
            let mut cr: Tensor<f64, _> = rt::zeros(([m, n], device));
            cr.fill(0.0);
            let mut f = |cv: &mut MaybeUninit<f64>, x: &f64, y: &f64| { cv.write(x + y); };
            op_mutc_refa_refb_func(&mut cr, &a, &bt.t(), &mut f).unwrap();
            check(&format!("add strided {m}x{n} reuse (tile path)"), all_close64(cr.raw().as_slice(), &want_st));

            // ---- [T4'] flip views -------------------------------------------
            let a_flip = a.flip(0);
            let want_f0: Vec<f64> = (0..m * n)
                .map(|k| av[(m - 1 - k / n) * n + k % n] + bv[k])
                .collect();
            let c = &a_flip + &b;
            check(&format!("add flip(0) operand {m}x{n}"), all_close64(c.raw().as_slice(), &want_f0));
            let b_flip = b.flip(1);
            let want_ff: Vec<f64> = (0..m * n)
                .map(|k| {
                    let (i, j) = (k / n, k % n);
                    av[(m - 1 - i) * n + j] + bv[i * n + (n - 1 - j)]
                })
                .collect();
            let c = &a_flip + &b_flip;
            check(&format!("add flip(0)+flip(1) {m}x{n}"), all_close64(c.raw().as_slice(), &want_ff));
            let mut cr: Tensor<f64, _> = rt::zeros(([m, n], device));
            cr.fill(0.0);
            let mut f = |cv: &mut MaybeUninit<f64>, x: &f64, y: &f64| { cv.write(x + y); };
            op_mutc_refa_refb_func(&mut cr, &a_flip, &bt.t(), &mut f).unwrap();
            let want_ft: Vec<f64> = (0..m * n)
                .map(|k| {
                    let (i, j) = (k / n, k % n);
                    av[(m - 1 - i) * n + j] + btv[j * m + i]
                })
                .collect();
            check(&format!("add flip(0)+t() {m}x{n} reuse"), all_close64(cr.raw().as_slice(), &want_ft));

            // ---- [T4'] i32 / Complex<f64> spots ------------------------------
            let avi: Vec<i32> = (0..m * n).map(|k| (k % 7) as i32 - 3).collect();
            let bvi: Vec<i32> = (0..m * n).map(|k| (k % 5) as i32 - 2).collect();
            let ai = rt::asarray((avi.clone(), [m, n], device));
            let bi = rt::asarray((bvi.clone(), [m, n], device));
            let want_addi: Vec<i32> = avi.iter().zip(bvi.iter()).map(|(x, y)| x + y).collect();
            let ci = &ai + &bi;
            check(&format!("add contig {m}x{n} alloc i32"), eq_slices(ci.raw().as_slice(), &want_addi));
            let mut cri: Tensor<i32, _> = rt::zeros(([m, n], device));
            cri.fill(0);
            let mut fi = |cv: &mut MaybeUninit<i32>, x: &i32, y: &i32| { cv.write(x + y); };
            op_mutc_refa_refb_func(&mut cri, &ai, &bi, &mut fi).unwrap();
            check(&format!("add contig {m}x{n} reuse i32"), eq_slices(cri.raw().as_slice(), &want_addi));

            let avc: Vec<Complex<f64>> =
                (0..m * n).map(|k| Complex::new(gen_value(k, 1), gen_value(k, 3))).collect();
            let bvc: Vec<Complex<f64>> =
                (0..m * n).map(|k| Complex::new(gen_value(k, 2), gen_value(k, 4))).collect();
            let ac = rt::asarray((avc.clone(), [m, n], device));
            let bc = rt::asarray((bvc.clone(), [m, n], device));
            let want_mulc: Vec<Complex<f64>> = avc.iter().zip(bvc.iter()).map(|(x, y)| x * y).collect();
            let cc = &ac * &bc;
            check(&format!("mul contig {m}x{n} alloc c64"), eq_slices(cc.raw().as_slice(), &want_mulc));
            let mut crc: Tensor<Complex<f64>, _> = rt::zeros(([m, n], device));
            crc.fill(Complex::new(0.0, 0.0));
            let mut fc = |cv: &mut MaybeUninit<Complex<f64>>, x: &Complex<f64>, y: &Complex<f64>| { cv.write(x + y); };
            op_mutc_refa_refb_func(&mut crc, &ac, &bc, &mut fc).unwrap();
            let want_addc: Vec<Complex<f64>> = avc.iter().zip(bvc.iter()).map(|(x, y)| x + y).collect();
            check(&format!("add contig {m}x{n} reuse c64"), eq_slices(crc.raw().as_slice(), &want_addc));

            // ---- [T4'] f32 spots ---------------------------------------------
            let av32 = gen_vec_f32(m * n, 1);
            let bv32 = gen_vec_f32(m * n, 2);
            let a32 = rt::asarray((av32.clone(), [m, n], device));
            let b32 = rt::asarray((bv32.clone(), [m, n], device));
            let want_add32: Vec<f32> = av32.iter().zip(bv32.iter()).map(|(x, y)| x + y).collect();
            let c = &a32 + &b32;
            check(&format!("add contig {m}x{n} alloc f32"), eq_slices(c.raw().as_slice(), &want_add32));
            let mut cr32: Tensor<f32, _> = rt::zeros(([m, n], device));
            cr32.fill(0.0);
            let mut f32_ = |cv: &mut MaybeUninit<f32>, x: &f32, y: &f32| { cv.write(x + y); };
            op_mutc_refa_refb_func(&mut cr32, &a32, &b32, &mut f32_).unwrap();
            check(&format!("add contig {m}x{n} reuse f32"), eq_slices(cr32.raw().as_slice(), &want_add32));

            // ---- [T1'] transpose/assign family -------------------------------
            let want_t: Vec<f64> = (0..n * m).map(|k| av[(k % m) * n + k / m]).collect();
            let out = a.t().to_contig(RowMajor).into_owned();
            check(&format!("A idiom t().to_contig {m}x{n} f64"), out.raw().as_slice() == &want_t);
            let outf = a.t().to_contig(ColMajor).into_owned();
            check(&format!("A idiom to_contig(ColMajor) {m}x{n} (r2c)"), outf.raw().as_slice() == &av);
            let mut c_t: Tensor<f64, _> = rt::zeros(([n, m], device));
            c_t.fill(0.0);
            c_t.assign(&a.t());
            check(&format!("B reuse c.assign(&a.t()) {m}x{n}"), c_t.raw().as_slice() == &want_t);

            // f-contig reuse form (r2c via assign)
            // f-contig [m, n] storage order: cf[i + j*m] = av[i*n + j]
            let want_r2c: Vec<f64> = (0..m * n).map(|p| av[(p % m) * n + p / m]).collect();
            let mut cf: Tensor<f64, _> = rt::zeros(([m, n], device));
            cf = cf.to_contig(ColMajor).into_owned();
            cf.assign(&a);
            check(&format!("assign into f-contig r2c {m}x{n}"), cf.raw().as_slice() == &want_r2c);

            // contig assign sanity
            let mut cc2: Tensor<f64, _> = rt::zeros(([m, n], device));
            cc2.fill(0.0);
            cc2.assign(&a);
            check(&format!("contig assign {m}x{n}"), cc2.raw().as_slice() == &av);
        }

        // ---- [T1'] flip-transposed, sliced, degenerate, bcast, 3-D ---------
        for (m, n) in [(37usize, 53usize), (777usize, 1000usize), (1usize, 7usize), (7usize, 1usize)] {
            let av = gen_vec_f64(m * n, 1);
            let a = rt::asarray((av.clone(), [m, n], device));
            let want_t: Vec<f64> = (0..n * m).map(|k| av[(k % m) * n + k / m]).collect();

            let out = a.t().to_contig(RowMajor).into_owned();
            check(&format!("A idiom {m}x{n} (degenerate set)"), out.raw().as_slice() == &want_t);

            if m > 1 && n > 1 {
                let want_f0: Vec<f64> = (0..n * m).map(|k| av[(m - 1 - k % m) * n + k / m]).collect();
                let outf0 = a.flip(0).t().to_contig(RowMajor).into_owned();
                check(&format!("flip(0).t() A {m}x{n}"), outf0.raw().as_slice() == &want_f0);
                let mut cf0: Tensor<f64, _> = rt::zeros(([n, m], device));
                cf0.fill(0.0);
                cf0.assign(&a.flip(0).t());
                check(&format!("flip(0).t() B reuse {m}x{n}"), cf0.raw().as_slice() == &want_f0);
                let want_f1: Vec<f64> = (0..n * m).map(|k| av[(k % m) * n + (n - 1 - k / m)]).collect();
                let outf1 = a.flip(1).t().to_contig(RowMajor).into_owned();
                check(&format!("flip(1).t() A {m}x{n}"), outf1.raw().as_slice() == &want_f1);

                // sliced fast-axis stride-2 view: must fall through generic
                let n2 = (n + 1) / 2;
                let a_sliced = a.i((.., slice!(0, n, 2)));
                let want_sliced: Vec<f64> =
                    (0..m * n2).map(|k| av[(k / n2) * n + (k % n2) * 2]).collect();
                let outs = a_sliced.to_contig(RowMajor).into_owned();
                check(&format!("sliced view to_contig {m}x{n2}"), outs.raw().as_slice() == &want_sliced);
                let mut cs: Tensor<f64, _> = rt::zeros(([n2, m], device));
                cs.fill(0.0);
                cs.assign(&a_sliced.t());
                let want_sliced_t: Vec<f64> =
                    (0..n2 * m).map(|k| av[(k % m) * n + (k / m) * 2]).collect();
                check(&format!("sliced.t() assign {n2}x{m}"), cs.raw().as_slice() == &want_sliced_t);
            }
        }

        // broadcast slow-axis + transposed + 3-D + dtype spots
        {
            let (m, n) = (37usize, 53usize);
            let rowv = gen_vec_f64(n, 3);
            let brow = rt::asarray((rowv.clone(), [1usize, n], device));
            let brow_bc = brow.broadcast_to(vec![m, n]);

            let mut c1: Tensor<f64, _> = rt::zeros(([m, n], device));
            c1.fill(0.0);
            c1.assign(&brow_bc);
            let want_bc: Vec<f64> = (0..m * n).map(|k| rowv[k % n]).collect();
            check("bcast assign [1,n]->[m,n]", c1.raw().as_slice() == &want_bc);

            let brow_bt = brow_bc.t();
            let out_bt = brow_bt.to_contig(RowMajor).into_owned();
            let want_bt: Vec<f64> = (0..n * m).map(|k| rowv[k / m]).collect();
            check("bcast.t() to_contig", out_bt.raw().as_slice() == &want_bt);
            let mut c2: Tensor<f64, _> = rt::zeros(([n, m], device));
            c2.fill(0.0);
            c2.assign(&brow_bt);
            check("bcast.t() assign", c2.raw().as_slice() == &want_bt);

            let a3v = gen_vec_f64(2 * 3 * 4, 5);
            let a3 = rt::asarray((a3v.clone(), [2usize, 3, 4], device));
            let out3 = a3.t().to_contig(RowMajor).into_owned();
            let want3: Vec<f64> = (0..24).map(|k| a3v[((k % 2) * 3 + (k / 2) % 3) * 4 + k / 6]).collect();
            check("3-D t().to_contig 2x3x4", out3.raw().as_slice() == &want3);

            let avi: Vec<i32> = (0..m * n).map(|k| (k % 7) as i32 - 3).collect();
            let ai = rt::asarray((avi.clone(), [m, n], device));
            let outi = ai.t().to_contig(RowMajor).into_owned();
            let wanti: Vec<i32> = (0..n * m).map(|k| avi[(k % m) * n + k / m]).collect();
            check("A idiom 37x53 i32", outi.raw().as_slice() == &wanti);

            let av32 = gen_vec_f32(1000 * 777, 1);
            let a32 = rt::asarray((av32.clone(), [1000usize, 777], device));
            let want32: Vec<f32> = (0..777 * 1000).map(|k| av32[(k % 1000) * 777 + k / 1000]).collect();
            let out32 = a32.t().to_contig(RowMajor).into_owned();
            check("A idiom 1000x777 f32", out32.raw().as_slice() == &want32);
            let mut c32: Tensor<f32, _> = rt::zeros(([777, 1000], device));
            c32.fill(0.0);
            c32.assign(&a32.t());
            check("B reuse 1000x777 f32", c32.raw().as_slice() == &want32);
        }
    }};
}

// ===========================================================================
// [T2' reductions]
// ===========================================================================
macro_rules! section_red {
    ($dev:expr) => {{
        let device = &$dev;

        for (m, n) in [(8usize, 6usize), (1000usize, 777usize)] {
            let a_data = gen_vec_f64(m * n, 1);
            let a = rt::asarray((a_data.clone(), [m, n], device));

            check(&format!("shape sum_axes(0) {m}x{n} == [{n}]"), a.sum_axes(0).shape().to_vec() == vec![n]);
            check(&format!("shape sum_axes(-1) {m}x{n} == [{m}]"), a.sum_axes(-1).shape().to_vec() == vec![m]);

            let sum: Vec<f64> = (0..n)
                .map(|j| (0..m).map(|i| a_data[i * n + j]).sum())
                .collect();
            let row_sum: Vec<f64> = (0..m)
                .map(|i| (0..n).map(|j| a_data[i * n + j]).sum())
                .collect();
            let got = a.sum_axes(0).to_vec();
            check(&format!("sum_axis0 {m}x{n}"), all_close64(&got, &sum));
            let got = a.sum_axes(-1).to_vec();
            check(&format!("sum_axis1 {m}x{n}"), all_close64(&got, &row_sum));

            let got = a.min_axes(0).to_vec();
            let want_min: Vec<f64> = (0..n)
                .map(|j| (0..m).map(|i| a_data[i * n + j]).fold(f64::INFINITY, f64::min))
                .collect();
            check(&format!("min_axis0 {m}x{n}"), eq_slices(&got, &want_min));
            let got = a.max_axes(-1).to_vec();
            let want_max: Vec<f64> = (0..m)
                .map(|i| (0..n).map(|j| a_data[i * n + j]).fold(f64::NEG_INFINITY, f64::max))
                .collect();
            check(&format!("max_axis1 {m}x{n}"), eq_slices(&got, &want_max));

            let total: f64 = a_data.iter().sum();
            let got: f64 = a.sum_all();
            check(&format!("sum_all {m}x{n}"), (got - total).abs() <= 1e-9 * (1.0 + total.abs()));
            let got: f64 = a.min_all();
            let want = a_data.iter().copied().fold(f64::INFINITY, f64::min);
            check(&format!("min_all {m}x{n}"), got == want);
            let got: f64 = a.max_all();
            let want = a_data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            check(&format!("max_all {m}x{n}"), got == want);
        }

        // 3-D spot
        {
            let (p, q, r_) = (4usize, 5usize, 7usize);
            let a_data = gen_vec_f64(p * q * r_, 3);
            let a = rt::asarray((a_data.clone(), [p, q, r_], device));
            let got = a.sum_axes(0).reshape(-1).to_vec();
            let want: Vec<f64> = (0..q * r_)
                .map(|idx| (0..p).map(|k| a_data[k * q * r_ + idx]).sum())
                .collect();
            check("sum_axis0 3-D 4x5x7", all_close64(&got, &want));
            check("shape sum_axes(0) 3-D == [q, r_]", a.sum_axes(0).shape().to_vec() == vec![q, r_]);
            let got = a.min_axes(0).reshape(-1).to_vec();
            let want: Vec<f64> = (0..q * r_)
                .map(|idx| (0..p).map(|k| a_data[k * q * r_ + idx]).fold(f64::INFINITY, f64::min))
                .collect();
            check("min_axis0 3-D 4x5x7", eq_slices(&got, &want));
        }

        // zero-stride broadcast locks (incl. documented upstream sum bug)
        {
            let (m, n) = (200usize, 777usize);
            let row_data = gen_vec_f64(n, 5);
            let row = rt::asarray((row_data.clone(), [1, n], device));
            let a_data = gen_vec_f64(m * n, 6);
            let a = rt::asarray((a_data.clone(), [m, n], device));
            let got = (&row + &a).sum_axes(0).to_vec();
            let want: Vec<f64> = (0..n)
                .map(|j| row_data[j] * m as f64 + (0..m).map(|i| a_data[i * n + j]).sum::<f64>())
                .collect();
            check("broadcast-row add+sum_axis0", all_close64(&got, &want));

            // CURRENT-BEHAVIOR lock: stride-0 summed axis multiplies by the
            // WRONG count (n instead of m) — documented upstream bug that
            // T2' preserved bit-for-bit.
            let row_view = row.broadcast_to(vec![m, n]);
            let got = row_view.sum_axes(0).to_vec();
            let want_current: Vec<f64> = row_data.iter().map(|&x| x * n as f64).collect();
            check("zero-stride sum_axis0 (CURRENT-BEHAVIOR lock, upstream bug)", all_close64(&got, &want_current));

            let got = row_view.min_axes(0).to_vec();
            check("zero-stride min_axis0", eq_slices(&got, &row_data));
            let got = row_view.sum_axes(-1).to_vec();
            let want_rs: Vec<f64> = (0..m).map(|_| row_data.iter().sum::<f64>()).collect();
            check("zero-stride sum_axis1", all_close64(&got, &want_rs));
        }

        // transposed input
        {
            let (m, n) = (256usize, 384usize);
            let a_data = gen_vec_f64(m * n, 8);
            let a = rt::asarray((a_data.clone(), [m, n], device));
            let at = a.t();
            let row_sum: Vec<f64> = (0..m)
                .map(|i| (0..n).map(|j| a_data[i * n + j]).sum())
                .collect();
            let col_sum: Vec<f64> = (0..n)
                .map(|j| (0..m).map(|i| a_data[i * n + j]).sum())
                .collect();
            let got = at.sum_axes(0).to_vec();
            check("sum_axis0 of t-view", all_close64(&got, &row_sum));
            let got = at.sum_axes(-1).to_vec();
            check("sum_axis1 of t-view", all_close64(&got, &col_sum));
        }

        // NaN poison (sum) vs NaN skip (min/max) + all-NaN seed
        {
            let mut a_data = gen_vec_f64(64, 9);
            a_data[3] = f64::NAN;
            let a = rt::asarray((a_data.clone(), [8, 8], device));
            let s1 = a.sum_axes(-1).to_vec();
            check("NaN poisons sum of its row", s1[0].is_nan() && s1[1..].iter().all(|x| !x.is_nan()));
            let s0 = a.sum_axes(0).to_vec();
            check(
                "NaN poisons sum of its column",
                s0[3].is_nan() && s0.iter().enumerate().all(|(j, x)| j == 3 || !x.is_nan()),
            );
            let m1 = a.min_axes(-1).to_vec();
            let want_min: Vec<f64> = (0..8usize)
                .map(|i| {
                    (0..8usize)
                        .map(|j| a_data[i * 8 + j])
                        .filter(|x| !x.is_nan())
                        .fold(f64::INFINITY, f64::min)
                })
                .collect();
            check("NaN skipped by min_axis1", eq_slices(&m1, &want_min));
            let mx1 = a.max_axes(-1).to_vec();
            let want_max: Vec<f64> = (0..8usize)
                .map(|i| {
                    (0..8usize)
                        .map(|j| a_data[i * 8 + j])
                        .filter(|x| !x.is_nan())
                        .fold(f64::NEG_INFINITY, f64::max)
                })
                .collect();
            check("NaN skipped by max_axis1", eq_slices(&mx1, &want_max));

            let an = rt::asarray((vec![f64::NAN; 16], [4usize, 4], device));
            let m0 = an.min_axes(0).to_vec();
            let mx0 = an.max_axes(0).to_vec();
            check(
                "all-NaN axis0: min -> +MAX seed, max -> -MIN seed",
                m0.iter().all(|&x| x == f64::MAX) && mx0.iter().all(|&x| x == f64::MIN),
            );
        }

        // signbit locks (strict-compare fold keeps the incumbent on ±0 ties)
        {
            let a = rt::asarray((vec![-0.0f64, 0.0], [1usize, 2], device));
            let m: f64 = a.min_all();
            check(
                "signbit: min_all([-0,+0]) keeps incumbent -0.0",
                m == 0.0 && m.is_sign_negative(),
            );
            let b = rt::asarray((vec![0.0f64, -0.0], [1usize, 2], device));
            let x: f64 = b.max_all();
            check(
                "signbit: max_all([+0,-0]) keeps incumbent +0.0",
                x == 0.0 && !x.is_sign_negative(),
            );
            let c = rt::asarray((vec![-0.0f64, 0.0, 0.0, -0.0], [2usize, 2], device));
            let mn = c.min_axes(-1).reshape(-1).to_vec();
            check(
                "signbit: min_axes(-1) [[-0,+0],[+0,-0]] -> [-0,+0]",
                mn[0].is_sign_negative() && !mn[1].is_sign_negative(),
            );
            let mn0 = c.min_axes(0).reshape(-1).to_vec();
            check(
                "signbit: min_axes(0) [[-0,+0],[+0,-0]] -> [-0,+0]",
                mn0[0].is_sign_negative() && !mn0[1].is_sign_negative(),
            );
        }

        // i32 min/max
        {
            let (m, n) = (64usize, 97usize);
            let a_data: Vec<i32> =
                (0..m * n).map(|i| ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) % 2003) as i32 - 1000).collect();
            let a = rt::asarray((a_data.clone(), [m, n], device));
            let got = a.min_axes(0).reshape(-1).to_vec();
            let want: Vec<i32> = (0..n)
                .map(|j| (0..m).map(|i| a_data[i * n + j]).fold(i32::MAX, i32::min))
                .collect();
            check("i32 min_axis0", got == want);
            let got = a.max_axes(-1).reshape(-1).to_vec();
            let want: Vec<i32> = (0..m)
                .map(|i| (0..n).map(|j| a_data[i * n + j]).fold(i32::MIN, i32::max))
                .collect();
            check("i32 max_axis1", got == want);
            let got: i32 = a.min_all();
            check("i32 min_all", got == a_data.iter().copied().fold(i32::MAX, i32::min));
        }

        // f32 spot
        {
            let (m, n) = (255usize, 769usize);
            let a_data = gen_vec_f32(m * n, 10);
            let a = rt::asarray((a_data.clone(), [m, n], device));
            let got = a.min_axes(0).to_vec();
            let want: Vec<f32> = (0..n)
                .map(|j| (0..m).map(|i| a_data[i * n + j]).fold(f32::INFINITY, f32::min))
                .collect();
            check("min_axis0 f32 255x769", got == want);
        }
    }};
}

// ===========================================================================
// [T3' vecdot]
// ===========================================================================
macro_rules! section_vec {
    ($dev:expr) => {{
        let device = &$dev;

        let naive_dot = |a: &[f64], b: &[f64]| -> f64 { a.iter().zip(b).map(|(x, y)| x * y).sum() };

        // 1-D dots + NaN
        let n = 777;
        let av: Vec<f64> = (0..n).map(|i| gen_value(i, 1)).collect();
        let bv: Vec<f64> = (0..n).map(|i| gen_value(i, 2)).collect();
        let a = rt::asarray((av.clone(), [n], device));
        let b = rt::asarray((bv.clone(), [n], device));
        let c = rt::vecdot(&a, &b, None).reshape(-1).to_vec();
        check("dot1d f64 odd", (c[0] - naive_dot(&av, &bv)).abs() <= 1e-9 * (1.0 + c[0].abs()));

        let av32: Vec<f32> = (0..n).map(|i| gen_value(i, 1) as f32).collect();
        let bv32: Vec<f32> = (0..n).map(|i| gen_value(i, 2) as f32).collect();
        let a32 = rt::asarray((av32.clone(), [n], device));
        let b32 = rt::asarray((bv32.clone(), [n], device));
        let c32 = rt::vecdot(&a32, &b32, None).reshape(-1).to_vec();
        let expect32: f32 = av32.iter().zip(&bv32).map(|(x, y)| x * y).sum();
        check("dot1d f32 odd", (c32[0] - expect32).abs() <= 1e-3 * expect32.abs());

        let avc: Vec<Complex<f64>> = (0..n).map(|i| Complex::new(gen_value(i, 1), gen_value(i, 31))).collect();
        let bvc: Vec<Complex<f64>> = (0..n).map(|i| Complex::new(gen_value(i, 2), gen_value(i, 32))).collect();
        let ac = rt::asarray((avc.clone(), [n], device));
        let bc = rt::asarray((bvc.clone(), [n], device));
        let cc = rt::vecdot(&ac, &bc, None).reshape(-1).to_vec();
        let want_cc: Complex<f64> = avc.iter().zip(&bvc).map(|(x, y)| x.conj() * y).sum();
        check(
            "dot1d c64 odd (conj)",
            (cc[0] - want_cc).re.abs() <= 1e-9 * (1.0 + want_cc.re.abs())
                && (cc[0] - want_cc).im.abs() <= 1e-9 * (1.0 + want_cc.im.abs()),
        );

        let mut avn = av.clone();
        avn[100] = f64::NAN;
        let an = rt::asarray((avn, [n], device));
        let cn = rt::vecdot(&an, &b, None).reshape(-1).to_vec();
        check("dot1d NaN in a -> NaN", cn[0].is_nan());
        let dn = &an % &b;
        check("innerdot NaN -> NaN", dn.reshape(-1).to_vec()[0].is_nan());

        // batched am1 + axis0
        let (m, k) = (1000usize, 777usize);
        let ma: Vec<f64> = (0..m * k).map(|i| gen_value(i, 11)).collect();
        let mb: Vec<f64> = (0..m * k).map(|i| gen_value(i, 12)).collect();
        let ta = rt::asarray((ma.clone(), [m, k], device));
        let tb = rt::asarray((mb.clone(), [m, k], device));

        let tc = rt::vecdot(&ta, &tb, None).reshape(-1).to_vec();
        let want_am1: Vec<f64> = (0..m)
            .map(|i| naive_dot(&ma[i * k..(i + 1) * k], &mb[i * k..(i + 1) * k]))
            .collect();
        check("batched am1 f64 1000x777", all_close64(&tc, &want_am1));

        let tc0 = rt::vecdot(&ta, &tb, 0).reshape(-1).to_vec();
        let want_axis0: Vec<f64> = (0..k)
            .map(|j| (0..m).map(|i| ma[i * k + j] * mb[i * k + j]).sum())
            .collect();
        check("batched axis0 f64 1000x777", all_close64(&tc0, &want_axis0));

        // axis0 fall-through: remaining run above the 32 KiB budget
        let (m2, k2) = (8usize, 20000usize);
        let ma2: Vec<f64> = (0..m2 * k2).map(|i| gen_value(i, 21)).collect();
        let mb2: Vec<f64> = (0..m2 * k2).map(|i| gen_value(i, 22)).collect();
        let ta2 = rt::asarray((ma2.clone(), [m2, k2], device));
        let tb2 = rt::asarray((mb2.clone(), [m2, k2], device));
        let tc2 = rt::vecdot(&ta2, &tb2, 0).reshape(-1).to_vec();
        let want2: Vec<f64> = (0..k2)
            .map(|j| (0..m2).map(|i| ma2[i * k2 + j] * mb2[i * k2 + j]).sum())
            .collect();
        check("batched axis0 large fall-through 8x20000", all_close64(&tc2, &want2));

        // 3-D ax1 fall-through
        let (d0, d1_, d2_) = (2usize, 3usize, 4usize);
        let m3: Vec<f64> = (0..d0 * d1_ * d2_).map(|i| gen_value(i, 23)).collect();
        let n3v: Vec<f64> = (0..d0 * d1_ * d2_).map(|i| gen_value(i, 24)).collect();
        let t3a = rt::asarray((m3.clone(), [d0, d1_, d2_], device));
        let t3b = rt::asarray((n3v.clone(), [d0, d1_, d2_], device));
        let t3c = rt::vecdot(&t3a, &t3b, None).reshape(-1).to_vec();
        let t3_expect: Vec<f64> = (0..d0 * d1_)
            .map(|r| naive_dot(&m3[r * d2_..(r + 1) * d2_], &n3v[r * d2_..(r + 1) * d2_]))
            .collect();
        check("batched 3-D ax1 fall-through", all_close64(&t3c, &t3_expect));

        // strided b (transposed view)
        let tbst = rt::asarray((mb.clone(), [k, m], device));
        let tbv = tbst.t();
        let tcs = rt::vecdot(&ta, &tbv, None).reshape(-1).to_vec();
        let expect_s: Vec<f64> = (0..m)
            .map(|i| (0..k).map(|j| ma[i * k + j] * mb[i + j * m]).sum::<f64>())
            .collect();
        check("batched strided b (t-view)", all_close64(&tcs, &expect_s));

        // f-contig a
        let taf = ta.to_contig(ColMajor);
        let tcf = rt::vecdot(&taf, &tb, None).reshape(-1).to_vec();
        check("batched am1 f-contig a", all_close64(&tcf, &want_am1));

        // broadcast cases
        let tv = rt::asarray((bv.clone(), [k], device));
        let tcb = rt::vecdot(&ta, &tv, None).reshape(-1).to_vec();
        let expect_b: Vec<f64> = (0..m)
            .map(|i| naive_dot(&ma[i * k..(i + 1) * k], &bv))
            .collect();
        check("batched am1 broadcast 1-D b", all_close64(&tcb, &expect_b));
        let tvr = tv.broadcast_to(vec![m, k]);
        let tcb2 = rt::vecdot(&ta, &tvr, None).reshape(-1).to_vec();
        check("batched am1 broadcast remaining-axis", all_close64(&tcb2, &expect_b));
        let v: Vec<f64> = (0..m).map(|i| gen_value(i, 14)).collect();
        let tb2 = rt::asarray((v.clone(), [m, 1], device));
        let tb2b = tb2.broadcast_to(vec![m, k]);
        let tcb4 = rt::vecdot(&ta, &tb2b, None).reshape(-1).to_vec();
        let expect4: Vec<f64> = (0..m)
            .map(|i| v[i] * (0..k).map(|j| ma[i * k + j]).sum::<f64>())
            .collect();
        check("batched am1 broadcast summed-axis", all_close64(&tcb4, &expect4));

        // inner_dot %: contig, n=300 sequential fallback, strided row views,
        // c64 no-conj, n=1
        let d1 = &a % &b;
        check("innerdot % contig", all_close64(&d1.reshape(-1).to_vec(), &[naive_dot(&av, &bv)]));
        let n3 = 300;
        let a3 = rt::asarray((av[..n3].to_vec(), [n3], device));
        let b3 = rt::asarray((bv[..n3].to_vec(), [n3], device));
        let d3 = &a3 % &b3;
        check(
            "innerdot % n=300 seq fallback",
            all_close64(&d3.reshape(-1).to_vec(), &[naive_dot(&av[..n3], &bv[..n3])]),
        );
        let row_a = ta.i(3);
        let row_b = tb.i(3);
        let d2 = &row_a % &row_b;
        check(
            "innerdot % strided row views",
            all_close64(&d2.reshape(-1).to_vec(), &[naive_dot(&ma[3 * k..4 * k], &mb[3 * k..4 * k])]),
        );
        let dc = &ac % &bc;
        let expect_c: Complex<f64> = avc.iter().zip(&bvc).map(|(x, y)| x * y).sum();
        let dcv = dc.reshape(-1).to_vec();
        check(
            "innerdot % c64 no-conj",
            (dcv[0] - expect_c).re.abs() <= 1e-9 * (1.0 + expect_c.re.abs())
                && (dcv[0] - expect_c).im.abs() <= 1e-9 * (1.0 + expect_c.im.abs()),
        );
        let a1 = rt::asarray((vec![2.5_f64], [1], device));
        let b1 = rt::asarray((vec![-4.0_f64], [1], device));
        check("innerdot % n=1", (&a1 % &b1).reshape(-1).to_vec()[0] == -10.0);
        check("dot1d n=1", rt::vecdot(&a1, &b1, None).reshape(-1).to_vec()[0] == -10.0);

        // alpha/beta fallback via rt::matmul_from (fast path requires
        // alpha == 1 && beta == 0; anything else must take the fallback)
        {
            let nn = 501usize;
            let xa: Vec<f64> = (0..nn).map(|i| gen_value(i, 61)).collect();
            let xb: Vec<f64> = (0..nn).map(|i| gen_value(i, 62)).collect();
            let xa_t = rt::asarray((xa.clone(), [nn], device));
            let xb_t = rt::asarray((xb.clone(), [nn], device));
            let want_dot = naive_dot(&xa, &xb);

            // alpha only (beta = 0)
            let mut c_ab: Tensor<f64, _> = rt::zeros(([], device));
            c_ab.fill(0.0);
            rt::matmul_from(&mut c_ab, &xa_t, &xb_t, 2.0, 0.0);
            let got = c_ab.reshape(-1).to_vec()[0];
            check("alpha/beta fallback (a=2.0, b=0)", (got - 2.0 * want_dot).abs() <= 1e-9 * (1.0 + want_dot.abs()));

            // alpha + beta accumulate (c pre-filled with 7.0)
            let mut c_ab2: Tensor<f64, _> = rt::zeros(([], device));
            c_ab2.fill(7.0);
            rt::matmul_from(&mut c_ab2, &xa_t, &xb_t, 2.0, 0.5);
            let got2 = c_ab2.reshape(-1).to_vec()[0];
            check(
                "alpha/beta fallback (a=2.0, b=0.5 accumulates)",
                (got2 - (2.0 * want_dot + 0.5 * 7.0)).abs() <= 1e-9 * (1.0 + want_dot.abs()),
            );
        }
    }};
}

fn main() {
    println!("=== T8 compose-smoke correctness gate (combined 5-patch tree) ===");
    println!("-- device: DeviceCpuSerial");
    let dev_serial = serial_device();
    section_arg!(dev_serial);
    section_elem_trans!(dev_serial);
    section_red!(dev_serial);
    section_vec!(dev_serial);

    println!("-- device: DeviceFaer (default, RAYON_NUM_THREADS=16)");
    let dev_faer = faer_device();
    assert_faer_threads(&dev_faer, 16);
    section_arg!(dev_faer);
    section_elem_trans!(dev_faer);
    section_red!(dev_faer);
    section_vec!(dev_faer);

    let n = N_CHECKS.load(Ordering::Relaxed);
    println!("\ncorrectness gate: ALL PASSED ({n} checks)");
}

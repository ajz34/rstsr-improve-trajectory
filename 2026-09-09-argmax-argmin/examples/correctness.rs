//! Correctness gate for T6 argmax/argmin (run BEFORE trusting any perf number,
//! and reused as the regression gate for the phase-2 kernel patch).
//!
//! Establishes and locks down the CURRENT (386948be) rstsr semantics:
//!
//! - **Tie-breaking**: the FIRST occurrence in row-major visit order wins
//!   (i.e. the lowest row-major flat index among equal extremes). Verified by
//!   all-equal fixtures and by duplicated-maximum fixtures (incl. sizes above
//!   the rayon PARALLEL_SWITCH, so the faer fold/reduce combine is covered).
//! - **NaN handling**: the first row-major element seeds the accumulator
//!   unconditionally; afterwards only a strictly-greater value replaces it and
//!   NaN compares are always false. Consequences (all verified below, and
//!   they coincide with numpy's documented argmax/argmin NaN behavior):
//!   NaN at the END/MIDDLE of real data never wins (the real max wins);
//!   a NaN at the FRONT poisons the result (returns flat index 0);
//!   an all-NaN input returns flat index 0 (no error).
//! - **Empty input**: the fallible API (`argmax_f`/`argmin_f`) returns
//!   Err(InvalidLayout) "empty sequence is not allowed for reduce_arg."; the
//!   infallible wrapper (`rt::argmax`) PANICS via `rstsr_unwrap`. Verified
//!   below with `catch_unwind`.
//!
//! The naive reference below is a literal transcription of the current fold
//! (seed first element; replace only on strict >/<; ties keep the earlier
//! index) applied to the row-major visit sequence. Cross-checked against
//! ndarray-stats `argmax`/`argmin` on the NaN-free cases (ndarray-stats errors
//! on NaN, so NaN cases rest on the rstsr-vs-reference comparison only).
//!
//! Coverage: contiguous odd size (999_983), all-equal, duplicated extremes,
//! NaN placements, empty, 2-D contiguous, 2-D transpose view (strided),
//! zero-stride broadcast view, axis reductions (argmax_axes/-1/0 incl. on a
//! t-view), f32 + f64, BOTH devices (DeviceCpuSerial; DeviceFaer with
//! RAYON_NUM_THREADS=16 asserted).
//!
//! Exits non-zero on any mismatch. Run under both RUSTFLAGS configs.

use std::panic::{catch_unwind, set_hook, take_hook};

use bench_argmax_argmin::{assert_faer_threads, faer_device, gen_vec_f32, gen_vec_f64, serial_device};
use ndarray::Array1;
use ndarray_stats::QuantileExt;
use rstsr_core::prelude::*;

fn check(name: &str, cond: bool) {
    if cond {
        println!("  [ok]   {name}");
    } else {
        eprintln!("  [FAIL] {name}");
        std::process::exit(1);
    }
}

/// Naive scalar reference: literal transcription of rstsr's current arg fold
/// over a row-major visit sequence `(flat_row_major_index, value)` pairs.
fn ref_arg(visit: &[(usize, f64)], want_max: bool) -> usize {
    assert!(!visit.is_empty(), "reference not defined for empty input");
    let mut best_idx = visit[0].0; // seed: first visited element, unconditionally
    let mut best_val = visit[0].1;
    for &(idx, x) in &visit[1..] {
        let better = if want_max { x > best_val } else { x < best_val };
        // NaN comparisons are false: NaN never replaces the accumulator, and
        // equal values keep the earlier (lower flat) index.
        if better {
            best_idx = idx;
            best_val = x;
        }
    }
    best_idx
}

/// Row-major visit sequence of a contiguous [m, n] matrix stored row-major.
fn visit_contig<T: Copy + Into<f64>>(data: &[T], m: usize, n: usize) -> Vec<(usize, f64)> {
    (0..m * n).map(|k| (k, data[k].into())).collect()
}

/// Row-major visit sequence of the transpose view of a contiguous [m, n]
/// matrix: view element (i, j) = data[j * n + i], visited i-major then j.
fn visit_tview<T: Copy + Into<f64>>(data: &[T], m: usize, n: usize) -> Vec<(usize, f64)> {
    let mut out = Vec::with_capacity(m * n);
    for i in 0..n {
        for j in 0..m {
            out.push((i * m + j, data[j * n + i].into()));
        }
    }
    out
}

/// The whole gate, instantiated per device via macro (concrete ops only).
macro_rules! check_device {
    ($dev_name:expr, $_device:expr) => {{
        let device = &$_device;
        println!("-- device: {}", $dev_name);

        // --------------------------------------------------------------
        // 1-D contiguous, odd size 999_983 (prime; catches lane-tail bugs)
        // --------------------------------------------------------------
        {
            let n = 999_983usize;
            let data = gen_vec_f64(n, 1);
            let a = rt::asarray((data.clone(), [n], device));
            let visit = visit_contig(&data, 1, n);
            check(&format!("argmax 1-D contig odd n={n}"), rt::argmax(&a) == ref_arg(&visit, true));
            check(&format!("argmin 1-D contig odd n={n}"), rt::argmin(&a) == ref_arg(&visit, false));
            // ndarray cross-check (NaN-free data)
            let nd = Array1::from(data.clone());
            check(
                &format!("argmax 1-D odd n={n} (ndarray cross-check)"),
                nd.argmax().unwrap() == rt::argmax(&a) && nd.argmin().unwrap() == rt::argmin(&a),
            );
        }

        // --------------------------------------------------------------
        // ties: all-equal values (both extremes everywhere -> index 0)
        // --------------------------------------------------------------
        for &n in &[127usize, 4099usize] {
            let data = vec![2.5_f64; n];
            let a = rt::asarray((data.clone(), [n], device));
            check(&format!("argmax all-equal n={n} -> 0"), rt::argmax(&a) == 0);
            check(&format!("argmin all-equal n={n} -> 0"), rt::argmin(&a) == 0);
        }
        // duplicated maximum at scattered positions: the LOWEST flat index wins
        {
            let n = 4099usize; // above PARALLEL_SWITCH=1024: covers rayon combine
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
            // duplicated minimum likewise
            let mut data3 = gen_vec_f64(n, 3);
            for &k in &[31usize, 777, 4098] {
                data3[k] = -100.0;
            }
            let a3 = rt::asarray((data3.clone(), [n], device));
            check("argmin duplicated min (first at 31) -> 31", rt::argmin(&a3) == 31);
        }

        // --------------------------------------------------------------
        // NaN placements (current semantics; matches numpy)
        // --------------------------------------------------------------
        {
            let n = 5003usize;
            let base = gen_vec_f64(n, 4);

            // NaN in the middle: real max wins
            let mut d = base.clone();
            d[n / 2] = f64::NAN;
            let a = rt::asarray((d.clone(), [n], device));
            let visit = visit_contig(&d, 1, n);
            check("argmax NaN middle (real max wins)", rt::argmax(&a) == ref_arg(&visit, true));

            // NaN at the end
            let mut d = base.clone();
            d[n - 1] = f64::NAN;
            let a = rt::asarray((d.clone(), [n], device));
            let visit = visit_contig(&d, 1, n);
            check("argmax NaN end (real max wins)", rt::argmax(&a) == ref_arg(&visit, true));

            // NaN at the front poisons: result is flat index 0
            let mut d = base.clone();
            d[0] = f64::NAN;
            let a = rt::asarray((d.clone(), [n], device));
            check("argmax NaN front -> 0 (NaN poisons)", rt::argmax(&a) == 0);

            // NaN at positions 1..8 with the true max later IN THE SAME LANE
            // of the 8-lane scan (position + 8): caught an early draft of the
            // phase-2 kernel where a NaN lane seed blocked its lane. Under
            // the exact semantics only index 0 may poison, so the real max
            // must win with its own index.
            {
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
                // NaN at 1..8 with the max far away in a different lane
                let mut dd = gen_vec_f64(m, 47);
                dd[3] = f64::NAN;
                dd[m / 2] = 1000.0;
                let a5 = rt::asarray((dd, [m], device));
                check("argmax NaN at 3, max mid (other lane) -> mid", rt::argmax(&a5) == m / 2);
            }

            // only NaN: index 0, no error
            let d = vec![f64::NAN; n];
            let a = rt::asarray((d.clone(), [n], device));
            check("argmax all-NaN -> 0 (no error)", rt::argmax(&a) == 0);
            check("argmin all-NaN -> 0 (no error)", rt::argmin(&a) == 0);

            // argmin NaN placements mirrored
            let mut d = base.clone();
            d[n / 2] = f64::NAN;
            let a = rt::asarray((d.clone(), [n], device));
            let visit = visit_contig(&d, 1, n);
            check("argmin NaN middle (real min wins)", rt::argmin(&a) == ref_arg(&visit, false));
            let mut d = base.clone();
            d[0] = f64::NAN;
            let a = rt::asarray((d.clone(), [n], device));
            check("argmin NaN front -> 0 (NaN poisons)", rt::argmin(&a) == 0);

            // ----------------------------------------------------------
            // NaN at chunk-start offsets, n > PARALLEL_SWITCH (review G1
            // item 2). The PRE-PATCH rayon kernel seeded each thread-chunk
            // independently, so a NaN at a chunk start could suppress the
            // true global max depending on fold/reduce pairing order (the
            // clean tree may return a NaN index here — that instability is
            // exactly what this fixture exercises). The patched kernel
            // combines chunk partials deterministically: NaN anywhere except
            // flat index 0 never wins, matching the serial fold exactly.
            // ----------------------------------------------------------
            {
                let m = 10_003usize;
                let mut dd = gen_vec_f64(m, 44);
                let mut max_idx = m / 2; // global max well inside the buffer
                dd[max_idx] = 1000.0;
                // scatter NaNs at offsets that land on chunk starts for
                // various plausible thread counts/chunkings
                let nan_positions = [156usize, 312, 625, 1250, 2500, 5000];
                for &k in &nan_positions {
                    dd[k] = f64::NAN;
                }
                if nan_positions.contains(&max_idx) {
                    max_idx = m / 2 + 1;
                    dd[max_idx] = 1000.0;
                }
                let a2 = rt::asarray((dd.clone(), [m], device));
                let got = rt::argmax(&a2);
                if got != max_idx {
                    println!(
                        "         [info] NaN-at-chunk-start case: expected {max_idx}, got {got} \
                         (NaN at {nan_positions:?}, n={m})"
                    );
                }
                check(
                    "argmax NaN at chunk-start offsets (real max wins, deterministic)",
                    got == max_idx,
                );
                // mirrored argmin
                let mut dd2 = gen_vec_f64(m, 45);
                let min_idx = m / 2;
                dd2[min_idx] = -1000.0;
                for &k in &nan_positions {
                    dd2[k] = f64::NAN;
                }
                let a3 = rt::asarray((dd2, [m], device));
                check(
                    "argmin NaN at chunk-start offsets (real min wins, deterministic)",
                    rt::argmin(&a3) == min_idx,
                );
            }
        }

        // --------------------------------------------------------------
        // empty tensor: _f returns Err; infallible wrapper panics
        // --------------------------------------------------------------
        {
            let empty = rt::asarray((Vec::<f64>::new(), [0usize], device));
            let e_max = rt::argmax_f(&empty);
            let e_min = rt::argmin_f(&empty);
            check(
                "argmax_f empty -> Err",
                e_max.is_err() && e_min.is_err(),
            );
            if let Err(e) = &e_max {
                println!("         [document] argmax_f(empty) error: {e}");
            }
            // the infallible wrapper panics via rstsr_unwrap
            let prev_hook = take_hook();
            set_hook(Box::new(|_| {})); // silence the panic printout
            let panicked = catch_unwind(std::panic::AssertUnwindSafe(|| rt::argmax(&empty)));
            set_hook(prev_hook);
            match panicked {
                Ok(_) => check("argmax empty -> panics (expected panic)", false),
                Err(payload) => {
                    let msg = payload
                        .downcast_ref::<String>()
                        .cloned()
                        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                        .unwrap_or_default();
                    let ok = msg.contains("empty sequence");
                    println!("         [document] argmax(empty) panic message: {msg}");
                    check("argmax empty -> panics mentioning 'empty sequence'", ok);
                },
            }
        }

        // --------------------------------------------------------------
        // 2-D contiguous + transpose view (strided), f64
        // --------------------------------------------------------------
        {
            let (m, n) = (37usize, 53usize);
            let data = gen_vec_f64(m * n, 5);
            let a = rt::asarray((data.clone(), [m, n], device));
            let visit = visit_contig(&data, m, n);
            check(
                &format!("argmax 2-D contig {m}x{n} (flat row-major)"),
                rt::argmax(&a) == ref_arg(&visit, true),
            );
            check(
                &format!("argmin 2-D contig {m}x{n} (flat row-major)"),
                rt::argmin(&a) == ref_arg(&visit, false),
            );

            let at = a.t(); // shape [n, m], strided view
            let visit_t = visit_tview(&data, m, n);
            check(
                &format!("argmax 2-D t-view {n}x{m} (flat row-major of view)"),
                rt::argmax(&at) == ref_arg(&visit_t, true),
            );
            check(
                &format!("argmin 2-D t-view {n}x{m} (flat row-major of view)"),
                rt::argmin(&at) == ref_arg(&visit_t, false),
            );

            // whole-tensor reduce of a 2-D must equal the 1-D reduce of the
            // raveled tensor (the API's documented flat-index contract)
            let a_raveled = rt::asarray((data.clone(), [m * n], device));
            check(
                "2-D whole argmax == raveled 1-D argmax",
                rt::argmax(&a) == rt::argmax(&a_raveled),
            );
        }

        // --------------------------------------------------------------
        // 2-D with ties spanning rows/cols (ties resolve row-major first)
        // --------------------------------------------------------------
        {
            let (m, n) = (17usize, 23usize);
            let data = vec![7.0_f64; m * n];
            let a = rt::asarray((data, [m, n], device));
            check(&format!("argmax 2-D all-equal {m}x{n} -> 0"), rt::argmax(&a) == 0);
        }

        // --------------------------------------------------------------
        // broadcast (zero-stride) input: exercises the generic fold over a
        // zero-stride layout. All-equal row -> every element ties -> 0;
        // arbitrary row -> the max sits in row 0 (duplicates below are
        // equal, never strictly greater, so the row-0 flat index wins).
        // --------------------------------------------------------------
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
            check("argmax broadcast view (max in row 0 at correct col)", rt::argmax(&b) == want);

            let row_eq = rt::asarray((vec![7.0_f64; n], [1, n], device));
            let b_eq = row_eq.broadcast_to(vec![m, n]);
            check("argmax broadcast all-equal (ties -> 0)", rt::argmax(&b_eq) == 0);
            check("argmin broadcast all-equal (ties -> 0)", rt::argmin(&b_eq) == 0);
        }

        // --------------------------------------------------------------
        // axis reductions (argmax_axes / argmin_axes): exercises the
        // reduce_axes_arg_cpu_serial / _rayon path that phase 2's kernel
        // change is shared by. Rows (last axis, contiguous inner) and
        // columns (axis 0, strided inner).
        // --------------------------------------------------------------
        {
            let (m, n) = (37usize, 53usize);
            let data = gen_vec_f64(m * n, 9);
            let a = rt::asarray((data.clone(), [m, n], device));

            let got_last_max = a.argmax_axes(-1).to_vec();
            let want_last_max: Vec<usize> = (0..m)
                .map(|i| {
                    let row = &data[i * n..(i + 1) * n];
                    ref_arg(&row.iter().enumerate().map(|(k, &x)| (k, x)).collect::<Vec<_>>(), true)
                })
                .collect();
            check(&format!("argmax_axes(-1) {m}x{n}"), got_last_max == want_last_max);

            let got_last_min = a.argmin_axes(-1).to_vec();
            let want_last_min: Vec<usize> = (0..m)
                .map(|i| {
                    let row = &data[i * n..(i + 1) * n];
                    ref_arg(&row.iter().enumerate().map(|(k, &x)| (k, x)).collect::<Vec<_>>(), false)
                })
                .collect();
            check(&format!("argmin_axes(-1) {m}x{n}"), got_last_min == want_last_min);

            let got_col_max = a.argmax_axes(0).to_vec();
            let want_col_max: Vec<usize> = (0..n)
                .map(|j| {
                    let col: Vec<(usize, f64)> =
                        (0..m).map(|i| (i, data[i * n + j])).collect();
                    ref_arg(&col, true)
                })
                .collect();
            check(&format!("argmax_axes(0) {m}x{n}"), got_col_max == want_col_max);

            // axis reduction over a strided view
            let at = a.t();
            let got_t_last = at.argmax_axes(-1).to_vec();
            let want_t_last: Vec<usize> = (0..n)
                .map(|i| {
                    let col: Vec<(usize, f64)> =
                        (0..m).map(|j| (j, data[j * n + i])).collect();
                    ref_arg(&col, true)
                })
                .collect();
            check(&format!("argmax_axes(-1) on t-view {n}x{m}"), got_t_last == want_t_last);
        }

        // --------------------------------------------------------------
        // f32 spots (odd 1-D, ties, NaN, 2-D, t-view)
        // --------------------------------------------------------------
        {
            let n = 999_983usize;
            let data = gen_vec_f32(n, 6);
            let a = rt::asarray((data.clone(), [n], device));
            let visit = visit_contig(&data, 1, n);
            check(&format!("argmax 1-D contig odd f32 n={n}"), rt::argmax(&a) == ref_arg(&visit, true));
            check(&format!("argmin 1-D contig odd f32 n={n}"), rt::argmin(&a) == ref_arg(&visit, false));

            let nd = Array1::from(data.clone());
            check(
                &format!("argmax/argmin 1-D odd f32 n={n} (ndarray cross-check)"),
                nd.argmax().unwrap() == rt::argmax(&a) && nd.argmin().unwrap() == rt::argmin(&a),
            );

            let mut d32 = gen_vec_f32(n, 6);
            d32[0] = f32::NAN;
            let a32 = rt::asarray((d32.clone(), [n], device));
            check("argmax f32 NaN front -> 0 (NaN poisons)", rt::argmax(&a32) == 0);

            let (m2, n2) = (41usize, 67usize);
            let d2 = gen_vec_f32(m2 * n2, 7);
            let a2 = rt::asarray((d2.clone(), [m2, n2], device));
            let visit2 = visit_contig(&d2, m2, n2);
            check(
                &format!("argmax 2-D contig f32 {m2}x{n2}"),
                rt::argmax(&a2) == ref_arg(&visit2, true),
            );
            let at2 = a2.t();
            let visit2t = visit_tview(&d2, m2, n2);
            check(
                &format!("argmin 2-D t-view f32 {n2}x{m2}"),
                rt::argmin(&at2) == ref_arg(&visit2t, false),
            );
        }
    }};
}

fn main() {
    println!("=== T6 argmax/argmin correctness gate (rstsr 386948be semantics) ===");
    check_device!("DeviceCpuSerial", serial_device());
    let dev = faer_device();
    assert_faer_threads(&dev, 16);
    check_device!("DeviceFaer (rayon, 16 threads)", dev);
    println!("correctness gate: ALL PASSED");
}

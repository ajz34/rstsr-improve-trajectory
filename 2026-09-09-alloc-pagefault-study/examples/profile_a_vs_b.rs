//! Fixed-iteration op runner for `perf stat` AND getrusage accounting
//! (T7's fault-quantification entry points).
//!
//! Each subcommand runs ONE variant for a FIXED number of iterations with
//! black_box on inputs and outputs. Around the timed loop only, it snapshots
//! `getrusage(RUSAGE_SELF)` and prints a machine-parsable RESULT line:
//! wall time, minor-page-fault delta (zero-page mapping churn lives in
//! ru_minflt), and user/sys CPU split. This quantifies the fault rider
//! without perf; `perf stat -d` wraps the same binary for full counters.
//!
//! The env-tunable experiments reuse `add_a_serial` under different
//! MALLOC_* environments (reproduce.sh stage `tunables`).
//!
//! Usage: profile_a_vs_b <case>   (2048x2048 f64 unless noted)
//!   add_a_serial | add_mutc_serial | add_2pass_serial | add_bound_serial
//!   tr_a_serial  | tr_assign_serial | tr_bound_serial
//!   sum_a_serial     (negative control: tiny output)
//!   full_a_serial | full_b_serial | zeros_a_serial
//!   faer16 (requires RAYON_NUM_THREADS=16):
//!   add_a_faer   | add_mutc_faer | add_bound_faer
//!   tr_a_faer    | tr_assign_faer

use std::hint::black_box;
use std::mem::MaybeUninit;
use std::time::{Duration, Instant};

use alloc_pagefault_study::{gen_mat_f64, gen_vec_f64, getrusage, report_accounting, Rusage};
use rstsr_core::prelude::*;
use rstsr_core::tensor::operators::op_with_func::op_mutc_refa_refb_func;

const M: usize = 2048;
const N: usize = 2048;

type CaseOut = (Duration, usize, Rusage);

fn faer_check(dev: &DeviceFaer) {
    assert_eq!(dev.get_num_threads(), 16, "set RAYON_NUM_THREADS=16");
}

// --------------------------------------------------------------- serial ----

fn case_add_a_serial() -> CaseOut {
    const ITERS: usize = 100;
    let dev = DeviceCpuSerial::default();
    let a = gen_mat_f64!(M, N, 1, &dev);
    let b = gen_mat_f64!(M, N, 2, &dev);
    black_box(&a + &b); // warmup
    let ru0 = getrusage();
    let t0 = Instant::now();
    for _ in 0..ITERS {
        black_box(black_box(&a) + black_box(&b));
    }
    (t0.elapsed(), ITERS, getrusage().delta(&ru0))
}

fn case_add_mutc_serial() -> CaseOut {
    const ITERS: usize = 100;
    let dev = DeviceCpuSerial::default();
    let a = gen_mat_f64!(M, N, 1, &dev);
    let b = gen_mat_f64!(M, N, 2, &dev);
    let mut c: Tensor<f64, _> = rt::zeros(([M, N], &dev));
    {
        let mut f = |cv: &mut MaybeUninit<f64>, x: &f64, y: &f64| { cv.write(x + y); };
        op_mutc_refa_refb_func(&mut c, &a, &b, &mut f).unwrap(); // warmup
    }
    let ru0 = getrusage();
    let t0 = Instant::now();
    for _ in 0..ITERS {
        let mut f = |cv: &mut MaybeUninit<f64>, x: &f64, y: &f64| { cv.write(x + y); };
        black_box(op_mutc_refa_refb_func(&mut c, black_box(&a), black_box(&b), &mut f).unwrap());
    }
    (t0.elapsed(), ITERS, getrusage().delta(&ru0))
}

fn case_add_2pass_serial() -> CaseOut {
    const ITERS: usize = 100;
    let dev = DeviceCpuSerial::default();
    let a = gen_mat_f64!(M, N, 1, &dev);
    let b = gen_mat_f64!(M, N, 2, &dev);
    let mut c: Tensor<f64, _> = rt::zeros(([M, N], &dev));
    c.assign(&a);
    c += &b;
    let ru0 = getrusage();
    let t0 = Instant::now();
    for _ in 0..ITERS {
        c.assign(black_box(&a));
        c += black_box(&b);
        black_box(&c);
    }
    (t0.elapsed(), ITERS, getrusage().delta(&ru0))
}

fn case_add_bound_serial() -> CaseOut {
    const ITERS: usize = 100;
    let va = gen_vec_f64(M * N, 1);
    let vb = gen_vec_f64(M * N, 2);
    let mut c = vec![0.0_f64; M * N];
    for ((cv, av), bv) in c.iter_mut().zip(va.iter()).zip(vb.iter()) {
        *cv = av + bv;
    }
    let ru0 = getrusage();
    let t0 = Instant::now();
    for _ in 0..ITERS {
        let (va, vb, c) = (black_box(&va), black_box(&vb), black_box(&mut c));
        for ((cv, av), bv) in c.iter_mut().zip(va.iter()).zip(vb.iter()) {
            *cv = av + bv;
        }
    }
    (t0.elapsed(), ITERS, getrusage().delta(&ru0))
}

fn case_tr_a_serial() -> CaseOut {
    const ITERS: usize = 40;
    let dev = DeviceCpuSerial::default();
    let a = gen_mat_f64!(M, N, 1, &dev);
    black_box(a.t().to_contig(RowMajor)); // warmup
    let ru0 = getrusage();
    let t0 = Instant::now();
    for _ in 0..ITERS {
        black_box(black_box(&a).t().to_contig(RowMajor));
    }
    (t0.elapsed(), ITERS, getrusage().delta(&ru0))
}

fn case_tr_assign_serial() -> CaseOut {
    const ITERS: usize = 40;
    let dev = DeviceCpuSerial::default();
    let a = gen_mat_f64!(M, N, 1, &dev);
    let mut c: Tensor<f64, _> = rt::zeros(([N, M], &dev));
    c.assign(&a.t());
    let ru0 = getrusage();
    let t0 = Instant::now();
    for _ in 0..ITERS {
        c.assign(black_box(&a.t()));
        black_box(&c);
    }
    (t0.elapsed(), ITERS, getrusage().delta(&ru0))
}

fn case_tr_bound_serial() -> CaseOut {
    const ITERS: usize = 40;
    let va = gen_vec_f64(M * N, 1);
    let mut c = vec![0.0_f64; N * M];
    for j in 0..N {
        for (i, cv) in c[j * M..(j + 1) * M].iter_mut().enumerate() {
            *cv = va[i * N + j];
        }
    }
    let ru0 = getrusage();
    let t0 = Instant::now();
    for _ in 0..ITERS {
        let (va, c) = (black_box(&va), black_box(&mut c));
        for j in 0..N {
            for (i, cv) in c[j * M..(j + 1) * M].iter_mut().enumerate() {
                *cv = va[i * N + j];
            }
        }
    }
    (t0.elapsed(), ITERS, getrusage().delta(&ru0))
}

fn case_sum_a_serial() -> CaseOut {
    const ITERS: usize = 80;
    let dev = DeviceCpuSerial::default();
    let a = gen_mat_f64!(M, N, 1, &dev);
    black_box(a.sum_axes(0)); // warmup
    let ru0 = getrusage();
    let t0 = Instant::now();
    for _ in 0..ITERS {
        black_box(a.sum_axes(0));
    }
    (t0.elapsed(), ITERS, getrusage().delta(&ru0))
}

fn case_full_a_serial() -> CaseOut {
    const ITERS: usize = 60;
    let dev = DeviceCpuSerial::default();
    let _f: Tensor<f64, _> = rt::full(([M, N], 3.5, &dev));
    black_box(_f); // warmup
    let ru0 = getrusage();
    let t0 = Instant::now();
    for _ in 0..ITERS {
        let f: Tensor<f64, _> = rt::full(([M, N], 3.5, &dev));
        black_box(f);
    }
    (t0.elapsed(), ITERS, getrusage().delta(&ru0))
}

fn case_full_b_serial() -> CaseOut {
    const ITERS: usize = 60;
    let dev = DeviceCpuSerial::default();
    let mut c: Tensor<f64, _> = rt::zeros(([M, N], &dev));
    c.fill(3.5);
    let ru0 = getrusage();
    let t0 = Instant::now();
    for _ in 0..ITERS {
        c.fill(3.5);
        black_box(&c);
    }
    (t0.elapsed(), ITERS, getrusage().delta(&ru0))
}

fn case_zeros_a_serial() -> CaseOut {
    const ITERS: usize = 400;
    let dev = DeviceCpuSerial::default();
    let _z: Tensor<f64, _> = rt::zeros(([M, N], &dev));
    black_box(_z); // warmup
    let ru0 = getrusage();
    let t0 = Instant::now();
    for _ in 0..ITERS {
        let z: Tensor<f64, _> = rt::zeros(([M, N], &dev));
                black_box(z);
    }
    (t0.elapsed(), ITERS, getrusage().delta(&ru0))
}

// --------------------------------------------------------------- faer16 ----

fn case_add_a_faer() -> CaseOut {
    const ITERS: usize = 200;
    let dev = DeviceFaer::new(0);
    faer_check(&dev);
    let a = gen_mat_f64!(M, N, 1, &dev);
    let b = gen_mat_f64!(M, N, 2, &dev);
    black_box(&a + &b);
    let ru0 = getrusage();
    let t0 = Instant::now();
    for _ in 0..ITERS {
        black_box(black_box(&a) + black_box(&b));
    }
    (t0.elapsed(), ITERS, getrusage().delta(&ru0))
}

fn case_add_mutc_faer() -> CaseOut {
    const ITERS: usize = 200;
    let dev = DeviceFaer::new(0);
    faer_check(&dev);
    let a = gen_mat_f64!(M, N, 1, &dev);
    let b = gen_mat_f64!(M, N, 2, &dev);
    let mut c: Tensor<f64, _> = rt::zeros(([M, N], &dev));
    {
        let mut f = |cv: &mut MaybeUninit<f64>, x: &f64, y: &f64| { cv.write(x + y); };
        op_mutc_refa_refb_func(&mut c, &a, &b, &mut f).unwrap();
    }
    let ru0 = getrusage();
    let t0 = Instant::now();
    for _ in 0..ITERS {
        let mut f = |cv: &mut MaybeUninit<f64>, x: &f64, y: &f64| { cv.write(x + y); };
        black_box(op_mutc_refa_refb_func(&mut c, black_box(&a), black_box(&b), &mut f).unwrap());
    }
    (t0.elapsed(), ITERS, getrusage().delta(&ru0))
}

fn case_add_bound_faer() -> CaseOut {
    const ITERS: usize = 200;
    let va = gen_vec_f64(M * N, 1);
    let vb = gen_vec_f64(M * N, 2);
    let mut c = vec![0.0_f64; M * N];
    let ru0 = getrusage();
    let t0 = Instant::now();
    for _ in 0..ITERS {
        let (va, vb, c) = (black_box(&va), black_box(&vb), black_box(&mut c));
        let chunk = c.len().div_ceil(16);
        std::thread::scope(|s| {
            for ((cc, aa), bb) in c.chunks_mut(chunk).zip(va.chunks(chunk)).zip(vb.chunks(chunk)) {
                s.spawn(move || {
                    for ((cv, av), bv) in cc.iter_mut().zip(aa).zip(bb) {
                        *cv = av + bv;
                    }
                });
            }
        });
    }
    (t0.elapsed(), ITERS, getrusage().delta(&ru0))
}

fn case_tr_a_faer() -> CaseOut {
    const ITERS: usize = 60;
    let dev = DeviceFaer::new(0);
    faer_check(&dev);
    let a = gen_mat_f64!(M, N, 1, &dev);
    black_box(a.t().to_contig(RowMajor));
    let ru0 = getrusage();
    let t0 = Instant::now();
    for _ in 0..ITERS {
        black_box(black_box(&a).t().to_contig(RowMajor));
    }
    (t0.elapsed(), ITERS, getrusage().delta(&ru0))
}

fn case_tr_assign_faer() -> CaseOut {
    const ITERS: usize = 60;
    let dev = DeviceFaer::new(0);
    faer_check(&dev);
    let a = gen_mat_f64!(M, N, 1, &dev);
    let mut c: Tensor<f64, _> = rt::zeros(([N, M], &dev));
    c.assign(&a.t());
    let ru0 = getrusage();
    let t0 = Instant::now();
    for _ in 0..ITERS {
        c.assign(black_box(&a.t()));
        black_box(&c);
    }
    (t0.elapsed(), ITERS, getrusage().delta(&ru0))
}

// ----------------------------------------------------------------- main ----

fn main() {
    let case = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: profile_a_vs_b <case> (see file header)");
        std::process::exit(2);
    });
    eprintln!("[profile_a_vs_b] case={case}");

    let (elapsed, iters, d) = match case.as_str() {
        "add_a_serial" => case_add_a_serial(),
        "add_mutc_serial" => case_add_mutc_serial(),
        "add_2pass_serial" => case_add_2pass_serial(),
        "add_bound_serial" => case_add_bound_serial(),
        "tr_a_serial" => case_tr_a_serial(),
        "tr_assign_serial" => case_tr_assign_serial(),
        "tr_bound_serial" => case_tr_bound_serial(),
        "sum_a_serial" => case_sum_a_serial(),
        "full_a_serial" => case_full_a_serial(),
        "full_b_serial" => case_full_b_serial(),
        "zeros_a_serial" => case_zeros_a_serial(),
        "add_a_faer" => case_add_a_faer(),
        "add_mutc_faer" => case_add_mutc_faer(),
        "add_bound_faer" => case_add_bound_faer(),
        "tr_a_faer" => case_tr_a_faer(),
        "tr_assign_faer" => case_tr_assign_faer(),
        other => {
            eprintln!("unknown case: {other}");
            std::process::exit(2);
        }
    };
    report_accounting(&case, iters, elapsed.as_secs_f64(), &d);
}

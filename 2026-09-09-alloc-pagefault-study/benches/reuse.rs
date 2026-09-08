//! T7 main suite: allocating variant (A) vs output-reuse variants (B) per
//! op/size/device.
//!
//! Variants:
//! - `A_alloc`       : idiomatic allocating call (T0's policy) — output
//!                     storage allocated inside the op on every iteration.
//! - `B_reuse_mutc`  : single pass into a buffer allocated ONCE below, via the
//!                     public rstsr-386948be fn `op_mutc_refa_refb_func`
//!                     (c <- a + b elementwise; add contig/strided only).
//! - `B_reuse_2pass` : existing tensor-method composition `c.assign(&a);
//!                     c += &b` (two kernel passes; add only).
//! - `B_reuse_*`     : other existing-API reuse (`assign` for transpose,
//!                     `fill` for full/zeros).
//! - `B_bound`       : "emulated kernel bound, not an rstsr API" — a raw
//!                     single-pass loop writing into a preallocated Vec
//!                     (serial id: 1 thread; faer id: 16 std threads —
//!                     labeled `B_bound_par16` in the analysis).
//!
//! All B variants keep the output buffer alive and warm across iterations;
//! the buffer is allocated before the timed loop.
//!
//! Ops × sizes (f64, criterion, devices serial + faer16):
//! - add contig    : 512×512, 2048×2048, 1000×777   (A, mutc, 2pass, bound)
//! - add strided   : 2048×2048 (a + bt.t())          (A, mutc, 2pass, bound)
//! - transpose copy: 2048×2048, 1000×777             (A, assign, bound)
//! - sum_axis(0)   : 2048×2048  (A, bound; no reduce-into API at 386948be)
//! - batched vecdot: (4096,512) (A, bound; output tiny — control)
//! - full          : 2048×2048  (A, fill)
//! - zeros         : 2048×2048  (A = calloc-lazy per T0; fill(0.) shown as
//!                   "what an eager zeros would cost")

use std::hint::black_box;
use std::mem::MaybeUninit;

use alloc_pagefault_study::{assert_faer_threads, configure_group, faer_device, gen_mat_f64, gen_vec_f64, serial_device};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rstsr_core::prelude::*;
use rstsr_core::tensor::operators::op_with_func::op_mutc_refa_refb_func;

const NTHREADS: usize = 16;

// ---------------------------------------------------------------------------
// emulated kernel bounds (raw loops into preallocated Vec; NOT rstsr APIs)
// ---------------------------------------------------------------------------

fn bound_add_contig(a: &[f64], b: &[f64], c: &mut [f64]) {
    for ((c_i, a_i), b_i) in c.iter_mut().zip(a.iter()).zip(b.iter()) {
        *c_i = a_i + b_i;
    }
}

fn bound_add_contig_par(a: &[f64], b: &[f64], c: &mut [f64]) {
    let chunk = c.len().div_ceil(NTHREADS);
    std::thread::scope(|s| {
        for ((cc, aa), bb) in c.chunks_mut(chunk).zip(a.chunks(chunk)).zip(b.chunks(chunk)) {
            s.spawn(move || {
                for ((cv, av), bv) in cc.iter_mut().zip(aa).zip(bb) {
                    *cv = av + bv;
                }
            });
        }
    });
}

/// c[i,j] = a[i,j] + bt[j,i]; a is [m,n] row-major, bt is [n,m] row-major.
fn bound_add_strided(a: &[f64], bt: &[f64], c: &mut [f64], m: usize, n: usize) {
    for i in 0..m {
        let a_row = &a[i * n..(i + 1) * n];
        let c_row = &mut c[i * n..(i + 1) * n];
        for (j, (cv, av)) in c_row.iter_mut().zip(a_row.iter()).enumerate() {
            *cv = av + bt[j * m + i];
        }
    }
}

fn bound_add_strided_par(a: &[f64], bt: &[f64], c: &mut [f64], m: usize, n: usize) {
    let rows_chunk = m.div_ceil(NTHREADS);
    std::thread::scope(|s| {
        for (cc, aa) in c.chunks_mut(rows_chunk * n).zip(a.chunks(rows_chunk * n)) {
            let rows = aa.len() / n;
            s.spawn(move || {
                for i in 0..rows {
                    let a_row = &aa[i * n..(i + 1) * n];
                    let c_row = &mut cc[i * n..(i + 1) * n];
                    for (j, (cv, av)) in c_row.iter_mut().zip(a_row.iter()).enumerate() {
                        *cv = av + bt[j * m + i];
                    }
                }
            });
        }
    });
}

/// c[j,i] = a[i,j]: output-row-chunked transpose into preallocated [n,m].
fn bound_transpose(a: &[f64], c: &mut [f64], m: usize, n: usize) {
    for j in 0..n {
        let c_row = &mut c[j * m..(j + 1) * m];
        for (i, cv) in c_row.iter_mut().enumerate() {
            *cv = a[i * n + j];
        }
    }
}

fn bound_transpose_par(a: &[f64], c: &mut [f64], m: usize, n: usize) {
    let rows_chunk = n.div_ceil(NTHREADS);
    std::thread::scope(|s| {
        for cc in c.chunks_mut(rows_chunk * m) {
            let j0 = cc.len() / m;
            s.spawn(move || {
                for j in 0..j0 {
                    let c_row = &mut cc[j * m..(j + 1) * m];
                    for (i, cv) in c_row.iter_mut().enumerate() {
                        *cv = a[i * n + j];
                    }
                }
            });
        }
    });
}

/// out[j] = sum_i a[i,j], accumulated in memory order.
fn bound_sum_axis0(a: &[f64], out: &mut [f64], m: usize, n: usize) {
    out.fill(0.0);
    for i in 0..m {
        let row = &a[i * n..(i + 1) * n];
        for (o, r) in out.iter_mut().zip(row.iter()) {
            *o += r;
        }
    }
}

fn bound_sum_axis0_par(a: &[f64], out: &mut [f64], m: usize, n: usize) {
    let col_chunk = n.div_ceil(NTHREADS);
    out.fill(0.0);
    std::thread::scope(|s| {
        for oo in out.chunks_mut(col_chunk) {
            s.spawn(move || {
                for i in 0..m {
                    let row = &a[i * n..(i + 1) * n];
                    for (o, r) in oo.iter_mut().zip(row.iter()) {
                        *o += r;
                    }
                }
            });
        }
    });
}

fn bound_vecdot_batched(a: &[f64], b: &[f64], out: &mut [f64], rows: usize, cols: usize) {
    for k in 0..rows {
        let a_row = &a[k * cols..(k + 1) * cols];
        let b_row = &b[k * cols..(k + 1) * cols];
        let mut acc = 0.0;
        for (av, bv) in a_row.iter().zip(b_row.iter()) {
            acc += av * bv;
        }
        out[k] = acc;
    }
}

fn bound_vecdot_batched_par(a: &[f64], b: &[f64], out: &mut [f64], rows: usize, cols: usize) {
    let chunk = rows.div_ceil(NTHREADS);
    std::thread::scope(|s| {
        for ((oc, ac), bc) in out
            .chunks_mut(chunk)
            .zip(a.chunks(chunk * cols))
            .zip(b.chunks(chunk * cols))
        {
            let nrows = ac.len() / cols;
            s.spawn(move || bound_vecdot_batched(ac, bc, oc, nrows, cols));
        }
    });
}

// ---------------------------------------------------------------------------
// benches
// ---------------------------------------------------------------------------

const ADD_SIZES: &[(&str, usize, usize)] =
    &[("medium", 512, 512), ("large", 2048, 2048), ("odd", 1000, 777)];
const TR_SIZES: &[(&str, usize, usize)] = &[("large", 2048, 2048), ("odd", 1000, 777)];

fn bench_reuse(c: &mut Criterion) {
    let dev_serial = serial_device();
    let dev_faer = faer_device();
    assert_faer_threads(&dev_faer, 16);

    // The two devices are distinct types; each op section is expressed once as
    // a closure and instantiated for both (T0's trick: fixture macros inherit
    // rstsr's trait bounds at the expansion site).
    macro_rules! both_devices {
        ($group:expr, $body:expr) => {{
            {
                let dev = &dev_serial;
                let devtag: &str = "serial";
                let body = $body;
                body($group, dev, devtag);
            }
            {
                let dev = &dev_faer;
                let devtag: &str = "faer16";
                let body = $body;
                body($group, dev, devtag);
            }
        }};
    }

    // ---------------- add contiguous ----------------------------------------
    let mut group = c.benchmark_group("add_contig");
    configure_group(&mut group);
    both_devices!(&mut group, |group: &mut criterion::BenchmarkGroup<_>, dev: &_, devtag: &str| {
        for &(label, m, n) in ADD_SIZES {
            let a = gen_mat_f64!(m, n, 1, dev);
            let b = gen_mat_f64!(m, n, 2, dev);
            let id = |v: &'static str| BenchmarkId::new(format!("{label}_{m}x{n}-{devtag}_f64"), v);

            group.bench_function(id("A_alloc"), |bench| {
                bench.iter(|| black_box(black_box(&a) + black_box(&b)))
            });

            // B_reuse_mutc: c allocated once HERE (outside timing); single
            // kernel pass per iteration via public op_mutc_refa_refb_func.
            let mut c: Tensor<f64, _> = rt::zeros(([m, n], dev));
            c.fill(0.0); // warm the buffer's pages once, outside timing
            group.bench_function(id("B_reuse_mutc"), |bench| {
                bench.iter(|| {
                    let mut f = |cv: &mut MaybeUninit<f64>, x: &f64, y: &f64| { cv.write(x + y); };
                    op_mutc_refa_refb_func(&mut c, black_box(&a), black_box(&b), &mut f).unwrap();
                    black_box(&c);
                })
            });

            // B_reuse_2pass: assign + add_assign (existing tensor methods)
            group.bench_function(id("B_reuse_2pass"), |bench| {
                bench.iter(|| {
                    c.assign(black_box(&a));
                    c += black_box(&b);
                    black_box(&c);
                })
            });

            // B_bound: emulated single-pass kernel into preallocated Vec
            let va = gen_vec_f64(m * n, 1);
            let vb = gen_vec_f64(m * n, 2);
            let mut cvec = vec![0.0_f64; m * n];
            group.bench_function(id("B_bound"), |bench| {
                bench.iter(|| {
                    if devtag == "serial" {
                        bound_add_contig(&va, &vb, black_box(&mut cvec));
                    } else {
                        bound_add_contig_par(&va, &vb, black_box(&mut cvec));
                    }
                    black_box(&cvec);
                })
            });
        }
    });
    group.finish();

    // ---------------- add strided (a + bt.t()) ------------------------------
    let mut group = c.benchmark_group("add_strided");
    configure_group(&mut group);
    both_devices!(&mut group, |group: &mut criterion::BenchmarkGroup<_>, dev: &_, devtag: &str| {
        let (m, n) = (2048usize, 2048usize);
        let a = gen_mat_f64!(m, n, 1, dev);
        let bt = gen_mat_f64!(m, n, 4, dev); // square: bt.t() is [m, n]
        let id = |v: &'static str| BenchmarkId::new(format!("large_{m}x{n}-{devtag}_f64"), v);

        group.bench_function(id("A_alloc"), |bench| {
            bench.iter(|| black_box(black_box(&a) + black_box(&bt.t())))
        });

        let mut c: Tensor<f64, _> = rt::zeros(([m, n], dev));
        c.fill(0.0);
        group.bench_function(id("B_reuse_mutc"), |bench| {
            bench.iter(|| {
                let mut f = |cv: &mut MaybeUninit<f64>, x: &f64, y: &f64| { cv.write(x + y); };
                op_mutc_refa_refb_func(&mut c, black_box(&a), black_box(&bt.t()), &mut f).unwrap();
                black_box(&c);
            })
        });

        group.bench_function(id("B_reuse_2pass"), |bench| {
            bench.iter(|| {
                c.assign(black_box(&a));
                c += black_box(&bt.t());
                black_box(&c);
            })
        });

        let va = gen_vec_f64(m * n, 1);
        let vbt = gen_vec_f64(m * n, 4);
        let mut cvec = vec![0.0_f64; m * n];
        group.bench_function(id("B_bound"), |bench| {
            bench.iter(|| {
                if devtag == "serial" {
                    bound_add_strided(&va, &vbt, black_box(&mut cvec), m, n);
                } else {
                    bound_add_strided_par(&va, &vbt, black_box(&mut cvec), m, n);
                }
                black_box(&cvec);
            })
        });
    });
    group.finish();

    // ---------------- transpose copy ----------------------------------------
    let mut group = c.benchmark_group("transpose_copy");
    configure_group(&mut group);
    both_devices!(&mut group, |group: &mut criterion::BenchmarkGroup<_>, dev: &_, devtag: &str| {
        for &(label, m, n) in TR_SIZES {
            let a = gen_mat_f64!(m, n, 1, dev);
            let id = |v: &'static str| BenchmarkId::new(format!("{label}_{m}x{n}-{devtag}_f64"), v);

            group.bench_function(id("A_alloc"), |bench| {
                bench.iter(|| {
                    let at = black_box(&a).t();
                    let out = at.to_contig(RowMajor).into_owned();
                    black_box(out)
                })
            });

            let mut c: Tensor<f64, _> = rt::zeros(([n, m], dev));
            c.assign(&a.t());
            group.bench_function(id("B_reuse_assign"), |bench| {
                bench.iter(|| {
                    c.assign(black_box(&a.t()));
                    black_box(&c);
                })
            });

            let va = gen_vec_f64(m * n, 1);
            let mut cvec = vec![0.0_f64; n * m];
            group.bench_function(id("B_bound"), |bench| {
                bench.iter(|| {
                    if devtag == "serial" {
                        bound_transpose(&va, black_box(&mut cvec), m, n);
                    } else {
                        bound_transpose_par(&va, black_box(&mut cvec), m, n);
                    }
                    black_box(&cvec);
                })
            });
        }
    });
    group.finish();

    // ---------------- sum_axis(0) (negative control; no reduce-into API) ----
    let mut group = c.benchmark_group("sum_axis0");
    configure_group(&mut group);
    both_devices!(&mut group, |group: &mut criterion::BenchmarkGroup<_>, dev: &_, devtag: &str| {
        let (m, n) = (2048usize, 2048usize);
        let a = gen_mat_f64!(m, n, 1, dev);
        let id = |v: &'static str| BenchmarkId::new(format!("large_{m}x{n}-{devtag}_f64"), v);

        group.bench_function(id("A_alloc"), |bench| {
            bench.iter(|| black_box(black_box(&a).sum_axes(0)))
        });

        let va = gen_vec_f64(m * n, 1);
        let mut out = vec![0.0_f64; n];
        group.bench_function(id("B_bound"), |bench| {
            bench.iter(|| {
                if devtag == "serial" {
                    bound_sum_axis0(&va, black_box(&mut out), m, n);
                } else {
                    bound_sum_axis0_par(&va, black_box(&mut out), m, n);
                }
                black_box(&out);
            })
        });
    });
    group.finish();

    // ---------------- batched vecdot (small-output control) -----------------
    let mut group = c.benchmark_group("vecdot_batched");
    configure_group(&mut group);
    both_devices!(&mut group, |group: &mut criterion::BenchmarkGroup<_>, dev: &_, devtag: &str| {
        let (rows, cols) = (4096usize, 512usize);
        let a = gen_mat_f64!(rows, cols, 1, dev);
        let b = gen_mat_f64!(rows, cols, 2, dev);
        let id = |v: &'static str| BenchmarkId::new(format!("batched_{rows}x{cols}-{devtag}_f64"), v);

        group.bench_function(id("A_alloc"), |bench| {
            bench.iter(|| black_box(rt::vecdot(black_box(&a), black_box(&b), None)))
        });

        let va = gen_vec_f64(rows * cols, 1);
        let vb = gen_vec_f64(rows * cols, 2);
        let mut out = vec![0.0_f64; rows];
        group.bench_function(id("B_bound"), |bench| {
            bench.iter(|| {
                if devtag == "serial" {
                    bound_vecdot_batched(&va, &vb, black_box(&mut out), rows, cols);
                } else {
                    bound_vecdot_batched_par(&va, &vb, black_box(&mut out), rows, cols);
                }
                black_box(&out);
            })
        });
    });
    group.finish();

    // ---------------- full (allocation IS the op) ----------------------------
    let mut group = c.benchmark_group("fill_full");
    configure_group(&mut group);
    both_devices!(&mut group, |group: &mut criterion::BenchmarkGroup<_>, dev: &_, devtag: &str| {
        let (m, n) = (2048usize, 2048usize);
        let v = 3.5_f64;
        let id = |v: &'static str| BenchmarkId::new(format!("large_{m}x{n}-{devtag}_f64"), v);

        group.bench_function(id("A_alloc"), |bench| {
            bench.iter(|| {
                let t: Tensor<f64, _> = rt::full(([m, n], v, dev));
                black_box(t)
            })
        });

        let mut c: Tensor<f64, _> = rt::zeros(([m, n], dev));
        c.fill(v);
        group.bench_function(id("B_reuse_fill"), |bench| {
            bench.iter(|| {
                c.fill(v);
                black_box(&c);
            })
        });
    });
    group.finish();

    // ---------------- zeros (calloc-lazy; fill(0) = eager-zeros cost) --------
    let mut group = c.benchmark_group("zeros");
    configure_group(&mut group);
    both_devices!(&mut group, |group: &mut criterion::BenchmarkGroup<_>, dev: &_, devtag: &str| {
        let (m, n) = (2048usize, 2048usize);
        let id = |v: &'static str| BenchmarkId::new(format!("large_{m}x{n}-{devtag}_f64"), v);

        group.bench_function(id("A_alloc"), |bench| {
            bench.iter(|| {
                let z: Tensor<f64, _> = rt::zeros(([m, n], dev));
                black_box(z)
            })
        });

        let mut c: Tensor<f64, _> = rt::zeros(([m, n], dev));
        c.fill(0.0);
        group.bench_function(id("B_reuse_fill"), |bench| {
            bench.iter(|| {
                c.fill(0.0);
                black_box(&c);
            })
        });
    });
    group.finish();
}

criterion_group!(benches, bench_reuse);
criterion_main!(benches);

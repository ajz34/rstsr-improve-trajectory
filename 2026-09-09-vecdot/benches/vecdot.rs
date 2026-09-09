//! vecdot benchmarks (T3' phase 1 baseline).
//!
//! - 1-D dot 1e3 / 1e5 / 1e7 f64 (regression gates — already memory-bound,
//!   must NOT regress; T0: 2.81 ms @1e7 serial ≈ 56.9 GB/s).
//! - 1-D f32 spot at 1e7 (D8-regression canary context).
//! - 1-D Complex<f64> spot at 1e5 (D5 secondary dtype).
//! - Batched vecdot, f64 primary + Complex<f64> on the primary case:
//!   - `batched_am1` (4096,512)·(4096,512) axis -1 (T0 primary; 466 µs serial
//!     vs 148 µs faer16) and (8192,256).
//!   - `batched_axis0` (512,4096)·(512,4096) axis 0 — the geometry that hits
//!     the contiguous-remaining MaybeUninit-RMW branch.
//!   - `batched_strided` (4096,512)·(512,4096).t() axis -1 — general branch.
//!   - odd (1000,777) and small (64,64) gates.
//!
//! Allocation included (rt::vecdot allocates its output; 4096 f64 = 32 KiB,
//! no page-fault rider per the T7 study). Efficiency framing: each row of
//! `batched_am1` is 512 mul + 512 add = 1024 FLOP over 2·512·8 = 8192 B read,
//! i.e. 0.125 FLOP/B — memory-bound framing (GB/s) first, FLOP/s-vs-peak as
//! the secondary lens.

use std::hint::black_box;

use exp_vecdot::{
    assert_faer_threads, configure_group, faer_device, serial_device, BATCHED_CASES, DOT1D_SIZES,
};
use exp_vecdot::{gen_mat_c64, gen_mat_f64, gen_vec_dev_c64, gen_vec_dev_f32, gen_vec_dev_f64};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rstsr_core::prelude::*;

fn bench_vecdot(c: &mut Criterion) {
    let dev_serial = serial_device();
    let dev_faer = faer_device();
    assert_faer_threads(&dev_faer, 16);

    let mut group = c.benchmark_group("vecdot");
    configure_group(&mut group);

    // ---- 1-D dot, f64 primary (gates) ----
    for &(label, n) in DOT1D_SIZES {
        group.bench_function(
            BenchmarkId::new(format!("dot1d_{label}_{n}-serial_f64"), "vecdot"),
            |b| {
                let a = gen_vec_dev_f64!(n, 1, &dev_serial);
                let bb = gen_vec_dev_f64!(n, 2, &dev_serial);
                b.iter(|| black_box(rt::vecdot(black_box(&a), black_box(&bb), None)))
            },
        );
        group.bench_function(
            BenchmarkId::new(format!("dot1d_{label}_{n}-faer16_f64"), "vecdot"),
            |b| {
                let a = gen_vec_dev_f64!(n, 1, &dev_faer);
                let bb = gen_vec_dev_f64!(n, 2, &dev_faer);
                b.iter(|| black_box(rt::vecdot(black_box(&a), black_box(&bb), None)))
            },
        );
    }

    // ---- 1-D f32 spot at 1e7 (D8-regression canary context) ----
    group.bench_function(BenchmarkId::new("dot1d_large_10000000-serial_f32", "vecdot"), |b| {
        let a = gen_vec_dev_f32!(10_000_000, 1, &dev_serial);
        let bb = gen_vec_dev_f32!(10_000_000, 2, &dev_serial);
        b.iter(|| black_box(rt::vecdot(black_box(&a), black_box(&bb), None)))
    });
    group.bench_function(BenchmarkId::new("dot1d_large_10000000-faer16_f32", "vecdot"), |b| {
        let a = gen_vec_dev_f32!(10_000_000, 1, &dev_faer);
        let bb = gen_vec_dev_f32!(10_000_000, 2, &dev_faer);
        b.iter(|| black_box(rt::vecdot(black_box(&a), black_box(&bb), None)))
    });

    // ---- 1-D Complex<f64> spot at 1e5 (D5 secondary dtype) ----
    group.bench_function(BenchmarkId::new("dot1d_medium_100000-serial_c64", "vecdot"), |b| {
        let a = gen_vec_dev_c64!(100_000, 1, &dev_serial);
        let bb = gen_vec_dev_c64!(100_000, 2, &dev_serial);
        b.iter(|| black_box(rt::vecdot(black_box(&a), black_box(&bb), None)))
    });
    group.bench_function(BenchmarkId::new("dot1d_medium_100000-faer16_c64", "vecdot"), |b| {
        let a = gen_vec_dev_c64!(100_000, 1, &dev_faer);
        let bb = gen_vec_dev_c64!(100_000, 2, &dev_faer);
        b.iter(|| black_box(rt::vecdot(black_box(&a), black_box(&bb), None)))
    });

    // ---- batched cases ----
    macro_rules! bench_batched_dev {
        ($dev_name:expr, $dev:expr) => {{
            let dev = $dev;
            for &(case, rows, cols, axis) in BATCHED_CASES {
                let a = gen_mat_f64!(rows, cols, 1, dev);
                let axis_name = if axis == -2 { -1 } else { axis };
                let id_f64 = format!("{case}_{rows}x{cols}-ax{axis_name}-{}", $dev_name);
                if axis == -2 {
                    // strided: b stored (cols, rows) contiguous, viewed
                    // transposed (view only — the transpose itself is not
                    // part of the op).
                    let bst = gen_mat_f64!(cols, rows, 2, dev);
                    let bv = bst.t();
                    group.bench_function(BenchmarkId::new(id_f64, "vecdot"), |b| {
                        b.iter(|| black_box(rt::vecdot(black_box(&a), black_box(&bv), None)))
                    });
                } else {
                    let bb = gen_mat_f64!(rows, cols, 2, dev);
                    group.bench_function(BenchmarkId::new(id_f64, "vecdot"), |b| {
                        if axis == 0 {
                            b.iter(|| black_box(rt::vecdot(black_box(&a), black_box(&bb), 0)))
                        } else {
                            b.iter(|| black_box(rt::vecdot(black_box(&a), black_box(&bb), None)))
                        }
                    });
                }

                // Complex<f64> on the primary case only (D5).
                if case == "batched_am1" && rows == 4096 {
                    let id_c64 = format!("{case}_{rows}x{cols}-ax{axis_name}-{}_c64", $dev_name);
                    group.bench_function(BenchmarkId::new(id_c64, "vecdot"), |b| {
                        let a = gen_mat_c64!(rows, cols, 1, dev);
                        let bb = gen_mat_c64!(rows, cols, 2, dev);
                        b.iter(|| black_box(rt::vecdot(black_box(&a), black_box(&bb), None)))
                    });
                }
            }
        }};
    }
    bench_batched_dev!("serial", &dev_serial);
    bench_batched_dev!("faer16", &dev_faer);

    group.finish();
}

criterion_group!(benches, bench_vecdot);
criterion_main!(benches);

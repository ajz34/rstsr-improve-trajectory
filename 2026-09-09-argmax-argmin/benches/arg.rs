//! argmax/argmin benchmarks (T6 phase 1, baseline on clean 386948be).
//!
//! Matrix:
//! - 1-D f64 at n = 64, 1000, 1e6, 1e7 (argmax AND argmin), devices
//!   DeviceCpuSerial (`serial`) and DeviceFaer (`faer16`, RAYON_NUM_THREADS=16).
//! - 1-D f32 spot at 1e7 (both ops, both devices).
//! - 2-D whole-tensor argmax/argmin (flat row-major index) at 64x64, 512x512,
//!   2048x2048 f64, both devices.
//! - Strided case: transpose view `a.t()` of a 2048x2048 (and a 64x64 gate)
//!   f64 tensor — the view keeps a[0,1] adjacent to a[0,0], so iteration is
//!   non-contiguous. Both ops, both devices.
//!
//! The `small` cases exist as the D3 regression gate (no >2-3% regression on
//! small inputs); `small_odd` (n=1000) also exercises the serial fold path
//! (around the PARALLEL_SWITCH=1024 boundary of the rayon kernel).

use std::hint::black_box;

use bench_argmax_argmin::{
    assert_faer_threads, configure_group, faer_device, gen_mat_f64, gen_vec_dev_f32, gen_vec_dev_f64, serial_device,
    ARG_MAT_SIZES, ARG_SIZES,
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rstsr_core::prelude::*;

fn bench_arg_1d(c: &mut Criterion) {
    let dev_serial = serial_device();
    let dev_faer = faer_device();
    assert_faer_threads(&dev_faer, 16);

    let mut group = c.benchmark_group("arg_1d");
    configure_group(&mut group);

    for &(label, n) in ARG_SIZES {
        let a = gen_vec_dev_f64!(n, 1, &dev_serial);
        group.bench_function(BenchmarkId::new(format!("{label}_{n}-serial_f64"), "argmax"), |bench| {
            bench.iter(|| black_box(rt::argmax(black_box(&a))))
        });
        group.bench_function(BenchmarkId::new(format!("{label}_{n}-serial_f64"), "argmin"), |bench| {
            bench.iter(|| black_box(rt::argmin(black_box(&a))))
        });
        let a = gen_vec_dev_f64!(n, 1, &dev_faer);
        group.bench_function(BenchmarkId::new(format!("{label}_{n}-faer16_f64"), "argmax"), |bench| {
            bench.iter(|| black_box(rt::argmax(black_box(&a))))
        });
        group.bench_function(BenchmarkId::new(format!("{label}_{n}-faer16_f64"), "argmin"), |bench| {
            bench.iter(|| black_box(rt::argmin(black_box(&a))))
        });
    }

    // f32 spot at 1e7 (D5 secondary dtype)
    {
        let n = 10_000_000usize;
        let a = gen_vec_dev_f32!(n, 1, &dev_serial);
        group.bench_function(BenchmarkId::new(format!("large_{n}-serial_f32"), "argmax"), |bench| {
            bench.iter(|| black_box(rt::argmax(black_box(&a))))
        });
        group.bench_function(BenchmarkId::new(format!("large_{n}-serial_f32"), "argmin"), |bench| {
            bench.iter(|| black_box(rt::argmin(black_box(&a))))
        });
        let a = gen_vec_dev_f32!(n, 1, &dev_faer);
        group.bench_function(BenchmarkId::new(format!("large_{n}-faer16_f32"), "argmax"), |bench| {
            bench.iter(|| black_box(rt::argmax(black_box(&a))))
        });
        group.bench_function(BenchmarkId::new(format!("large_{n}-faer16_f32"), "argmin"), |bench| {
            bench.iter(|| black_box(rt::argmin(black_box(&a))))
        });
    }

    group.finish();
}

fn bench_arg_2d_whole(c: &mut Criterion) {
    let dev_serial = serial_device();
    let dev_faer = faer_device();
    assert_faer_threads(&dev_faer, 16);

    let mut group = c.benchmark_group("arg_2d_whole");
    configure_group(&mut group);

    // Whole-tensor argmax/argmin on 2-D contiguous row-major tensors.
    // The API supports any-rank whole-tensor reduce directly: rt::argmax(&a)
    // returns the flat ROW-MAJOR index (usize) regardless of input layout.
    for &(label, m, n) in ARG_MAT_SIZES {
        let a = gen_mat_f64!(m, n, 1, &dev_serial);
        group.bench_function(BenchmarkId::new(format!("{label}_{m}x{n}-serial_f64"), "argmax"), |bench| {
            bench.iter(|| black_box(rt::argmax(black_box(&a))))
        });
        group.bench_function(BenchmarkId::new(format!("{label}_{m}x{n}-serial_f64"), "argmin"), |bench| {
            bench.iter(|| black_box(rt::argmin(black_box(&a))))
        });
        let a = gen_mat_f64!(m, n, 1, &dev_faer);
        group.bench_function(BenchmarkId::new(format!("{label}_{m}x{n}-faer16_f64"), "argmax"), |bench| {
            bench.iter(|| black_box(rt::argmax(black_box(&a))))
        });
        group.bench_function(BenchmarkId::new(format!("{label}_{m}x{n}-faer16_f64"), "argmin"), |bench| {
            bench.iter(|| black_box(rt::argmin(black_box(&a))))
        });
    }

    group.finish();
}

fn bench_arg_strided(c: &mut Criterion) {
    let dev_serial = serial_device();
    let dev_faer = faer_device();
    assert_faer_threads(&dev_faer, 16);

    let mut group = c.benchmark_group("arg_strided");
    configure_group(&mut group);

    // Strided case: transpose view. `a.t()` is a free view; the arg* op walks
    // it through IndexedIterLayout (RowMajor) with per-element offset
    // arithmetic — the non-contiguous fallback path this task must not
    // regress. (No `.to_contig()` here: the VIEW is the benched op.)
    // The small 64x64 case gates the fallback against wiring overhead.
    let cases: [(&str, usize, usize); 2] = [("small", 64, 64), ("large", 2048, 2048)];
    for (label, m, n) in cases {
        let a = gen_mat_f64!(m, n, 1, &dev_serial);
        let at = a.t(); // shape [n, m], strided view
        group.bench_function(BenchmarkId::new(format!("{label}_{n}x{m}-tview-serial_f64"), "argmax"), |bench| {
            bench.iter(|| black_box(rt::argmax(black_box(&at))))
        });
        group.bench_function(BenchmarkId::new(format!("{label}_{n}x{m}-tview-serial_f64"), "argmin"), |bench| {
            bench.iter(|| black_box(rt::argmin(black_box(&at))))
        });
        let a = gen_mat_f64!(m, n, 1, &dev_faer);
        let at = a.t();
        group.bench_function(BenchmarkId::new(format!("{label}_{n}x{m}-tview-faer16_f64"), "argmax"), |bench| {
            bench.iter(|| black_box(rt::argmax(black_box(&at))))
        });
        group.bench_function(BenchmarkId::new(format!("{label}_{n}x{m}-tview-faer16_f64"), "argmin"), |bench| {
            bench.iter(|| black_box(rt::argmin(black_box(&at))))
        });
    }

    group.finish();
}

criterion_group!(benches, bench_arg_1d, bench_arg_2d_whole, bench_arg_strided);
criterion_main!(benches);

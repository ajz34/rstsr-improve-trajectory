//! ndarray anchors (dev-dep, single-threaded) — context numbers for the same
//! ops/sizes as the rstsr arg* suites. No rstsr involvement.
//!
//! argmax/argmin come from `ndarray-stats 0.6` (`QuantileExt`; ndarray 0.16 has
//! no built-in argmax). ndarray-stats semantics on NaN-free data match the
//! first-occurrence rule; fixtures are NaN-free by construction (deterministic
//! hash in [-0.5, 0.5)).

use std::hint::black_box;

use bench_argmax_argmin::{configure_group, gen_vec_f32, gen_vec_f64, ARG_MAT_SIZES, ARG_SIZES};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array1, Array2};
use ndarray_stats::QuantileExt;

fn bench_anchor_arg(c: &mut Criterion) {
    let mut group = c.benchmark_group("anchor_ndarray_arg");
    configure_group(&mut group);

    // 1-D
    for &(label, n) in ARG_SIZES {
        let a = Array1::from(gen_vec_f64(n, 1));
        group.bench_function(BenchmarkId::new(format!("{label}_{n}-f64"), "argmax"), |bench| {
            bench.iter(|| black_box(black_box(&a).argmax()))
        });
        group.bench_function(BenchmarkId::new(format!("{label}_{n}-f64"), "argmin"), |bench| {
            bench.iter(|| black_box(black_box(&a).argmin()))
        });
    }

    // 1-D f32 spot at 1e7
    {
        let n = 10_000_000usize;
        let a = Array1::from(gen_vec_f32(n, 1));
        group.bench_function(BenchmarkId::new(format!("large_{n}-f32"), "argmax"), |bench| {
            bench.iter(|| black_box(black_box(&a).argmax()))
        });
        group.bench_function(BenchmarkId::new(format!("large_{n}-f32"), "argmin"), |bench| {
            bench.iter(|| black_box(black_box(&a).argmin()))
        });
    }

    // 2-D whole-tensor (returns (i, j) pattern)
    for &(label, m, n) in ARG_MAT_SIZES {
        let a = Array2::from_shape_vec((m, n), gen_vec_f64(m * n, 1)).unwrap();
        group.bench_function(BenchmarkId::new(format!("{label}_{m}x{n}-f64"), "argmax"), |bench| {
            bench.iter(|| black_box(black_box(&a).argmax()))
        });
        group.bench_function(BenchmarkId::new(format!("{label}_{m}x{n}-f64"), "argmin"), |bench| {
            bench.iter(|| black_box(black_box(&a).argmin()))
        });
    }

    // strided transpose-view case at large
    {
        let (m, n) = (2048usize, 2048usize);
        let a = Array2::from_shape_vec((m, n), gen_vec_f64(m * n, 1)).unwrap();
        let at = a.t(); // shape [n, m], strided view
        group.bench_function(BenchmarkId::new(format!("large_{n}x{m}-tview-f64"), "argmax"), |bench| {
            bench.iter(|| black_box(black_box(&at).argmax()))
        });
        group.bench_function(BenchmarkId::new(format!("large_{n}x{m}-tview-f64"), "argmin"), |bench| {
            bench.iter(|| black_box(black_box(&at).argmin()))
        });
    }

    group.finish();
}

criterion_group!(benches, bench_anchor_arg);
criterion_main!(benches);

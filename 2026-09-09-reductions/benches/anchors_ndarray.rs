//! ndarray anchors for the T2' reduction benches (same fixtures, same sizes).
//!
//! ndarray is single-threaded; these anchor the serial device. Numbers are
//! also context for faer16 (ndarray has no rayon reduction).

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array2;
use reductions::{configure_group, gen_vec_f64};

fn arr2(m: usize, n: usize) -> Array2<f64> {
    Array2::from_shape_vec((m, n), gen_vec_f64(m * n, 1)).unwrap()
}

fn bench_anchors(c: &mut Criterion) {
    let mut group = c.benchmark_group("anchors_ndarray");
    configure_group(&mut group);

    for &(label, m, n) in [("medium", 512usize, 512usize), ("large", 2048, 2048), ("odd", 1000, 777)].iter() {
        let a = arr2(m, n);
        let id = move |op: &str| BenchmarkId::new(format!("{label}_{m}x{n}-serial_f64"), op);
        group.bench_function(id("sum_axis0"), |bench| {
            bench.iter(|| black_box(black_box(&a).sum_axis(ndarray::Axis(0))))
        });
        group.bench_function(id("sum_axis1"), |bench| {
            bench.iter(|| black_box(black_box(&a).sum_axis(ndarray::Axis(1))))
        });
        group.bench_function(id("sum_all"), |bench| bench.iter(|| black_box(black_box(&a).sum())));
        group.bench_function(id("mean_axis1"), |bench| {
            bench.iter(|| black_box(black_box(&a).mean_axis(ndarray::Axis(1))))
        });
        group.bench_function(id("var_axis1"), |bench| {
            bench.iter(|| black_box(black_box(&a).var_axis(ndarray::Axis(1), 0.0)))
        });
    }

    // 1-D full-reduction anchors at n = 1e7
    {
        let n = 10_000_000usize;
        let a = ndarray::Array1::from_vec(gen_vec_f64(n, 1));
        let id = move |op: &str| BenchmarkId::new("large_1e7-serial_f64", op);
        group.bench_function(id("sum_all"), |bench| bench.iter(|| black_box(black_box(&a).sum())));
        group.bench_function(id("mean_all"), |bench| bench.iter(|| black_box(black_box(&a).mean())));
        group.bench_function(id("var_all"), |bench| {
            bench.iter(|| black_box(black_box(&a).var_axis(ndarray::Axis(0), 0.0)))
        });
    }

    group.finish();
}

criterion_group!(benches, bench_anchors);
criterion_main!(benches);

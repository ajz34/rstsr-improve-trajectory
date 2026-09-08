//! ndarray anchors (dev-dep, single-threaded) — context numbers for the same
//! ops/sizes as the rstsr suites. These run on plain Vec-backed Array2/Array1;
//! no rstsr involvement. Allocation policy mirrors the rstsr benches:
//! `to_owned`/`+`/`sum_axis`/`dot`/`argmax` allocate inside the op; the
//! explicit-Zip add variant reuses a preallocated output (shown separately,
//! labeled `zip_prealloc`, as ndarray's in-place ceiling).

use std::hint::black_box;

use bench_harness_baseline::{configure_group, gen_vec_f64, ARGMAX_SIZES, MAT_SIZES, VECDOT_SIZES};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array1, Array2, Axis, Zip};
use ndarray_stats::QuantileExt;

fn mat(m: usize, n: usize, salt: usize) -> Array2<f64> {
    Array2::from_shape_vec((m, n), gen_vec_f64(m * n, salt)).unwrap()
}

fn bench_anchors(c: &mut Criterion) {
    let mut group = c.benchmark_group("anchor_ndarray");
    configure_group(&mut group);

    // transpose copy + reductions on all four 2-D classes
    for &(label, m, n) in MAT_SIZES {
        let a = mat(m, n, 1);

        group.bench_function(BenchmarkId::new(format!("{label}_{m}x{n}-f64"), "transpose_copy"), |b| {
            b.iter(|| black_box(black_box(&a).t().to_owned()))
        });
        group.bench_function(BenchmarkId::new(format!("{label}_{m}x{n}-f64"), "sum_axis0"), |b| {
            b.iter(|| black_box(a.sum_axis(Axis(0))))
        });
        group.bench_function(BenchmarkId::new(format!("{label}_{m}x{n}-f64"), "sum_axislast"), |b| {
            b.iter(|| black_box(a.sum_axis(Axis(1))))
        });
        group.bench_function(BenchmarkId::new(format!("{label}_{m}x{n}-f64"), "sum_all"), |b| {
            b.iter(|| black_box(a.sum()))
        });
    }

    // add: operator form (allocating) and explicit zip into preallocated out
    let a = mat(2048, 2048, 1);
    let b = mat(2048, 2048, 2);
    let mut c = Array2::<f64>::zeros((2048, 2048));
    group.bench_function(BenchmarkId::new("large_2048x2048-contig-f64", "add_alloc"), |bench| {
        bench.iter(|| black_box(black_box(&a) + black_box(&b)))
    });
    group.bench_function(BenchmarkId::new("large_2048x2048-contig-f64", "add_zip_prealloc"), |bench| {
        bench.iter(|| {
            let cv = black_box(c.view_mut());
            Zip::from(cv).and(black_box(&a)).and(black_box(&b)).for_each(|c_i, &a_i, &b_i| *c_i = a_i + b_i);
            black_box(c[[0, 0]]);
            c.len()
        })
    });

    // vecdot 1-D
    for &(label, n) in VECDOT_SIZES {
        let a = Array1::from(gen_vec_f64(n, 1));
        let b = Array1::from(gen_vec_f64(n, 2));
        group.bench_function(BenchmarkId::new(format!("dot1d_{label}_{n}-f64"), "dot"), |bench| {
            bench.iter(|| black_box(a.dot(black_box(&b))))
        });
    }

    // argmax 1-D
    for &(label, n) in ARGMAX_SIZES {
        let a = Array1::from(gen_vec_f64(n, 1));
        group.bench_function(BenchmarkId::new(format!("{label}_{n}-f64"), "argmax"), |bench| {
            bench.iter(|| black_box(a.argmax()))
        });
    }

    // fill anchors: zeros + full at large
    group.bench_function(BenchmarkId::new("large_2048x2048-f64", "zeros"), |b| {
        b.iter(|| black_box(Array2::<f64>::zeros((2048, 2048))))
    });
    group.bench_function(BenchmarkId::new("large_2048x2048-f64", "full"), |b| {
        b.iter(|| black_box(Array2::from_elem((2048, 2048), 3.25_f64)))
    });

    group.finish();
}

criterion_group!(benches, bench_anchors);
criterion_main!(benches);

//! ndarray anchors for T1' transpose copy: `a.t().to_owned()` on the same
//! sizes and fixtures (ndarray is row-major; `.t()` is a view; `.to_owned()`
//! materializes the transposed copy). Serial in-process — same meaning as
//! T0's anchor column. Allocation included (fresh output each iteration).

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array2;
use transpose_assign::{configure_group, gen_vec_f64, MAT_SIZES};

fn bench_ndarray_transpose(c: &mut Criterion) {
    let mut group = c.benchmark_group("ndarray_transpose");
    configure_group(&mut group);

    for &(label, m, n) in MAT_SIZES {
        let v = gen_vec_f64(m * n, 1);
        let a = Array2::from_shape_vec((m, n), v).unwrap();
        group.bench_function(
            BenchmarkId::new(format!("{label}_{m}x{n}-serial_f64"), "t_to_owned"),
            |bench| {
                bench.iter(|| {
                    let out = black_box(&a).t().to_owned();
                    black_box(out)
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_ndarray_transpose);
criterion_main!(benches);

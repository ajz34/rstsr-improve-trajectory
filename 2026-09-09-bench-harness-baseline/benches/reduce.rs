//! Reduction benchmarks: sum over axis 0 (down columns, stride n for
//! row-major), sum over the last axis (within rows, contiguous runs), and
//! full-tensor sum. Allocation included (output vector/scalar allocated by
//! the op each iteration).

use std::hint::black_box;

use bench_harness_baseline::{assert_faer_threads, configure_group, faer_device, serial_device, MAT_SIZES};
use bench_harness_baseline::{gen_mat_f32, gen_mat_f64};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

macro_rules! bench_reduce_one {
    ($group:expr, $prefix:expr, $a:expr) => {{
        let a = $a;
        $group.bench_function(BenchmarkId::new($prefix, "sum_axis0"), |bench| {
            bench.iter(|| black_box(a.sum_axes(0)))
        });
        // sum over the last axis: contiguous runs within rows
        $group.bench_function(BenchmarkId::new($prefix, "sum_axislast"), |bench| {
            bench.iter(|| black_box(a.sum_axes(-1)))
        });
        // full-tensor reduction -> scalar
        $group.bench_function(BenchmarkId::new($prefix, "sum_all"), |bench| {
            bench.iter(|| black_box(a.sum_all()))
        });
    }};
}

fn bench_reduce(c: &mut Criterion) {
    let dev_serial = serial_device();
    let dev_faer = faer_device();
    assert_faer_threads(&dev_faer, 16);

    let mut group = c.benchmark_group("reduce");
    configure_group(&mut group);

    // f64 primary, all four size classes, both devices
    for &(label, m, n) in MAT_SIZES {
        bench_reduce_one!(&mut group, format!("{label}_{m}x{n}-serial_f64"), gen_mat_f64!(m, n, 1, &dev_serial));
        bench_reduce_one!(&mut group, format!("{label}_{m}x{n}-faer16_f64"), gen_mat_f64!(m, n, 1, &dev_faer));
    }

    // f32 secondary spot-check (large, axis 0 measured for both devices)
    {
        let a = gen_mat_f32!(2048, 2048, 1, &dev_serial);
        group.bench_function(BenchmarkId::new("large_2048x2048-serial_f32", "sum_axis0"), |bench| {
            bench.iter(|| black_box(a.sum_axes(0)))
        });
        let a = gen_mat_f32!(2048, 2048, 1, &dev_faer);
        group.bench_function(BenchmarkId::new("large_2048x2048-faer16_f32", "sum_axis0"), |bench| {
            bench.iter(|| black_box(a.sum_axes(0)))
        });
    }

    group.finish();
}

criterion_group!(benches, bench_reduce);
criterion_main!(benches);

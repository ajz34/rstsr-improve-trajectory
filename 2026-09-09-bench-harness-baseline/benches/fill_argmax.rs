//! fill/creation and argmax benchmarks.
//!
//! - `zeros` / `full`: the `fill_promote` path (allocate + fill with scalar).
//!   Allocation obviously included — it IS the op.
//! - `argmax` on 1-D f64 at 1e6 / 1e7 (flat index, first max wins).

use std::hint::black_box;

use bench_harness_baseline::{assert_faer_threads, configure_group, faer_device, serial_device, ARGMAX_SIZES};
use bench_harness_baseline::gen_vec_dev_f64;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rstsr_core::prelude::*;

fn bench_fill(c: &mut Criterion) {
    let dev_serial = serial_device();
    let dev_faer = faer_device();
    assert_faer_threads(&dev_faer, 16);

    let mut group = c.benchmark_group("fill");
    configure_group(&mut group);

    group.bench_function(BenchmarkId::new("large_2048x2048-zeros", "serial_f64"), |bench| {
        bench.iter(|| {
            let t: Tensor<f64, _> = rt::zeros(([2048, 2048], &dev_serial));
            black_box(t)
        })
    });
    group.bench_function(BenchmarkId::new("large_2048x2048-zeros", "faer16_f64"), |bench| {
        bench.iter(|| {
            let t: Tensor<f64, _> = rt::zeros(([2048, 2048], &dev_faer));
            black_box(t)
        })
    });
    group.bench_function(BenchmarkId::new("large_2048x2048-full", "serial_f64"), |bench| {
        bench.iter(|| {
            let t: Tensor<f64, _> = rt::full(([2048, 2048], 3.25_f64, &dev_serial));
            black_box(t)
        })
    });
    group.bench_function(BenchmarkId::new("large_2048x2048-full", "faer16_f64"), |bench| {
        bench.iter(|| {
            let t: Tensor<f64, _> = rt::full(([2048, 2048], 3.25_f64, &dev_faer));
            black_box(t)
        })
    });

    group.finish();
}

fn bench_argmax(c: &mut Criterion) {
    let dev_serial = serial_device();
    let dev_faer = faer_device();
    assert_faer_threads(&dev_faer, 16);

    let mut group = c.benchmark_group("argmax");
    configure_group(&mut group);

    for &(label, n) in ARGMAX_SIZES {
        let a = gen_vec_dev_f64!(n, 1, &dev_serial);
        group.bench_function(BenchmarkId::new(format!("{label}_{n}-serial_f64"), "argmax"), |bench| {
            bench.iter(|| black_box(rt::argmax(black_box(&a))))
        });
        let a = gen_vec_dev_f64!(n, 1, &dev_faer);
        group.bench_function(BenchmarkId::new(format!("{label}_{n}-faer16_f64"), "argmax"), |bench| {
            bench.iter(|| black_box(rt::argmax(black_box(&a))))
        });
    }

    group.finish();
}

criterion_group!(benches, bench_fill, bench_argmax);
criterion_main!(benches);

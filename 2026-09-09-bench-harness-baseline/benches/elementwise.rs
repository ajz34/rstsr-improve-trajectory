//! Elementwise add `c = a + b` benchmarks.
//!
//! Variants (all at 2048x2048 large unless noted):
//! - `contig`: both operands contiguous row-major.
//! - `broadcast`: b is a [1, 2048] row vector auto-broadcast over [2048, 2048]
//!   (zero-stride on the leading axis). Choice documented in README.
//! - `strided`: b is the transpose VIEW of a contiguous tensor, i.e. `a + b.t()`
//!   style (fully strided second operand).
//! - medium 512x512 contiguous spot, f32 large contiguous spot.
//!
//! The `+` operator ALLOCATES the output inside the op on every iteration —
//! allocation included, identical policy on both devices and both configs.

use std::hint::black_box;

use bench_harness_baseline::{assert_faer_threads, configure_group, faer_device, serial_device};
use bench_harness_baseline::{gen_mat_f32, gen_mat_f64};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

fn bench_add(c: &mut Criterion) {
    let dev_serial = serial_device();
    let dev_faer = faer_device();
    assert_faer_threads(&dev_faer, 16);

    let mut group = c.benchmark_group("elementwise_add");
    configure_group(&mut group);

    // --- large contiguous + broadcast + strided (f64 primary) --------------
    {
        let a = gen_mat_f64!(2048, 2048, 1, &dev_serial);
        let b = gen_mat_f64!(2048, 2048, 2, &dev_serial);
        let brow = gen_mat_f64!(1, 2048, 3, &dev_serial);
        let bt = gen_mat_f64!(2048, 2048, 4, &dev_serial);

        group.bench_function(BenchmarkId::new("large_2048x2048-contig", "serial_f64"), |bench| {
            bench.iter(|| black_box(black_box(&a) + black_box(&b)))
        });
        group.bench_function(BenchmarkId::new("large_2048x2048-broadcast", "serial_f64"), |bench| {
            bench.iter(|| black_box(black_box(&a) + black_box(&brow)))
        });
        group.bench_function(BenchmarkId::new("large_2048x2048-strided", "serial_f64"), |bench| {
            bench.iter(|| black_box(black_box(&a) + black_box(&bt.t())))
        });
    }
    {
        let a = gen_mat_f64!(2048, 2048, 1, &dev_faer);
        let b = gen_mat_f64!(2048, 2048, 2, &dev_faer);
        let brow = gen_mat_f64!(1, 2048, 3, &dev_faer);
        let bt = gen_mat_f64!(2048, 2048, 4, &dev_faer);

        group.bench_function(BenchmarkId::new("large_2048x2048-contig", "faer16_f64"), |bench| {
            bench.iter(|| black_box(black_box(&a) + black_box(&b)))
        });
        group.bench_function(BenchmarkId::new("large_2048x2048-broadcast", "faer16_f64"), |bench| {
            bench.iter(|| black_box(black_box(&a) + black_box(&brow)))
        });
        group.bench_function(BenchmarkId::new("large_2048x2048-strided", "faer16_f64"), |bench| {
            bench.iter(|| black_box(black_box(&a) + black_box(&bt.t())))
        });
    }

    // --- medium contiguous spot (f64) ---------------------------------------
    {
        let a = gen_mat_f64!(512, 512, 1, &dev_serial);
        let b = gen_mat_f64!(512, 512, 2, &dev_serial);
        group.bench_function(BenchmarkId::new("medium_512x512-contig", "serial_f64"), |bench| {
            bench.iter(|| black_box(black_box(&a) + black_box(&b)))
        });
        let a = gen_mat_f64!(512, 512, 1, &dev_faer);
        let b = gen_mat_f64!(512, 512, 2, &dev_faer);
        group.bench_function(BenchmarkId::new("medium_512x512-contig", "faer16_f64"), |bench| {
            bench.iter(|| black_box(black_box(&a) + black_box(&b)))
        });
    }

    // --- f32 large contiguous spot ------------------------------------------
    {
        let a = gen_mat_f32!(2048, 2048, 1, &dev_serial);
        let b = gen_mat_f32!(2048, 2048, 2, &dev_serial);
        group.bench_function(BenchmarkId::new("large_2048x2048-contig", "serial_f32"), |bench| {
            bench.iter(|| black_box(black_box(&a) + black_box(&b)))
        });
        let a = gen_mat_f32!(2048, 2048, 1, &dev_faer);
        let b = gen_mat_f32!(2048, 2048, 2, &dev_faer);
        group.bench_function(BenchmarkId::new("large_2048x2048-contig", "faer16_f32"), |bench| {
            bench.iter(|| black_box(black_box(&a) + black_box(&b)))
        });
    }

    group.finish();
}

criterion_group!(benches, bench_add);
criterion_main!(benches);

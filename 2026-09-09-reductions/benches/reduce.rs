//! T2' reduction benchmarks (phase-1 baseline).
//!
//! Value reductions: 2-D axis reductions (sum over axis 0 = down columns,
//! stride-n reads for row-major; sum over the last axis = contiguous runs),
//! full-tensor sum, and spot checks for mean/var/l2_norm/min/max.
//! 1-D full-reduce spots at n=1e7.
//!
//! Allocation policy: T7 measured the fault rider for reductions at exactly
//! 0% (outputs are scalars or [n] <= 16 KiB vectors — no >=32 MiB fresh
//! output), so the idiomatic allocating call is the kernel-only denominator
//! here; no MALLOC column (a page-faults sanity row comes from `perf stat` on
//! the profile runs instead).

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use reductions::{assert_faer_threads, configure_group, faer_device, gen_mat_f32, gen_mat_f64, gen_vec_dev_f32, gen_vec_dev_f64, serial_device, MAT_SIZES};

/// Per-device 2-D benches; `$dev` is a device reference expression.
macro_rules! bench_2d_device {
    ($group:expr, $dev_name:expr, $dev:expr) => {{
        let group = $group;
        let dev = &$dev;
        for &(label, m, n) in MAT_SIZES {
            let a = gen_mat_f64!(m, n, 1, dev);
            let id = |op: &str| BenchmarkId::new(format!("{label}_{m}x{n}-{}_f64", $dev_name), op);

            // headline ops: all size classes (these are also the gates)
            group.bench_function(id("sum_axis0"), |bench| bench.iter(|| black_box(black_box(&a).sum_axes(0))));
            group.bench_function(id("sum_axis1"), |bench| bench.iter(|| black_box(black_box(&a).sum_axes(-1))));
            group.bench_function(id("sum_all"), |bench| bench.iter(|| black_box(black_box(&a).sum_all())));

            // spots (mean/var/norm/min/max) on medium + large only
            if matches!(label, "medium" | "large") {
                group.bench_function(id("mean_axis1"), |bench| {
                    bench.iter(|| black_box(black_box(&a).mean_axes(-1)))
                });
                group.bench_function(id("var_axis1"), |bench| {
                    bench.iter(|| black_box(black_box(&a).var_axes(-1)))
                });
                group.bench_function(id("norm_all"), |bench| {
                    bench.iter(|| black_box(black_box(&a).l2_norm_all()))
                });
                group.bench_function(id("min_axis0"), |bench| {
                    bench.iter(|| black_box(black_box(&a).min_axes(0)))
                });
                group.bench_function(id("max_axis0"), |bench| {
                    bench.iter(|| black_box(black_box(&a).max_axes(0)))
                });
            }
        }
    }};
}

fn bench_reduce(c: &mut Criterion) {
    let dev_serial = serial_device();
    let dev_faer = faer_device();
    assert_faer_threads(&dev_faer, 16);

    // ------------------------------------------------------------------
    // 2-D axis reductions + spots, f64, four size classes, both devices
    // ------------------------------------------------------------------
    let mut group = c.benchmark_group("reduce2d");
    configure_group(&mut group);
    bench_2d_device!(&mut group, "serial", dev_serial);
    bench_2d_device!(&mut group, "faer16", dev_faer);
    group.finish();

    // ------------------------------------------------------------------
    // 1-D full-reduction spots at n = 1e7
    // ------------------------------------------------------------------
    let mut group = c.benchmark_group("reduce1d");
    configure_group(&mut group);
    {
        let n = 10_000_000usize;
        {
            let a = gen_vec_dev_f64!(n, 1, &dev_serial);
            let id = |op: &str| BenchmarkId::new("large_1e7-serial_f64", op);
            group.bench_function(id("sum_all"), |bench| bench.iter(|| black_box(black_box(&a).sum_all())));
            group.bench_function(id("mean_all"), |bench| bench.iter(|| black_box(black_box(&a).mean_all())));
            group.bench_function(id("var_all"), |bench| bench.iter(|| black_box(black_box(&a).var_all())));
            group.bench_function(id("norm_all"), |bench| bench.iter(|| black_box(black_box(&a).l2_norm_all())));
            group.bench_function(id("min_all"), |bench| bench.iter(|| black_box(black_box(&a).min_all())));
            group.bench_function(id("max_all"), |bench| bench.iter(|| black_box(black_box(&a).max_all())));
        }
        {
            let a = gen_vec_dev_f64!(n, 1, &dev_faer);
            let id = |op: &str| BenchmarkId::new("large_1e7-faer16_f64", op);
            group.bench_function(id("sum_all"), |bench| bench.iter(|| black_box(black_box(&a).sum_all())));
            group.bench_function(id("mean_all"), |bench| bench.iter(|| black_box(black_box(&a).mean_all())));
            group.bench_function(id("var_all"), |bench| bench.iter(|| black_box(black_box(&a).var_all())));
            group.bench_function(id("norm_all"), |bench| bench.iter(|| black_box(black_box(&a).l2_norm_all())));
            group.bench_function(id("min_all"), |bench| bench.iter(|| black_box(black_box(&a).min_all())));
            group.bench_function(id("max_all"), |bench| bench.iter(|| black_box(black_box(&a).max_all())));
        }
    }
    group.finish();

    // ------------------------------------------------------------------
    // f32 secondary spots (D5)
    // ------------------------------------------------------------------
    let mut group = c.benchmark_group("reduce2d_f32");
    configure_group(&mut group);
    {
        let (m, n) = (2048usize, 2048usize);
        let a = gen_mat_f32!(m, n, 1, &dev_serial);
        let id = |op: &str| BenchmarkId::new(format!("large_{m}x{n}-serial_f32"), op);
        group.bench_function(id("sum_axis0"), |bench| bench.iter(|| black_box(black_box(&a).sum_axes(0))));
        group.bench_function(id("sum_axis1"), |bench| bench.iter(|| black_box(black_box(&a).sum_axes(-1))));
        group.bench_function(id("sum_all"), |bench| bench.iter(|| black_box(black_box(&a).sum_all())));
        group.bench_function(id("min_axis0"), |bench| bench.iter(|| black_box(black_box(&a).min_axes(0))));

        let a = gen_mat_f32!(m, n, 1, &dev_faer);
        let id = |op: &str| BenchmarkId::new(format!("large_{m}x{n}-faer16_f32"), op);
        group.bench_function(id("sum_axis0"), |bench| bench.iter(|| black_box(black_box(&a).sum_axes(0))));
        group.bench_function(id("sum_axis1"), |bench| bench.iter(|| black_box(black_box(&a).sum_axes(-1))));

        let n = 10_000_000usize;
        let a = gen_vec_dev_f32!(n, 1, &dev_serial);
        group.bench_function(
            BenchmarkId::new("large_1e7-serial_f32", "sum_all"),
            |bench| bench.iter(|| black_box(black_box(&a).sum_all())),
        );
    }
    group.finish();
}

criterion_group!(benches, bench_reduce);
criterion_main!(benches);

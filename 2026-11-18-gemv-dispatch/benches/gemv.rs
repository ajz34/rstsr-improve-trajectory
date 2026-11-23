//! P0 benchmarks: current rstsr matmul vs raw cblas gemv, per device and cell.
//!
//! `rstsr` rows use the public `%` operator (fresh output, `beta = 0`, i.e. the
//! `matmul_uninit` path users hit). `gemv-anchor` rows call `cblas_?gemv`
//! directly on the same buffer.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use gemv_dispatch_p0::*;
use std::hint::black_box;

macro_rules! bench_cells {
    ($g:expr, $dev:expr, $ty:ty) => {{
        let dev = &$dev;
        for &(p, q, name) in CELLS {
            let a = linspace((0 as $ty, 1 as $ty, p * q, dev)).into_shape([p, q]);
            let xp = linspace((0 as $ty, 1 as $ty, p, dev));
            let xq = linspace((0 as $ty, 1 as $ty, q, dev));
            // column-major view of the row-major fixture: shape (q, p), lda = q
            let ap = a.as_ptr();

            $g.bench_function(BenchmarkId::new(format!("{name}/matvec"), "rstsr"), |b| {
                b.iter(|| black_box(&a % &xq));
            });
            $g.bench_function(BenchmarkId::new(format!("{name}/matvec"), "gemv-anchor"), |b| {
                let mut y = vec![0 as $ty; p];
                b.iter(|| unsafe {
                    <$ty as CblasGemv>::cblas_gemv(true, q, p, ap, q, xq.as_ptr(), y.as_mut_ptr());
                    black_box(&y);
                });
            });

            $g.bench_function(BenchmarkId::new(format!("{name}/vecmat"), "rstsr"), |b| {
                b.iter(|| black_box(&xp % &a));
            });
            $g.bench_function(BenchmarkId::new(format!("{name}/vecmat"), "gemv-anchor"), |b| {
                let mut y = vec![0 as $ty; q];
                b.iter(|| unsafe {
                    <$ty as CblasGemv>::cblas_gemv(false, q, p, ap, q, xp.as_ptr(), y.as_mut_ptr());
                    black_box(&y);
                });
            });
        }
    }};
}

macro_rules! bench_extras {
    ($g:expr, $dev:expr, $ty:ty) => {{
        let dev = &$dev;
        // beta != 0 with a caller-provided output vector
        let a = linspace((0 as $ty, 1 as $ty, 1024 * 1024, dev)).into_shape([1024, 1024]);
        let x = linspace((0 as $ty, 1 as $ty, 1024, dev));
        let mut c = linspace((0 as $ty, 1 as $ty, 1024, dev));
        $g.bench_function("extra/matmul_from-beta/1024x1024", |b| {
            b.iter(|| {
                matmul_from_f(c.view_mut(), a.view(), x.view(), 1 as $ty, 1 as $ty).unwrap();
            });
        });

        // batched rule 6: (64, 1024, 1024) @ (1024,) -> (64, 1024)
        let ab = linspace((0 as $ty, 1 as $ty, 64 * 1024 * 1024, dev)).into_shape([64, 1024, 1024]);
        let x = linspace((0 as $ty, 1 as $ty, 1024, dev));
        $g.bench_function("extra/batched-gemv/64x1024x1024", |b| {
            b.iter(|| black_box(&ab % &x));
        });

        // batched rule 7 gemm: (64, 512, 512) @ (64, 512, 512) — D29's regression datum
        let a7 = linspace((0 as $ty, 1 as $ty, 64 * 512 * 512, dev)).into_shape([64, 512, 512]);
        let b7 = linspace((0 as $ty, 1 as $ty, 64 * 512 * 512, dev)).into_shape([64, 512, 512]);
        $g.bench_function("extra/batched-gemm/64x512x512", |b| {
            b.iter(|| black_box(&a7 % &b7));
        });
    }};
}

fn bench_all(c: &mut Criterion) {
    assert_threads();
    pin_blas_threads(NTHREADS);
    eprintln!("blas={}", blas_config());

    let mut g = c.benchmark_group("openblas/f64");
    g.sample_size(10).measurement_time(std::time::Duration::from_secs(3));
    bench_cells!(g, DeviceOpenBLAS::new(NTHREADS), f64);
    bench_extras!(g, DeviceOpenBLAS::new(NTHREADS), f64);
    g.finish();

    let mut g = c.benchmark_group("faer/f64");
    g.sample_size(10).measurement_time(std::time::Duration::from_secs(3));
    bench_cells!(g, DeviceFaer::new(NTHREADS), f64);
    bench_extras!(g, DeviceFaer::new(NTHREADS), f64);
    g.finish();

    let mut g = c.benchmark_group("openblas/f32");
    g.sample_size(10).measurement_time(std::time::Duration::from_secs(3));
    {
        let dev = DeviceOpenBLAS::new(NTHREADS);
        let (p, q) = (1024usize, 1024usize);
        let a = linspace((0f32, 1f32, p * q, &dev)).into_shape([p, q]);
        let xq = linspace((0f32, 1f32, q, &dev));
        let ap = a.as_ptr();
        g.bench_function(BenchmarkId::new("1024x1024/matvec", "rstsr-f32"), |b| {
            b.iter(|| black_box(&a % &xq));
        });
        g.bench_function(BenchmarkId::new("1024x1024/matvec", "gemv-anchor-f32"), |b| {
            let mut y = vec![0f32; p];
            b.iter(|| unsafe {
                <f32 as CblasGemv>::cblas_gemv(true, q, p, ap, q, xq.as_ptr(), y.as_mut_ptr());
                black_box(&y);
            });
        });
    }
    g.finish();
}

criterion_group!(benches, bench_all);
criterion_main!(benches);

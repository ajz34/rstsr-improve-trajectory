//! inner_dot benchmarks (T3' phase 1 baseline).
//!
//! API discovery (verified against the 386948be source): the ONLY user-level
//! operation that routes to `inner_dot_naive_cpu_rayon` is matmul `%` with
//! both operands 1-D (rule `(1, 1, 0)` in `matmul_row_major_faer`,
//! rstsr-core/src/device_faer/matmul.rs:114-122 — and the mirrored serial
//! rule in `matmul_naive_cpu_serial`). Notably, under DeviceFaer this path is
//! taken for ALL dtypes — f64 included — so 1-D `%` never reaches faer's BLAS
//! dot; it always runs rstsr's own rayon fold with per-element
//! `index_uncheck` arithmetic. Under DeviceCpuSerial the twin
//! `inner_dot_naive_cpu_serial` runs a scalar fold with `alpha.clone() *`
//! inside the loop.
//!
//! Larger matvec-like products (`(m,n) % (n,)`) do NOT hit inner_dot; they
//! take the broadcasted-rules path into gemm/gemv, out of T3' scope.

use std::hint::black_box;

use exp_vecdot::{assert_faer_threads, configure_group, faer_device, serial_device, INNERDOT_SIZES};
use exp_vecdot::{gen_vec_dev_c64, gen_vec_dev_f64};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

fn bench_innerdot(c: &mut Criterion) {
    let dev_serial = serial_device();
    let dev_faer = faer_device();
    assert_faer_threads(&dev_faer, 16);

    let mut group = c.benchmark_group("innerdot");
    configure_group(&mut group);

    // ---- `%` on 1-D · 1-D, f64 primary ----
    for &(label, n) in INNERDOT_SIZES {
        group.bench_function(
            BenchmarkId::new(format!("dot1d_{label}_{n}-serial_f64"), "percent"),
            |b| {
                let a = gen_vec_dev_f64!(n, 1, &dev_serial);
                let bb = gen_vec_dev_f64!(n, 2, &dev_serial);
                b.iter(|| black_box(black_box(&a) % black_box(&bb)))
            },
        );
        group.bench_function(
            BenchmarkId::new(format!("dot1d_{label}_{n}-faer16_f64"), "percent"),
            |b| {
                let a = gen_vec_dev_f64!(n, 1, &dev_faer);
                let bb = gen_vec_dev_f64!(n, 2, &dev_faer);
                b.iter(|| black_box(black_box(&a) % black_box(&bb)))
            },
        );
    }

    // ---- Complex<f64> spot at 1e5 (D5 secondary dtype) ----
    group.bench_function(BenchmarkId::new("dot1d_medium_100000-serial_c64", "percent"), |b| {
        let a = gen_vec_dev_c64!(100_000, 1, &dev_serial);
        let bb = gen_vec_dev_c64!(100_000, 2, &dev_serial);
        b.iter(|| black_box(black_box(&a) % black_box(&bb)))
    });
    group.bench_function(BenchmarkId::new("dot1d_medium_100000-faer16_c64", "percent"), |b| {
        let a = gen_vec_dev_c64!(100_000, 1, &dev_faer);
        let bb = gen_vec_dev_c64!(100_000, 2, &dev_faer);
        b.iter(|| black_box(black_box(&a) % black_box(&bb)))
    });

    group.finish();
}

criterion_group!(benches, bench_innerdot);
criterion_main!(benches);

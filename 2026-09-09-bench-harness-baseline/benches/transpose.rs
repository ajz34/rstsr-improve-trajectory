//! Transpose-copy benchmark: `a.t().to_contig(RowMajor)`.
//!
//! `.t()` is a VIEW (permuted strides); `to_contig(RowMajor)` on that view
//! cannot borrow and must materialize a new contiguous tensor — this measures
//! the actual data-movement path (`change_layout_f` -> `assign_arbitary_*`),
//! not the free view. Allocation included (output tensor is allocated by the
//! op on every iteration — same policy on both devices and both configs).

use std::hint::black_box;

use bench_harness_baseline::{assert_faer_threads, configure_group, faer_device, serial_device, MAT_SIZES};
use bench_harness_baseline::{gen_mat_f32, gen_mat_f64};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rstsr_core::prelude::*;

macro_rules! bench_transpose_one {
    ($group:expr, $id:expr, $a:expr) => {{
        let a = $a;
        $group.bench_function($id, |bench| {
            bench.iter(|| {
                let at = black_box(&a).t();
                let out = at.to_contig(RowMajor).into_owned();
                black_box(out)
            })
        });
    }};
}

fn bench_transpose_copy(c: &mut Criterion) {
    let dev_serial = serial_device();
    let dev_faer = faer_device();
    assert_faer_threads(&dev_faer, 16);

    let mut group = c.benchmark_group("transpose_copy");
    configure_group(&mut group);

    // f64 primary, all four size classes, both devices
    for &(label, m, n) in MAT_SIZES {
        bench_transpose_one!(&mut group, BenchmarkId::new(format!("{label}_{m}x{n}-serial_f64"), "to_contig"), gen_mat_f64!(m, n, 1, &dev_serial));
        bench_transpose_one!(&mut group, BenchmarkId::new(format!("{label}_{m}x{n}-faer16_f64"), "to_contig"), gen_mat_f64!(m, n, 1, &dev_faer));
    }

    // f32 secondary spot-check (large)
    bench_transpose_one!(&mut group, BenchmarkId::new("large_2048x2048-serial_f32", "to_contig"), gen_mat_f32!(2048, 2048, 1, &dev_serial));
    bench_transpose_one!(&mut group, BenchmarkId::new("large_2048x2048-faer16_f32", "to_contig"), gen_mat_f32!(2048, 2048, 1, &dev_faer));

    group.finish();
}

criterion_group!(benches, bench_transpose_copy);
criterion_main!(benches);

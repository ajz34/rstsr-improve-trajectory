//! vecdot benchmarks.
//!
//! - 1-D dot at 1e3 / 1e5 / 1e7 (f64 primary, f32 spot at 1e7).
//! - One batched vecdot: shape (4096, 512) . (4096, 512) -> 4096 outputs,
//!   contraction over the LAST axis (contiguous per output; row-major
//!   contiguous inputs => this is the "contiguous-remaining" branch of
//!   `vecdot_naive_*`, the #1-ranked code-map target). Choice documented in
//!   README. Allocation included (output tensor allocated per iteration).

use std::hint::black_box;

use bench_harness_baseline::{assert_faer_threads, configure_group, faer_device, serial_device, VECDOT_SIZES};
use bench_harness_baseline::{gen_mat_f64, gen_vec_dev_f32, gen_vec_dev_f64};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rstsr_core::prelude::*;

macro_rules! bench_dot1d_one {
    ($group:expr, $id:expr, $a:expr, $b:expr) => {{
        let (a, b) = ($a, $b);
        $group.bench_function($id, |bench| {
            bench.iter(|| black_box(rt::vecdot(black_box(&a), black_box(&b), None)))
        });
    }};
}

fn bench_vecdot(c: &mut Criterion) {
    let dev_serial = serial_device();
    let dev_faer = faer_device();
    assert_faer_threads(&dev_faer, 16);

    let mut group = c.benchmark_group("vecdot");
    configure_group(&mut group);

    // 1-D dot, f64 primary
    for &(label, n) in VECDOT_SIZES {
        bench_dot1d_one!(&mut group, BenchmarkId::new(format!("dot1d_{label}_{n}-serial_f64"), "vecdot"),
            gen_vec_dev_f64!(n, 1, &dev_serial), gen_vec_dev_f64!(n, 2, &dev_serial));
        bench_dot1d_one!(&mut group, BenchmarkId::new(format!("dot1d_{label}_{n}-faer16_f64"), "vecdot"),
            gen_vec_dev_f64!(n, 1, &dev_faer), gen_vec_dev_f64!(n, 2, &dev_faer));
    }

    // 1-D dot, f32 spot at 1e7
    bench_dot1d_one!(&mut group, BenchmarkId::new("dot1d_large_10000000-serial_f32", "vecdot"),
        gen_vec_dev_f32!(10_000_000, 1, &dev_serial), gen_vec_dev_f32!(10_000_000, 2, &dev_serial));
    bench_dot1d_one!(&mut group, BenchmarkId::new("dot1d_large_10000000-faer16_f32", "vecdot"),
        gen_vec_dev_f32!(10_000_000, 1, &dev_faer), gen_vec_dev_f32!(10_000_000, 2, &dev_faer));

    // batched: (4096, 512) . (4096, 512) -> 4096 outputs, contract last axis
    bench_dot1d_one!(&mut group, BenchmarkId::new("batched_4096x512-serial_f64", "vecdot"),
        gen_mat_f64!(4096, 512, 1, &dev_serial), gen_mat_f64!(4096, 512, 2, &dev_serial));
    bench_dot1d_one!(&mut group, BenchmarkId::new("batched_4096x512-faer16_f64", "vecdot"),
        gen_mat_f64!(4096, 512, 1, &dev_faer), gen_mat_f64!(4096, 512, 2, &dev_faer));

    group.finish();
}

criterion_group!(benches, bench_vecdot);
criterion_main!(benches);

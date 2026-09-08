//! Triad + memcpy bandwidth ceiling — pure Rust, no rstsr.
//!
//! This is the measured DRAM bandwidth ceiling that every streaming rstsr op
//! is framed against (plan D10). `c = a + 3.0*b` over f64 vectors sized so
//! the three buffers total >= 100 MB (n = 1e7 -> 240 MB traffic per pass).

use std::hint::black_box;

use bench_harness_baseline::{configure_group, gen_vec_f64};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

const N_LARGE: usize = 10_000_000; // 3 buffers = 240 MB f64 traffic
const N_MEDIUM: usize = 1_000_000; // 24 MB: L2/L3-boundary reference

fn bench_triad(c: &mut Criterion) {
    let mut group = c.benchmark_group("triad");
    configure_group(&mut group);

    for (label, n) in [("large", N_LARGE), ("medium", N_MEDIUM)] {
        let a = gen_vec_f64(n, 1);
        let b = gen_vec_f64(n, 2);
        let mut c_buf = vec![0.0_f64; n];

        group.bench_function(BenchmarkId::new("triad_f64", label), |bench| {
            bench.iter(|| {
                let a = black_box(&a);
                let b = black_box(&b);
                let c_buf = black_box(&mut c_buf);
                for (c_i, (a_i, b_i)) in c_buf.iter_mut().zip(a.iter().zip(b.iter())) {
                    *c_i = a_i + 3.0 * b_i;
                }
                black_box(c_buf[0]); // writes through the black-boxed &mut are observable
                c_buf.len()
            })
        });
    }

    group.finish();
}

fn bench_copy(c: &mut Criterion) {
    let mut group = c.benchmark_group("memcpy");
    configure_group(&mut group);

    for (label, n) in [("large", N_LARGE), ("medium", N_MEDIUM)] {
        let a = gen_vec_f64(n, 1);
        let mut c_buf = vec![0.0_f64; n];

        group.bench_function(BenchmarkId::new("copy_f64", label), |bench| {
            bench.iter(|| {
                let a = black_box(&a);
                let c_buf = black_box(&mut c_buf);
                c_buf.copy_from_slice(a);
                black_box(c_buf[0]);
                c_buf.len()
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_triad, bench_copy);
criterion_main!(benches);

//! ndarray anchors for the T4' elementwise study (f64).
//!
//! - `owned`: `&a + &b` with owned arrays — ALLOCATION INCLUDED, same policy
//!   as rstsr variant A (compare against A, not against B).
//! - `zip_prealloc`: `Zip::from(&mut c).and(&a).and(&b)` into a pre-allocated
//!   array — the KERNEL BOUND that a fixed-shape zip loop achieves on this
//!   machine (compare against rstsr variant B).
//!
//! Single-threaded (plain ndarray, no rayon feature).

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array2;

use elementwise::{configure_group, gen_vec_f64, gbps};

fn arr(m: usize, n: usize, salt: usize) -> Array2<f64> {
    Array2::from_shape_vec((m, n), gen_vec_f64(m * n, salt)).unwrap()
}

fn bench_anchors(c: &mut Criterion) {
    let mut group = c.benchmark_group("ndarray_add");
    configure_group(&mut group);

    for &(label, m, n) in [("medium", 512usize, 512usize), ("large", 2048, 2048)].iter() {
        let (a, b) = (arr(m, n, 1), arr(m, n, 2));
        group.bench_function(BenchmarkId::new(format!("{label}_{m}x{n}"), "owned"), |bench| {
            bench.iter(|| black_box(black_box(&a) + black_box(&b)))
        });

        let mut c = arr(m, n, 9);
        group.bench_function(BenchmarkId::new(format!("{label}_{m}x{n}"), "zip_prealloc"), |bench| {
            bench.iter(|| {
                ndarray::Zip::from(&mut c)
                    .and(black_box(&a))
                    .and(black_box(&b))
                    .for_each(|c_i, &a_i, &b_i| *c_i = a_i + b_i);
                black_box(&c);
            })
        });
    }
    group.finish();

    // quick derived-GB/s note (criterion prints times; README derives GB/s)
    let _ = gbps(24.0 * 2048.0 * 2048.0, 2e-3);
}

criterion_group!(benches, bench_anchors);
criterion_main!(benches);

//! T5 fill/creation benchmarks.
//!
//! Groups:
//! - `creation` (A, alloc-inclusive): `rt::full` / `rt::zeros` / `rt::ones`.
//!   These route through the device `*_impl` constructors (`vec![fill; len]`),
//!   NOT through `fill_promote` — verified by call-chain reading at 386948be
//!   (`device_cpu_serial/creation.rs`: full_impl/zeros_impl/ones_impl).
//! - `reuse_fill` (B, kernel-only): `c.fill(v)` into a pre-allocated,
//!   pre-warmed tensor — the tensor-level route into
//!   `fill_promote_cpu_serial/_rayon`. The tensor is captured by the closure
//!   and mutated in place; NO clone/copy per iteration.
//! - `fill_bound` (raw floor): plain slice broadcast loop into a pre-warmed
//!   `Vec` (write-only streaming floor; no layout machinery). Single-threaded
//!   only — the faer16 reuse rows themselves show parallel fill does not beat
//!   serial (write bandwidth saturates per-core, cf. T7 where faer16 fill
//!   0.535 ms ≈ serial 0.542 ms at 2048²).
//! - `strided_fill` (B_strided, kernel-only spot): device-level fill on
//!   non-contiguous layouts — the `else` (layout-iterator) branch of
//!   `fill_promote`: a diagonal layout (the `eye` creation path,
//!   `tensor/creation.rs:829`) and a stride-2-on-both-axes layout.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use fill_misc_structural::{assert_faer_threads, configure_group, faer_device, serial_device, FILL_SIZES, FILL_VALUE_F32, FILL_VALUE_F64};
use rstsr_core::operators::assignment::OpAssignAPI;
use rstsr_core::prelude::*;

fn bench_creation(c: &mut Criterion) {
    let dev_serial = serial_device();
    let dev_faer = faer_device();
    assert_faer_threads(&dev_faer, 16);

    let mut group = c.benchmark_group("creation");
    configure_group(&mut group);

    for &(label, m, n) in FILL_SIZES {
        for (dev_name, op) in [("serial", "full"), ("serial", "zeros"), ("serial", "ones"), ("faer16", "full"), ("faer16", "zeros"), ("faer16", "ones")] {
            let id = BenchmarkId::new(format!("{op}_{label}_{m}x{n}"), format!("{dev_name}_f64"));
            group.bench_function(id, |bench| match (dev_name, op) {
                ("serial", "full") => bench.iter(|| {
                    let t: Tensor<f64, _> = rt::full(([m, n], FILL_VALUE_F64, &dev_serial));
                    black_box(t)
                }),
                ("serial", "zeros") => bench.iter(|| {
                    let t: Tensor<f64, _> = rt::zeros(([m, n], &dev_serial));
                    black_box(t)
                }),
                ("serial", "ones") => bench.iter(|| {
                    let t: Tensor<f64, _> = rt::ones(([m, n], &dev_serial));
                    black_box(t)
                }),
                ("faer16", "full") => bench.iter(|| {
                    let t: Tensor<f64, _> = rt::full(([m, n], FILL_VALUE_F64, &dev_faer));
                    black_box(t)
                }),
                ("faer16", "zeros") => bench.iter(|| {
                    let t: Tensor<f64, _> = rt::zeros(([m, n], &dev_faer));
                    black_box(t)
                }),
                ("faer16", "ones") => bench.iter(|| {
                    let t: Tensor<f64, _> = rt::ones(([m, n], &dev_faer));
                    black_box(t)
                }),
                _ => unreachable!(),
            });
        }
    }

    // f32 spot (large only): 16 MiB output — below the glibc 32 MiB cap, so
    // A is expected to be kernel-faithful at this size (T4' finding).
    for dev_name in ["serial", "faer16"] {
        let id = BenchmarkId::new("full_large_2048x2048", format!("{dev_name}_f32"));
        group.bench_function(id, |bench| match dev_name {
            "serial" => bench.iter(|| {
                let t: Tensor<f32, _> = rt::full(([2048, 2048], FILL_VALUE_F32, &dev_serial));
                black_box(t)
            }),
            _ => bench.iter(|| {
                let t: Tensor<f32, _> = rt::full(([2048, 2048], FILL_VALUE_F32, &dev_faer));
                black_box(t)
            }),
        });
    }

    group.finish();
}

fn bench_reuse_fill(c: &mut Criterion) {
    let dev_serial = serial_device();
    let dev_faer = faer_device();
    assert_faer_threads(&dev_faer, 16);

    let mut group = c.benchmark_group("reuse_fill");
    configure_group(&mut group);

    for &(label, m, n) in FILL_SIZES {
        let mut cs: Tensor<f64, _> = rt::zeros(([m, n], &dev_serial));
        cs.fill(FILL_VALUE_F64); // pre-warm pages (zeros is calloc-lazy)
        let mut cf: Tensor<f64, _> = rt::zeros(([m, n], &dev_faer));
        cf.fill(FILL_VALUE_F64);

        let id = BenchmarkId::new(format!("fill_{label}_{m}x{n}"), "serial_f64");
        group.bench_function(id, |bench| {
            bench.iter(|| {
                let c = black_box(&mut cs);
                c.fill(FILL_VALUE_F64);
                black_box(())
            })
        });

        let id = BenchmarkId::new(format!("fill_{label}_{m}x{n}"), "faer16_f64");
        group.bench_function(id, |bench| {
            bench.iter(|| {
                let c = black_box(&mut cf);
                c.fill(FILL_VALUE_F64);
                black_box(())
            })
        });
    }

    // f32 spot (large only, kernel-only).
    let mut cs32: Tensor<f32, _> = rt::zeros(([2048, 2048], &dev_serial));
    cs32.fill(0.0f32);
    let id = BenchmarkId::new("fill_large_2048x2048", "serial_f32");
    group.bench_function(id, |bench| {
        bench.iter(|| {
            let c = black_box(&mut cs32);
            c.fill(FILL_VALUE_F32);
            black_box(())
        })
    });

    group.finish();
}

fn bench_fill_bound(c: &mut Criterion) {
    let mut group = c.benchmark_group("fill_bound");
    configure_group(&mut group);

    for &(label, m, n) in FILL_SIZES {
        let mut v = vec![FILL_VALUE_F64; m * n]; // alloc + warm pages immediately
        let id = BenchmarkId::new(format!("slice_{label}_{m}x{n}"), "serial_f64");
        group.bench_function(id, |bench| {
            bench.iter(|| {
                let v = black_box(&mut v);
                v.iter_mut().for_each(|x| *x = FILL_VALUE_F64);
                black_box(())
            })
        });
    }

    group.finish();
}

fn bench_strided_fill(c: &mut Criterion) {
    // Device-level fill on NON-contiguous layouts — exercises the `else`
    // (layout-iterator) branch of fill_promote. Kernel-only: buffers pre-warmed.
    let dev_serial = serial_device();
    let dev_faer = faer_device();
    assert_faer_threads(&dev_faer, 16);

    let mut group = c.benchmark_group("strided_fill");
    configure_group(&mut group);

    // (a) diagonal layout of a 2048x2048 matrix: shape [2048], stride [2049].
    //     This is exactly the layout `eye` fills (tensor/creation.rs:829).
    {
        let n = 2048usize;
        let mut vs = vec![0.0f64; n * n + 1]; // +1: the [2049] stride's last element
        vs.iter_mut().for_each(|x| *x = 0.0); // pre-warm pages
        let mut vf = vs.clone();
        let ls = Layout::new([n], [(n + 1) as isize], 0).unwrap();
        let lf = Layout::new([n], [(n + 1) as isize], 0).unwrap();

        let id = BenchmarkId::new("diag_2048x2048", "serial_f64");
        group.bench_function(id, |bench| {
            bench.iter(|| {
                dev_serial.fill(black_box(&mut vs), black_box(&ls), FILL_VALUE_F64).unwrap();
                black_box(())
            })
        });
        let id = BenchmarkId::new("diag_2048x2048", "faer16_f64");
        group.bench_function(id, |bench| {
            bench.iter(|| {
                dev_faer.fill(black_box(&mut vf), black_box(&lf), FILL_VALUE_F64).unwrap();
                black_box(())
            })
        });
    }

    // (b) stride-2 on both axes: shape [1024, 1024] over a [2048, 2048] buffer
    //     (strides [4096, 2] in elements).
    {
        let (m, n) = (1024usize, 1024usize);
        let mut vs = vec![0.0f64; 2048 * 2048];
        vs.iter_mut().for_each(|x| *x = 0.0);
        let mut vf = vs.clone();
        let ls = Layout::new([m, n], [4096isize, 2isize], 0).unwrap();
        let lf = Layout::new([m, n], [4096isize, 2isize], 0).unwrap();

        let id = BenchmarkId::new("stride2_1024x1024", "serial_f64");
        group.bench_function(id, |bench| {
            bench.iter(|| {
                dev_serial.fill(black_box(&mut vs), black_box(&ls), FILL_VALUE_F64).unwrap();
                black_box(())
            })
        });
        let id = BenchmarkId::new("stride2_1024x1024", "faer16_f64");
        group.bench_function(id, |bench| {
            bench.iter(|| {
                dev_faer.fill(black_box(&mut vf), black_box(&lf), FILL_VALUE_F64).unwrap();
                black_box(())
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_creation, bench_reuse_fill, bench_fill_bound, bench_strided_fill);
criterion_main!(benches);

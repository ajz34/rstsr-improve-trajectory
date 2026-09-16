// Criterion benches for the §4.5 size-cache question.
// Groups model the call frequencies actually found in rstsr at aa24643:
// every live size() call site is once-per-op (dispatch/allocation/validation);
// none is per-element or per-task. The frequency sweep quantifies at what
// (hypothetical) frequency caching would start to pay.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use size_cache_mwe::*;
use std::hint::black_box;

fn make_shape(n: usize) -> Vec<usize> {
    // product ≈ 1e6 for all ndim, realistic-ish dims
    match n {
        2 => vec![1000, 1000],
        4 => vec![32, 32, 32, 32],
        8 => vec![4, 4, 5, 5, 5, 5, 5, 10],
        _ => unreachable!(),
    }
}

fn make_shape_arr<const N: usize>() -> [usize; N] {
    let v = make_shape(N);
    let mut a = [0usize; N];
    a.copy_from_slice(&v);
    a
}

fn bench_size_call(c: &mut Criterion) {
    let mut g = c.benchmark_group("size_call_raw");
    for ndim in [2usize, 4, 8] {
        let shape = make_shape(ndim);
        let stride: Vec<isize> = shape
            .iter()
            .scan(1usize, |st, &d| {
                let s = *st as isize;
                *st *= d;
                Some(s)
            })
            .collect();
        let lr = LayoutVecR::new(shape.clone(), stride.clone(), 0);
        let lc = LayoutVecC::new(shape, stride, 0);
        g.bench_with_input(BenchmarkId::new("vec_recompute", ndim), &lr, |b, l| {
            b.iter(|| black_box(l.size()))
        });
        g.bench_with_input(BenchmarkId::new("vec_cached", ndim), &lc, |b, l| {
            b.iter(|| black_box(l.size()))
        });
    }
    // inline-array regime
    {
        let lr = LayoutArrR::<4>::new(make_shape_arr::<4>(), [1024, 32, 1024, 32].map(|_| 1), 0);
        let lc = LayoutArrC::<4>::new(make_shape_arr::<4>(), [1024, 32, 1024, 32].map(|_| 1), 0);
        g.bench_function("arr4_recompute", |b| b.iter(|| black_box(lr.size())));
        g.bench_function("arr4_cached", |b| b.iter(|| black_box(lc.size())));
    }
    g.finish();
}

/// Creation path, fairly paired: recompute pays the product at the (single)
/// size() call; cached pays it in the constructor. One product either way.
fn bench_construct(c: &mut Criterion) {
    let mut g = c.benchmark_group("construct_plus_one_size");
    for ndim in [2usize, 4, 8] {
        let shape = make_shape(ndim);
        let stride: Vec<isize> = shape
            .iter()
            .scan(1usize, |st, &d| {
                let s = *st as isize;
                *st *= d;
                Some(s)
            })
            .collect();
        g.bench_with_input(BenchmarkId::new("vec_recompute", ndim), &(shape.clone(), stride.clone()), |b, (s, st)| {
            b.iter(|| {
                let l = LayoutVecR::new(black_box(s.clone()), black_box(st.clone()), 0);
                black_box(l.size())
            })
        });
        g.bench_with_input(BenchmarkId::new("vec_cached", ndim), &(shape.clone(), stride.clone()), |b, (s, st)| {
            b.iter(|| {
                let l = LayoutVecC::new(black_box(s.clone()), black_box(st.clone()), 0);
                black_box(l.size())
            })
        });
    }
    g.finish();
}

fn bench_clone(c: &mut Criterion) {
    let mut g = c.benchmark_group("clone_layout");
    let shape = make_shape(4);
    let stride = vec![1isize; 4];
    let lr = LayoutVecR::new(shape.clone(), stride.clone(), 0);
    let lc = LayoutVecC::new(shape, stride, 0);
    g.bench_function("vec_recompute", |b| b.iter(|| black_box(lr.clone())));
    g.bench_function("vec_cached", |b| b.iter(|| black_box(lc.clone())));
    let ar = LayoutArrR::<4>::new(make_shape_arr::<4>(), [1; 4], 0);
    let ac = LayoutArrC::<4>::new(make_shape_arr::<4>(), [1; 4], 0);
    g.bench_function("arr4_recompute", |b| b.iter(|| black_box(ar.clone())));
    g.bench_function("arr4_cached", |b| b.iter(|| black_box(ac.clone())));
    g.finish();
}

/// IterAxesView::next analog: clone the step state (layout inside) per step.
fn bench_per_step_clone(c: &mut Criterion) {
    let mut g = c.benchmark_group("per_step_clone_iter1000");
    let shape = make_shape(4);
    let stride = vec![1isize; 4];
    let sr = AxisStep { layout: LayoutVecR::new(shape.clone(), stride.clone(), 0), index: [0; 4] };
    let sc = AxisStep { layout: LayoutVecC::new(shape, stride, 0), index: [0; 4] };
    g.bench_function("vec_recompute", |b| {
        b.iter(|| {
            let mut st = black_box(sr.clone());
            for _ in 0..1000 {
                st = st.clone().advanced();
            }
            black_box(st)
        })
    });
    g.bench_function("vec_cached", |b| {
        b.iter(|| {
            let mut st = black_box(sc.clone());
            for _ in 0..1000 {
                st = st.clone().advanced();
            }
            black_box(st)
        })
    });
    g.finish();
}

/// Realistic kernel: 1e6 f64 adds in 64 chunks, one size() per chunk
/// (matches every live size() call site in rstsr today).
fn bench_kernel_per_task(c: &mut Criterion) {
    let mut g = c.benchmark_group("kernel_per_task_64");
    let n = 1_000_000usize;
    let a = vec![1.0f64; n];
    let mut out = vec![0.0f64; n];
    let shape = make_shape(2);
    let stride = vec![1isize; 2];
    let lr = LayoutVecR::new(shape.clone(), stride.clone(), 0);
    let lc = LayoutVecC::new(shape, stride, 0);
    g.bench_function("vec_recompute", |b| {
        b.iter(|| {
            kernel_per_task(black_box(&lr), black_box(&a), black_box(&mut out), 64);
            black_box(out[0]) // keep the writes observable
        })
    });
    g.bench_function("vec_cached", |b| {
        b.iter(|| {
            kernel_per_task(black_box(&lc), black_box(&a), black_box(&mut out), 64);
            black_box(out[0]) // keep the writes observable
        })
    });
    g.finish();
}

/// Frequency sweep: size() every {256, 64, 1} elements while walking 1e6
/// elements. Shows the break-even slope; "every 1" is the pathological bound.
fn bench_freq_sweep(c: &mut Criterion) {
    let mut g = c.benchmark_group("walk_size_every");
    let n = 1_000_000usize;
    let a = vec![0.5f64; n];
    let lr = LayoutVecR::new(make_shape(4), vec![1isize; 4], 0);
    let lc = LayoutVecC::new(make_shape(4), vec![1isize; 4], 0);
    for every in [256usize, 64, 1] {
        g.bench_with_input(BenchmarkId::new("vec_recompute", every), &every, |b, &e| {
            b.iter(|| black_box(walk_with_size_freq(&lr, &a, e)))
        });
        g.bench_with_input(BenchmarkId::new("vec_cached", every), &every, |b, &e| {
            b.iter(|| black_box(walk_with_size_freq(&lc, &a, e)))
        });
    }
    g.finish();
}

criterion_group!(
    benches,
    bench_size_call,
    bench_construct,
    bench_clone,
    bench_per_step_clone,
    bench_kernel_per_task,
    bench_freq_sweep,
);
criterion_main!(benches);

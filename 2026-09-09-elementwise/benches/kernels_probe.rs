//! Kernel-structure probes: WHAT exactly blocks vectorization in the rstsr
//! elementwise contiguous branch, and what the strided branch leaves on the
//! table. Phase-1 evidence for PLAN.md — all probes are LOCAL loop shapes
//! (this crate), serial, f64, outputs pre-allocated (kernel-only timing).
//!
//! Contig probes (c = a + b, flat slices):
//! - `idx_generic_mu_closure` — current contig-branch shape: generic fn,
//!   `impl FnMut(&mut MaybeUninit<T>, &T, &T)` closure, index loop
//!   `for i in 0..n { f(&mut c[i], &a[i], &b[i]) }` (3 bounds checks/elem,
//!   3 index adds/elem). Compare with the main-bench add B number to judge
//!   whether cross-crate generic instantiation loses further codegen.
//! - `idx_scalar_bounds` — plain `for i { c[i] = a[i] + b[i] }`.
//! - `idx_unchecked` — get_unchecked: isolates the bounds-check cost.
//! - `zip_direct` — `iter_mut().zip(..).zip(..)` with inline body: the
//!   canonical auto-vectorization form.
//! - `zip_generic_mu_closure` — zip slices but keep the MaybeUninit closure
//!   interface: isolates the INTERFACE (not the bounds checks).
//! - `zip_dyn_mu_closure` — zip with `&mut dyn FnMut(...)`: isolates dyn
//!   dispatch.
//! - `chunks_exact8` — fixed-size-8 batching (plan §5 "plain fixed-size
//!   batching first").
//!
//! Strided probes (c[i,j] = a[i,j] + b[j,i], 2048x2048, row-major flats):
//! - `strided_bound_rowmajor` — T7's emulated bound shape (17.2 ms ref).
//! - `strided_colmajor_walk` — the current branch's effective walk (c/a
//!   stride-2048, b contiguous inner). Expect ~23 ms (rstsr B reference).
//! - `strided_blocked64` / `strided_blocked128` — 2-D tiled kernels
//!   (candidate phase-2 shape).

use std::hint::black_box;
use std::mem::MaybeUninit;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use elementwise::{configure_group, gen_vec_f64};

// ---------------------------------------------------------------------------
// contig probes
// ---------------------------------------------------------------------------

#[inline(never)]
fn probe_idx_generic_mu_closure<T>(c: &mut [MaybeUninit<T>], a: &[T], b: &[T], mut f: impl FnMut(&mut MaybeUninit<T>, &T, &T)) {
    let n = c.len();
    let (idx_c, idx_a, idx_b) = (0usize, 0usize, 0usize);
    for i in 0..n {
        f(&mut c[idx_c + i], &a[idx_a + i], &b[idx_b + i]);
    }
}

#[inline(never)]
fn probe_idx_scalar_bounds(c: &mut [f64], a: &[f64], b: &[f64]) {
    let n = c.len();
    for i in 0..n {
        c[i] = a[i] + b[i];
    }
}

#[inline(never)]
fn probe_idx_unchecked(c: &mut [f64], a: &[f64], b: &[f64]) {
    let n = c.len();
    unsafe {
        for i in 0..n {
            *c.get_unchecked_mut(i) = *a.get_unchecked(i) + *b.get_unchecked(i);
        }
    }
}

#[inline(never)]
fn probe_zip_direct(c: &mut [f64], a: &[f64], b: &[f64]) {
    for ((c_i, a_i), b_i) in c.iter_mut().zip(a.iter()).zip(b.iter()) {
        *c_i = a_i + b_i;
    }
}

#[inline(never)]
fn probe_zip_generic_mu_closure<T>(
    c: &mut [MaybeUninit<T>],
    a: &[T],
    b: &[T],
    f: &mut impl FnMut(&mut MaybeUninit<T>, &T, &T),
) {
    let n = c.len();
    for (c_i, (a_i, b_i)) in c[..n].iter_mut().zip(a[..n].iter().zip(b[..n].iter())) {
        f(c_i, a_i, b_i);
    }
}

#[inline(never)]
fn probe_zip_dyn_mu_closure(c: &mut [MaybeUninit<f64>], a: &[f64], b: &[f64], f: &mut dyn FnMut(&mut MaybeUninit<f64>, &f64, &f64)) {
    let n = c.len();
    for (c_i, (a_i, b_i)) in c[..n].iter_mut().zip(a[..n].iter().zip(b[..n].iter())) {
        f(c_i, a_i, b_i);
    }
}

#[inline(never)]
fn probe_chunks_exact8(c: &mut [f64], a: &[f64], b: &[f64]) {
    let n = c.len();
    let mut cs = c.chunks_exact_mut(8);
    let mut as_ = a[..n].chunks_exact(8);
    let mut bs = b[..n].chunks_exact(8);
    for ((cv, av), bv) in cs.by_ref().zip(as_.by_ref()).zip(bs.by_ref()) {
        for k in 0..8 {
            cv[k] = av[k] + bv[k];
        }
    }
    let r = cs.into_remainder();
    let ra = as_.remainder();
    let rb = bs.remainder();
    for ((cv, av), bv) in r.iter_mut().zip(ra.iter()).zip(rb.iter()) {
        *cv = av + bv;
    }
}

// ---------------------------------------------------------------------------
// strided probes (c[i*n+j] = a[i*n+j] + b[j*n+i], n x n row-major flats)
// ---------------------------------------------------------------------------

#[inline(never)]
fn probe_strided_bound_rowmajor(c: &mut [f64], a: &[f64], b: &[f64], n: usize) {
    for i in 0..n {
        for j in 0..n {
            c[i * n + j] = a[i * n + j] + b[j * n + i];
        }
    }
}

#[inline(never)]
fn probe_strided_colmajor_walk(c: &mut [f64], a: &[f64], b: &[f64], n: usize) {
    // what the current generic branch effectively does after greedy_layout:
    // iteration axis i fastest; c and a hop by n (16 KiB), b contiguous.
    for j in 0..n {
        for i in 0..n {
            c[i * n + j] = a[i * n + j] + b[j * n + i];
        }
    }
}

#[inline(never)]
fn probe_strided_blocked64(c: &mut [f64], a: &[f64], b: &[f64], n: usize) {
    const T: usize = 64;
    for j0 in (0..n).step_by(T) {
        for i0 in (0..n).step_by(T) {
            for i in i0..i0 + T {
                for j in j0..j0 + T {
                    c[i * n + j] = a[i * n + j] + b[j * n + i];
                }
            }
        }
    }
}

#[inline(never)]
fn probe_strided_blocked128(c: &mut [f64], a: &[f64], b: &[f64], n: usize) {
    const T: usize = 128;
    for j0 in (0..n).step_by(T) {
        for i0 in (0..n).step_by(T) {
            for i in i0..i0 + T {
                for j in j0..j0 + T {
                    c[i * n + j] = a[i * n + j] + b[j * n + i];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------

fn bench_probe(c: &mut Criterion) {
    let mut group = c.benchmark_group("probe_contig");
    configure_group(&mut group);

    for &(label, n) in [("medium", 512usize * 512usize), ("large", 2048usize * 2048usize)].iter() {
        let a = gen_vec_f64(n, 1);
        let b = gen_vec_f64(n, 2);
        let mut c = vec![0.0f64; n];
        let mut c_mu: Vec<MaybeUninit<f64>> = (0..n).map(|_| MaybeUninit::new(0.0)).collect();

        let id = |v: &str| BenchmarkId::new(format!("{label}_{n}"), v);

        group.bench_function(id("idx_generic_mu_closure"), |bench| {
            bench.iter(|| {
                probe_idx_generic_mu_closure(black_box(&mut c_mu), black_box(&a), black_box(&b), |cv, x, y| {
                    cv.write(x + y);
                });
                black_box(&c_mu);
            })
        });
        group.bench_function(id("idx_scalar_bounds"), |bench| {
            bench.iter(|| {
                probe_idx_scalar_bounds(black_box(&mut c), black_box(&a), black_box(&b));
                black_box(&c);
            })
        });
        group.bench_function(id("idx_unchecked"), |bench| {
            bench.iter(|| {
                probe_idx_unchecked(black_box(&mut c), black_box(&a), black_box(&b));
                black_box(&c);
            })
        });
        group.bench_function(id("zip_direct"), |bench| {
            bench.iter(|| {
                probe_zip_direct(black_box(&mut c), black_box(&a), black_box(&b));
                black_box(&c);
            })
        });
        group.bench_function(id("zip_generic_mu_closure"), |bench| {
            bench.iter(|| {
                let mut f = |cv: &mut MaybeUninit<f64>, x: &f64, y: &f64| { cv.write(x + y); };
                probe_zip_generic_mu_closure(black_box(&mut c_mu), black_box(&a), black_box(&b), &mut f);
                black_box(&c_mu);
            })
        });
        group.bench_function(id("zip_dyn_mu_closure"), |bench| {
            bench.iter(|| {
                let mut f = |cv: &mut MaybeUninit<f64>, x: &f64, y: &f64| { cv.write(x + y); };
                probe_zip_dyn_mu_closure(black_box(&mut c_mu), black_box(&a), black_box(&b), &mut f);
                black_box(&c_mu);
            })
        });
        group.bench_function(id("chunks_exact8"), |bench| {
            bench.iter(|| {
                probe_chunks_exact8(black_box(&mut c), black_box(&a), black_box(&b));
                black_box(&c);
            })
        });
    }
    group.finish();

    // strided probes (large square only)
    let n = 2048usize;
    let a = gen_vec_f64(n * n, 1);
    let b = gen_vec_f64(n * n, 4);
    let mut cbuf = vec![0.0f64; n * n];

    let mut group = c.benchmark_group("probe_strided");
    configure_group(&mut group);
    group.bench_function("large_2048x2048-bound_rowmajor", |bench| {
        bench.iter(|| {
            probe_strided_bound_rowmajor(black_box(&mut cbuf), black_box(&a), black_box(&b), n);
            black_box(&cbuf);
        })
    });
    group.bench_function("large_2048x2048-colmajor_walk", |bench| {
        bench.iter(|| {
            probe_strided_colmajor_walk(black_box(&mut cbuf), black_box(&a), black_box(&b), n);
            black_box(&cbuf);
        })
    });
    group.bench_function("large_2048x2048-blocked64", |bench| {
        bench.iter(|| {
            probe_strided_blocked64(black_box(&mut cbuf), black_box(&a), black_box(&b), n);
            black_box(&cbuf);
        })
    });
    group.bench_function("large_2048x2048-blocked128", |bench| {
        bench.iter(|| {
            probe_strided_blocked128(black_box(&mut cbuf), black_box(&a), black_box(&b), n);
            black_box(&cbuf);
        })
    });
    group.finish();
}

criterion_group!(benches, bench_probe);
criterion_main!(benches);

//! Candidate-kernel probes on plain `Vec<f64>` (no rstsr), serial core.
//!
//! These anchor the design options for the PLAN before touching rstsr:
//! what would each restructure of the three vecdot branches + inner_dot buy
//! under both target configs? Numbers feed PLAN.md §"probe anchors".
//!
//! Probes (f64 only; Complex assessed by compiling a c64 variant of the best
//! shape in the same binary):
//! - am1 (row-dot, branch-1 shape, (4096,512)):
//!     A0 replica of the current branch-1 body (unrolled_binary_reduce per
//!       row through closures, copied from cpu_serial/reduction.rs:96-140);
//!     A1 plain per-row zip fold `s += x*y` (scalar);
//!     A2 per-row explicit 8-lane accumulators ([f64;8], no closures);
//!     A3 A2 with two rows interleaved (ILP across rows);
//!     A5 complex c64 plain-fold (the D5 secondary evidence).
//! - axis0 (column-accumulation, branch-2 RMW shape, (512,4096)):
//!     B0 replica of the current RMW body (CHUNK=64 bands, per-element
//!       write(read+val), per row re-walk);
//!     B1 full output vacc (32 KiB, L1) + row walks `v[j] += x*y`;
//!     B2 B1 with 8-lane split of j.
//! - strided (general branch shape, b stride = m):
//!     C0 replica of the stride-fold (idx_m + i*step arithmetic per element);
//!     C1 unrolled-8 accumulator variant of the same walks.
//! - inner_dot:
//!     D0 replica of inner_dot_naive_cpu_serial (index_uncheck arithmetic
//!       + alpha.clone() per element);
//!     D1 zip fold;
//!     D2 8-lane unrolled.
//!
//! Timing: warmup pass + best-of-3 fixed-iteration blocks; prints ms and
//! derived GB/s + cycles/elem at nominal 5.7 GHz boost (approx label only).

use std::hint::black_box;
use std::time::Instant;

use num::Complex;

// ---------------------------------------------------------------------------
// replica of unrolled_binary_reduce specialized to f64 add/mul (as the
// compiler sees it after monomorphization + inlining of |acc,(x,y)| acc + x*y)
// ---------------------------------------------------------------------------
#[inline(never)]
fn ubr_f64(xs1: &[f64], xs2: &[f64]) -> f64 {
    let mut acc = 0.0;
    let (mut p0, mut p1, mut p2, mut p3, mut p4, mut p5, mut p6, mut p7) =
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let mut xs1 = xs1;
    let mut xs2 = xs2;
    while xs1.len() >= 8 && xs2.len() >= 8 {
        p0 = p0 + xs1[0] * xs2[0];
        p1 = p1 + xs1[1] * xs2[1];
        p2 = p2 + xs1[2] * xs2[2];
        p3 = p3 + xs1[3] * xs2[3];
        p4 = p4 + xs1[4] * xs2[4];
        p5 = p5 + xs1[5] * xs2[5];
        p6 = p6 + xs1[6] * xs2[6];
        p7 = p7 + xs1[7] * xs2[7];
        xs1 = &xs1[8..];
        xs2 = &xs2[8..];
    }
    acc = acc + (p0 + p4);
    acc = acc + (p1 + p5);
    acc = acc + (p2 + p6);
    acc = acc + (p3 + p7);
    for i in 0..xs1.len().min(7) {
        acc = acc + xs1[i] * xs2[i];
    }
    acc
}

fn bench<F: FnMut() -> f64>(name: &str, flops: f64, bytes: f64, mut f: F) -> f64 {
    let _ = f(); // warmup + correctness-of-execution
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let iters = 40;
        let t0 = Instant::now();
        for _ in 0..iters {
            black_box(f());
        }
        let dt = t0.elapsed().as_secs_f64() / iters as f64;
        if dt < best {
            best = dt;
        }
    }
    let gbps = bytes / best / 1e9;
    let gflops = flops / best / 1e9;
    let ms = best * 1e3;
    println!("{name:52} {ms:10.3} ms  {gbps:7.1} GB/s  {gflops:7.2} GFLOP/s");
    best
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let group = args.get(1).map(|s| s.as_str()).unwrap_or("all");
    let (m, k) = (4096usize, 512usize);

    if group == "all" || group == "am1" {
        println!("=== am1 row-dot (branch-1 shape)  a({m},{k}).b({m},{k}) -> ({m},) ===");
        let a: Vec<f64> = (0..m * k).map(|i| (i % 2003) as f64 / 2003.0 - 0.5).collect();
        let b: Vec<f64> = (0..m * k).map(|i| (i % 2009) as f64 / 2009.0 - 0.5).collect();
        let mut c = vec![0.0_f64; m];
        let flops = (m * k * 2) as f64;
        let bytes = (2 * m * k * 8) as f64;

        // A0: current branch-1 body shape (closure ubr per row)
        bench("A0 branch1-replica (ubr per row)", flops, bytes, || {
            for i in 0..m {
                c[i] = ubr_f64(&a[i * k..(i + 1) * k], &b[i * k..(i + 1) * k]);
            }
            c[m - 1]
        });

        // A1: plain per-row fold
        bench("A1 per-row plain fold s+=x*y", flops, bytes, || {
            let mut s = 0.0;
            for i in 0..m {
                let (ra, rb) = (&a[i * k..(i + 1) * k], &b[i * k..(i + 1) * k]);
                let mut acc = 0.0;
                for j in 0..k {
                    acc += ra[j] * rb[j];
                }
                c[i] = acc;
                s += acc;
            }
            s
        });

        // A2: per-row 8-lane accumulators
        bench("A2 per-row 8-lane [f64;8]", flops, bytes, || {
            let mut s = 0.0;
            for i in 0..m {
                let (ra, rb) = (&a[i * k..(i + 1) * k], &b[i * k..(i + 1) * k]);
                let mut acc = [0.0_f64; 8];
                let n8 = k / 8 * 8;
                for j in (0..n8).step_by(8) {
                    for l in 0..8 {
                        acc[l] += ra[j + l] * rb[j + l];
                    }
                }
                let mut accv = 0.0;
                for l in 0..8 {
                    accv += acc[l];
                }
                for j in n8..k {
                    accv += ra[j] * rb[j];
                }
                c[i] = accv;
                s += accv;
            }
            s
        });

        // A3: two rows interleaved (ILP across rows)
        bench("A3 two-rows interleaved 8-lane", flops, bytes, || {
            let mut s = 0.0;
            for i in (0..m / 2 * 2).step_by(2) {
                let (ra0, rb0) = (&a[i * k..(i + 1) * k], &b[i * k..(i + 1) * k]);
                let (ra1, rb1) = (&a[(i + 1) * k..(i + 2) * k], &b[(i + 1) * k..(i + 2) * k]);
                let mut acc0 = [0.0_f64; 8];
                let mut acc1 = [0.0_f64; 8];
                let n8 = k / 8 * 8;
                for j in (0..n8).step_by(8) {
                    for l in 0..8 {
                        acc0[l] += ra0[j + l] * rb0[j + l];
                        acc1[l] += ra1[j + l] * rb1[j + l];
                    }
                }
                let (mut v0, mut v1) = (0.0, 0.0);
                for l in 0..8 {
                    v0 += acc0[l];
                    v1 += acc1[l];
                }
                c[i] = v0;
                c[i + 1] = v1;
                s += v0 + v1;
            }
            s
        });

        // A2-small: same 8-lane shape at (8192,256) to see short-row effects
        {
            let (m2, k2) = (8192usize, 256usize);
            let a2: Vec<f64> = (0..m2 * k2).map(|i| (i % 2003) as f64 / 2003.0 - 0.5).collect();
            let b2: Vec<f64> = (0..m2 * k2).map(|i| (i % 2009) as f64 / 2009.0 - 0.5).collect();
            let flops2 = (m2 * k2 * 2) as f64;
            let bytes2 = (2 * m2 * k2 * 8) as f64;
            bench("A2 @ (8192,256) 8-lane", flops2, bytes2, || {
                let mut s = 0.0;
                for i in 0..m2 {
                    let (ra, rb) = (&a2[i * k2..(i + 1) * k2], &b2[i * k2..(i + 1) * k2]);
                    let mut acc = [0.0_f64; 8];
                    let n8 = k2 / 8 * 8;
                    for j in (0..n8).step_by(8) {
                        for l in 0..8 {
                            acc[l] += ra[j + l] * rb[j + l];
                        }
                    }
                    let mut accv = 0.0;
                    for l in 0..8 {
                        accv += acc[l];
                    }
                    for j in n8..k2 {
                        accv += ra[j] * rb[j];
                    }
                    s += accv;
                }
                s
            });
        }

        // A5: complex plain fold (D5 evidence) at (4096,512)
        {
            let ac: Vec<Complex<f64>> =
                (0..m * k).map(|i| Complex::new((i % 2003) as f64 / 2003.0 - 0.5, 0.1)).collect();
            let bc: Vec<Complex<f64>> =
                (0..m * k).map(|i| Complex::new((i % 2009) as f64 / 2009.0 - 0.5, 0.2)).collect();
            let mut cc = vec![Complex::new(0.0, 0.0); m];
            // c64: 4 mul + 2 add per element ~ 6 FLOP-equivalents, 32 B/elem
            let flopsc = (m * k * 6) as f64;
            let bytesc = (2 * m * k * 16) as f64;
            bench("A5 c64 per-row plain fold (conj(a)*b)", flopsc, bytesc, || {
                let mut s = Complex::new(0.0, 0.0);
                for i in 0..m {
                    let (ra, rb) = (&ac[i * k..(i + 1) * k], &bc[i * k..(i + 1) * k]);
                    let mut acc = Complex::new(0.0, 0.0);
                    for j in 0..k {
                        acc += ra[j].conj() * rb[j];
                    }
                    cc[i] = acc;
                    s += acc;
                }
                s.re
            });

            // A6: c64 per-row 8-lane accumulators (chain-breaking; informs the
            // D6 complex-in-dispatch_simd decision)
            bench("A6 c64 per-row 8-lane", flopsc, bytesc, || {
                let mut s = Complex::new(0.0, 0.0);
                for i in 0..m {
                    let (ra, rb) = (&ac[i * k..(i + 1) * k], &bc[i * k..(i + 1) * k]);
                    let mut acc = [Complex::new(0.0, 0.0); 8];
                    let n8 = k / 8 * 8;
                    for j in (0..n8).step_by(8) {
                        for l in 0..8 {
                            acc[l] += ra[j + l].conj() * rb[j + l];
                        }
                    }
                    let mut accv = Complex::new(0.0, 0.0);
                    for l in 0..8 {
                        accv += acc[l];
                    }
                    for j in n8..k {
                        accv += ra[j].conj() * rb[j];
                    }
                    cc[i] = accv;
                    s += accv;
                }
                s.re
            });
        }
    }

    if group == "all" || group == "axis0" {
        println!("=== axis0 column-accumulation (branch-2 RMW shape)  a({m},{k}).b({m},{k}) ax0 -> ({m},) ===");
        // NOTE: bench geometry batched_axis0 is (512,4096) contracting axis 0;
        // here a is stored (k=512 rows, m=4096 cols) row-major.
        let (kk, mm) = (512usize, 4096usize);
        let a: Vec<f64> = (0..kk * mm).map(|i| (i % 2003) as f64 / 2003.0 - 0.5).collect();
        let b: Vec<f64> = (0..kk * mm).map(|i| (i % 2009) as f64 / 2009.0 - 0.5).collect();
        let flops = (kk * mm * 2) as f64;
        let bytes = (2 * kk * mm * 8) as f64;

        // B0: current branch-2 RMW body shape (CHUNK=64, MaybeUninit-style
        // write(read+val) per element per row — modeled on uninit buffer)
        bench("B0 branch2-replica (RMW per elem per row)", flops, bytes, || {
            let mut c = vec![0.0_f64; mm]; // stands in for the uninit buffer
            const CHUNK: usize = 64;
            for ichunk in 0..(mm + CHUNK - 1) / CHUNK {
                let j0 = ichunk * CHUNK;
                let j1 = (j0 + CHUNK).min(mm);
                for i in 0..kk {
                    for j in j0..j1 {
                        c[j] += a[i * mm + j] * b[i * mm + j];
                    }
                }
            }
            black_box(&c);
            c[mm - 1]
        });

        // B1: full vacc + row walks
        bench("B1 full vacc (32KiB L1) row walks", flops, bytes, || {
            let mut c = vec![0.0_f64; mm];
            for i in 0..kk {
                let (ra, rb) = (&a[i * mm..(i + 1) * mm], &b[i * mm..(i + 1) * mm]);
                for j in 0..mm {
                    c[j] += ra[j] * rb[j];
                }
            }
            black_box(&c);
            c[mm - 1]
        });

        // B2: full vacc + 8-lane j split
        bench("B2 full vacc 8-lane j split", flops, bytes, || {
            let mut c = vec![0.0_f64; mm];
            let n8 = mm / 8 * 8;
            for i in 0..kk {
                let (ra, rb) = (&a[i * mm..(i + 1) * mm], &b[i * mm..(i + 1) * mm]);
                for j in (0..n8).step_by(8) {
                    for l in 0..8 {
                        c[j + l] += ra[j + l] * rb[j + l];
                    }
                }
                for j in n8..mm {
                    c[j] += ra[j] * rb[j];
                }
            }
            black_box(&c);
            c[mm - 1]
        });
    }

    if group == "all" || group == "strided" {
        println!("=== strided general-branch shape  a({m},{k}) contig . b({k},{m})-view -> ({m},) ===");
        let a: Vec<f64> = (0..m * k).map(|i| (i % 2003) as f64 / 2003.0 - 0.5).collect();
        let b: Vec<f64> = (0..m * k).map(|i| (i % 2009) as f64 / 2009.0 - 0.5).collect(); // stored (k,m)
        let flops = (m * k * 2) as f64;
        let bytes = (2 * m * k * 8) as f64;

        // C0: current general-branch shape (per-element index arithmetic)
        bench("C0 general-replica (idx + i*step per elem)", flops, bytes, || {
            let mut s = 0.0;
            for i in 0..m {
                let mut acc = 0.0;
                let (base_a, base_b) = (i * k, i); // b row i is column i of (k,m) -> stride m
                for j in 0..k {
                    acc += a[base_a + j] * b[base_b + j * m];
                }
                s += acc;
            }
            s
        });

        // C1: 8-lane accumulator variant
        bench("C1 8-lane stride fold", flops, bytes, || {
            let mut s = 0.0;
            for i in 0..m {
                let mut acc = [0.0_f64; 8];
                let n8 = k / 8 * 8;
                let (base_a, base_b) = (i * k, i);
                for j in (0..n8).step_by(8) {
                    for l in 0..8 {
                        acc[l] += a[base_a + j + l] * b[base_b + (j + l) * m];
                    }
                }
                let mut accv = 0.0;
                for l in 0..8 {
                    accv += acc[l];
                }
                for j in n8..k {
                    accv += a[base_a + j] * b[base_b + j * m];
                }
                s += accv;
            }
            s
        });
    }

    if group == "all" || group == "innerdot" {
        println!("=== inner_dot 1-D ===");
        for (label, n) in [("n=4096", 4096usize), ("n=1e7", 10_000_000usize)] {
            let a: Vec<f64> = (0..n).map(|i| (i % 2003) as f64 / 2003.0 - 0.5).collect();
            let b: Vec<f64> = (0..n).map(|i| (i % 2009) as f64 / 2009.0 - 0.5).collect();
            let flops = (n * 2) as f64;
            let bytes = (2 * n * 8) as f64;
            println!("-- {label} --");

            // D0: inner_dot_naive_cpu_serial replica (index arithmetic +
            // alpha* per element as in the source)
            bench("D0 serial-kernel replica (idx_uncheck + alpha*)", flops, bytes, || {
                let alpha = 1.0;
                let mut sum = 0.0;
                for i in 0..n {
                    sum = sum + alpha * (a[i] * b[i]);
                }
                sum
            });

            // D1: zip fold
            bench("D1 zip fold s+=x*y", flops, bytes, || {
                let mut s = 0.0;
                for i in 0..n {
                    s += a[i] * b[i];
                }
                s
            });

            // D2: 8-lane unrolled
            bench("D2 8-lane unrolled", flops, bytes, || {
                let mut acc = [0.0_f64; 8];
                let n8 = n / 8 * 8;
                for i in (0..n8).step_by(8) {
                    for l in 0..8 {
                        acc[l] += a[i + l] * b[i + l];
                    }
                }
                let mut s = 0.0;
                for l in 0..8 {
                    s += acc[l];
                }
                for i in n8..n {
                    s += a[i] * b[i];
                }
                s
            });
        }
    }

    println!("\n(nominal cycles/elem labels assume ~5.7 GHz; treat GB/s as primary)");
}

//! Review probes for the rstsr-cuda matmul chain (`261021/cuda` @ `67a0c79`).
//!
//! Nothing asserts: every test prints `ok` / `FAIL` lines so one run shows every cell.
//! T1 complex operands with imaginary parts, every driver alone; T2 the symmetric form
//! on shapes the crate's tests do not reach; T3 the symmetric form and the compute mode;
//! T4 degenerate inner dots; T5 cross-stream in-place matmul; T6 empty and `k = 0`
//! products; T7 the per-call-mode entries; T8 mixed operands through the mode entries.

use num::Complex;
use rstsr::prelude::*;
use rstsr_cuda::dtype::CudaDType;
use rstsr_cuda::matmul::{ComputeType32F, ComputeType64F, MatMulDrivers, MatMulUseSyrk, TensorCudaMatMulAPI};
use rstsr_cuda::prelude_dev::*;
use rstsr_cuda::stream::CudaStreamObj;
use std::sync::Arc;
use std::time::Instant;

type Cpu<T> = Tensor<T, DeviceCpuSerial, IxD>;
type Dev<T> = Tensor<T, DeviceCuda, IxD>;

trait Scalar:
    CudaDType
    + CudaMatMulMode
    + Copy
    + Clone
    + Send
    + Sync
    + 'static
    + num::Zero
    + num::One
    + std::ops::Mul<Output = Self>
    + std::ops::Sub<Output = Self>
{
    fn mk(re: f64, im: f64) -> Self;
    fn re(self) -> f64;
    fn im(self) -> f64;
}

macro_rules! scalar_complex {
    ($t:ty, $f:ty) => {
        impl Scalar for $t {
            fn mk(re: f64, im: f64) -> Self {
                Complex::new(re as $f, im as $f)
            }
            fn re(self) -> f64 {
                self.re as f64
            }
            fn im(self) -> f64 {
                self.im as f64
            }
        }
    };
}
scalar_complex!(Complex<f32>, f32);
scalar_complex!(Complex<f64>, f64);

macro_rules! scalar_real {
    ($t:ty, $c:expr) => {
        impl Scalar for $t {
            fn mk(re: f64, _: f64) -> Self {
                re as $t
            }
            fn re(self) -> f64 {
                self as f64
            }
            fn im(self) -> f64 {
                0.0
            }
        }
    };
}
scalar_real!(f32, true);
scalar_real!(f64, false);

fn data(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect()
}

fn rnd<S: Scalar>(n: usize, seed: u64) -> Vec<S> {
    data(n, seed).into_iter().zip(data(n, seed.wrapping_mul(7).wrapping_add(3))).map(|(x, y)| S::mk(x, y)).collect()
}

fn constv<S: Scalar>(n: usize, re: f64, im: f64) -> Vec<S> {
    vec![S::mk(re, im); n]
}

fn close<S: Scalar>(a: S, b: S, tol: f64) -> bool {
    (a.re() - b.re()).abs() + (a.im() - b.im()).abs() <= tol * (1.0 + b.re().abs() + b.im().abs())
}

/// Number of mismatching elements (`usize::MAX` when the lengths differ).
fn count_bad<S: Scalar>(got: &[S], want: &[S], tol: f64) -> usize {
    if got.len() != want.len() {
        return usize::MAX;
    }
    got.iter().zip(want).filter(|(g, w)| !close(**g, **w, tol)).count()
}

/// A fresh output (offset 0) back on the host, in row-major logical order.
fn read<S: CudaDType + Copy>(t: &Dev<S>) -> Vec<S> {
    DeviceToCpuAPI::<S>::to_cpu_vec(t.to_contig(RowMajor).storage()).rstsr_unwrap()
}

fn read_cpu<S: CudaDType + Copy>(t: &Cpu<S>) -> Vec<S> {
    DeviceToCpuAPI::<S>::to_cpu_vec(t.to_contig(RowMajor).storage()).rstsr_unwrap()
}

/// Any device tensor (a view, an offset) back on the host, in row-major logical order.
fn read_owned<S: CudaDType + Copy>(t: &Dev<S>) -> Vec<S> {
    let fresh = t.view().to_contig(RowMajor).to_owned();
    read(&fresh)
}

fn positions<S: CudaDType + Copy>(cpu: &Cpu<S>, order: FlagOrder) -> Vec<S> {
    match order {
        FlagOrder::RowMajor => DeviceToCpuAPI::<S>::to_cpu_vec(cpu.to_contig(RowMajor).storage()).rstsr_unwrap(),
        FlagOrder::ColMajor => DeviceToCpuAPI::<S>::to_cpu_vec(cpu.to_contig(ColMajor).storage()).rstsr_unwrap(),
        _ => unreachable!(),
    }
}

/// The logical values of `cpu` on the device, laid out in the device's default `order`.
fn to_dev<S: CudaDType + Copy>(cpu: &Cpu<S>, order: FlagOrder, d: &DeviceCuda) -> Dev<S> {
    rt::asarray((positions(cpu, order), d)).into_shape(cpu.shape().to_vec())
}

fn dev(order: FlagOrder, drivers: MatMulDrivers, edit: impl FnOnce(&mut CudaRuntimeFlags)) -> DeviceCuda {
    let mut flags = CudaRuntimeFlags::default();
    flags.matmul_drivers = drivers;
    edit(&mut flags);
    let mut d = DeviceCuda::new(0).with_flags(flags);
    d.set_default_order(order);
    d
}

fn drivers() -> Vec<MatMulDrivers> {
    use MatMulDrivers as M;
    vec![M::CUBLAS_LT, M::CUBLAS_EX, M::CUBLAS, M::TILED, M::NAIVE, M::all()]
}

/* #region T1: complex operands with imaginary parts, each driver alone */

struct Bases<S, B: DeviceRawAPI<S>> {
    a: Tensor<S, B, IxD>,
    at: Tensor<S, B, IxD>,
    bt: Tensor<S, B, IxD>,
    b: Tensor<S, B, IxD>,
    ab: Tensor<S, B, IxD>,
    bb: Tensor<S, B, IxD>,
    abig: Tensor<S, B, IxD>,
    abig2: Tensor<S, B, IxD>,
}

macro_rules! cases {
    ($b:expr, $s:expr, $s2:expr) => {{
        let b = &$b;
        vec![
            ("plain", b.a.view(), b.bt.t()),
            ("a stored as its transpose, used as a.t()", b.at.t(), b.bt.t()),
            ("b in row-contiguous storage", b.a.view(), b.b.view()),
            ("batched", b.ab.view(), b.bb.view()),
            ("b broadcast over the batch", b.ab.view(), b.bt.t()),
            ("a strided view (offset, lda)", $s.view(), b.bt.t()),
            ("a.t() of a strided view (offset, lda)", $s2.t(), b.bt.t()),
        ]
    }};
}

fn t1<S: Scalar>(what: &str, tol: f64) -> usize {
    let (m, k, n, p) = (12usize, 32usize, 8usize, 3usize);
    let mut bad = 0;
    for order in [RowMajor, ColMajor] {
        let mut dcpu = DeviceCpuSerial::default();
        dcpu.set_default_order(order);
        let (sab, sbb) = match order {
            RowMajor => (vec![p, m, k], vec![p, k, n]),
            ColMajor => (vec![m, k, p], vec![k, n, p]),
            _ => unreachable!(),
        };
        let cpu = Bases {
            a: rt::asarray((rnd::<S>(m * k, 1), &dcpu)).into_shape([m, k]),
            at: rt::asarray((rnd::<S>(m * k, 7), &dcpu)).into_shape([k, m]),
            bt: rt::asarray((rnd::<S>(n * k, 2), &dcpu)).into_shape([n, k]),
            b: rt::asarray((rnd::<S>(k * n, 3), &dcpu)).into_shape([k, n]),
            ab: rt::asarray((rnd::<S>(p * m * k, 4), &dcpu)).into_shape(sab),
            bb: rt::asarray((rnd::<S>(p * k * n, 5), &dcpu)).into_shape(sbb),
            abig: rt::asarray((rnd::<S>((m + 2) * (k + 3), 6), &dcpu)).into_shape([m + 2, k + 3]),
            abig2: rt::asarray((rnd::<S>((k + 2) * (m + 3), 8), &dcpu)).into_shape([k + 2, m + 3]),
        };
        let s_cpu = cpu.abig.i(s![1..m + 1, 2..k + 2]);
        let s2_cpu = cpu.abig2.i(s![1..k + 1, 2..m + 2]);
        let cpu_cases = cases!(cpu, s_cpu, s2_cpu);
        for drivers in drivers() {
            let d = dev(order, drivers, |_| {});
            let g = Bases {
                a: to_dev(&cpu.a, order, &d),
                at: to_dev(&cpu.at, order, &d),
                bt: to_dev(&cpu.bt, order, &d),
                b: to_dev(&cpu.b, order, &d),
                ab: to_dev(&cpu.ab, order, &d),
                bb: to_dev(&cpu.bb, order, &d),
                abig: to_dev(&cpu.abig, order, &d),
                abig2: to_dev(&cpu.abig2, order, &d),
            };
            let s_dev = g.abig.i(s![1..m + 1, 2..k + 2]);
            let s2_dev = g.abig2.i(s![1..k + 1, 2..m + 2]);
            let dev_cases = cases!(g, s_dev, s2_dev);
            for ((name, ca, cb), (ga, gb)) in cpu_cases.iter().zip(dev_cases.iter().map(|(_, x, y)| (x, y))) {
                let want = read_cpu(&rt::matmul(ca, cb));
                match rt::matmul_f(ga, gb) {
                    Ok(got) => {
                        let nb = count_bad(&read(&got), &want, tol);
                        if nb == 0 {
                            println!("ok   T1 {what} {order:?} {drivers:?} {name}");
                        } else {
                            bad += 1;
                            println!("FAIL T1 {what} {order:?} {drivers:?} {name}: {nb} of {} wrong", want.len());
                        }
                    },
                    Err(e) => {
                        bad += 1;
                        println!("ERR  T1 {what} {order:?} {drivers:?} {name}: {e:?}");
                    },
                }
            }
        }
    }
    bad
}

#[test]
fn t1_complex_imaginary_parts() {
    let mut bad = 0;
    bad += t1::<Complex<f32>>("c32", 3e-5);
    bad += t1::<Complex<f64>>("c64", 1e-12);
    bad += t1::<f32>("f32", 1e-5);
    bad += t1::<f64>("f64", 1e-12);
    println!("T1 bad cells: {bad}");
}

/* #endregion */

/* #region T2: the symmetric form on shapes the crate's tests do not reach */

fn t2<S: Scalar>(what: &str, tol: f64) -> usize {
    let (n, k) = (37usize, 19usize);
    let dcpu = DeviceCpuSerial::default();
    let cpu_abig: Cpu<S> = rt::asarray((rnd::<S>((n + 2) * (k + 3), 11), &dcpu)).into_shape([n + 2, k + 3]);
    let cpu_zero: Cpu<S> = rt::asarray((constv::<S>(4 * n * n, 0.0, 0.0), &dcpu)).into_shape([2 * n, 2 * n]);
    let mut bad = 0;
    for order in [RowMajor, ColMajor] {
        for drivers in [MatMulDrivers::CUBLAS, MatMulDrivers::all()] {
            let tag = format!("{what} {order:?} {drivers:?}");
            let d = dev(order, drivers, |f| {
                f.trace_matmul = true;
                f.matmul_use_syrk = MatMulUseSyrk::Allow;
            });
            let abig = to_dev(&cpu_abig, order, &d);
            let (cpu_abig_t, abig_t) = (cpu_abig.t(), abig.t());
            let (ca, ca2) = (cpu_abig.i(s![1..n + 1, 2..k + 2]), cpu_abig_t.i(s![2..k + 2, 1..n + 1]));
            let (a, a2) = (abig.i(s![1..n + 1, 2..k + 2]), abig_t.i(s![2..k + 2, 1..n + 1]));

            // fresh products through the public entry: four operand orientations
            for (name, cv, gv) in [("a strided view", ca.view(), a.view()), ("a other orientation", ca2.view(), a2.view())] {
                let want = read_cpu(&rt::matmul(&cv, cv.t()));
                let got = read_owned(&rt::matmul(&gv, gv.t()));
                report("T2", &tag, name, count_bad(&got, &want, tol), &mut bad);
            }
            // (the two orientations swap with the device order; repeat with the roles exchanged)
            let (want, got) = (read_cpu(&rt::matmul(&ca, ca.t())), read_owned(&rt::matmul(&a, a.t())));
            report("T2", &tag, "a of the device order's own orientation", count_bad(&got, &want, tol), &mut bad);

            // k = 0: the product is the zero matrix
            let (c0, g0) = (cpu_abig.i(s![1..n + 1, 2..2]), abig.i(s![1..n + 1, 2..2]));
            let want = read_cpu(&rt::matmul(&c0, c0.t()));
            match rt::matmul_f(&g0, g0.t()) {
                Ok(got) => {
                    let got = read(&got);
                    let nonzero = got.iter().filter(|x| x.re() != 0.0 || x.im() != 0.0).count();
                    report("T2", &tag, "k = 0 gives zeros", nonzero, &mut bad);
                    let _ = want;
                },
                Err(e) => report("T2", &tag, &format!("k = 0: error {e:?}"), 1, &mut bad),
            }
            // n = 1 and k = 1
            for (name, cs, gs) in [
                ("n = 1", cpu_abig.i(s![1..2, 2..k + 2]), abig.i(s![1..2, 2..k + 2])),
                ("k = 1", cpu_abig.i(s![1..n + 1, 2..3]), abig.i(s![1..n + 1, 2..3])),
            ] {
                let want = read_cpu(&rt::matmul(&cs, cs.t()));
                let got = read_owned(&rt::matmul(&gs, gs.t()));
                report("T2", &tag, name, count_bad(&got, &want, tol), &mut bad);
            }
            // alpha = 0 with beta = 0 into a caller's buffer
            {
                let want = vec![S::mk(0.0, 0.0); n * n];
                let mut c = to_dev(&rt::asarray((constv::<S>(n * n, 0.5, -0.25), &dcpu)).into_shape([n, n]), order, &d);
                rt::matmul_from(&mut c, &a, a.t(), S::mk(0.0, 0.0), S::mk(0.0, 0.0));
                report("T2", &tag, "alpha = 0, beta = 0", count_bad(&read(&c), &want, tol), &mut bad);
            }
            // a strided output view inside a larger buffer: the block is written, nothing else is
            // the transposed view has no column-major expression: it is the tiled/naive drivers' case
            for (name, transposed) in [("c strided view", false), ("c strided view, transposed", drivers == MatMulDrivers::all())] {
                let mut cbig = to_dev(&cpu_zero, order, &d);
                let want = read_cpu(&rt::matmul(&ca, ca.t()));
                {
                    let (one, zero) = (S::mk(1.0, 0.0), S::mk(0.0, 0.0));
                    if transposed {
                        let mut whole = cbig.view_mut().into_reverse_axes();
                        let mut block = whole.i_mut(s![..n, ..n]);
                        rt::matmul_from(&mut block, &a, a.t(), one, zero);
                    } else {
                        let mut block = cbig.i_mut(s![..n, ..n]);
                        rt::matmul_from(&mut block, &a, a.t(), one, zero);
                    }
                }
                let got = read(&cbig);
                let zero = vec![S::mk(0.0, 0.0); 4 * n * n];
                let mut touched_outside = 0;
                for (i, x) in got.iter().enumerate() {
                    let (r, c) = (i / (2 * n), i % (2 * n));
                    let inside = r < n && c < n;
                    if !inside && (x.re() != 0.0 || x.im() != 0.0) {
                        touched_outside += 1;
                    }
                }
                let mut block_got = Vec::new();
                for r in 0..n {
                    for c in 0..n {
                        block_got.push(got[r * 2 * n + c]);
                    }
                }
                let _ = zero;
                report("T2", &tag, name, count_bad(&block_got, &want, tol) + touched_outside, &mut bad);
            }
        }
    }
    bad
}

fn report(t: &str, tag: &str, name: &str, nb: usize, bad: &mut usize) {
    if nb == 0 {
        println!("ok   {t} {tag} {name}");
    } else {
        *bad += 1;
        println!("FAIL {t} {tag} {name}: {nb}");
    }
}

#[test]
fn t2_syrk_shapes() {
    let mut bad = 0;
    bad += t2::<f64>("f64", 1e-12);
    bad += t2::<Complex<f64>>("c64", 1e-12);
    bad += t2::<f32>("f32 with MatMulUseSyrk::Allow", 1e-4);
    println!("T2 bad cells: {bad}");
}

/* #endregion */

/* #region T3: the symmetric form and the compute mode */

fn time<F: FnMut()>(mut f: F, reps: u32, d: &DeviceCuda) -> f64 {
    f();
    d.synchronize().unwrap();
    let t = Instant::now();
    for _ in 0..reps {
        f();
    }
    d.synchronize().unwrap();
    t.elapsed().as_secs_f64() * 1e3 / reps as f64
}

#[test]
fn t3_syrk_and_compute_mode() {
    let n = 1024usize;
    let dcpu = DeviceCpuSerial::default();
    let v = data(n * n, 21);
    let a: Cpu<f64> = rt::asarray((v.clone(), &dcpu)).into_shape([n, n]);
    let b: Cpu<f64> = rt::asarray((data(n * n, 22), &dcpu)).into_shape([n, n]);

    for (label, mode, syrk, bits) in [
        ("EmulatedFixedPoint, syrk Scientific (default)", ComputeType64F::EmulatedFixedPoint, MatMulUseSyrk::Scientific, 0),
        ("EmulatedFixedPoint, syrk Disallow", ComputeType64F::EmulatedFixedPoint, MatMulUseSyrk::Disallow, 0),
        ("Default, syrk Scientific (default)", ComputeType64F::Default, MatMulUseSyrk::Scientific, 0),
        ("Pedantic, syrk Scientific (default)", ComputeType64F::Pedantic, MatMulUseSyrk::Scientific, 0),
    ] {
        let d = dev(RowMajor, MatMulDrivers::all(), |f| {
            f.compute_type_64f = mode;
            f.matmul_use_syrk = syrk;
            f.trace_matmul = true;
        });
        let (ga, gb) = (to_dev(&a, RowMajor, &d), to_dev(&b, RowMajor, &d));
        println!("--- {label}");
        let _ = rt::matmul(&ga, &gb);
        let _ = rt::matmul(&ga, ga.t());
        d.synchronize().unwrap();
        let (t_gemm, t_syrk) = (time(|| { rt::matmul(&ga, &gb); }, 5, &d), time(|| { rt::matmul(&ga, ga.t()); }, 5, &d));
        println!("T3 {label}: a @ b {t_gemm:.3} ms, a @ a.t() {t_syrk:.3} ms");
    }

    // the documented error of a non-zero mantissa bit count
    let d = dev(RowMajor, MatMulDrivers::all(), |f| {
        f.compute_type_64f = ComputeType64F::EmulatedFixedPoint;
        f.fixedpoint_mantissa_bits = 30;
    });
    let (ga, gb) = (to_dev(&a, RowMajor, &d), to_dev(&b, RowMajor, &d));
    println!("T3 mantissa bits = 30: a @ b -> {:?}", rt::matmul_f(&ga, &gb).err().map(|e| e.to_string()));
    println!("T3 mantissa bits = 30: a @ a.t() -> {:?}", rt::matmul_f(&ga, ga.t()).err().map(|e| e.to_string()));
}

/* #endregion */

/* #region T4: degenerate inner dots */

#[test]
fn t4_degenerate_dots() {
    let dcpu = DeviceCpuSerial::default();
    for drivers in drivers() {
        let d = dev(RowMajor, drivers, |_| {});
        for (k, val) in [(0usize, 0.0f64), (1, 3.0)] {
            let cpu: Cpu<f64> = rt::asarray((vec![val; k], &dcpu)).into_shape(vec![k]);
            let g = to_dev(&cpu, RowMajor, &d);
            match rt::matmul_f(&g, g.t()) {
                Ok(got) => println!("{} T4 {drivers:?} k = {k}: dot = {} (want {})", if got.to_scalar() == val * val * k as f64 { "ok  " } else { "FAIL" }, got.to_scalar(), val * val * k as f64),
                Err(e) => println!("ERR  T4 {drivers:?} k = {k}: {e:?}"),
            }
        }
    }
}

/* #endregion */

/* #region T5: cross-stream in-place matmul */

fn t5_control(n: usize, iters: usize) {
    let d = dev(RowMajor, MatMulDrivers::all(), |_| {});
    let a: Dev<f64> = rt::asarray((vec![0.001; n * n], &d)).into_shape([n, n]);
    let b: Dev<f64> = rt::asarray((vec![1.0; n * n], &d)).into_shape([n, n]);
    let mut c: Dev<f64> = rt::asarray((vec![0.0; n * n], &d)).into_shape([n, n]);
    let mut stale_iters = 0;
    for i in 1..=iters {
        rt::matmul_from(&mut c, &a, &b, 1.0, 1.0);
        let v = read(&c);
        let want = i as f64 * 0.001 * n as f64;
        if v.iter().any(|&x| (x - want).abs() > 1e-9) {
            stale_iters += 1;
        }
    }
    println!("T5 same-stream control: {stale_iters} of {iters} iterations with stale elements");
}

#[test]
fn t5_cross_stream_inplace() {
    t5_control(1536, 10);
    let d1 = DeviceCuda::new(0);
    let s2 = d1.ordinal_obj().context().new_stream().unwrap();
    let mut d2 = d1.clone();
    d2.set_stream(Arc::new(CudaStreamObj::wrap(d1.ordinal(), s2)));
    let (m, k, n) = (1536usize, 1536, 1536);
    for (label, drivers) in [("chain", MatMulDrivers::all()), ("cublasLt", MatMulDrivers::CUBLAS_LT), ("tiled", MatMulDrivers::TILED)] {
        let mut flags = CudaRuntimeFlags::default();
        flags.matmul_drivers = drivers;
        let (mut d1, mut d2) = (d1.clone().with_flags(flags), DeviceCuda::new(0).with_flags(flags));
        let s2 = d2.ordinal_obj().context().new_stream().unwrap();
        d2.set_stream(Arc::new(CudaStreamObj::wrap(d1.ordinal(), s2)));
        let a: Dev<f64> = rt::asarray((vec![0.001; m * k], &d2)).into_shape([m, k]);
        let b: Dev<f64> = rt::asarray((vec![1.0; k * n], &d2)).into_shape([k, n]);
        let c: Dev<f64> = rt::asarray((vec![0.0; m * n], &d1)).into_shape([m, n]);
        let mut c = c.into_device(&d2);
        let mut stale_iters = 0;
        let iters = 30;
        for i in 1..=iters {
            rt::matmul_from(&mut c, &a, &b, 1.0, 1.0);
            let v = read(&c);
            let want = i as f64 * 0.001 * k as f64;
            let stale = v.iter().filter(|&&x| (x - want).abs() > 1e-9).count();
            if stale != 0 {
                stale_iters += 1;
                if stale_iters <= 3 {
                    let shown: Vec<f64> = v.iter().filter(|&&x| (x - want).abs() > 1e-9).take(4).map(|&x| x / (0.001 * k as f64)).collect();
                    println!("T5 cross-stream {label}: iteration {i}: {stale} of {} elements not up to date (in units of one update: {shown:?})", v.len());
                }
            }
        }
        println!("T5 cross-stream {label}: {stale_iters} of {iters} iterations with stale elements");
    }
}

/* #endregion */

/* #region T6: empty and k = 0 products */

#[test]
fn t6_empty_and_k0() {
    let dcpu = DeviceCpuSerial::default();
    for drivers in [MatMulDrivers::CUBLAS_LT, MatMulDrivers::CUBLAS, MatMulDrivers::NAIVE, MatMulDrivers::all()] {
        let d = dev(RowMajor, drivers, |_| {});
        // an empty batch axis
        let a: Cpu<f64> = rt::asarray((vec![1.0; 2 * 3], &dcpu)).into_shape([2, 3]);
        let be: Cpu<f64> = rt::asarray((vec![], &dcpu)).into_shape([0, 3, 4]);
        match rt::matmul_f(&to_dev(&a, RowMajor, &d), &to_dev(&be, RowMajor, &d)) {
            Ok(got) => println!("ok   T6 {drivers:?} [2,3] @ [0,3,4] -> shape {:?}", got.shape()),
            Err(e) => println!("ERR  T6 {drivers:?} [2,3] @ [0,3,4]: {e:?}"),
        }
        // k = 0 with beta = 0, 1, 2
        let (a0, b0): (Cpu<f64>, Cpu<f64>) =
            (rt::asarray((vec![], &dcpu)).into_shape([2, 0]), rt::asarray((vec![], &dcpu)).into_shape([0, 4]));
        let (ga0, gb0) = (to_dev(&a0, RowMajor, &d), to_dev(&b0, RowMajor, &d));
        for beta in [0.0f64, 1.0, 2.0] {
            let mut c = to_dev(&rt::asarray((vec![0.5; 8], &dcpu)).into_shape([2, 4]), RowMajor, &d);
            match rt::matmul_from_f(&mut c, &ga0, &gb0, 1.0, beta) {
                Ok(()) => {
                    let v = read(&c);
                    let want = beta * 0.5;
                    let bad = v.iter().filter(|&&x| x != want).count();
                    println!("{} T6 {drivers:?} k = 0, beta = {beta}: {} of 8 wrong (first values {:?})", if bad == 0 { "ok  " } else { "FAIL" }, bad, &v[..v.len().min(4)]);
                },
                Err(e) => println!("ERR  T6 {drivers:?} k = 0, beta = {beta}: {e:?}"),
            }
        }
        match rt::matmul_f(&ga0, &gb0) {
            Ok(got) => println!("{} T6 {drivers:?} k = 0, fresh output: {:?}", if read(&got).iter().all(|&x| x == 0.0) { "ok  " } else { "FAIL" }, read(&got)),
            Err(e) => println!("ERR  T6 {drivers:?} k = 0, fresh: {e:?}"),
        }
    }
}

/* #endregion */

/* #region T7: the per-call-mode entries */

#[test]
fn t7_mode_entries() {
    let (m, k, n) = (24usize, 32usize, 16usize);
    let dcpu = DeviceCpuSerial::default();
    let v = data(m * k, 31);
    let a: Cpu<f32> = rt::asarray((v.iter().map(|&x| x as f32).collect::<Vec<_>>(), &dcpu)).into_shape([m, k]);
    let b: Cpu<f32> = rt::asarray((data(k * n, 32).into_iter().map(|x| x as f32).collect::<Vec<_>>(), &dcpu)).into_shape([k, n]);
    let c0: Cpu<f32> = rt::asarray((data(m * n, 33).into_iter().map(|x| x as f32).collect::<Vec<_>>(), &dcpu)).into_shape([m, n]);
    let mut cw = c0.clone();
    rt::matmul_from(&mut cw, &a, &b, 2.0f32, 3.0f32);
    let want = read_cpu(&cw);

    let d = dev(RowMajor, MatMulDrivers::all(), |f| f.compute_type_32f = ComputeType32F::Fast16BF);
    let (ga, gb, gc) = (to_dev(&a, RowMajor, &d), to_dev(&b, RowMajor, &d), to_dev(&c0, RowMajor, &d));

    let mut c = gc.clone();
    ga.matmul_with_output_mode(&gb, &mut c.view_mut(), ComputeType32F::Pedantic);
    // the entry writes alpha = 1, beta = 0
    let mut cw2: Cpu<f32> = rt::asarray((vec![0.0f32; m * n], &dcpu)).into_shape([m, n]);
    rt::matmul_from(&mut cw2, &a, &b, 1.0f32, 0.0f32);
    let want2 = read_cpu(&cw2);
    println!("T7 matmul_with_output_mode (Pedantic per call): {} wrong", count_bad(&read(&c), &want2, 1e-5));

    let mut c = gc.clone();
    c.matmul_from_mode(&ga, &gb, 2.0f32, 3.0f32, ComputeType32F::Pedantic);
    println!("T7 matmul_from_mode (Pedantic per call): {} wrong", count_bad(&read(&c), &want, 1e-5));

    let got = ga.matmul_with_mode(&gb, ComputeType32F::Pedantic).out::<f32>();
    println!("T7 matmul_with_mode(..).out(): {} wrong", count_bad(&read(&got), &want2, 1e-5));

    // the device flag is ignored by the per-call entries: Fast16BF would be far less accurate than this
    let d = dev(RowMajor, MatMulDrivers::CUBLAS_EX, |f| f.compute_type_32f = ComputeType32F::Fast16BF);
    let (ga, gb) = (to_dev(&a, RowMajor, &d), to_dev(&b, RowMajor, &d));
    let via_flag = read(&rt::matmul(&ga, &gb));
    let via_call = read(&ga.matmul_with_mode(&gb, ComputeType32F::Pedantic).out::<f32>());
    println!("T7 flag Fast16BF: {} wrong;  per-call Pedantic on the same device: {} wrong", count_bad(&via_flag, &want2, 1e-5), count_bad(&via_call, &want2, 1e-5));

    // mixed operands through the mode entries (naive driver)
    let a64: Cpu<f64> = rt::asarray((v.clone(), &dcpu)).into_shape([m, k]);
    let b64: Cpu<f64> = rt::asarray((data(k * n, 32).into_iter().map(|x| x as f32 as f64).collect::<Vec<_>>(), &dcpu)).into_shape([k, n]);
    let mut cw3: Cpu<f64> = rt::asarray((vec![0.0f64; m * n], &dcpu)).into_shape([m, n]);
    rt::matmul_from(&mut cw3, &a64, &b64, 1.0f64, 0.0f64);
    let want3 = read_cpu(&cw3);
    let d = dev(RowMajor, MatMulDrivers::all(), |_| {});
    let (ga64, gb32) = (to_dev(&a64, RowMajor, &d), to_dev(&b, RowMajor, &d));
    let got = ga64.matmul_with_mode(&gb32, ComputeType64F::Pedantic).out::<f64>();
    println!("T7 mixed f64 @ f32 -> f64 through matmul_with_mode: {} wrong", count_bad(&read(&got), &want3, 1e-12));
}

/* #endregion */

/* #region T9: operands broadcast along a matmul axis (stride 0, extent > 1), and flipped batches */

/// `cases` returns `(name, a, b, logical a, logical b)`: the device operands and their
/// materialized (owned) forms, which give the reference product on the CPU.
macro_rules! bcases {
    ($b:expr) => {{
        let b = &$b;
        vec![
            ("b broadcast along k: [1,4] -> [3,4]", b.a.view(), b.b14.broadcast_to(vec![3, 4])),
            ("b broadcast along n: [3,1] -> [3,4]", b.a.view(), b.b31.broadcast_to(vec![3, 4])),
            ("a broadcast along m: [1,2] -> [3,2]", b.a12.broadcast_to(vec![3, 2]), b.b24.view()),
            ("a broadcast along k: [2,1] -> [2,3]", b.a21.broadcast_to(vec![2, 3]), b.b.view()),
        ]
    }};
}

struct Bases9<S, B: DeviceRawAPI<S>> {
    a: Tensor<S, B, IxD>,
    b: Tensor<S, B, IxD>,
    bt: Tensor<S, B, IxD>,
    b14: Tensor<S, B, IxD>,
    b31: Tensor<S, B, IxD>,
    a12: Tensor<S, B, IxD>,
    a21: Tensor<S, B, IxD>,
    b24: Tensor<S, B, IxD>,
    ab: Tensor<S, B, IxD>,
    bb: Tensor<S, B, IxD>,
    abig: Tensor<S, B, IxD>,
}

fn t9<S: Scalar>(what: &str, tol: f64) -> usize {
    let dcpu = DeviceCpuSerial::default();
    let cpu = Bases9::<S, DeviceCpuSerial> {
        a: rt::asarray((rnd::<S>(2 * 3, 41), &dcpu)).into_shape([2, 3]),
        b: rt::asarray((rnd::<S>(3 * 4, 42), &dcpu)).into_shape([3, 4]),
        bt: rt::asarray((rnd::<S>(1 * 4, 43), &dcpu)).into_shape([1, 4]),
        b14: rt::asarray((rnd::<S>(1 * 4, 43), &dcpu)).into_shape([1, 4]),
        b31: rt::asarray((rnd::<S>(3 * 1, 44), &dcpu)).into_shape([3, 1]),
        a12: rt::asarray((rnd::<S>(1 * 2, 45), &dcpu)).into_shape([1, 2]),
        a21: rt::asarray((rnd::<S>(2 * 1, 46), &dcpu)).into_shape([2, 1]),
        b24: rt::asarray((rnd::<S>(2 * 4, 47), &dcpu)).into_shape([2, 4]),
        ab: rt::asarray((rnd::<S>(6, 48), &dcpu)).into_shape([1, 2, 3]),
        bb: rt::asarray((rnd::<S>(6, 49), &dcpu)).into_shape([1, 3, 2]),
        abig: rt::asarray((rnd::<S>(12, 50), &dcpu)).into_shape([3, 4]),
    };
    let mut bad = 0;
    for order in [RowMajor, ColMajor] {
        for drivers in drivers() {
            let d = dev(order, drivers, |_| {});
            let g = Bases9::<S, DeviceCuda> {
                a: to_dev(&cpu.a, order, &d),
                b: to_dev(&cpu.b, order, &d),
                bt: to_dev(&cpu.bt, order, &d),
                b14: to_dev(&cpu.b14, order, &d),
                b31: to_dev(&cpu.b31, order, &d),
                a12: to_dev(&cpu.a12, order, &d),
                a21: to_dev(&cpu.a21, order, &d),
                b24: to_dev(&cpu.b24, order, &d),
                ab: to_dev(&cpu.ab, order, &d),
                bb: to_dev(&cpu.bb, order, &d),
                abig: to_dev(&cpu.abig, order, &d),
            };
            let cc = bcases!(cpu);
            let gc = bcases!(g);
            for ((name, ca, cb), (ga, gb)) in cc.iter().zip(gc.iter().map(|(_, x, y)| (x, y))) {
                let want = read_cpu(&rt::matmul(&ca.to_owned(), &cb.to_owned()));
                match rt::matmul_f(ga, gb) {
                    Ok(got) => {
                        let (got, nb) = (read(&got), count_bad(&read(&got), &want, tol));
                        if nb == 0 {
                            println!("ok   T9 {what} {order:?} {drivers:?} {name}");
                        } else {
                            bad += 1;
                            println!("FAIL T9 {what} {order:?} {drivers:?} {name}: {nb} of {} wrong; got[:3] = {:?}", want.len(), &got[..got.len().min(3)]);
                        }
                    },
                    Err(e) => println!("ref  T9 {what} {order:?} {drivers:?} {name}: {e:?}"),
                }
            }
        }
    }
    bad
}

#[test]
fn t11_syrk_of_a_broadcast_operand() {
    // a column [n, 1] broadcast along k to [n, k], times its own transpose: the form's condition holds
    let (n, k) = (37usize, 19usize);
    let dcpu = DeviceCpuSerial::default();
    let col: Cpu<f64> = rt::asarray((data(n, 71), &dcpu)).into_shape([n, 1]);
    for order in [RowMajor, ColMajor] {
        for drivers in [MatMulDrivers::CUBLAS, MatMulDrivers::all()] {
            let d = dev(order, drivers, |_| {});
            let g = to_dev(&col, order, &d);
            let a = g.broadcast_to(vec![n, k]);
            match rt::matmul_f(&a, a.t()) {
                Ok(got) => {
                    // a = col broadcast along k, so a @ a^T = k * (col col^T)
                    let want = read_cpu(&(rt::matmul(&col, col.t()) * (k as f64)));
                    let nb = count_bad(&read(&got), &want, 1e-12);
                    println!("{} T11 {order:?} {drivers:?} a @ a.t() with a broadcast along k: {nb} of {} wrong", if nb == 0 { "ok  " } else { "FAIL" }, want.len());
                },
                Err(e) => println!("ref  T11 {order:?} {drivers:?}: {}", format!("{e:?}").chars().filter(|&c| c != '\n').take(230).collect::<String>()),
            }
        }
    }
}

#[test]
fn t9_broadcast_matmul_axis() {
    let mut bad = 0;
    bad += t9::<f64>("f64", 1e-12);
    bad += t9::<Complex<f64>>("c64", 1e-12);
    println!("T9 bad cells: {bad}");

    // a batch axis reversed (negative batch stride)
    for order in [RowMajor, ColMajor] {
        let mut dcpu = DeviceCpuSerial::default();
        dcpu.set_default_order(order);
        let (_sab, _sbb): (Vec<usize>, Vec<usize>) = match order {
            RowMajor => (vec![3usize, 4usize, 5usize], vec![3, 5, 6]),
            ColMajor => (vec![4, 5, 3], vec![5, 6, 3]),
            _ => unreachable!(),
        };
        let axis = if order == RowMajor { 0 } else { 2 };
        let cpu_b: Cpu<f64> = rt::asarray((data(90, 62), &dcpu)).into_shape(_sbb);
        let cpu_a = rt::asarray((data(3 * 4 * 5, 61), &dcpu)).into_shape(if order == RowMajor { vec![3, 4, 5] } else { vec![4, 5, 3] });
        let want = read_cpu(&rt::matmul(&cpu_a.flip(axis), &cpu_b.flip(axis)));
        for drivers in drivers() {
            let d = dev(order, drivers, |_| {});
            let (ga, gb) = (to_dev(&cpu_a, order, &d), to_dev(&cpu_b, order, &d));
            match rt::matmul_f(&ga.flip(axis), &gb.flip(axis)) {
                Ok(got) => {
                    let nb = count_bad(&read(&got), &want, 1e-12);
                    println!("{} T10 {order:?} {drivers:?} reversed batch axis: {nb} of {} wrong", if nb == 0 { "ok  " } else { "FAIL" }, want.len());
                },
                Err(e) => println!("ref  T10 {order:?} {drivers:?} reversed batch axis: {e:?}"),
            }
        }
    }
}

/* #endregion */

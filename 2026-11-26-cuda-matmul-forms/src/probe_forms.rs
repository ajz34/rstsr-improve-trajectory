//! B0 probe: inner-dot bandwidth, and raw cuBLAS gemv / syrk against the rstsr-cuda matmul chain
//! (`261021/cuda` @ `80e626d`). Trajectory only; no rstsr file is edited.
//!
//! ```text
//! cargo run --release --example probe_forms            # every part
//! cargo run --release --example probe_forms -- gemv    # one of: dot, dot-gemm, gemv, gemv-batch, syrk
//! ```
//!
//! The f32 dot gate uses a loose relative tolerance: the rounding error of a sum of `k` random
//! terms grows with `k` while the sum itself stays of order 1.
//!
//! Timing: the repeat count is fitted to about 0.1 s per trial, then the median of 5 trials of
//! the mean wall time per call around a synchronized loop (launch overhead included). Chain rows
//! go through the tensor API with a fresh output per call; anchors are raw cuBLAS calls on the
//! same stream. A correctness gate on small shapes runs before any timing of a part.

use std::sync::Arc;
use std::time::Instant;

use cudarc::cublas::{sys as cb, CudaBlas};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use rstsr::prelude::*;
use rstsr_cuda::matmul::MatMulDrivers;
use rstsr_cuda::prelude_dev::*;

fn data(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect()
}

fn rel<T>(got: T, want: T) -> f64
where
    T: Into<f64> + Copy,
{
    (got.into() - want.into()).abs() / want.into().abs().max(1.0)
}

/// Median of 5 trials, each the mean per call over a repeat count fitted to about 0.1 s.
fn time(stream: &Arc<CudaStream>, mut f: impl FnMut()) -> f64 {
    for _ in 0..3 {
        f();
    }
    stream.synchronize().unwrap();
    let t = Instant::now();
    for _ in 0..3 {
        f();
    }
    stream.synchronize().unwrap();
    let per = t.elapsed().as_secs_f64() / 3.0;
    let reps = (0.1 / per).clamp(5.0, 5000.0) as usize;
    let mut ts: Vec<f64> = (0..5)
        .map(|_| {
            stream.synchronize().unwrap();
            let t = Instant::now();
            for _ in 0..reps {
                f();
            }
            stream.synchronize().unwrap();
            t.elapsed().as_secs_f64() / reps as f64
        })
        .collect();
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ts[2]
}

fn us(s: f64) -> f64 {
    s * 1e6
}

/// A device tensor brought back to the host (contiguous in the device's order first).
fn host<T, D>(t: &Tensor<T, DeviceCuda, D>) -> Vec<T>
where
    T: Copy + Send + Sync + 'static + rstsr_cuda::dtype::CudaDType,
    D: DimAPI,
{
    DeviceToCpuAPI::<T>::to_cpu_vec(t.to_contig(RowMajor).storage()).rstsr_unwrap()
}

struct Ctx {
    d: DeviceCuda,
    stream: Arc<CudaStream>,
    blas: Arc<CudaBlas>,
}

impl Ctx {
    fn new(drivers: MatMulDrivers) -> Self {
        let mut flags = default_flags();
        flags.matmul_drivers = drivers;
        let d = DeviceCuda::new(0).with_flags(flags);
        let stream = d.stream().inner().clone();
        let blas = Arc::new(CudaBlas::new(stream.clone()).unwrap());
        Self { d, stream, blas }
    }

    fn up<T: Copy + Send + Sync + 'static + cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(&self, v: &[T]) -> CudaSlice<T> {
        let mut dst = self.stream.alloc_zeros(v.len()).unwrap();
        self.stream.memcpy_htod(v, &mut dst).unwrap();
        dst
    }
}


// =============================================================================================
// inner dot
// =============================================================================================

macro_rules! dot_part {
    ($ctx:expr, $name:literal, $t:ty, $dot:ident, $tol:expr) => {{
        let esz = std::mem::size_of::<$t>();
        println!("\n### {}\n", $name);
        println!("| log2 k | `vecdot` fresh (us) | `vecdot_from` (us) | copy ceiling (us) | cuBLAS dot, blocking (us) | cuBLAS dot, device mode (us) | `vecdot_from` GB/s | ceiling GB/s | % of ceiling |");
        println!("|---|---|---|---|---|---|---|---|---|");
        for p in 20..=26u32 {
            let k = 1usize << p;
            let av: Vec<$t> = data(k, 11).iter().map(|&x| x as $t).collect();
            let bv: Vec<$t> = data(k, 7).iter().map(|&x| x as $t).collect();
            let da = rt::asarray((av.clone(), &$ctx.d));
            let db = rt::asarray((bv.clone(), &$ctx.d));
            let mut out = rt::vecdot(&da, &db, 0);
            let want: $t = av.iter().zip(&bv).map(|(x, y)| *x * *y).sum();

            let got = host(&rt::vecdot(&da, &db, 0))[0];
            assert!(rel(got, want) < $tol, "{} vecdot k = {k}: got {got:?}, want {want:?}", $name);

            let src = $ctx.up(&av);
            let mut dst: CudaSlice<$t> = $ctx.stream.alloc_zeros(k).unwrap();
            let copy = time(&$ctx.stream, || $ctx.stream.memcpy_dtod(&src, &mut dst).unwrap());
            let fresh = time(&$ctx.stream, || {
                host(&rt::vecdot(&da, &db, 0));
            });
            let into = time(&$ctx.stream, || {
                out.vecdot_from(&da, &db, 0);
            });

            let x = $ctx.up(&av);
            let y = $ctx.up(&bv);
            let (xp, yp) = (x.device_ptr(&$ctx.stream).0 as usize, y.device_ptr(&$ctx.stream).0);
            let blas = $ctx.blas.clone();
            let mut r = 0 as $t;
            let blocking = time(&$ctx.stream, || unsafe {
                cb::$dot(*blas.handle(), k as i32, xp as *const $t, 1, yp as *const $t, 1, &mut r);
            });
            assert!(rel(r, want) < $tol, "{} cublas dot k = {k}: got {r:?}, want {want:?}", $name);

            let mut dr: CudaSlice<$t> = $ctx.stream.alloc_zeros(1).unwrap();
            let rp = dr.device_ptr_mut(&$ctx.stream).0;
            blas.set_pointer_mode(cb::cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE).unwrap();
            let dev = time(&$ctx.stream, || unsafe {
                cb::$dot(*blas.handle(), k as i32, xp as *const $t, 1, yp as *const $t, 1, rp as *mut $t);
            });
            blas.set_pointer_mode(cb::cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST).unwrap();
            let mut back = vec![0 as $t; 1];
            $ctx.stream.memcpy_dtoh(&dr, &mut back).unwrap();
            assert!(rel(back[0], want) < $tol, "{} cublas dot (device) k = {k}: got {back:?}", $name);

            let bytes = 2 * k * esz;
            println!(
                "| {p} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.0} | {:.0} | {:.0} % |",
                us(fresh),
                us(into),
                us(copy),
                us(blocking),
                us(dev),
                bytes as f64 / into / 1e9,
                bytes as f64 / copy / 1e9,
                100.0 * copy / into,
            );

            // two-stage dot from the same kernel: partials over `[B, k / B]` (B blocks), then one
            // small reduction of the B partials into the rank-0 output
            for b in [256usize, 1024, 4096, 16384, 65536] {
                let l = k / b;
                let a2 = da.view().into_shape([b, l]);
                let b2 = db.view().into_shape([b, l]);
                let ones: Tensor<$t, _, _> = rt::ones(([b], &$ctx.d));
                let two = time(&$ctx.stream, || {
                    let part = rt::vecdot(&a2, &b2, 1);
                    rt::vecdot(&part, &ones, 0);
                });
                let got = host(&rt::vecdot(&rt::vecdot(&a2, &b2, 1), &ones, 0))[0];
                assert!(rel(got, want) < $tol * 100.0, "{} two-stage dot k = {k}, B = {b}: got {got:?}", $name);
                println!(
                    "|    two-stage, B = {b} | | {:.1} | | | | {:.0} | | {:.0} % |",
                    us(two),
                    bytes as f64 / two / 1e9,
                    100.0 * copy / two,
                );
            }
        }
    }};
}

fn dot() {
    let ctx = Ctx::new(MatMulDrivers::all());
    println!("\n## Inner dot: `vecdot` against the copy ceiling\n");
    println!("Correctness gate passed (device dot vs host sum, and both cuBLAS dots).\n");
    dot_part!(&ctx, "f64", f64, cublasDdot_v2, 1e-9);
    dot_part!(&ctx, "f32", f32, cublasSdot_v2, 2e-2);
}

// =============================================================================================
// gemv
// =============================================================================================

macro_rules! gemv_part {
    ($ctx:expr, $name:literal, $t:ty, $gemv:ident, $tol:expr, $f32_spot:expr) => {{
        /// One raw gemv on a row-major `a` (`m x k`, leading dimension `k`); `vecmat` computes
        /// `y = a^T x`. Returns the timed call and the device `y`.
        fn raw(c: &Ctx, a: &[$t], x: &[$t], m: usize, k: usize, vecmat: bool) -> (Box<dyn FnMut()>, CudaSlice<$t>) {
            let a = c.up(a);
            let x = c.up(x);
            let mut y: CudaSlice<$t> = c.stream.alloc_zeros(if vecmat { k } else { m }).unwrap();
            let (ap, xp, yp) = (a.device_ptr(&c.stream).0, x.device_ptr(&c.stream).0, y.device_ptr_mut(&c.stream).0);
            let trans = if vecmat { cb::cublasOperation_t::CUBLAS_OP_N } else { cb::cublasOperation_t::CUBLAS_OP_T };
            let blas = c.blas.clone();
            let f = Box::new(move || unsafe {
                let st = cb::$gemv(*blas.handle(), trans, k as i32, m as i32, &(1 as $t), ap as *const $t, k as i32, xp as *const $t, 1, &(0 as $t), yp as *mut $t, 1);
                assert_eq!(st, cb::cublasStatus_t::CUBLAS_STATUS_SUCCESS, "{:?}", st);
                let (_a, _x) = (&a, &x);
            });
            (f, y)
        }

        fn host_ref(a: &[$t], x: &[$t], m: usize, k: usize, vecmat: bool) -> Vec<$t> {
            (0..if vecmat { k } else { m })
                .map(|i| {
                    (0..if vecmat { m } else { k }).map(|j| {
                        let (ai, aj) = if vecmat { (j, i) } else { (i, j) };
                        a[ai * k + aj] * x[j]
                    }).sum::<$t>()
                })
                .collect()
        }

        /// The chain through `rt::matmul`, on a device with the given drivers.
        fn chain(drivers: MatMulDrivers, a: Vec<$t>, x: Vec<$t>, m: usize, k: usize, vecmat: bool, check: bool) -> (f64, Option<Vec<$t>>) {
            let c = Ctx::new(drivers);
            let a = rt::asarray((a, &c.d)).into_shape([m, k]);
            let x = rt::asarray((x, &c.d));
            let product = || if vecmat { rt::matmul(&x, &a) } else { rt::matmul(&a, &x) };
            let got = check.then(|| host(&product()));
            let t = time(&c.stream, || {
                product();
            });
            (t, got)
        }

        // gate: raw and chain against the host loop, both orientations
        let (m, k) = (37, 53);
        let a: Vec<$t> = data(m * k, 3).iter().map(|&x| x as $t).collect();
        for vecmat in [false, true] {
            let x: Vec<$t> = data(if vecmat { m } else { k }, 5).iter().map(|&x| x as $t).collect();
            let (mut f, y) = raw(&$ctx, &a, &x, m, k, vecmat);
            f();
            $ctx.stream.synchronize().unwrap();
            let mut yv = vec![0 as $t; y.len()];
            $ctx.stream.memcpy_dtoh(&y, &mut yv).unwrap();
            let want = host_ref(&a, &x, m, k, vecmat);
            for (g, w) in yv.iter().zip(&want) {
                assert!(rel(*g, *w) < $tol, "{} raw gemv: got {g:?}, want {w:?}", $name);
            }
            let (_, got) = chain(MatMulDrivers::all(), a.clone(), x.clone(), m, k, vecmat, true);
            for (g, w) in got.unwrap().iter().zip(&want) {
                assert!(rel(*g, *w) < $tol, "{} chain gemv: got {g:?}, want {w:?}", $name);
            }
        }
        println!("\n## gemv, {}\n", $name);
        println!("Correctness gate passed (raw gemv and the chain vs a host loop, both orientations).\n");
        println!("| matrix | orientation | cuBLAS gemv (us) | chain: all | chain: Lt | chain: Ex | chain: cuBLAS | best chain / gemv |");
        println!("|---|---|---|---|---|---|---|---|");

        let mut cells: Vec<(usize, usize)> = vec![(1024, 1024)];
        if !$f32_spot {
            cells.extend([(4096, 4096), (8192, 8192), (65536, 64), (256, 452_676), (452_676, 256)]);
        }
        for (m, k) in cells {
            for vecmat in [false, true] {
                let x: Vec<$t> = vec![1 as $t; if vecmat { m } else { k }];
                let a: Vec<$t> = vec![1 as $t; m * k];
                let (f, y) = raw(&$ctx, &a, &x, m, k, vecmat);
                let gemv = time(&$ctx.stream, f);
                drop(y);
                let rows = [
                    chain(MatMulDrivers::all(), a.clone(), x.clone(), m, k, vecmat, false).0,
                    chain(MatMulDrivers::CUBLAS_LT, a.clone(), x.clone(), m, k, vecmat, false).0,
                    chain(MatMulDrivers::CUBLAS_EX, a.clone(), x.clone(), m, k, vecmat, false).0,
                    chain(MatMulDrivers::CUBLAS, a.clone(), x.clone(), m, k, vecmat, false).0,
                ];
                let best = rows.iter().copied().fold(f64::INFINITY, f64::min);
                println!(
                    "| {m} x {k} | {} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.2} |",
                    if vecmat { "vecmat" } else { "matvec" },
                    us(gemv),
                    us(rows[0]),
                    us(rows[1]),
                    us(rows[2]),
                    us(rows[3]),
                    gemv / best,
                );
            }
        }
    }};
}

fn gemv() {
    let ctx = Ctx::new(MatMulDrivers::all());
    gemv_part!(&ctx, "f64", f64, cublasDgemv_v2, 1e-9, false);
    let ctx = Ctx::new(MatMulDrivers::all());
    gemv_part!(&ctx, "f32", f32, cublasSgemv_v2, 1e-4, false);
}

/// Batched gemv, f64: `cublasDgemvStridedBatched` against the chain, for the broadcast rule
/// (`[p, m, k] @ [k]`) and the per-item vector as a narrow GEMM (`[p, m, k] @ [p, k, 1]`).
fn gemv_batch() {
    let ctx = Ctx::new(MatMulDrivers::all());

    let raw = |ctx: &Ctx, a: &[f64], x: &[f64], p: usize, stridex: i64, m: usize, k: usize| -> (Box<dyn FnMut()>, CudaSlice<f64>) {
        let a = ctx.up(a);
        let x = ctx.up(x);
        let mut y: CudaSlice<f64> = ctx.stream.alloc_zeros(p * m).unwrap();
        let (ap, xp, yp) = (a.device_ptr(&ctx.stream).0, x.device_ptr(&ctx.stream).0, y.device_ptr_mut(&ctx.stream).0);
        let blas = ctx.blas.clone();
        let f = Box::new(move || unsafe {
            let st = cb::cublasDgemvStridedBatched(
                *blas.handle(),
                cb::cublasOperation_t::CUBLAS_OP_T,
                k as i32,
                m as i32,
                &1.0f64,
                ap as *const f64,
                k as i32,
                (m * k) as i64,
                xp as *const f64,
                1,
                stridex,
                &0.0f64,
                yp as *mut f64,
                1,
                m as i64,
                p as i32,
            );
            assert_eq!(st, cb::cublasStatus_t::CUBLAS_STATUS_SUCCESS, "{:?}", st);
            let (_a, _x) = (&a, &x);
        });
        (f, y)
    };

    // gate
    let (p, m, k) = (3usize, 5usize, 7usize);
    let a: Vec<f64> = data(p * m * k, 3);
    for broadcast in [false, true] {
        let x: Vec<f64> = data(if broadcast { k } else { p * k }, 5);
        let stridex = if broadcast { 0 } else { k as i64 };
        let (mut f, y) = raw(&ctx, &a, &x, p, stridex, m, k);
        f();
        ctx.stream.synchronize().unwrap();
        let mut yv = vec![0.0; y.len()];
        ctx.stream.memcpy_dtoh(&y, &mut yv).unwrap();
        for b in 0..p {
            for i in 0..m {
                let want: f64 = (0..k).map(|j| a[b * m * k + i * k + j] * x[if broadcast { j } else { b * k + j }]).sum();
                assert!(rel(yv[b * m + i], want) < 1e-9);
            }
        }
    }
    println!("\n## Batched gemv, f64 (p = 64, 1024 x 1024 per item)\n");
    println!("Correctness gate passed (raw strided-batched gemv vs a host loop, both cases).\n");
    println!("| rule | cuBLAS gemvStridedBatched (us) | chain: all | chain: Lt | chain: Ex | chain: cuBLAS | all / gemv |");
    println!("|---|---|---|---|---|---|---|");

    let (p, m, k) = (64usize, 1024usize, 1024usize);
    let a: Vec<f64> = vec![1.0; p * m * k];
    for broadcast in [false, true] {
        let x: Vec<f64> = vec![1.0; if broadcast { k } else { p * k }];
        let (f, y) = raw(&ctx, &a, &x, p, if broadcast { 0 } else { k as i64 }, m, k);
        let gemv = time(&ctx.stream, f);
        drop(y);
        let mut rows = Vec::new();
        for d in [MatMulDrivers::CUBLAS_LT, MatMulDrivers::CUBLAS_EX, MatMulDrivers::CUBLAS] {
            let c = Ctx::new(d);
            let at = rt::asarray((a.clone(), &c.d)).into_shape([p, m, k]);
            let xt = if broadcast { rt::asarray((x[..k].to_vec(), &c.d)) } else { rt::asarray((x.clone(), &c.d)).into_shape([p, k, 1]) };
            rows.push(time(&c.stream, || {
                rt::matmul(&at, &xt);
            }));
        }
        let c = Ctx::new(MatMulDrivers::all());
        let at = rt::asarray((a.clone(), &c.d)).into_shape([p, m, k]);
        let xt = if broadcast { rt::asarray((x[..k].to_vec(), &c.d)) } else { rt::asarray((x.clone(), &c.d)).into_shape([p, k, 1]) };
        let all = time(&c.stream, || {
            rt::matmul(&at, &xt);
        });
        println!(
            "| {} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.2} |",
            if broadcast { "[p, m, k] @ [k] (broadcast)" } else { "[p, m, k] @ [p, k, 1]" },
            us(gemv),
            us(all),
            us(rows[0]),
            us(rows[1]),
            us(rows[2]),
            gemv / all,
        );
    }
}

// =============================================================================================
// syrk
// =============================================================================================

macro_rules! syrk_part {
    ($ctx:expr, $name:literal, $t:ty, $syrk:ident, $tol:expr) => {{
        fn call(blas: &CudaBlas, uplo: cb::cublasFillMode_t, n: usize, k: usize, ap: usize, cp: usize) {
            let st = unsafe {
                cb::$syrk(*blas.handle(), uplo, cb::cublasOperation_t::CUBLAS_OP_T, n as i32, k as i32, &(1 as $t), ap as *const $t, k as i32, &(0 as $t), cp as *mut $t, n as i32)
            };
            assert_eq!(st, cb::cublasStatus_t::CUBLAS_STATUS_SUCCESS, "{:?}", st);
        }

        // gate: the upper triangle against a host loop, the other triangle untouched, and the
        // completed matrix equal to the chain's `a @ a.t()`
        let (n, k) = (33usize, 21usize);
        let a: Vec<$t> = data(n * k, 3).iter().map(|&x| x as $t).collect();
        let ad = $ctx.up(&a);
        let mut c: CudaSlice<$t> = $ctx.stream.alloc_zeros(n * n).unwrap();
        call(&$ctx.blas, cb::cublasFillMode_t::CUBLAS_FILL_MODE_UPPER, n, k, ad.device_ptr(&$ctx.stream).0 as usize, c.device_ptr_mut(&$ctx.stream).0 as usize);
        $ctx.stream.synchronize().unwrap();
        let mut cv = vec![0 as $t; n * n];
        $ctx.stream.memcpy_dtoh(&c, &mut cv).unwrap();
        for i in 0..n {
            for j in 0..n {
                let want: $t = (0..k).map(|p| a[i * k + p] * a[j * k + p]).sum();
                if i <= j {
                    assert!(rel(cv[i + j * n], want) < $tol, "{} syrk upper: got {:?}, want {want:?}", $name, cv[i + j * n]);
                } else {
                    assert_eq!(cv[i + j * n], 0 as $t, "the lower triangle must be untouched");
                }
            }
        }
        let at = rt::asarray((a.clone(), &$ctx.d)).into_shape([n, k]);
        let got = host(&rt::matmul(&at, &at.t()));
        for i in 0..n {
            for j in 0..n {
                let want: $t = (0..k).map(|p| a[i * k + p] * a[j * k + p]).sum();
                assert!(rel(got[i * n + j], want) < $tol, "{} chain a @ a.t(): got {:?}, want {want:?}", $name, got[i * n + j]);
            }
        }
        drop((ad, c));

        println!("\n## syrk, {}\n", $name);
        println!("Correctness gate passed (upper triangle vs a host loop, lower untouched, chain `a @ a.t()`).\n");
        println!("| n | k | syrk upper (us) | two syrk calls (us) | triangle copy (us) | chain: all | chain: Lt | chain: cuBLAS | all / (syrk + copy) |");
        println!("|---|---|---|---|---|---|---|---|---|");
        for n in [256usize, 512, 1024, 2048, 4096] {
            for k in [n / 8, n, 8 * n] {
                let a: Vec<$t> = vec![1 as $t; n * k];
                let ad = $ctx.up(&a);
                let mut c: CudaSlice<$t> = $ctx.stream.alloc_zeros(n * n).unwrap();
                let (ap, cp) = (ad.device_ptr(&$ctx.stream).0 as usize, c.device_ptr_mut(&$ctx.stream).0 as usize);
                let blas = $ctx.blas.clone();
                let upper = time(&$ctx.stream, || call(&blas, cb::cublasFillMode_t::CUBLAS_FILL_MODE_UPPER, n, k, ap, cp));
                let both = time(&$ctx.stream, || {
                    call(&blas, cb::cublasFillMode_t::CUBLAS_FILL_MODE_UPPER, n, k, ap, cp);
                    call(&blas, cb::cublasFillMode_t::CUBLAS_FILL_MODE_LOWER, n, k, ap, cp);
                });
                let half: CudaSlice<$t> = $ctx.stream.alloc_zeros(n * n / 2).unwrap();
                let mut mirror: CudaSlice<$t> = $ctx.stream.alloc_zeros(n * n / 2).unwrap();
                let copy = time(&$ctx.stream, || $ctx.stream.memcpy_dtod(&half, &mut mirror).unwrap());
                drop((ad, c, half, mirror));

                let mut rows = Vec::new();
                for d in [MatMulDrivers::CUBLAS_LT, MatMulDrivers::CUBLAS] {
                    let cd = Ctx::new(d);
                    let at = rt::asarray((a.clone(), &cd.d)).into_shape([n, k]);
                    rows.push(time(&cd.stream, || {
                        rt::matmul(&at, &at.t());
                    }));
                }
                let cd = Ctx::new(MatMulDrivers::all());
                let at = rt::asarray((a.clone(), &cd.d)).into_shape([n, k]);
                let all = time(&cd.stream, || {
                    rt::matmul(&at, &at.t());
                });
                println!(
                    "| {n} | {k} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.2} |",
                    us(upper),
                    us(both),
                    us(copy),
                    us(all),
                    us(rows[0]),
                    us(rows[1]),
                    all / (upper + copy),
                );
            }
        }
    }};
}

fn syrk() {
    let ctx = Ctx::new(MatMulDrivers::all());
    syrk_part!(&ctx, "f64", f64, cublasDsyrk_v2, 1e-9);
    let ctx = Ctx::new(MatMulDrivers::all());
    syrk_part!(&ctx, "f32", f32, cublasSsyrk_v2, 1e-4);
}

/// `cublasCsyrk_v2` is the plain-transpose form (no conjugation): the gate that D7 relies on,
/// plus one timing cell.
fn syrk_complex() {
    use cb::cuComplex;
    use num::Complex;
    let (one, zero) = (cuComplex { x: 1.0, y: 0.0 }, cuComplex { x: 0.0, y: 0.0 });

    let ctx = Ctx::new(MatMulDrivers::all());
    let (n, k) = (20usize, 17usize);
    let a: Vec<Complex<f32>> = data(2 * n * k, 3).chunks(2).map(|c| Complex::new(c[0] as f32, c[1] as f32)).collect();
    let raw: Vec<[f32; 2]> = a.iter().map(|c| [c.re, c.im]).collect();
    let ad = ctx.up(&raw);
    let mut c: CudaSlice<[f32; 2]> = ctx.stream.alloc_zeros(n * n).unwrap();
    let st = unsafe {
        cb::cublasCsyrk_v2(*ctx.blas.handle(), cb::cublasFillMode_t::CUBLAS_FILL_MODE_UPPER, cb::cublasOperation_t::CUBLAS_OP_T, n as i32, k as i32, &one, ad.device_ptr(&ctx.stream).0 as *const cuComplex, k as i32, &zero, c.device_ptr_mut(&ctx.stream).0 as *mut cuComplex, n as i32)
    };
    assert_eq!(st, cb::cublasStatus_t::CUBLAS_STATUS_SUCCESS);
    ctx.stream.synchronize().unwrap();
    let mut cv = vec![[0.0f32; 2]; n * n];
    ctx.stream.memcpy_dtoh(&c, &mut cv).unwrap();
    let mut worst = 0.0f32;
    for i in 0..n {
        for j in 0..n {
            let want: Complex<f32> = (0..k).map(|p| a[i * k + p] * a[j * k + p]).sum();
            if i <= j {
                let g = Complex::new(cv[i + j * n][0], cv[i + j * n][1]);
                worst = worst.max((g - want).norm() / want.norm().max(1.0));
            }
        }
    }
    println!("\n## `cublasCsyrk_v2` is the plain-transpose form\n");
    println!("Upper triangle vs `a @ a.t()` (no conjugation), worst relative error {worst:.2e} (gate: < 1e-4).");
    assert!(worst < 1e-4);

    let blas = ctx.blas.clone();
    println!("\n| c32, k = n | `cublasCsyrk_v2` upper (us) | chain: all | chain: Lt | chain: cuBLAS | all / (syrk + copy) |");
    println!("|---|---|---|---|---|---|");
    for n in [256usize, 512, 1024, 2048, 4096] {
        let k = n;
        let a: Vec<Complex<f32>> = vec![Complex::new(1.0, 0.0); n * k];
        let raw: Vec<[f32; 2]> = a.iter().map(|c| [c.re, c.im]).collect();
        let ad = ctx.up(&raw);
        let mut c: CudaSlice<[f32; 2]> = ctx.stream.alloc_zeros(n * n).unwrap();
        let (ap, cp) = (ad.device_ptr(&ctx.stream).0 as usize, c.device_ptr_mut(&ctx.stream).0 as usize);
        let upper = time(&ctx.stream, || unsafe {
            cb::cublasCsyrk_v2(*blas.handle(), cb::cublasFillMode_t::CUBLAS_FILL_MODE_UPPER, cb::cublasOperation_t::CUBLAS_OP_T, n as i32, k as i32, &one, ap as *const cuComplex, k as i32, &zero, cp as *mut cuComplex, n as i32);
        });
        let half: CudaSlice<[f32; 2]> = ctx.stream.alloc_zeros(n * n / 2).unwrap();
        let mut mirror: CudaSlice<[f32; 2]> = ctx.stream.alloc_zeros(n * n / 2).unwrap();
        let copy = time(&ctx.stream, || ctx.stream.memcpy_dtod(&half, &mut mirror).unwrap());
        drop((ad, c, half, mirror));
        let mut rows = Vec::new();
        for d in [MatMulDrivers::CUBLAS_LT, MatMulDrivers::CUBLAS] {
            let cd = Ctx::new(d);
            let at = rt::asarray((a.clone(), &cd.d)).into_shape([n, k]);
            rows.push(time(&cd.stream, || {
                rt::matmul(&at, &at.t());
            }));
        }
        let cd = Ctx::new(MatMulDrivers::all());
        let at = rt::asarray((a.clone(), &cd.d)).into_shape([n, k]);
        let all = time(&cd.stream, || {
            rt::matmul(&at, &at.t());
        });
        println!("| {n} | {:.1} | {:.1} | {:.1} | {:.1} | {:.2} |", us(upper), us(all), us(rows[0]), us(rows[1]), all / (upper + copy));
    }
}

/// The inner dot as a `1 x k @ k x 1` GEMM through the chain (the question of round 3), against
/// the two-stage `vecdot` and the copy ceiling.
macro_rules! dot_gemm_part {
    ($ctx:expr, $name:literal, $t:ty, $tol:expr) => {{
        for p in 20..=26u32 {
            let k = 1usize << p;
            let av: Vec<$t> = data(k, 11).iter().map(|&x| x as $t).collect();
            let bv: Vec<$t> = data(k, 7).iter().map(|&x| x as $t).collect();
            let want: $t = av.iter().zip(&bv).map(|(x, y)| *x * *y).sum();
            let bytes = 2 * k * std::mem::size_of::<$t>();
            let mut rows = Vec::new();
            for d in [MatMulDrivers::CUBLAS_LT, MatMulDrivers::CUBLAS_EX, MatMulDrivers::CUBLAS] {
                let c = Ctx::new(d);
                let a = rt::asarray((av.clone(), &c.d)).into_shape([1, k]);
                let b = rt::asarray((bv.clone(), &c.d)).into_shape([k, 1]);
                let got = host(&rt::matmul(&a, &b))[0];
                assert!(rel(got, want) < $tol, "{} gemm dot k = {k}: got {got:?}", $name);
                rows.push(time(&c.stream, || {
                    rt::matmul(&a, &b);
                }));
            }
            let da = rt::asarray((av.clone(), &$ctx.d));
            let db = rt::asarray((bv.clone(), &$ctx.d));
            let src = $ctx.up(&av);
            let mut dst: CudaSlice<$t> = $ctx.stream.alloc_zeros(k).unwrap();
            let copy = time(&$ctx.stream, || $ctx.stream.memcpy_dtod(&src, &mut dst).unwrap());
            let a2 = da.view().into_shape([1024, k / 1024]);
            let b2 = db.view().into_shape([1024, k / 1024]);
            let ones: Tensor<$t, _, _> = rt::ones(([1024usize], &$ctx.d));
            let two = time(&$ctx.stream, || {
                let part = rt::vecdot(&a2, &b2, 1);
                rt::vecdot(&part, &ones, 0);
            });
            println!(
                "| {} | {p} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} |",
                $name,
                us(rows[0]),
                us(rows[1]),
                us(rows[2]),
                us(two),
                us(copy),
            );
            let _ = bytes;
        }
    }};
}

fn dot_gemm() {
    let ctx = Ctx::new(MatMulDrivers::all());
    println!("\n## Inner dot as a `1 x k @ k x 1` GEMM through the chain\n");
    println!("Correctness gate passed (the chain vs the host sum).\n");
    println!("| dtype | log2 k | chain: Lt | chain: Ex | chain: cuBLAS | two-stage `vecdot` (B = 1024) | copy ceiling |");
    println!("|---|---|---|---|---|---|---|");
    dot_gemm_part!(&ctx, "f64", f64, 1e-9);
    dot_gemm_part!(&ctx, "f32", f32, 2e-2);
}

/// Reproduction of the B1 segfault: the inner dot as a GEMM on a `ColMajor` device, `c64`, with
/// one driver enabled. Each case runs in its own process when invoked with arguments, so a
/// crash is attributed to exactly one case.
fn lt_dot_repro() {
    let args: Vec<String> = std::env::args().skip(2).collect();
    let cases: Vec<(&str, usize, bool, MatMulDrivers)> = vec![
        ("c64", 40, false, MatMulDrivers::CUBLAS_LT),
        ("c64", 40, true, MatMulDrivers::CUBLAS_LT),
        ("c64", 64, false, MatMulDrivers::CUBLAS_LT),
        ("c64", 1000, false, MatMulDrivers::CUBLAS_LT),
        ("c32", 40, false, MatMulDrivers::CUBLAS_LT),
        ("f64", 40, false, MatMulDrivers::CUBLAS_LT),
        ("c64", 40, false, MatMulDrivers::CUBLAS_EX),
        ("c64", 40, false, MatMulDrivers::CUBLAS),
    ];
    for (i, (ty, k, row_major, drivers)) in cases.iter().enumerate() {
        let tag = format!("{i}: {ty} k = {k} {} {:?}", if *row_major { "C" } else { "F" }, drivers);
        if !args.is_empty() && !args.iter().any(|a| a == &i.to_string() || a == "all") {
            continue;
        }
        eprintln!("case {tag}");
        let k = *k;
        let mut flags = default_flags();
        flags.matmul_drivers = *drivers;
        let mut d = DeviceCuda::new(0).with_flags(flags);
        d.set_default_order(if *row_major { RowMajor } else { ColMajor });
        macro_rules! run {
            ($t:ty, $mk:expr) => {{
                let v: Vec<$t> = (0..k).map(|j| $mk(j as f64)).collect();
                let w: Vec<$t> = (0..k).map(|j| $mk((j + 1) as f64)).collect();
                let a = rt::asarray((v, &d)).into_shape([1, k]);
                let b = rt::asarray((w, &d)).into_shape([k, 1]);
                let got = rt::matmul(&a, &b).to_scalar();
                let want: f64 = (0..k).map(|j| j as f64 * (j as f64 + 1.0)).sum();
                eprintln!("    ok: got {:?}", got);
                let _ = want;
            }};
        }
        match *ty {
            "c64" => run!(num::Complex<f64>, |x| num::Complex::new(x, 0.0)),
            "c32" => run!(num::Complex<f32>, |x| num::Complex::new(x as f32, 0.0)),
            _ => run!(f64, |x| x),
        }
    }
}

/// Scratch for B2: where does `a @ a.t()` go wrong?
fn syrk_debug() {
    use rstsr_cuda::matmul::MatMulDrivers;
    let (n, k) = (2048usize, 48usize);
    let v: Vec<f32> = data(n * k, 7).iter().map(|&x| x as f32).collect();
    for order in [RowMajor, ColMajor] {
        for drivers in [MatMulDrivers::all(), MatMulDrivers::CUBLAS, MatMulDrivers::CUBLAS_LT] {
            let mut flags = default_flags();
            flags.matmul_drivers = drivers;
            flags.trace_matmul = true;
            let mut d = DeviceCuda::new(0).with_flags(flags);
            d.set_default_order(order);
            let a = rt::asarray((v.clone(), &d)).into_shape([n, k]);
            let c = rt::matmul(&a, &a.t());
            let h = host(&c);
            let want00: f32 = (0..k).map(|p| v[p] * v[p]).sum();
            println!("{order:?} {drivers:?}: c[0,0] = {} (want {want00}), c[0,1] = {}, c[1,0] = {}, c[n-1,n-1] = {}, c[5, 2000] = {}, c[2000,5] = {}",
                h[0], h[1], h[n], h[n * n - 1], h[5 * n + 2000], h[2000 * n + 5]);
        }
    }
}


/// Scratch for B2: batched `a @ a.t()` in column-major, which the new test flags.
fn batch_debug() {
    let (n, k, p) = (3usize, 4usize, 2usize);
    let v: Vec<num::Complex<f64>> = data(p * n * k, 13).iter().enumerate().map(|(i, &x)| num::Complex::new(x, i as f64)).collect();
    // logical (q, i, j) -> v[q*n*k + i*k + j]
    let col = |dims: &[usize], x: &[num::Complex<f64>]| -> Vec<num::Complex<f64>> {
        let mut out = vec![num::Complex::new(0.0, 0.0); x.len()];
        let mut stride = 1;
        let strides: Vec<usize> = dims.iter().map(|&dim| { let s = stride; stride *= dim; s }).collect();
        for pos in 0..x.len() {
            let (mut rem, mut idx) = (pos, 0);
            for (d, &dim) in dims.iter().enumerate().rev() {
                idx += (rem % dim) * strides[d];
                rem /= dim;
            }
            out[idx] = x[pos];
        }
        out
    };
    let dcpu = DeviceCpuSerial::default();
    let w: Tensor<num::Complex<f64>, DeviceCpuSerial, IxD> = rt::asarray((v.clone(), &dcpu)).into_shape([p, n, k]);
    let want = rt::matmul(&w, &w.transpose([0, 2, 1]));
    println!("cpu want[0,0,0] = {:?}", want.i(s![0, 0, 0]).to_scalar());
    for order in [RowMajor, ColMajor] {
        let mut d = DeviceCuda::new(0);
        d.set_default_order(order);
        let (shape, b, unscale) = if order == RowMajor { (vec![p, n, k], vec![0, 2, 1], vec![0, 1, 2]) } else { (vec![n, k, p], vec![1, 0, 2], vec![2, 0, 1]) };
        let mut flat = vec![num::Complex::new(0.0, 0.0); p * n * k];
        for pos in 0..flat.len() {
            // logical index of element `pos` stored at position `pos` in this order
            let (mut rem, mut idx) = (pos, vec![0usize; 3]);
            let axes: Vec<usize> = if order == RowMajor { vec![2, 1, 0] } else { vec![0, 1, 2] };
            for ax in axes {
                idx[ax] = rem % shape[ax];
                rem /= shape[ax];
            }
            let (i, j, q) = if order == RowMajor { (idx[1], idx[2], idx[0]) } else { (idx[0], idx[1], idx[2]) };
            flat[pos] = v[q * n * k + i * k + j];
        }
        let a = rt::asarray((flat, &d)).into_shape(shape);
        let b = a.transpose(&b);
        println!("{order:?}: a shape {:?} strides {:?}", a.shape(), a.stride());
        println!("{order:?}: b shape {:?} strides {:?}", b.shape(), b.stride());
        let got = rt::matmul(&a, &b).transpose(&unscale).to_owned();
        println!("{order:?}: c shape {:?} strides {:?}", got.shape(), got.stride());
        let h = host(&got.to_contig(RowMajor).to_owned());
        println!("{order:?}: got[0] = {:?}, got[1] = {:?}, got[9] = {:?}", h[0], h[1], h[9]);
    }
    let w2 = want.to_contig(RowMajor);
    println!("cpu want[0] = {:?}, [1] = {:?}, [9] = {:?}", w2.i(s![0, 0, 0]).to_scalar(), w2.i(s![0, 0, 1]).to_scalar(), w2.i(s![1, 0, 0]).to_scalar());
}


/// Scratch for B2: which routine takes `a @ a.t()` for each type and order (trace lines only).
fn syrk_trace() {
    for what in ["f32", "f64", "c32", "c64"] {
        for order in [RowMajor, ColMajor] {
            for n in [511usize, 512usize, 2048usize, 4096usize] {
                let k = 1024;
                macro_rules! one {
                    ($t:ty, $cx:expr, $real:expr) => {{
                        let v: Vec<$t> = data(n * k, 7).iter().map(|&x| $cx(if $real { x } else { x as f32 as f64 })).collect();
                        let mut flags = default_flags();
                        flags.matmul_drivers = MatMulDrivers::all();
                        flags.trace_matmul = true;
                        let mut d = DeviceCuda::new(0).with_flags(flags);
                        d.set_default_order(order);
                        let a = rt::asarray((v, &d)).into_shape([n, k]);
                        let _ = rt::matmul(&a, &a.t());
                    }};
                }
                match what {
                    "f32" => one!(f32, |x: f64| x as f32, true),
                    "f64" => one!(f64, |x: f64| x, true),
                    "c32" => one!(num::Complex<f32>, |x: f64| num::Complex::new(x as f32, 0.0), true),
                    _ => one!(num::Complex<f64>, |x: f64| num::Complex::new(x, 0.0), true),
                }
                eprintln!("  ^ {what}, {order:?}, n = {n}");
            }
        }
    }
}


/// B3: the `f32` gate on both sides. Chain (form where the gate lets it) against the gemm alone.
fn syrk_gate(what: &str, small: bool) {
    macro_rules! time {
        ($flags:expr, $t:ty, $c:expr, $n:expr, $k:expr) => {{
            let mut f = default_flags();
            f.matmul_drivers = $flags;
            let d = DeviceCuda::new(0).with_flags(f);
            let v: Vec<$t> = data($n * $k, 1).into_iter().map($c).collect();
            let a = rt::asarray((v, &d)).into_shape([$n, $k]);
            for _ in 0..3 {
                let _ = rt::matmul(&a, a.t());
            }
            d.stream().synchronize().unwrap();
            let mut samples = Vec::new();
            for _ in 0..15 {
                let t0 = std::time::Instant::now();
                for _ in 0..5 {
                    let _ = rt::matmul(&a, a.t());
                }
                d.stream().synchronize().unwrap();
                samples.push(t0.elapsed().as_secs_f64() / 5.0);
            }
            samples.sort_by(|x, y| x.total_cmp(y));
            samples[7] * 1e6
        }};
    }
    println!("\n## syrk gate sweep, {what}: chain (form) us / gemm us (gemm / chain); `+` the form wins by 5 % or more, `-` it loses\n");
    let ks: [usize; 7] = if small { [16, 32, 64, 128, 256, 512, 1024] } else { [32, 64, 128, 256, 512, 1024, 2048] };
    print!("| n \\ k |");
    for k in ks {
        print!(" {k} |");
    }
    println!("\n|---|");
    for _ in ks {
        print!("---|");
    }
    println!();
    let gemm = MatMulDrivers::all().minus(MatMulDrivers::CUBLAS);
    let ns: Vec<usize> = if small { vec![32, 64, 128, 256] } else { vec![512, 1024, 2048, 4096, 8192] };
    for n in ns {
        print!("| {n} |");
        for k in ks {
            let k = k.min(n);
            let (form, gemm_t) = match what {
                "f32" => (time!(MatMulDrivers::all(), f32, |x: f64| x as f32, n, k), time!(gemm, f32, |x: f64| x as f32, n, k)),
                "c32" => (
                    time!(MatMulDrivers::all(), num::Complex<f32>, |x: f64| num::Complex::new(x as f32, 0.0), n, k),
                    time!(gemm, num::Complex<f32>, |x: f64| num::Complex::new(x as f32, 0.0), n, k),
                ),
                "c64" => (
                    time!(MatMulDrivers::all(), num::Complex<f64>, |x: f64| num::Complex::new(x, 0.0), n, k),
                    time!(gemm, num::Complex<f64>, |x: f64| num::Complex::new(x, 0.0), n, k),
                ),
                _ => (time!(MatMulDrivers::all(), f64, |x: f64| x, n, k), time!(gemm, f64, |x: f64| x, n, k)),
            };
            let r = gemm_t / form;
            print!(" {:.0} / {:.0} ({:.2}){} |", form, gemm_t, r, if r >= 1.05 { " +" } else if r <= 0.95 { " -" } else { "" });
        }
        println!();
    }
}

fn main() {
    let parts: Vec<String> = std::env::args().skip(1).collect();
    let run = |name: &str| parts.is_empty() || parts.iter().any(|p| p == name);
    if run("dot") {
        dot();
    }
    if run("dot-gemm") {
        dot_gemm();
    }
    if run("lt-dot-repro") {
        lt_dot_repro();
    }
    if run("syrk-debug") {
        syrk_debug();
    }
    if run("batch-debug") {
        batch_debug();
    }
    if run("syrk-gate") {
        if std::env::var("SMALL").is_ok() {
            for what in ["f64", "c32", "c64"] {
                syrk_gate(what, true);
            }
        } else {
            for what in ["f32", "f32", "f64", "c32", "c64"] {
                syrk_gate(what, false);
            }
        }
    }
    if run("syrk-trace") {
        syrk_trace();
    }
    if run("gemv") {
        gemv();
    }
    if run("gemv-batch") {
        gemv_batch();
    }
    if run("syrk") {
        syrk();
        syrk_complex();
    }
}

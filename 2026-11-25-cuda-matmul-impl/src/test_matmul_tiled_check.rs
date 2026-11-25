//! Matmul driver check: the layout matrix against the CPU device, one module per `(dtype, driver)`.
//!
//! Complements the in-tree `test_matmul_tiled.rs` / `test_matmul_naive.rs`. Every operand is
//! integer-valued, so every product and partial sum is exact in every dtype and the comparison is
//! exact — this file detects *addressing* errors, not rounding. The driver is forced through the
//! device flags (`CudaRuntimeFlags::matmul_drivers`): each module enables one driver alone.
//!
//! Run with `--nocapture`: each case prints `ok`, `REFUSED <error>` (raised by name, never a wrong
//! answer) or fails the test on a mismatch.

use rstsr::prelude::*;
use rstsr_cuda::funcs;
use rstsr_cuda::prelude_dev::*;

/// An operand value of the check: real types take the integer as is; complex takes half-integers
/// with a small imaginary part, all exactly representable in `f32`.
trait CheckData {
    fn of(x: f64) -> Self;
}

macro_rules! impl_check_data {
    ($($t:ty),*) => {$(
        impl CheckData for $t {
            fn of(x: f64) -> Self {
                x as $t
            }
        }
    )*};
}
impl_check_data!(f32, f64, i32, i64);

impl CheckData for num::Complex<f64> {
    fn of(x: f64) -> Self {
        num::Complex::new(x / 2.0, (x as i64 % 5) as f64 - 2.0)
    }
}

impl CheckData for num::Complex<f32> {
    fn of(x: f64) -> Self {
        num::Complex::new((x / 2.0) as f32, ((x as i64 % 5) as f32) - 2.0)
    }
}

macro_rules! require_cuda {
    () => {
        let available = std::panic::catch_unwind(funcs::is_available).unwrap_or(false);
        if !available {
            eprintln!("skipping: no CUDA device visible");
            return;
        }
    };
}

/// Integer-valued data in `[-4, 4]`.
fn exact_data(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) % 9) as f64 - 4.0
        })
        .collect()
}

/// The layout matrix for dtype `$t` with the driver `$drivers` forced alone. `$refuses` says
/// whether the two documented tiled refusals are errors for this driver (true) or answers (false).
macro_rules! check_mod {
    ($t:ty, $suffix:ident, $drivers:expr, $refuses:expr) => {
        mod $suffix {
            use super::*;

            const DRV: MatMulDrivers = $drivers;
            const REFUSES: bool = $refuses;

            /// A device on which `DRV` is the whole matmul enable set.
            fn dev() -> DeviceCuda {
                let mut flags = CudaRuntimeFlags::default();
                flags.matmul_drivers = DRV;
                DeviceCuda::new(0).with_flags(flags)
            }

            type T = $t;
            type Tc = Tensor<T, DeviceCpuSerial, IxD>;

            fn data(n: usize, seed: u64) -> Vec<T> {
                exact_data(n, seed).into_iter().map(CheckData::of).collect()
            }

            fn cpu(values: Vec<T>, shape: &[usize]) -> Tc {
                let dcpu = DeviceCpuSerial::default();
                rt::asarray((values, &dcpu)).into_shape(shape.to_vec())
            }

            fn host<'a, Tr, B, D>(t: &TensorAny<Tr, T, B, D>) -> Vec<T>
            where
                Tr: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
                D: DimAPI,
                B: DeviceAPI<T> + DeviceToCpuAPI<T> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, D, D>,
            {
                let contig = t.to_contig(RowMajor);
                DeviceToCpuAPI::<T>::to_cpu_vec(contig.storage()).rstsr_unwrap()
            }

            /// Compare two host vectors; panics with the first differing element.
            fn expect_eq(what: &str, got: &[T], want: &[T])
            where
                T: PartialEq + core::fmt::Debug,
            {
                assert_eq!(got.len(), want.len(), "{what}: size");
                if let Some(i) = (0..got.len()).find(|&i| got[i] != want[i]) {
                    panic!("{what}: first mismatch at flat index {i}: got {:?}, want {:?}", got[i], want[i]);
                }
                eprintln!("  ok       {what}");
            }

            /// `a @ b` on both devices: exact match, or a refusal reported by name.
            macro_rules! check {
                ($what:expr, $a:expr, $b:expr) => {{
                    let what: &str = $what;
                    let want = rt::matmul_f($a, $b).rstsr_unwrap();
                    let want = host(&want);
                    let d = dev();
                    let (a, b) = ($a.to_device(&d), $b.to_device(&d));
                    match rt::matmul_f(&a, &b) {
                        Ok(got) => expect_eq(what, &host(&got), &want),
                        Err(e) => eprintln!("  REFUSED  {what}: {}", short(&e)),
                    }
                }};
            }

            /// `c = alpha * a @ b + beta * c` on both devices; `c` is re-viewed through `$idx`.
            macro_rules! check_from {
                ($what:expr, $c:expr, $idx:expr, $a:expr, $b:expr, $alpha:expr, $beta:expr) => {{
                    let what: &str = $what;
                    let d = dev();
                    let (mut c, mut cg) = ($c.clone(), $c.to_device(&d));
                    let (ag, bg) = ($a.to_device(&d), $b.to_device(&d));
                    rt::matmul_from_f(&mut c.slice_mut($idx), $a, $b, $alpha, $beta).rstsr_unwrap();
                    match rt::matmul_from_f(&mut cg.slice_mut($idx), &ag, &bg, $alpha, $beta) {
                        Ok(()) => {
                            let (c, cg) = (c.to_contig(RowMajor), cg.to_contig(RowMajor));
                            expect_eq(what, &host(&cg), &host(&c))
                        },
                        Err(e) => eprintln!("  REFUSED  {what}: {}", short(&e)),
                    }
                }};
            }

            fn short(e: &Error) -> String {
                let s = format!("{e:?}");
                s.chars().filter(|&c| c != '\n').take(160).collect()
            }

            
            #[test]
            fn contiguous_and_tile_edges() {
                require_cuda!();
                for &(m, k, n) in &[(64usize, 64usize, 64usize), (65, 65, 65), (1, 1, 1), (1, 37, 1), (17, 33, 5), (130, 70, 129)] {
                    let a = cpu(data(m * k, 1), &[m, k]);
                    let b = cpu(data(k * n, 2), &[k, n]);
                    check!(&format!("contiguous {m}x{k}x{n}"), &a, &b);
                }
            }

            #[test]
            fn transposed_operands() {
                require_cuda!();
                let (m, k, n) = (40usize, 24usize, 20usize);
                let a0 = cpu(data(k * m, 3), &[k, m]); // a0.t() is [m, k], m-contiguous
                let b0 = cpu(data(n * k, 4), &[n, k]); // b0.t() is [k, n], k-contiguous
                let b = cpu(data(k * n, 5), &[k, n]);
                let a = cpu(data(m * k, 6), &[m, k]);
                check!("a transposed", &a0.t(), &b);
                check!("b transposed", &a, &b0.t());
                check!("both transposed", &a0.t(), &b0.t());
            }

            #[test]
            fn stepped_operands() {
                require_cuda!();
                let (m, k, n) = (20usize, 16usize, 12usize);
                // Outer-axis step: the other axis keeps its unit stride.
                let a_m0 = cpu(data(2 * m * k, 7), &[2 * m, k]);
                let a_m = a_m0.i(s![slice!(None, None, 2), ..]);
                let b_k0 = cpu(data(2 * k * n, 8), &[2 * k, n]);
                let b_k = b_k0.i(s![slice!(None, None, 2), ..]);
                let a = cpu(data(m * k, 9), &[m, k]);
                let b = cpu(data(k * n, 10), &[k, n]);
                check!("a stepped on m", &a_m, &b);
                check!("b stepped on k", &a, &b_k);
                check!("both stepped on outer axes", &a_m, &b_k);
                // Inner-axis step: no unit-stride axis left in that operand.
                let a_k0 = cpu(data(m * 2 * k, 11), &[m, 2 * k]);
                let a_k = a_k0.i(s![.., slice!(None, None, 2)]);
                let b_n0 = cpu(data(k * 2 * n, 12), &[k, 2 * n]);
                let b_n = b_n0.i(s![.., slice!(None, None, 2)]);
                check!("a stepped on k", &a_k, &b);
                check!("b stepped on n", &a, &b_n);
            }

            #[test]
            fn negative_strides() {
                require_cuda!();
                let (m, k, n) = (20usize, 16usize, 12usize);
                let a = cpu(data(m * k, 13), &[m, k]);
                let b = cpu(data(k * n, 14), &[k, n]);
                check!("a flipped on m", &a.flip(0), &b);
                check!("a flipped on k", &a.flip(1), &b);
                check!("b flipped on k", &a, &b.flip(0));
                check!("b flipped on n", &a, &b.flip(1));
                check!("a.flip(1) @ b.flip(0)", &a.flip(1), &b.flip(0));
                let a0 = cpu(data(k * m, 15), &[k, m]);
                check!("a0.transpose(1,0).flip(0) @ b (unit stride -1)", &a0.transpose([1, 0]).flip(0), &b);
            }

            #[test]
            fn zero_strides() {
                require_cuda!();
                let (m, k, n) = (20usize, 16usize, 12usize);
                let a = cpu(data(m * k, 16), &[m, k]);
                let b = cpu(data(k * n, 17), &[k, n]);
                let a_row0 = cpu(data(k, 18), &[1, k]);
                let b_col0 = cpu(data(k, 19), &[k, 1]);
                let a_const0 = cpu(data(1, 20), &[1, 1]);
                eprintln!("  (broadcast a rows strides {:?}, b cols strides {:?})", a_row0.broadcast_to(vec![m, k]).stride(), b_col0.broadcast_to(vec![k, n]).stride());
                check!("a rows broadcast", &a_row0.broadcast_to(vec![m, k]), &b);
                check!("b cols broadcast", &a, &b_col0.broadcast_to(vec![k, n]));
                check!("both broadcast", &a_row0.broadcast_to(vec![m, k]), &b_col0.broadcast_to(vec![k, n]));
                check!("a scalar broadcast", &a_const0.broadcast_to(vec![m, k]), &b);
            }

            #[test]
            fn batched_and_broadcast_batch() {
                require_cuda!();
                let (p, q, m, k, n) = (3usize, 2usize, 12usize, 16usize, 8usize);
                let a = cpu(data(p * m * k, 21), &[p, m, k]);
                let b = cpu(data(p * k * n, 22), &[p, k, n]);
                let a1 = cpu(data(m * k, 23), &[m, k]);
                let b1 = cpu(data(k * n, 24), &[k, n]);
                check!("batched [p]", &a, &b);
                check!("a unbatched [m,k] @ [p,k,n]", &a1, &b);
                check!("b unbatched [p,m,k] @ [k,n]", &a, &b1);
                check!("rank-4 batched [p,q]", &cpu(data(p * q * m * k, 25), &[p, q, m, k]), &cpu(data(p * q * k * n, 26), &[p, q, k, n]));
                check!("batch broadcast [p,1] vs [1,q]", &cpu(data(p * m * k, 27), &[p, 1, m, k]), &cpu(data(q * k * n, 28), &[1, q, k, n]));
                check!("explicit [1,m,k] @ [p,k,n]", &a1.i(s![None, .., ..]), &b);
                // A transposed batch block: one step per batch index still.
                let base = cpu(data(m * p * k, 29), &[m, p, k]);
                check!("batch block not leading", &base.transpose([1, 0, 2]), &b);
                // Stepped batch.
                let a20 = cpu(data(2 * p * m * k, 30), &[2 * p, m, k]);
                let a2 = a20.i(s![slice!(None, None, 2), .., ..]);
                check!("stepped batch", &a2, &b);
            }

            #[test]
            fn col_major() {
                require_cuda!();
                let mut d = dev();
                d.set_default_order(ColMajor);
                let mut dcpu = DeviceCpuSerial::default();
                dcpu.set_default_order(ColMajor);
                let (p, m, k, n) = (3usize, 20usize, 12usize, 32usize);
                let mk = |n: usize, seed: u64, shape: &[usize]| -> Tc { rt::asarray((data(n, seed), &dcpu)).into_shape(shape.to_vec()) };
                // Matrix dims are the first two axes; trailing axes broadcast.
                let a3 = mk(m * k * p, 40, &[m, k, p]);
                let b3 = mk(k * n * p, 41, &[k, n, p]);
                let d3 = mk(m * n * p, 42, &[m, n, p]);
                let e3 = mk(n * k * p, 43, &[n, k, p]);
                let b1 = mk(k * n, 44, &[k, n]);
                eprintln!("  (col-major a strides {:?})", a3.stride());
                macro_rules! col_check {
                    ($what:expr, $a:expr, $b:expr) => {{
                        let want = rt::matmul_f($a, $b).rstsr_unwrap();
                        let (ag, bg) = ($a.to_device(&d), $b.to_device(&d));
                        match rt::matmul_f(&ag, &bg) {
                            Ok(got) => expect_eq($what, &host(&got), &host(&want)),
                            Err(e) => eprintln!("  REFUSED  {}: {}", $what, short(&e)),
                        }
                    }};
                }
                col_check!("col batched [m,k,p] @ [k,n,p]", &a3, &b3);
                col_check!("col a unbatched [m,k] @ [k,n,p]", &a3.i(s![.., .., 0]), &b3);
                col_check!("col transposed a: [k,m,p] @ [m,n,p]", &a3.transpose([1, 0, 2]), &d3);
                                col_check!("col transposed b: [m,k,p] @ [k,n,p] from [n,k,p]", &a3, &e3.transpose([1, 0, 2]));
                col_check!("col b unbatched [m,k,p] @ [k,n]", &a3, &b1);
                col_check!("col a unbatched-batch [m,k] @ [k,n,p]", &a3.i(s![.., .., 0]), &b3);
            }

            #[test]
            fn strided_output_matmul_from() {
                require_cuda!();
                let (m, k, n) = (20usize, 16usize, 12usize);
                let a = cpu(data(m * k, 50), &[m, k]);
                let b = cpu(data(k * n, 51), &[k, n]);
                let c2 = cpu(data(m * 2 * n, 52), &[m, 2 * n]);
                let c = cpu(data(m * n, 53), &[m, n]);
                check_from!("c stepped on n, beta != 0", &c2, s![.., slice!(None, None, 2)], &a, &b, CheckData::of(2.0), CheckData::of(3.0));
                check_from!("c stepped on n, beta = 0", &c2, s![.., slice!(None, None, 2)], &a, &b, CheckData::of(1.0), CheckData::of(0.0));
                check_from!("c flipped on m, beta != 0", &c, s![slice!(None, None, -1), ..], &a, &b, CheckData::of(2.0), CheckData::of(3.0));
                check_from!("c flipped on n, beta != 0", &c, s![.., slice!(None, None, -1)], &a, &b, CheckData::of(2.0), CheckData::of(3.0));
                check_from!("c contiguous, beta != 0", &c, s![.., ..], &a, &b, CheckData::of(2.0), CheckData::of(3.0));
            }

            /// The two layouts the tiled kernel refuses: errors there, answers on a driver that
            /// takes them.
            #[test]
            fn tiled_refusals() {
                require_cuda!();
                let d = dev();
                let (m, k, n) = (20usize, 16usize, 12usize);
                let b = cpu(data(k * n, 60), &[
                    k,
                    n,
                ]);
                let a2 = cpu(data(2 * m * 2 * k, 61), &[2 * m, 2 * k]);
                let a_step = a2.i(s![slice!(None, None, 2), slice!(None, None, 2)]);
                assert_eq!(a_step.stride(), &[4 * k as isize, 2]);
                match rt::matmul_f(&a_step.to_device(&d), &b.to_device(&d)) {
                    Err(err) => {
                        eprintln!("  [refusal: no unit-stride axis] {}", short(&err));
                        assert!(REFUSES, "this driver was expected to answer");
                        assert!(format!("{err:?}").contains("no unit-stride axis"));
                    },
                    Ok(got) => {
                        check!("no unit-stride operand (tiled refuses it)", &a_step, &b);
                        let _ = got;
                    },
                }

                // Non-affine batch: base [q, m, p, k] transposed to [p, q, m, k].
                let (p, q) = (3usize, 2usize);
                let base = cpu(data(q * m * p * k, 62), &[q, m, p, k]);
                let a_np = base.transpose([2, 0, 1, 3]);
                assert_eq!(a_np.shape(), &[p, q, m, k]);
                let b_np = cpu(data(p * q * k * n, 63), &[p, q, k, n]);
                match rt::matmul_f(&a_np.to_device(&d), &b_np.to_device(&d)) {
                    Err(err) => {
                        eprintln!("  [refusal: non-affine batch] {}", short(&err));
                        assert!(REFUSES, "this driver was expected to answer");
                        assert!(format!("{err:?}").contains("linear run"), "{err:?}");
                    },
                    Ok(_) => check!("non-affine batch (tiled refuses it)", &a_np, &b_np),
                }
            }
        }
    };
}

// tiled: the two refusals are errors
check_mod!(f64, tiled_f64, MatMulDrivers::TILED, true);
check_mod!(f32, tiled_f32, MatMulDrivers::TILED, true);
// naive: every layout is an answer
check_mod!(f64, naive_f64, MatMulDrivers::NAIVE, false);
check_mod!(f32, naive_f32, MatMulDrivers::NAIVE, false);
check_mod!(i32, naive_i32, MatMulDrivers::NAIVE, false);
check_mod!(i64, naive_i64, MatMulDrivers::NAIVE, false);
// complex: exact binary fractions (halves), served by naive only
check_mod!(num::Complex<f64>, naive_c64, MatMulDrivers::NAIVE, false);
check_mod!(num::Complex<f32>, naive_c32, MatMulDrivers::NAIVE, false);

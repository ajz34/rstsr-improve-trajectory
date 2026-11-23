//! P0 harness for the GEMV-dispatch campaign (`2026-11-18-gemv-dispatch`).
//!
//! Measures the **current** rstsr matmul path for row-vector / column-vector
//! products on `DeviceOpenBLAS` and `DeviceFaer`, anchored by raw
//! `cblas_?gemv`. No rstsr code is modified by this crate; see `README.md` for
//! the measurement contract and `DECISIONS.md` for what the table decides.

pub use rstsr_core::prelude::rt::*;
pub use rstsr_core::prelude_dev::DeviceFaer;
pub use rstsr_openblas::DeviceOpenBLAS;

pub use rstsr_openblas_ffi::cblas;

/// Thread count used for every measurement (physical cores of the bench box).
pub const NTHREADS: usize = 16;

/// Matrix cells `(rows, cols, label)`.
///
/// For `a` of shape `(rows, cols)`:
/// - `matvec` is `a @ xq -> (rows)` with `xq` of length `cols`;
/// - `vecmat` is `xp @ a -> (cols)` with `xp` of length `rows`.
pub const CELLS: &[(usize, usize, &str)] = &[
    (1024, 1024, "1024x1024"),
    (4096, 4096, "4096x4096"),
    (8192, 8192, "8192x8192"),
    (65536, 64, "65536x64"),
    (256, 452676, "256x452676"),
    (452676, 256, "452676x256"),
];

/// Raw cblas gemv, column-major, unit increments, `y = op(a) x`.
///
/// `a` is the column-major view of the row-major fixture: shape `(n, m)` with
/// `lda = n`.
pub trait CblasGemv: Copy {
    /// `trans = false`: `y(m) = a(m,n) x(n)`; `trans = true`:
    /// `y(n) = a(m,n)^T x(m)`.
    unsafe fn cblas_gemv(trans: bool, m: usize, n: usize, a: *const Self, lda: usize, x: *const Self, y: *mut Self);
}

impl CblasGemv for f64 {
    unsafe fn cblas_gemv(trans: bool, m: usize, n: usize, a: *const f64, lda: usize, x: *const f64, y: *mut f64) {
        unsafe {
            cblas::cblas_dgemv(
                cblas::CBLAS_LAYOUT::CblasColMajor,
                if trans { cblas::CBLAS_TRANSPOSE::CblasTrans } else { cblas::CBLAS_TRANSPOSE::CblasNoTrans },
                m as cblas::blas_int,
                n as cblas::blas_int,
                1.0,
                a,
                lda as cblas::blas_int,
                x,
                1,
                0.0,
                y,
                1,
            )
        }
    }
}

impl CblasGemv for f32 {
    unsafe fn cblas_gemv(trans: bool, m: usize, n: usize, a: *const f32, lda: usize, x: *const f32, y: *mut f32) {
        unsafe {
            cblas::cblas_sgemv(
                cblas::CBLAS_LAYOUT::CblasColMajor,
                if trans { cblas::CBLAS_TRANSPOSE::CblasTrans } else { cblas::CBLAS_TRANSPOSE::CblasNoTrans },
                m as cblas::blas_int,
                n as cblas::blas_int,
                1.0,
                a,
                lda as cblas::blas_int,
                x,
                1,
                0.0,
                y,
                1,
            )
        }
    }
}

/// Panic unless the process is pinned to [`NTHREADS`] rayon threads.
pub fn assert_threads() {
    assert_eq!(
        std::env::var("RAYON_NUM_THREADS").as_deref(),
        Ok("16"),
        "set RAYON_NUM_THREADS=16 for the P0 measurements"
    );
}

/// Pin the BLAS thread count through the mechanism the linked library honours.
///
/// OpenMP-built distributions ignore `OPENBLAS_NUM_THREADS`; both routes are
/// covered here (requires the `openmp` feature of `rstsr-openblas`, which links
/// `gomp`, i.e. build with `RSTSR_DEV=1`).
pub fn pin_blas_threads(n: usize) {
    unsafe {
        let parallel = cblas::openblas_get_parallel() as u32;
        if parallel == cblas::OPENBLAS_OPENMP {
            cblas::omp_set_num_threads(n as i32);
        } else {
            cblas::openblas_set_num_threads(n as i32);
        }
    }
}

/// Record the linked OpenBLAS identity for the README.
pub fn blas_config() -> String {
    unsafe {
        let cfg = cblas::openblas_get_config();
        let core = cblas::openblas_get_corename();
        let s = |p: *const std::os::raw::c_char| {
            if p.is_null() { String::new() } else { std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned() }
        };
        format!("{} | core={}", s(cfg), s(core))
    }
}

/// Max relative difference `|a - b| / max(1, |b|)` over the slices.
pub fn max_rel_diff<A, B>(a: &[A], b: &[B]) -> f64
where
    A: Copy + Into<f64>,
    B: Copy + Into<f64>,
{
    assert_eq!(a.len(), b.len());
    let mut worst = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let x: f64 = (*x).into();
        let y: f64 = (*y).into();
        let rel = (x - y).abs() / y.abs().max(1.0);
        if rel > worst {
            worst = rel;
        }
    }
    worst
}

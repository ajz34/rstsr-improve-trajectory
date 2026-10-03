//! DLPack data types and the Rust element types they map to.

use dlpack_ffi::{DLDataType, DLDataTypeCode};
use rstsr_common::error::Result;

use crate::err;

pub use half::{bf16, f16};
pub use num_complex::Complex;

/// Complex with two `f32` components (DLPack `kDLComplex`, 64 bits).
pub type Complex32 = Complex<f32>;
/// Complex with two `f64` components (DLPack `kDLComplex`, 128 bits).
pub type Complex64 = Complex<f64>;

/// Rust element type expressible as a DLPack [`DLDataType`] with a single lane.
///
/// Implemented for the numeric dtypes rstsr supports. `i128`/`u128`/`bf16` are
/// valid DLPack but stock NumPy cannot consume them (see the crate docs).
pub trait DlpackDtype: Copy + 'static {
    /// The DLPack data type of `Self`.
    const DLTYPE: DLDataType;

    /// Check that `dtype` is exactly `Self::DLTYPE`.
    fn from_dltype(dtype: DLDataType) -> Result<()> {
        let expected = Self::DLTYPE;
        let same = dtype.code == expected.code && dtype.bits == expected.bits && dtype.lanes == expected.lanes;
        if same {
            return Ok(());
        }
        Err(err::invalid_value(format!(
            "DLPack dtype {{code: {}, bits: {}, lanes: {}}} does not match the requested Rust type {{code: {}, bits: {}, lanes: {}}}",
            dtype.code, dtype.bits, dtype.lanes, expected.code, expected.bits, expected.lanes
        )))
    }
}

/// Error for a DLPack dtype the crate cannot represent.
#[doc(hidden)]
pub fn unsupported_dtype_error(dtype: DLDataType) -> rstsr_common::error::Error {
    err::unimplemented(format!(
        "unsupported DLPack dtype {{code: {}, bits: {}, lanes: {}}}",
        dtype.code, dtype.bits, dtype.lanes
    ))
}

macro_rules! impl_dlpack_dtype {
    ($code:expr => $($T:ty),* $(,)?) => {
        $(
            impl DlpackDtype for $T {
                const DLTYPE: DLDataType = DLDataType {
                    code: $code,
                    bits: (core::mem::size_of::<$T>() * 8) as u8,
                    lanes: 1,
                };
            }
        )*
    };
}

const CODE_INT: u8 = DLDataTypeCode::kDLInt.0 as u8;
const CODE_UINT: u8 = DLDataTypeCode::kDLUInt.0 as u8;
const CODE_FLOAT: u8 = DLDataTypeCode::kDLFloat.0 as u8;
const CODE_BFLOAT: u8 = DLDataTypeCode::kDLBfloat.0 as u8;
const CODE_COMPLEX: u8 = DLDataTypeCode::kDLComplex.0 as u8;
const CODE_BOOL: u8 = DLDataTypeCode::kDLBool.0 as u8;

impl_dlpack_dtype!(CODE_INT => i8, i16, i32, i64, i128);
impl_dlpack_dtype!(CODE_UINT => u8, u16, u32, u64, u128);
impl_dlpack_dtype!(CODE_FLOAT => f16, f32, f64);
impl_dlpack_dtype!(CODE_BFLOAT => bf16);
impl_dlpack_dtype!(CODE_COMPLEX => Complex32, Complex64);
impl_dlpack_dtype!(CODE_BOOL => bool);

/// Dispatch a body over the Rust type matching a DLPack data type.
///
/// Evaluates to `Result<R>`: `Ok(body)` for every supported dtype, `Err` for a
/// dtype the crate cannot represent (vector lanes, sub-byte floats, opaque
/// handles). The body must be valid for each `T`; the bound `T: DlpackDtype`
/// holds by construction.
///
/// ```ignore
/// let value: f64 = with_dlpack_dtype!(dtype, |T| {
///     unsafe { from_dlpack_versioned_f::<T, DeviceCpuSerial, IxD>(ptr) }?.into_raw_vec_sum()
/// })?;
/// ```
#[macro_export]
macro_rules! with_dlpack_dtype {
    ($dtype:expr, |$T:ident| $body:expr $(,)?) => {{
        let dtype: $crate::dlpack_ffi::DLDataType = $dtype;
        match (dtype.code, dtype.bits, dtype.lanes) {
            (0, 8, 1) => { type $T = i8; ::core::result::Result::Ok($body) }
            (0, 16, 1) => { type $T = i16; ::core::result::Result::Ok($body) }
            (0, 32, 1) => { type $T = i32; ::core::result::Result::Ok($body) }
            (0, 64, 1) => { type $T = i64; ::core::result::Result::Ok($body) }
            (0, 128, 1) => { type $T = i128; ::core::result::Result::Ok($body) }
            (1, 8, 1) => { type $T = u8; ::core::result::Result::Ok($body) }
            (1, 16, 1) => { type $T = u16; ::core::result::Result::Ok($body) }
            (1, 32, 1) => { type $T = u32; ::core::result::Result::Ok($body) }
            (1, 64, 1) => { type $T = u64; ::core::result::Result::Ok($body) }
            (1, 128, 1) => { type $T = u128; ::core::result::Result::Ok($body) }
            (2, 16, 1) => { type $T = $crate::dtype::f16; ::core::result::Result::Ok($body) }
            (2, 32, 1) => { type $T = f32; ::core::result::Result::Ok($body) }
            (2, 64, 1) => { type $T = f64; ::core::result::Result::Ok($body) }
            (4, 16, 1) => { type $T = $crate::dtype::bf16; ::core::result::Result::Ok($body) }
            (5, 64, 1) => { type $T = $crate::dtype::Complex32; ::core::result::Result::Ok($body) }
            (5, 128, 1) => { type $T = $crate::dtype::Complex64; ::core::result::Result::Ok($body) }
            (6, 8, 1) => { type $T = bool; ::core::result::Result::Ok($body) }
            _ => ::core::result::Result::Err($crate::dtype::unsupported_dtype_error(dtype)),
        }
    }};
}

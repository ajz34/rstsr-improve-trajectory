//! Error constructors; DLPack failures are reported through rstsr's error type.

use rstsr_common::error::{Error, RSTSRError, rstsr_backtrace};

pub(crate) fn dlpack_error(inner: RSTSRError) -> Error {
    Error { inner, backtrace: rstsr_backtrace() }
}

pub(crate) fn invalid_value(msg: impl Into<String>) -> Error {
    dlpack_error(RSTSRError::InvalidValue(msg.into()))
}

pub(crate) fn unimplemented(msg: impl Into<String>) -> Error {
    dlpack_error(RSTSRError::UnImplemented(msg.into()))
}

pub(crate) fn value_out_of_range(msg: impl Into<String>) -> Error {
    dlpack_error(RSTSRError::ValueOutOfRange(msg.into()))
}

//! Tensor dtype conversion: [`astype`] / [`into_astype`].
//!
//! A dtype change always transforms every element. The borrowing rule
//! mirrors [`reshape`]: if the target dtype equals the input dtype (`T ==
//! TOut`) the result borrows the input as a view ([`TensorCow`] borrowed
//! variant, no copy); otherwise a converted copy is created (owned variant).
//! This is NumPy's `astype(copy=False)` semantics.
//!
//! The name `astype` deliberately steps outside RSTSR's `to_` (zero-copy
//! view) / `into_` (consumes ownership) naming convention: it matches
//! NumPy/array-api's `astype`, and neither convention fits a function that
//! conditionally copies. [`into_astype`] is the consuming variant.

use crate::prelude_dev::*;

/* #region cast trait */

/// Per-dtype-pair cast implementation for [`astype`].
///
/// Implemented for `TensorAny<R, TI, B, D>` per (`TI`, `TO`) dtype pair that
/// has a scalar [`DTypeCastAPI`] conversion, plus a same-type blanket impl
/// (view, no copy). Use the free functions [`astype_f`]/[`astype`] instead
/// of calling this trait directly.
pub trait TensorCastAPI<'a, R, T, TOut, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    B: DeviceRawAPI<T> + DeviceRawAPI<TOut>,
    Self: Sized,
{
    #[doc(hidden)]
    fn astype_impl(tensor: &'a TensorAny<R, T, B, D>) -> Result<TensorCow<'a, TOut, B, D>>;
}

/// Same-dtype cast: return a borrowed view, no copy.
impl<'a, R, T, B, D> TensorCastAPI<'a, R, T, T, B, D> for TensorAny<R, T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    B: DeviceAPI<T>,
{
    fn astype_impl(tensor: &'a TensorAny<R, T, B, D>) -> Result<TensorCow<'a, T, B, D>> {
        Ok(tensor.view().into_cow())
    }
}

macro_rules! impl_tensor_cast {
    ($TI: ty, $TO: ty) => {
        impl<'a, R, B, D> TensorCastAPI<'a, R, $TI, $TO, B, D> for TensorAny<R, $TI, B, D>
        where
            R: DataAPI<Data = <B as DeviceRawAPI<$TI>>::Raw>,
            D: DimAPI,
            B: DeviceAPI<$TI>
                + DeviceAPI<$TO>
                + DeviceRawAPI<MaybeUninit<$TO>>
                + DeviceCreationAnyAPI<$TO>
                + OpAssignArbitaryAPI<$TO, IxD, D, $TI>,
        {
            fn astype_impl(tensor: &'a TensorAny<R, $TI, B, D>) -> Result<TensorCow<'a, $TO, B, D>> {
                // convert-copy: contiguous output in the device default order,
                // casting element-wise inside the assign kernel (no layout quirks
                // such as broadcast strides are carried over)
                let device = tensor.device();
                let shape: Vec<usize> = tensor.layout().shape().as_ref().to_vec();
                let layout_new = match device.default_order() {
                    RowMajor => shape.new_c_contig(None),
                    ColMajor => shape.new_f_contig(None),
                };
                let mut storage_new = device.uninit_impl(layout_new.size())?;
                device.assign_arbitary_uninit(
                    storage_new.raw_mut(),
                    &layout_new,
                    tensor.raw(),
                    &tensor.layout().to_dim()?,
                )?;
                // SAFETY: `assign_arbitary_uninit` above filled the fresh storage over
                // the contiguous `layout_new` completely.
                let storage_new = unsafe { B::assume_init_impl(storage_new)? };
                // SAFETY: fresh contiguous storage, fully written above; layout and
                // storage match by construction.
                unsafe { Ok(TensorBase::new_unchecked(storage_new, layout_new.into_dim::<D>()?).into_cow()) }
            }
        }
    };
}

/* #endregion */

/* #region cross-type cast impls (mirror rstsr-dtype-traits DTypeCastAPI) */

impl_tensor_cast!(i8, i32);
impl_tensor_cast!(i8, i64);
impl_tensor_cast!(i8, u8);
impl_tensor_cast!(i8, u16);
impl_tensor_cast!(i8, u32);
impl_tensor_cast!(i8, u64);
impl_tensor_cast!(i8, f32);
impl_tensor_cast!(i8, f64);
impl_tensor_cast!(i16, i8);
impl_tensor_cast!(i16, i32);
impl_tensor_cast!(i16, i64);
impl_tensor_cast!(i16, u8);
impl_tensor_cast!(i16, u16);
impl_tensor_cast!(i16, u32);
impl_tensor_cast!(i16, u64);
impl_tensor_cast!(i16, f32);
impl_tensor_cast!(i16, f64);
impl_tensor_cast!(i32, i8);
impl_tensor_cast!(i32, i16);
impl_tensor_cast!(i32, i64);
impl_tensor_cast!(i32, u8);
impl_tensor_cast!(i32, u16);
impl_tensor_cast!(i32, u32);
impl_tensor_cast!(i32, u64);
impl_tensor_cast!(i32, f32);
impl_tensor_cast!(i32, f64);
impl_tensor_cast!(i64, i8);
impl_tensor_cast!(i64, i16);
impl_tensor_cast!(i64, i32);
impl_tensor_cast!(i64, u8);
impl_tensor_cast!(i64, u16);
impl_tensor_cast!(i64, u32);
impl_tensor_cast!(i64, u64);
impl_tensor_cast!(i64, f32);
impl_tensor_cast!(i64, f64);
impl_tensor_cast!(u8, i8);
impl_tensor_cast!(u8, i16);
impl_tensor_cast!(u8, i32);
impl_tensor_cast!(u8, i64);
impl_tensor_cast!(u8, u16);
impl_tensor_cast!(u8, u32);
impl_tensor_cast!(u8, u64);
impl_tensor_cast!(u8, f32);
impl_tensor_cast!(u8, f64);
impl_tensor_cast!(u16, i8);
impl_tensor_cast!(u16, i16);
impl_tensor_cast!(u16, i32);
impl_tensor_cast!(u16, i64);
impl_tensor_cast!(u16, u8);
impl_tensor_cast!(u16, u32);
impl_tensor_cast!(u16, u64);
impl_tensor_cast!(u16, f32);
impl_tensor_cast!(u16, f64);
impl_tensor_cast!(u32, i8);
impl_tensor_cast!(u32, i16);
impl_tensor_cast!(u32, i32);
impl_tensor_cast!(u32, i64);
impl_tensor_cast!(u32, u8);
impl_tensor_cast!(u32, u16);
impl_tensor_cast!(u32, u64);
impl_tensor_cast!(u32, f32);
impl_tensor_cast!(u32, f64);
impl_tensor_cast!(u64, i8);
impl_tensor_cast!(u64, i16);
impl_tensor_cast!(u64, i32);
impl_tensor_cast!(u64, i64);
impl_tensor_cast!(u64, u8);
impl_tensor_cast!(u64, u16);
impl_tensor_cast!(u64, u32);
impl_tensor_cast!(u64, f32);
impl_tensor_cast!(u64, f64);
impl_tensor_cast!(f32, i8);
impl_tensor_cast!(f32, i16);
impl_tensor_cast!(f32, i32);
impl_tensor_cast!(f32, i64);
impl_tensor_cast!(f32, u8);
impl_tensor_cast!(f32, u16);
impl_tensor_cast!(f32, u32);
impl_tensor_cast!(f32, u64);
impl_tensor_cast!(f32, f64);
impl_tensor_cast!(f64, i8);
impl_tensor_cast!(f64, i16);
impl_tensor_cast!(f64, i32);
impl_tensor_cast!(f64, i64);
impl_tensor_cast!(f64, u8);
impl_tensor_cast!(f64, u16);
impl_tensor_cast!(f64, u32);
impl_tensor_cast!(f64, u64);
impl_tensor_cast!(f64, f32);
impl_tensor_cast!(isize, i8);
impl_tensor_cast!(isize, i16);
impl_tensor_cast!(isize, i32);
impl_tensor_cast!(isize, i64);
impl_tensor_cast!(isize, u8);
impl_tensor_cast!(isize, u16);
impl_tensor_cast!(isize, u32);
impl_tensor_cast!(isize, u64);
impl_tensor_cast!(isize, f32);
impl_tensor_cast!(isize, f64);
impl_tensor_cast!(i8, isize);
impl_tensor_cast!(i16, isize);
impl_tensor_cast!(i32, isize);
impl_tensor_cast!(i64, isize);
impl_tensor_cast!(u8, isize);
impl_tensor_cast!(u16, isize);
impl_tensor_cast!(u32, isize);
impl_tensor_cast!(u64, isize);
impl_tensor_cast!(f32, isize);
impl_tensor_cast!(f64, isize);
impl_tensor_cast!(usize, i8);
impl_tensor_cast!(usize, i16);
impl_tensor_cast!(usize, i32);
impl_tensor_cast!(usize, i64);
impl_tensor_cast!(usize, u8);
impl_tensor_cast!(usize, u16);
impl_tensor_cast!(usize, u32);
impl_tensor_cast!(usize, u64);
impl_tensor_cast!(usize, f32);
impl_tensor_cast!(usize, f64);
impl_tensor_cast!(i8, usize);
impl_tensor_cast!(i16, usize);
impl_tensor_cast!(i32, usize);
impl_tensor_cast!(i64, usize);
impl_tensor_cast!(u8, usize);
impl_tensor_cast!(u16, usize);
impl_tensor_cast!(u32, usize);
impl_tensor_cast!(u64, usize);
impl_tensor_cast!(f32, usize);
impl_tensor_cast!(f64, usize);
impl_tensor_cast!(bool, u8);
impl_tensor_cast!(u8, bool);
impl_tensor_cast!(bool, u16);
impl_tensor_cast!(u16, bool);
impl_tensor_cast!(bool, u32);
impl_tensor_cast!(u32, bool);
impl_tensor_cast!(bool, u64);
impl_tensor_cast!(u64, bool);
impl_tensor_cast!(bool, i8);
impl_tensor_cast!(i8, bool);
impl_tensor_cast!(bool, i16);
impl_tensor_cast!(i16, bool);
impl_tensor_cast!(bool, i32);
impl_tensor_cast!(i32, bool);
impl_tensor_cast!(bool, i64);
impl_tensor_cast!(i64, bool);
impl_tensor_cast!(bool, f32);
impl_tensor_cast!(f32, bool);
impl_tensor_cast!(bool, f64);
impl_tensor_cast!(f64, bool);
impl_tensor_cast!(bool, usize);
impl_tensor_cast!(usize, bool);
impl_tensor_cast!(bool, isize);
impl_tensor_cast!(isize, bool);
impl_tensor_cast!(bool, num::complex::Complex<f32>);
impl_tensor_cast!(num::complex::Complex<f32>, bool);
impl_tensor_cast!(bool, num::complex::Complex<f64>);
impl_tensor_cast!(num::complex::Complex<f64>, bool);
impl_tensor_cast!(i8, num::complex::Complex<f32>);
impl_tensor_cast!(i16, num::complex::Complex<f32>);
impl_tensor_cast!(u8, num::complex::Complex<f32>);
impl_tensor_cast!(u16, num::complex::Complex<f32>);
impl_tensor_cast!(f32, num::complex::Complex<f32>);
impl_tensor_cast!(i8, num::complex::Complex<f64>);
impl_tensor_cast!(i16, num::complex::Complex<f64>);
impl_tensor_cast!(i32, num::complex::Complex<f64>);
impl_tensor_cast!(i64, num::complex::Complex<f64>);
impl_tensor_cast!(isize, num::complex::Complex<f64>);
impl_tensor_cast!(u8, num::complex::Complex<f64>);
impl_tensor_cast!(u16, num::complex::Complex<f64>);
impl_tensor_cast!(u32, num::complex::Complex<f64>);
impl_tensor_cast!(u64, num::complex::Complex<f64>);
impl_tensor_cast!(usize, num::complex::Complex<f64>);
impl_tensor_cast!(f32, num::complex::Complex<f64>);
impl_tensor_cast!(f64, num::complex::Complex<f64>);
impl_tensor_cast!(i32, num::complex::Complex<f32>);
impl_tensor_cast!(i64, num::complex::Complex<f32>);
impl_tensor_cast!(isize, num::complex::Complex<f32>);
impl_tensor_cast!(u32, num::complex::Complex<f32>);
impl_tensor_cast!(u64, num::complex::Complex<f32>);
impl_tensor_cast!(usize, num::complex::Complex<f32>);
impl_tensor_cast!(f64, num::complex::Complex<f32>);
impl_tensor_cast!(num::complex::Complex<f64>, num::complex::Complex<f32>);
impl_tensor_cast!(num::complex::Complex<f32>, num::complex::Complex<f64>);

/* #endregion */

/* #region astype functions */

/// Cast the tensor to dtype `TOut`, with reshape-like conditional copy:
/// a view if `TOut` is the same dtype, else a converted copy.
///
/// # Parameters
///
/// - `tensor`: the tensor to cast (any representation).
/// - `TOut`: target dtype; specify by turbofish or output type annotation.
///
/// # Returns
///
/// - [`TensorCow`]: borrowed (view) iff `T == TOut`, otherwise owned copy.
///
/// # Notes of API accordance
///
/// - NumPy/array-api: `x.astype(dtype, copy=False)`; RSTSR always behaves as `copy=False` (a
///   same-dtype cast is a free view).
///
/// # Examples
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::tensor_from_nested!([[1, 2, 3], [4, 5, 6]], &device);
/// let b = a.astype::<f64>().into_owned();
/// println!("{}", b.sum_axes(1));
/// // [ 6.0 15.0]
/// # assert_eq!(b[[0, 0]], 1.0f64);
/// ```
pub fn astype_f<'a, TOut, R, T, B, D>(tensor: &'a TensorAny<R, T, B, D>) -> Result<TensorCow<'a, TOut, B, D>>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceAPI<TOut>,
    TensorAny<R, T, B, D>: TensorCastAPI<'a, R, T, TOut, B, D>,
{
    <TensorAny<R, T, B, D> as TensorCastAPI<'a, R, T, TOut, B, D>>::astype_impl(tensor)
}

/// Cast the tensor to dtype `TOut`; panicking version of [`astype_f`].
pub fn astype<'a, TOut, R, T, B, D>(tensor: &'a TensorAny<R, T, B, D>) -> TensorCow<'a, TOut, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceAPI<TOut>,
    TensorAny<R, T, B, D>: TensorCastAPI<'a, R, T, TOut, B, D>,
{
    astype_f(tensor).rstsr_unwrap()
}

/// Consuming cast to dtype `TOut` returning an owned [`Tensor`]; reuses the
/// input storage when `T == TOut`, otherwise converts.
pub fn into_astype_f<R, T, TOut, B, D>(tensor: TensorAny<R, T, B, D>) -> Result<Tensor<TOut, B, D>>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    TOut: Clone,
    B: DeviceAPI<T>
        + DeviceAPI<TOut>
        + DeviceRawAPI<MaybeUninit<TOut>>
        + DeviceCreationAnyAPI<TOut>
        + OpAssignAPI<TOut, D>,
    TensorAny<R, T, B, D>: for<'x> TensorCastAPI<'x, R, T, TOut, B, D>,
    <B as DeviceRawAPI<TOut>>::Raw: Clone,
{
    let cow = <TensorAny<R, T, B, D> as TensorCastAPI<'_, R, T, TOut, B, D>>::astype_impl(&tensor)?;
    Ok(cow.into_owned())
}

/// Consuming cast to dtype `TOut`; panicking version of [`into_astype_f`].
pub fn into_astype<R, T, TOut, B, D>(tensor: TensorAny<R, T, B, D>) -> Tensor<TOut, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    TOut: Clone,
    B: DeviceAPI<T>
        + DeviceAPI<TOut>
        + DeviceRawAPI<MaybeUninit<TOut>>
        + DeviceCreationAnyAPI<TOut>
        + OpAssignAPI<TOut, D>,
    TensorAny<R, T, B, D>: for<'x> TensorCastAPI<'x, R, T, TOut, B, D>,
    <B as DeviceRawAPI<TOut>>::Raw: Clone,
{
    into_astype_f(tensor).rstsr_unwrap()
}

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    B: DeviceAPI<T>,
{
    pub fn astype_f<'a, TOut>(&'a self) -> Result<TensorCow<'a, TOut, B, D>>
    where
        B: DeviceAPI<TOut>,
        TensorAny<R, T, B, D>: TensorCastAPI<'a, R, T, TOut, B, D>,
    {
        astype_f(self)
    }

    pub fn astype<'a, TOut>(&'a self) -> TensorCow<'a, TOut, B, D>
    where
        B: DeviceAPI<TOut>,
        TensorAny<R, T, B, D>: TensorCastAPI<'a, R, T, TOut, B, D>,
    {
        astype(self)
    }

    pub fn into_astype_f<TOut>(self) -> Result<Tensor<TOut, B, D>>
    where
        TOut: Clone,
        B: DeviceAPI<TOut> + DeviceRawAPI<MaybeUninit<TOut>> + DeviceCreationAnyAPI<TOut> + OpAssignAPI<TOut, D>,
        <B as DeviceRawAPI<TOut>>::Raw: Clone,
        TensorAny<R, T, B, D>: for<'x> TensorCastAPI<'x, R, T, TOut, B, D>,
    {
        into_astype_f(self)
    }

    pub fn into_astype<TOut>(self) -> Tensor<TOut, B, D>
    where
        TOut: Clone,
        B: DeviceAPI<TOut> + DeviceRawAPI<MaybeUninit<TOut>> + DeviceCreationAnyAPI<TOut> + OpAssignAPI<TOut, D>,
        <B as DeviceRawAPI<TOut>>::Raw: Clone,
        TensorAny<R, T, B, D>: for<'x> TensorCastAPI<'x, R, T, TOut, B, D>,
    {
        into_astype(self)
    }
}

/* #endregion */

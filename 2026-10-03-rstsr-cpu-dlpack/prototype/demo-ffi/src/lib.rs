//! Prototype `cdylib` host driving the Python end-to-end tests of
//! `rstsr-cpu-dlpack` (`../../../rstsr/crates-interop/rstsr-cpu-dlpack`).
//!
//! It exposes a tiny C ABI: a handle table of rstsr tensors, export entry
//! points (shared / move / basic-indexed view), a dtype-dispatching import
//! entry point, simple readers (values, layout, buffer addresses), and — the
//! interesting part — a `PyCapsule` destructor written in Rust, so the Python
//! side only has to box the capsule. Production hosts would use the crate
//! directly (or through a pyo3 adapter); this crate exists only to exercise
//! the protocol from real CPython + NumPy.

use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;

use dlpack_ffi::{DLDataType, DLManagedTensor, DLManagedTensorVersioned};
use rstsr_common::error::Result as RstsrResult;
use rstsr_common::layout::exports::{Indexer, Slice};
use rstsr_core::prelude::*;
use rstsr_core::storage::exports::{DataOwned, Storage};
use rstsr_cpu_dlpack::{
    DlpackDtype, TensorDlpack, TensorDlpackShared, from_dlpack_legacy_f, from_dlpack_versioned_f, into_dlpack_f,
    into_shared_dlpack_f, to_dlpack_copy_f, to_dlpack_shared_f, to_dlpack_shared_view_f,
};

type Cpu = DeviceCpuSerial;
type OwnedF64 = Tensor<f64, Cpu, IxD>;
type SharedF64 = TensorDlpackShared<f64, Cpu, IxD>;

enum Handle {
    Owned(OwnedF64),
    Shared(SharedF64),
    ImportedF64(TensorDlpack<f64, Cpu, IxD>),
    ImportedF32(TensorDlpack<f32, Cpu, IxD>),
    ImportedI64(TensorDlpack<i64, Cpu, IxD>),
    ImportedI32(TensorDlpack<i32, Cpu, IxD>),
}

/* #region helpers */

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_error(msg: impl Into<String>) {
    let msg = msg.into();
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = Some(CString::new(msg).unwrap_or_default());
    });
}

/// Run `f`, converting a panic into `default` plus an error message.
fn guarded<T>(default: T, f: impl FnOnce() -> T) -> T {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(value) => value,
        Err(_) => {
            set_error("demo-ffi: panic (see stderr)");
            default
        },
    }
}

/// Consume a handle box; `None` (plus an error) when the pointer is NULL.
unsafe fn take_handle(handle: *mut c_void) -> Option<Handle> {
    let handle = handle as *mut Handle;
    if handle.is_null() {
        set_error("demo-ffi: NULL handle");
        return None;
    }
    Some(unsafe { *Box::from_raw(handle) })
}

/// Element-zero address of the tensor behind a handle.
fn data_ptr_of<R, T, B, D>(tensor: &TensorBase<Storage<R, T, B>, D>) -> usize
where
    R: DataAPI<Data = Vec<T>>,
    B: DeviceAPI<T, Raw = Vec<T>>,
    D: DimAPI,
{
    let base = tensor.storage().raw().as_ptr();
    unsafe { base.add(tensor.layout().offset()) as usize }
}

/// Base address of the buffer span behind a tensor (for a negative-stride
/// import this is the lowest addressed element, not necessarily element zero).
fn buffer_ptr_of<R, T, B, D>(tensor: &TensorBase<Storage<R, T, B>, D>) -> usize
where
    R: DataAPI<Data = Vec<T>>,
    B: DeviceAPI<T, Raw = Vec<T>>,
    D: DimAPI,
{
    tensor.storage().raw().as_ptr() as usize
}

/// `(shape, strides, offset)` of a tensor; strides in elements.
fn layout_of<R, T, B, D>(tensor: &TensorBase<Storage<R, T, B>, D>) -> (Vec<usize>, Vec<isize>, usize)
where
    R: DataAPI<Data = Vec<T>>,
    B: DeviceAPI<T, Raw = Vec<T>>,
    D: DimAPI,
{
    let layout = tensor.layout();
    (layout.shape().as_ref().to_vec(), layout.stride().as_ref().to_vec(), layout.offset())
}

/// Values convertible to `f64` for the demo's readers.
trait ToF64 {
    fn to_f64(self) -> f64;
}

macro_rules! impl_to_f64 {
    ($($T:ty),*) => { $(impl ToF64 for $T { fn to_f64(self) -> f64 { self as f64 } })* };
}

impl_to_f64!(f32, i64, i32);

impl ToF64 for f64 {
    fn to_f64(self) -> f64 {
        self
    }
}

/// Logical values in row-major order (`.raw()` would be address order).
fn gather_f64<R, T, B, D>(tensor: &TensorBase<Storage<R, T, B>, D>) -> Vec<f64>
where
    R: DataAPI<Data = Vec<T>>,
    B: DeviceAPI<T, Raw = Vec<T>>,
    D: DimAPI,
    T: Copy + ToF64,
{
    let shape_slice: &[usize] = tensor.layout().shape().as_ref();
    let shape = shape_slice.to_vec();
    let numel: usize = shape.iter().product();
    let mut out = Vec::with_capacity(numel);
    for flat in 0..numel {
        let mut remainder = flat;
        let mut index = vec![0isize; shape.len()];
        for (dim, idx) in shape.iter().zip(index.iter_mut()).rev() {
            *idx = (remainder % dim) as isize;
            remainder /= dim;
        }
        out.push(tensor.storage().get_index(tensor.layout().index(&index)).to_f64());
    }
    out
}

/* #endregion */

/* #region Python C-API: capsule destructor */

/// Opaque `PyObject` (the demo never touches anything but capsules).
type PyObject = c_void;

unsafe extern "C" {
    fn PyCapsule_GetName(capsule: *mut PyObject) -> *const c_char;
    fn PyCapsule_GetPointer(capsule: *mut PyObject, name: *const c_char) -> *mut c_void;
}

const CAPSULE_VERSIONED: &[u8] = b"dltensor_versioned\0";

/// Capsule destructor: frees an export that the consumer never took.
///
/// A consumed capsule has been renamed by the consumer (`used_dltensor_*`), so
/// checking the name is exactly what prevents a double free.
///
/// # Safety
///
/// Called by CPython with a live capsule.
unsafe extern "C" fn dlpack_capsule_destructor(capsule: *mut PyObject) {
    let name = unsafe { PyCapsule_GetName(capsule) };
    if name.is_null() {
        return;
    }
    if unsafe { CStr::from_ptr(name) }.to_bytes() != b"dltensor_versioned" {
        return; // consumed (or a foreign capsule we do not own)
    }
    let managed = unsafe { PyCapsule_GetPointer(capsule, CAPSULE_VERSIONED.as_ptr() as *const c_char) }
        as *mut DLManagedTensorVersioned;
    if managed.is_null() {
        return;
    }
    if let Some(deleter) = unsafe { (*managed).deleter } {
        unsafe { deleter(managed) };
    }
}

/// Address of the capsule destructor, for `PyCapsule_New` on the Python side.
#[no_mangle]
pub extern "C" fn rstsr_demo_capsule_destructor() -> *const c_void {
    dlpack_capsule_destructor as *const c_void
}

/* #endregion */

/* #region C ABI entry points */

#[no_mangle]
pub extern "C" fn rstsr_demo_version() -> *const c_char {
    static VERSION: &[u8] = b"rstsr-cpu-dlpack-demo-ffi/0.1\0";
    VERSION.as_ptr() as *const c_char
}

#[no_mangle]
pub extern "C" fn rstsr_demo_last_error() -> *const c_char {
    LAST_ERROR.with(|slot| match slot.borrow().as_ref() {
        Some(msg) => msg.as_ptr(),
        None => ptr::null(),
    })
}

/// Create a 1-D `f64` tensor `[0, len)`.
#[no_mangle]
pub extern "C" fn rstsr_demo_arange_f64(len: usize) -> *mut c_void {
    guarded(ptr::null_mut(), || {
        let device = Cpu::default();
        match rt::arange_f((0.0, len as f64, 1.0, &device)) {
            Ok(tensor) => Box::into_raw(Box::new(Handle::Owned(tensor))) as *mut c_void,
            Err(e) => {
                set_error(format!("arange failed: {e:?}"));
                ptr::null_mut()
            },
        }
    })
}

/// Create a 2-D row-major `f64` tensor with values `0..rows*cols`.
#[no_mangle]
pub extern "C" fn rstsr_demo_arange2d_f64(rows: usize, cols: usize) -> *mut c_void {
    guarded(ptr::null_mut(), || {
        let Some(numel) = rows.checked_mul(cols) else {
            set_error("arange2d: shape overflows");
            return ptr::null_mut();
        };
        let layout = match Layout::new(vec![rows, cols], vec![cols as isize, 1], 0) {
            Ok(layout) => layout,
            Err(e) => {
                set_error(format!("arange2d: invalid layout: {e:?}"));
                return ptr::null_mut();
            },
        };
        let values: Vec<f64> = (0..numel).map(|v| v as f64).collect();
        let storage = Storage::new(DataOwned::from(values), Cpu::default());
        let tensor = Tensor::new(storage, layout);
        Box::into_raw(Box::new(Handle::Owned(tensor))) as *mut c_void
    })
}

/// Move an owned handle into the shareable representation (no data copy).
#[no_mangle]
pub extern "C" fn rstsr_demo_to_shared(handle: *mut c_void) -> *mut c_void {
    guarded(ptr::null_mut(), || match unsafe { take_handle(handle) } {
        Some(Handle::Owned(tensor)) => match into_shared_dlpack_f(tensor) {
            Ok(shared) => Box::into_raw(Box::new(Handle::Shared(shared))) as *mut c_void,
            Err(e) => {
                set_error(format!("into_shared_dlpack failed: {e:?}"));
                ptr::null_mut()
            },
        },
        Some(_) => {
            set_error("rstsr_demo_to_shared expects an owned handle");
            ptr::null_mut()
        },
        None => ptr::null_mut(),
    })
}

/// Fresh `DLManagedTensorVersioned` over a shared handle (read-only, zero-copy).
///
/// Repeatable: each call clones the `Arc`, so the handle stays valid.
#[no_mangle]
pub extern "C" fn rstsr_demo_export(handle: *mut c_void) -> *mut c_void {
    guarded(ptr::null_mut(), || {
        let handle = handle as *mut Handle;
        if handle.is_null() {
            set_error("demo-ffi: NULL handle");
            return ptr::null_mut();
        }
        match unsafe { &*handle } {
            Handle::Shared(shared) => match to_dlpack_shared_f(shared) {
                Ok(export) => export.into_raw() as *mut c_void,
                Err(e) => {
                    set_error(format!("to_dlpack_shared failed: {e:?}"));
                    ptr::null_mut()
                },
            },
            _ => {
                set_error("rstsr_demo_export expects a shared handle");
                ptr::null_mut()
            },
        }
    })
}

/// Fresh export over a deep copy of a shared handle (`__dlpack__(copy=True)`).
///
/// The consumer solely owns the copy (`IS_COPIED`), so NumPy shows a
/// writeable array that shares nothing with the original.
#[no_mangle]
pub extern "C" fn rstsr_demo_export_copy(handle: *mut c_void) -> *mut c_void {
    guarded(ptr::null_mut(), || {
        let handle = handle as *mut Handle;
        if handle.is_null() {
            set_error("demo-ffi: NULL handle");
            return ptr::null_mut();
        }
        match unsafe { &*handle } {
            Handle::Shared(shared) => match to_dlpack_copy_f(shared) {
                Ok(export) => export.into_raw() as *mut c_void,
                Err(e) => {
                    set_error(format!("to_dlpack_copy failed: {e:?}"));
                    ptr::null_mut()
                },
            },
            _ => {
                set_error("rstsr_demo_export_copy expects a shared handle");
                ptr::null_mut()
            },
        }
    })
}

/// Hand an owned handle to the consumer (`IS_COPIED`, writeable in NumPy).
#[no_mangle]
pub extern "C" fn rstsr_demo_export_move(handle: *mut c_void) -> *mut c_void {
    guarded(ptr::null_mut(), || match unsafe { take_handle(handle) } {
        Some(Handle::Owned(tensor)) => match into_dlpack_f(tensor) {
            Ok(export) => export.into_raw() as *mut c_void,
            Err(e) => {
                set_error(format!("into_dlpack failed: {e:?}"));
                ptr::null_mut()
            },
        },
        Some(_) => {
            set_error("rstsr_demo_export_move expects an owned handle");
            ptr::null_mut()
        },
        None => ptr::null_mut(),
    })
}

/// "No bound" sentinel for [`rstsr_demo_export_slice`] (Python `None`).
const SLICE_NONE: isize = isize::MIN;

/// Export a basic-indexed view of a shared `f64` handle, zero-copy.
///
/// `params` holds three `isize` per axis — `(start, stop, step)` with NumPy
/// slice conventions, `SLICE_NONE` for an absent bound — mirroring
/// `Slice::new`. Returns a fresh `DLManagedTensorVersioned` pointer per call.
#[no_mangle]
pub extern "C" fn rstsr_demo_export_slice(handle: *mut c_void, params: *const isize, n_axes: usize) -> *mut c_void {
    guarded(ptr::null_mut(), || {
        let handle = handle as *mut Handle;
        if handle.is_null() {
            set_error("demo-ffi: NULL handle");
            return ptr::null_mut();
        }
        if params.is_null() || n_axes == 0 {
            set_error("rstsr_demo_export_slice needs at least one axis");
            return ptr::null_mut();
        }
        let flat = unsafe { core::slice::from_raw_parts(params, n_axes * 3) };
        let bound = |v: isize| if v == SLICE_NONE { None } else { Some(v) };
        let indexers: Vec<Indexer> = flat
            .chunks_exact(3)
            .map(|p| Indexer::Slice(Slice::<isize>::new(bound(p[0]), bound(p[1]), bound(p[2]))))
            .collect();
        match unsafe { &*handle } {
            Handle::Shared(shared) => {
                let view = shared.i(indexers);
                match to_dlpack_shared_view_f(shared, &view) {
                    Ok(export) => export.into_raw() as *mut c_void,
                    Err(e) => {
                        set_error(format!("to_dlpack_shared_view failed: {e:?}"));
                        ptr::null_mut()
                    },
                }
            },
            _ => {
                set_error("rstsr_demo_export_slice expects a shared handle");
                ptr::null_mut()
            },
        }
    })
}

fn import_typed<T: DlpackDtype>(ptr: *mut c_void, legacy: bool) -> RstsrResult<TensorDlpack<T, Cpu, IxD>> {
    unsafe {
        if legacy {
            from_dlpack_legacy_f::<T, Cpu, IxD>(ptr as *mut DLManagedTensor)
        } else {
            from_dlpack_versioned_f::<T, Cpu, IxD>(ptr as *mut DLManagedTensorVersioned)
        }
    }
}

/// Import a foreign managed tensor; the dtype decides the stored variant.
#[no_mangle]
pub extern "C" fn rstsr_demo_import(ptr: *mut c_void, legacy: i32) -> *mut c_void {
    guarded(ptr::null_mut(), || {
        if ptr.is_null() {
            set_error("demo-ffi: NULL managed tensor");
            return ptr::null_mut();
        }
        let legacy = legacy != 0;
        let dtype: DLDataType = if legacy {
            unsafe { (*(ptr as *mut DLManagedTensor)).dl_tensor.dtype }
        } else {
            unsafe { (*(ptr as *mut DLManagedTensorVersioned)).dl_tensor.dtype }
        };
        let result = match (dtype.code, dtype.bits, dtype.lanes) {
            (2, 64, 1) => import_typed::<f64>(ptr, legacy).map(Handle::ImportedF64),
            (2, 32, 1) => import_typed::<f32>(ptr, legacy).map(Handle::ImportedF32),
            (0, 64, 1) => import_typed::<i64>(ptr, legacy).map(Handle::ImportedI64),
            (0, 32, 1) => import_typed::<i32>(ptr, legacy).map(Handle::ImportedI32),
            _ => {
                set_error(format!("unsupported dtype {:?}", (dtype.code, dtype.bits, dtype.lanes)));
                return ptr::null_mut();
            },
        };
        match result {
            Ok(handle) => Box::into_raw(Box::new(handle)) as *mut c_void,
            Err(e) => {
                set_error(format!("import failed: {e:?}"));
                ptr::null_mut()
            },
        }
    })
}

/// Element-zero address of the tensor (for zero-copy pointer checks).
#[no_mangle]
pub extern "C" fn rstsr_demo_data_ptr(handle: *mut c_void) -> usize {
    guarded(usize::MAX, || {
        let handle = handle as *mut Handle;
        if handle.is_null() {
            set_error("demo-ffi: NULL handle");
            return usize::MAX;
        }
        match unsafe { &*handle } {
            Handle::Owned(t) => data_ptr_of(t),
            Handle::Shared(t) => data_ptr_of(t),
            Handle::ImportedF64(t) => data_ptr_of(t),
            Handle::ImportedF32(t) => data_ptr_of(t),
            Handle::ImportedI64(t) => data_ptr_of(t),
            Handle::ImportedI32(t) => data_ptr_of(t),
        }
    })
}

/// Base address of the buffer span behind a handle (an import with negative
/// strides starts below element zero; see the crate's import normalisation).
#[no_mangle]
pub extern "C" fn rstsr_demo_buffer_ptr(handle: *mut c_void) -> usize {
    guarded(usize::MAX, || {
        let handle = handle as *mut Handle;
        if handle.is_null() {
            set_error("demo-ffi: NULL handle");
            return usize::MAX;
        }
        match unsafe { &*handle } {
            Handle::Owned(t) => buffer_ptr_of(t),
            Handle::Shared(t) => buffer_ptr_of(t),
            Handle::ImportedF64(t) => buffer_ptr_of(t),
            Handle::ImportedF32(t) => buffer_ptr_of(t),
            Handle::ImportedI64(t) => buffer_ptr_of(t),
            Handle::ImportedI32(t) => buffer_ptr_of(t),
        }
    })
}

/// Write `(shape, strides)` as `i64` into the caller buffers (capacity `cap`
/// each, `cap >= ndim`) and the offset into `*offset_out`.
///
/// Returns `ndim`, or `usize::MAX` on error.
#[no_mangle]
pub extern "C" fn rstsr_demo_layout(
    handle: *mut c_void,
    shape_out: *mut i64,
    strides_out: *mut i64,
    cap: usize,
    offset_out: *mut usize,
) -> usize {
    guarded(usize::MAX, || {
        let handle = handle as *mut Handle;
        if handle.is_null() {
            set_error("demo-ffi: NULL handle");
            return usize::MAX;
        }
        let (shape, strides, offset) = match unsafe { &*handle } {
            Handle::Owned(t) => layout_of(t),
            Handle::Shared(t) => layout_of(t),
            Handle::ImportedF64(t) => layout_of(t),
            Handle::ImportedF32(t) => layout_of(t),
            Handle::ImportedI64(t) => layout_of(t),
            Handle::ImportedI32(t) => layout_of(t),
        };
        if shape.len() > cap || shape_out.is_null() || strides_out.is_null() {
            set_error(format!("layout buffer too small: need {}, have {cap}", shape.len()));
            return usize::MAX;
        }
        for (i, (&dim, &stride)) in shape.iter().zip(strides.iter()).enumerate() {
            unsafe {
                *shape_out.add(i) = dim as i64;
                *strides_out.add(i) = stride as i64;
            }
        }
        if !offset_out.is_null() {
            unsafe { *offset_out = offset };
        }
        shape.len()
    })
}

/// Sum of the logical values (any stored dtype).
#[no_mangle]
pub extern "C" fn rstsr_demo_sum_f64(handle: *mut c_void) -> f64 {
    guarded(f64::NAN, || {
        let handle = handle as *mut Handle;
        if handle.is_null() {
            set_error("demo-ffi: NULL handle");
            return f64::NAN;
        }
        match unsafe { &*handle } {
            Handle::Owned(t) => t.raw().iter().copied().sum(),
            Handle::Shared(t) => gather_f64(t).iter().sum(),
            Handle::ImportedF64(t) => gather_f64(t).iter().sum(),
            Handle::ImportedF32(t) => gather_f64(t).iter().sum(),
            Handle::ImportedI64(t) => gather_f64(t).iter().sum(),
            Handle::ImportedI32(t) => gather_f64(t).iter().sum(),
        }
    })
}

/// Write the logical values (row-major) into a caller-provided buffer.
///
/// Returns the number of values written, or `usize::MAX` on error.
#[no_mangle]
pub extern "C" fn rstsr_demo_dump_f64(handle: *mut c_void, out: *mut f64, cap: usize) -> usize {
    guarded(usize::MAX, || {
        let handle = handle as *mut Handle;
        if handle.is_null() {
            set_error("demo-ffi: NULL handle");
            return usize::MAX;
        }
        let values = match unsafe { &*handle } {
            Handle::Owned(t) => t.raw().to_vec(),
            Handle::Shared(t) => gather_f64(t),
            Handle::ImportedF64(t) => gather_f64(t),
            Handle::ImportedF32(t) => gather_f64(t),
            Handle::ImportedI64(t) => gather_f64(t),
            Handle::ImportedI32(t) => gather_f64(t),
        };
        if values.len() > cap || (out.is_null() && !values.is_empty()) {
            set_error(format!("dump buffer too small: need {}, have {cap}", values.len()));
            return usize::MAX;
        }
        if !values.is_empty() {
            unsafe { ptr::copy_nonoverlapping(values.as_ptr(), out, values.len()) };
        }
        values.len()
    })
}

/// Number of logical elements, or `usize::MAX` on error.
#[no_mangle]
pub extern "C" fn rstsr_demo_len(handle: *mut c_void) -> usize {
    guarded(usize::MAX, || {
        let handle = handle as *mut Handle;
        if handle.is_null() {
            set_error("demo-ffi: NULL handle");
            return usize::MAX;
        }
        match unsafe { &*handle } {
            Handle::Owned(t) => t.layout().size(),
            Handle::Shared(t) => t.layout().size(),
            Handle::ImportedF64(t) => t.layout().size(),
            Handle::ImportedF32(t) => t.layout().size(),
            Handle::ImportedI64(t) => t.layout().size(),
            Handle::ImportedI32(t) => t.layout().size(),
        }
    })
}

/// Dimension count of the tensor behind a handle, or `usize::MAX` on error.
#[no_mangle]
pub extern "C" fn rstsr_demo_ndim(handle: *mut c_void) -> usize {
    guarded(usize::MAX, || {
        let handle = handle as *mut Handle;
        if handle.is_null() {
            set_error("demo-ffi: NULL handle");
            return usize::MAX;
        }
        match unsafe { &*handle } {
            Handle::Owned(t) => t.layout().ndim(),
            Handle::Shared(t) => t.layout().ndim(),
            Handle::ImportedF64(t) => t.layout().ndim(),
            Handle::ImportedF32(t) => t.layout().ndim(),
            Handle::ImportedI64(t) => t.layout().ndim(),
            Handle::ImportedI32(t) => t.layout().ndim(),
        }
    })
}

/// Drop a handle. Imported handles free the producer at this point.
#[no_mangle]
pub extern "C" fn rstsr_demo_free(handle: *mut c_void) {
    if handle.is_null() {
        return;
    }
    drop(unsafe { Box::from_raw(handle as *mut Handle) });
}

/* #endregion */

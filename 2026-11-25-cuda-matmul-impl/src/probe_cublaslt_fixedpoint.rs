//! Probe (trajectory only): does cuBLASLt serve the FP64 fixed-point emulation through cudarc's
//! 13.x bindings, and does the emulation descriptor's mantissa bit count change the accuracy?
//! Direct `cublaslt::result` calls; nothing here is driver code.
//!
//! ```text
//! cargo run --release -p rstsr-cuda --example probe_cublaslt_fixedpoint -- <n>
//! ```

use core::ffi::c_void;
use core::mem::size_of;
use cudarc::cublas::sys as cb;
use cudarc::cublaslt::{result as r, sys as lt};
use cudarc::driver::CudaContext;

fn set_desc(desc: lt::cublasLtMatmulDesc_t, attr: lt::cublasLtMatmulDescAttributes_t, v: &impl Sized) -> lt::cublasStatus_t {
    unsafe { lt::cublasLtMatmulDescSetAttribute(desc, attr, v as *const _ as *const c_void, size_of_val(v)) }
}

fn data(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed | 1;
    (0..n).map(|_| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }).collect()
}

/// `control`: `None` = no emulation descriptor; `Some((mantissa_control, max_bits))`. `plain` uses
/// `COMPUTE_64F` instead of the emulated compute type.
fn run(n: usize, reps: usize, control: Option<(i32, i32)>, plain: bool, strategy: i32, av: &[f64], bv: &[f64], want: &[f64], da: &cudarc::driver::CudaSlice<f64>, db: &cudarc::driver::CudaSlice<f64>, dd: &mut cudarc::driver::CudaSlice<f64>, stream: &std::sync::Arc<cudarc::driver::CudaStream>) -> Result<(f64, f64), lt::cublasStatus_t> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    let handle = r::create_handle().map_err(|e| e.0)?;
    let desc = r::create_matmul_desc(if plain { lt::cublasComputeType_t::CUBLAS_COMPUTE_64F } else { lt::cublasComputeType_t::CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT }, lt::cudaDataType_t::CUDA_R_64F).map_err(|e| e.0)?;
    let mut emu: Option<lt::cublasLtEmulationDesc_t> = None;
    if let Some((mantissa_control, max_bits)) = control {
        let mut e = core::mem::MaybeUninit::uninit();
        unsafe { lt::cublasLtEmulationDescCreate(e.as_mut_ptr()) }.result().map_err(|e| e.0)?;
        let e = unsafe { e.assume_init() };
        let st = unsafe { lt::cublasLtEmulationDescSetAttribute(e, lt::cublasLtEmulationDescAttributes_t::CUBLASLT_EMULATION_DESC_STRATEGY, &strategy as *const _ as *const _, 4) };
        st.result().map_err(|e| e.0)?;
        let st = unsafe { lt::cublasLtEmulationDescSetAttribute(e, lt::cublasLtEmulationDescAttributes_t::CUBLASLT_EMULATION_DESC_FIXEDPOINT_MANTISSA_CONTROL, &mantissa_control as *const _ as *const _, 4) };
        st.result().map_err(|e| e.0)?;
        let st = unsafe { lt::cublasLtEmulationDescSetAttribute(e, lt::cublasLtEmulationDescAttributes_t::CUBLASLT_EMULATION_DESC_FIXEDPOINT_MAX_MANTISSA_BIT_COUNT, &max_bits as *const _ as *const _, 4) };
        st.result().map_err(|e| e.0)?;
        set_desc(desc, lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EMULATION_DESCRIPTOR, &e).result().map_err(|e| e.0)?;
        emu = Some(e);
    }
    let la = r::create_matrix_layout(lt::cudaDataType_t::CUDA_R_64F, n as u64, n as u64, n as i64).map_err(|e| e.0)?;
    let lb = r::create_matrix_layout(lt::cudaDataType_t::CUDA_R_64F, n as u64, n as u64, n as i64).map_err(|e| e.0)?;
    let lc = r::create_matrix_layout(lt::cudaDataType_t::CUDA_R_64F, n as u64, n as u64, n as i64).map_err(|e| e.0)?;
    let pref = r::create_matmul_pref().map_err(|e| e.0)?;
    let ws: usize = 32 << 20;
    unsafe { r::set_matmul_pref_attribute(pref, lt::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws as *const _ as *const _, size_of::<usize>()) }.map_err(|e| e.0)?;
    let h = match unsafe { r::get_matmul_algo_heuristic(handle, desc, la, lb, lc, lc, pref) } {
        Ok(h) => h,
        Err(e) => return Err(e.0),
    };
    let mut wslice = stream.alloc_zeros::<u8>(ws).map_err(|_| lt::cublasStatus_t::CUBLAS_STATUS_ALLOC_FAILED).ok().unwrap();
    let (alpha, beta) = (1.0f64, 0.0f64);
    let dptr = dd.device_ptr_mut(&stream).0 as usize;
    let aptr = da.device_ptr(&stream).0 as usize;
    let bptr = db.device_ptr(&stream).0 as usize;
    let wptr = wslice.device_ptr_mut(&stream).0 as usize;
    let call = || unsafe {
        r::matmul(
            handle,
            desc,
            &alpha as *const _ as *const c_void,
            &beta as *const _ as *const c_void,
            bptr as *const c_void,
            la,
            aptr as *const c_void,
            lb,
            dptr as *const c_void,
            lc,
            dptr as *mut c_void,
            lc,
            &h.algo,
            wptr as *mut c_void,
            h.workspaceSize,
            stream.cu_stream() as *mut cudarc::cublaslt::sys::CUstream_st,
        )
    };
    // slot A holds our `b` and slot B our `a`, both read column-major: D = b^T a^T = (a b)^T.
    // A (slot) holds our `b`: C = A B with the swapped roles gives the transposed product, so
    // compute D = b^T a^T instead... keep it simple: square n, compare against (b a)^T.
    call().map_err(|e| e.0)?;
    stream.synchronize().unwrap();
    let t = std::time::Instant::now();
    for _ in 0..reps {
        call().map_err(|e| e.0)?;
    }
    stream.synchronize().unwrap();
    let secs = t.elapsed().as_secs_f64() / reps as f64;
    let mut host = vec![0.0f64; n * n];
    stream.memcpy_dtoh(dd, &mut host).unwrap();
    let got = host;
    // D = A_slot * B_slot = b * a, compared with (b a)[i, j] = sum_k b[i,k] a[j,k]
    let mut err = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let e = (got[i * n + j] - want_t(i, j, n, bv, av)).abs();
            err = err.max(e);
        }
    }
    if let Some(e) = emu {
        unsafe { lt::cublasLtEmulationDescDestroy(e) };
    }
    unsafe {
        lt::cublasLtMatmulPreferenceDestroy(pref);
        lt::cublasLtMatrixLayoutDestroy(la);
        lt::cublasLtMatrixLayoutDestroy(lb);
        lt::cublasLtMatrixLayoutDestroy(lc);
        lt::cublasLtMatmulDescDestroy(desc);
        lt::cublasLtDestroy(handle);
    }
    let _ = (da, av, want);
    Ok((err, 2.0 * n as f64 * n as f64 * n as f64 / secs / 1e12))
}

fn want_t(i: usize, j: usize, n: usize, bv: &[f64], av: &[f64]) -> f64 {
    // the row-major buffers declared column-major are the transposes, so D read row-major is a b
    (0..n).map(|k| av[i * n + k] * bv[k * n + j]).sum()
}

trait Res {
    fn result(self) -> Result<(), cudarc::cublaslt::result::CublasError>;
}
impl Res for lt::cublasStatus_t {
    fn result(self) -> Result<(), cudarc::cublaslt::result::CublasError> {
        match self {
            lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS => Ok(()),
            s => Err(cudarc::cublaslt::result::CublasError(s)),
        }
    }
}

fn main() {
    let n: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(2048);
    let reps = 5;
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let av = data(n * n, 3);
    let bv = data(n * n, 4);
    let mut want = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            want[i * n + j] = want_t(i, j, n, &bv, &av);
        }
    }
    let mut da = stream.alloc_zeros::<f64>(n * n).unwrap();
    let mut db = stream.alloc_zeros::<f64>(n * n).unwrap();
    stream.memcpy_htod(&av, &mut da).unwrap();
    stream.memcpy_htod(&bv, &mut db).unwrap();
    let mut dd = stream.alloc_zeros::<f64>(n * n).unwrap();
    let cases: [(String, Option<(i32, i32)>, bool, i32); 8] = [
        ("plain COMPUTE_64F (reference speed)".into(), None, true, 0),
        ("no emulation descriptor".into(), None, false, 0),
        ("dynamic control, max bits 0 (library default)".into(), Some((0, 0)), false, 0),
        ("dynamic control, max bits 24".into(), Some((0, 24)), false, 0),
        ("fixed control (1), max bits 16".into(), Some((1, 16)), false, 0),
        ("fixed control (1), max bits 24".into(), Some((1, 24)), false, 0),
        ("strategy PERFORMANT (1), default bits".into(), Some((0, 0)), false, 1),
        ("strategy EAGER (2), default bits".into(), Some((0, 0)), false, 2),
    ];
    println!("n = {n}; error = max |D - a b| (slot A holds b, slot B holds a)");
    println!("| case | max abs err | TFLOP/s |");
    println!("|---|---|---|");
    for (label, control, plain, strategy) in cases {
        match run(n, reps, control, plain, strategy, &av, &bv, &want, &da, &db, &mut dd, &stream) {
            Ok((err, tf)) => println!("| {label} | {err:.3e} | {tf:.2} |"),
            Err(s) => println!("| {label} | status {s:?} | - |"),
        }
    }
}

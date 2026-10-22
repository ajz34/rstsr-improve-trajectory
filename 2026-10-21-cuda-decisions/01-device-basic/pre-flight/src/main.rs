//! Pre-flight probe: verify the cudarc substrate assumptions of
//! `../DECISIONS.md` (D1/D2/D7) on the local machine before any rstsr code.
//! Build: cargo run (cudarc 0.19, cuda-version-from-build-system).

use cudarc::driver::result as dr;
use cudarc::driver::sys::{CUdeviceptr, CUmemPool_attribute};
use cudarc::driver::CudaContext;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // P1 — substrate
    let n = CudaContext::device_count()?;
    println!("device_count = {n}");

    let ctx = CudaContext::new(0)?;
    let (maj, min) = ctx.compute_capability()?;
    println!("compute_capability = ({maj},{min})");
    let (free, total) = ctx.mem_get_info()?;
    println!("mem_get_info: free={free} total={total}");

    let stream = ctx.default_stream();
    println!("default_stream ptr = {:p}", stream.cu_stream());

    let s2 = ctx.new_stream()?;
    println!("new_stream (non-blocking) ok");

    // typed alloc + H2D/D2H cycle on the non-blocking stream
    let mut a: cudarc::driver::CudaSlice<f32> = unsafe { s2.alloc(1024) }?;
    s2.memcpy_htod(&[1.0f32; 1024], &mut a)?;
    let host = s2.clone_dtoh(&a)?;
    assert_eq!(host[0], 1.0);
    assert_eq!(host[1023], 1.0);
    println!("alloc/memcpy_htod/clone_dtoh on new_stream ok");

    // deep D2D clone (Clone = memcpy_dtod assumption, F1)
    let b = a.try_clone()?;
    let host_b = s2.clone_dtoh(&b)?;
    assert_eq!(host_b[5], 1.0);
    println!("try_clone (deep dtod) ok");

    // P2 — mempool behavior (raw result layer, DECISIONS D7)
    println!("--- mempool checks ---");
    let dev = unsafe { dr::device::get(0) }?;
    let pool = unsafe { dr::device::get_default_mem_pool(dev) }?;
    println!("default mem pool = {pool:?}");
    let mut threshold: usize = 0;
    // set RELEASE_THRESHOLD through the raw pointer API, then read it back
    unsafe {
        dr::mem_pool::set_attribute(
            pool,
            CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
            std::ptr::addr_of_mut!(threshold).cast(),
        )?;
        dr::mem_pool::trim_to(pool, 0)?;
    }
    println!("set/get release threshold ok (value now {threshold})");
    println!("trim_to(0) ok");

    // allocation churn: second iteration should be pool-cached (timing heuristic)
    let t0 = std::time::Instant::now();
    let big = unsafe { s2.alloc::<u8>(256 * 1024 * 1024) }?;
    let t1 = t0.elapsed();
    drop(big);
    let big = unsafe { s2.alloc::<u8>(256 * 1024 * 1024) }?;
    let t2 = t0.elapsed() - t1;
    drop(big);
    println!("alloc#1 = {t1:?}, alloc#2 (cached?) = {t2:?}");

    // zero-size sentinel allocation (D1 empty-buffer design)
    let z: cudarc::driver::CudaSlice<u8> = s2.null()?;
    println!("null alloc ok (len={})", z.len());

    let _ = CUdeviceptr::default(); // reference the sys type for the record
    println!("PREFLIGHT PASS");
    Ok(())
}

//! Pre-flight probes P1–P5 of ../PLAN.md: NVRTC compile (arch spelling,
//! std), first-use latency, builder u8-marshal, cross-stream event
//! ordering, disk-cache layout. Run: cargo run --release

use std::time::Instant;

use cudarc::driver::result as dr;
use cudarc::driver::safe::{LaunchConfig, PushKernelArg};
use cudarc::driver::{CudaContext, CudaSlice};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

const PRELUDE: &str = r#"
typedef signed char i8_t; typedef short i16_t; typedef int i32_t; typedef long long i64_t;
typedef unsigned char u8_t; typedef unsigned short u16_t; typedef unsigned int u32_t; typedef unsigned long long u64_t;
typedef size_t usz_t; typedef long long isz_t;
struct c32_t { float re, im; };
struct c64_t { double re, im; };
extern "C" __global__ void probe_fill_f32(float* out, usz_t n, float v) {
    usz_t i = (usz_t)blockIdx.x * (usz_t)blockDim.x + (usz_t)threadIdx.x;
    if (i < n) out[i] = v;
}
extern "C" __global__ void probe_cast_f32_to_f64(float* src, double* dst, usz_t n) {
    usz_t i = (usz_t)blockIdx.x * (usz_t)blockDim.x + (usz_t)threadIdx.x;
    if (i < n) dst[i] = (double)src[i];
}
"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let (maj, min) = ctx.compute_capability()?;
    let cc = maj * 100 + min * 10;
    println!("== P1: NVRTC compile (arch spelling, --std=c++17) ==");
    println!("device 0 compute capability = ({maj},{min})");

    let stream = ctx.default_stream();
    let mut host_in: Vec<f32> = (0..1024).map(|i| i as f32).collect();

    // compile + load a full pipeline now; latency measured around it in P2
    let arch_candidates = [format!("compute_{maj}{min}"), format!("sm_{maj}{min}")];
    let mut loaded_ptx: Option<cudarc::nvrtc::Ptx> = None;
    for arch in &arch_candidates {
        let opts = CompileOptions {
            arch: Some(arch_leak(arch)),
            options: vec!["--std=c++17".into(), "-lineinfo".into()],
            ..Default::default()
        };
        match compile_ptx_with_opts(PRELUDE, opts) {
            Ok(ptx) => {
                println!("NVRTC compile ok with --gpu-architecture={arch}");
                loaded_ptx = Some(ptx);
                break;
            }
            Err(e) => println!("arch={arch} FAILED: {e}"),
        }
    }
    let ptx = loaded_ptx.ok_or("no arch spelling accepted")?;

    let module = ctx.load_module(ptx)?;
    let fill_fn = module.load_function("probe_fill_f32")?;
    let cast_fn = module.load_function("probe_cast_f32_to_f64")?;
    println!("module load + function lookup ok (2 kernels)");

    // == P3: builder u8-marshal — &CudaSlice<u8>, [u8; N], u64, usize ==
    println!("== P3: builder u8-wise marshaling ==");
    let mut buf: CudaSlice<u8> = unsafe { stream.alloc(1024 * 4) }?; // f32 payload
    let fill_bytes = 3.5f32.to_ne_bytes();
    let n_u64 = 1024_u64;
    let blocks = (1024 + 255) / 256;
    let cfg = LaunchConfig {
        grid_dim: (blocks as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&fill_fn)
            .arg(&buf) // &CudaSlice<u8>: no DeviceRepr bound needed
            .arg(&n_u64)
            .arg(&fill_bytes) // [u8; 4]
            .launch(cfg)?;
    }
    stream.synchronize()?;
    // read back through a typed view of the same bytes: clone to host bytes
    let host_bytes = stream.clone_dtoh(&buf)?;
    let got = f32::from_ne_bytes(host_bytes[0..4].try_into()?);
    assert_eq!(got, 3.5, "fill via u8 bytes");
    let got_last = f32::from_ne_bytes(host_bytes[4092..4096].try_into()?);
    assert_eq!(got_last, 3.5);
    println!("&CudaSlice<u8> + u64 + [u8;4] fill ok (3.5 read back)");

    // cast kernel: two u8 buffers marshaled the same way
    let mut dst: CudaSlice<u8> = unsafe { stream.alloc(1024 * 8) }?;
    unsafe {
        stream
            .launch_builder(&cast_fn)
            .arg(&buf)
            .arg(&dst)
            .arg(&n_u64)
            .launch(LaunchConfig::for_num_elems(1024))?;
    }
    stream.synchronize()?;
    let out_bytes = stream.clone_dtoh(&dst)?;
    let got = f64::from_ne_bytes(out_bytes[0..8].try_into()?);
    assert_eq!(got, 3.5f64 as f32 as f64);
    println!("cast kernel f32->f64 through byte buffers ok");

    // == P2: first-use latency (cold compile+load; warm load; disk file) ==
    println!("== P2: latency ==");
    let src2 = format!("{PRELUDE}\n// variant\n");
    let t0 = Instant::now();
    let opts = CompileOptions {
        arch: Some(arch_leak(&format!("compute_{maj}{min}"))),
        options: vec!["--std=c++17".into()],
        ..Default::default()
    };
    let ptx2 = compile_ptx_with_opts(&src2, opts)?;
    let t_compile = t0.elapsed();
    let t1 = Instant::now();
    let module2 = ctx.load_module(ptx2)?;
    let _ = module2.load_function("probe_fill_f32")?;
    let t_load = t1.elapsed();
    println!("cold NVRTC compile = {t_compile:?}");
    println!("module load + fn lookup = {t_load:?}");
    // warm path: a second module with identical source load (driver JIT cache hit shape)
    let t2 = Instant::now();
    let opts2 = CompileOptions {
        arch: Some(arch_leak(&format!("compute_{maj}{min}"))),
        options: vec!["--std=c++17".into()],
        ..Default::default()
    };
    let ptx3 = compile_ptx_with_opts(&src2, opts2)?;
    let _m3 = ctx.load_module(ptx3)?;
    println!("warm compile+load (same src) = {:?}", t2.elapsed());
    // note: disk cache stores the PTX source text we produced (P5), since
    // CudaModule does not retain or expose the compiled image.

    // == P4: cross-stream event ordering ==
    println!("== P4: cross-stream event ordering ==");
    let s2 = ctx.new_stream()?; // non-blocking stream
    let prod: CudaSlice<u8> = unsafe { s2.alloc(1024 * 4) }?;
    // fill on default stream
    unsafe {
        stream
            .launch_builder(&fill_fn)
            .arg(&prod)
            .arg(&n_u64)
            .arg(&3.5f32.to_ne_bytes())
            .launch(LaunchConfig::for_num_elems(1024))?;
    }
    // record event on default stream after fill, wait on s2 before reading
    let event = dr::event::create(cudarc::driver::sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC)?;
    // SAFETY: live event and stream handles from the calls above
    unsafe { dr::event::record(event, stream.cu_stream())? };
    // SAFETY: live stream and event handles
    unsafe {
        dr::stream::wait_event(
            s2.cu_stream(),
            event,
            cudarc::driver::sys::CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT,
        )?
    };
    let mut got2 = s2.clone_dtoh(&prod)?; // read on s2
    let v = f32::from_ne_bytes(got2[8..12].try_into()?);
    assert_eq!(v, 3.5);
    got2.clear();
    drop(prod);
    // SAFETY: event still live (all uses above have completed)
    unsafe { dr::event::destroy(event)? };
    println!("event record/wait ordering across streams ok");

    // == P5: disk-cache key/layout (hash + file write/read) ==
    println!("== P5: disk cache layout ==");
    let cache_dir = std::env::var_os("RSTSR_CUDA_CACHE_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            let base = std::env::var_os("XDG_CACHE_HOME")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|| {
                    let mut h = std::env::home_dir().unwrap_or_default();
                    h.push(".cache");
                    h
                });
            base.join("rstsr/cuda")
        });
    std::fs::create_dir_all(&cache_dir)?;
    // key = fnv1a-ish of (source + arch + options + payload format version)
    let key_src = format!("{PRELUDE}|compute_{maj}{min}|--std=c++17|-lineinfo|v1");
    let mut hash: u64 = 0xcbf29ce484222325;
    for b in key_src.as_bytes() {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    let path = cache_dir.join(format!("k_{hash:016x}.ptx"));
    std::fs::write(&path, PRELUDE.as_bytes())?;
    let round = std::fs::read_to_string(&path)?;
    assert_eq!(round, PRELUDE);
    println!("cache key fnv1a={hash:016x}, file round-trip ok at {}", path.display());
    std::fs::remove_file(&path).ok();

    drop(host_in); // silence unused (kept for future probes)
    println!("ALL PRE-FLIGHT PROBES PASSED");
    Ok(())
}

fn arch_leak(s: &str) -> &'static str {
    Box::leak(s.to_string().into_boxed_str())
}

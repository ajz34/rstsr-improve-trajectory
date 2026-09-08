//! THP (transparent huge pages) probe — RQ2d.
//!
//! System THP state is `madvise` (checked: /sys/kernel/mm/transparent_hugepage/
//! enabled), so a userspace buffer only gets 2 MiB pages after an explicit
//! `madvise(MADV_HUGEPAGE)`. rstsr at 386948be never issues madvise, so every
//! fresh 32 MiB output faults in 4 KiB pages (T0: ~8.2k faults/op).
//!
//! This probe allocates 32 MiB through the SAME path rstsr uses
//! (`std::alloc::alloc`, 64-byte alignment, via the `aligned_alloc` feature),
//! then measures a write-fill of a fresh allocation:
//!   plain     : alloc -> fill (rstsr's implicit behavior)
//!   madvise   : alloc -> madvise(MADV_HUGEPAGE) -> fill
//!   reuse     : one allocation, filled repeatedly (the theoretical floor)
//! plus a combined variant: fresh alloc each iteration WITH madvise, i.e.
//! "what a THP-hinting allocator would give".
//!
//! Reports wall time + ru_minflt deltas. Run under `perf stat -e
//! page-faults` (reproduce.sh) for the kernel-side confirmation.

use std::alloc::{alloc, dealloc, Layout};
use std::hint::black_box;
use std::time::{Duration, Instant};

use alloc_pagefault_study::{getrusage, report_accounting, Rusage};

const BYTES: usize = 2048 * 2048 * 8; // 32 MiB, exactly rstsr's large-class output
const ALIGN: usize = 64; // rstsr aligned_alloc alignment for len > 128
const ITERS: usize = 200;

fn fill(buf: &mut [f64], v: f64) {
    for x in buf.iter_mut() {
        *x = v;
    }
}

/// rust-simd-style: transmute the raw allocation into a &mut [f64] slice
unsafe fn as_f64_slice(ptr: *mut u8, n: usize) -> &'static mut [f64] {
    std::slice::from_raw_parts_mut(ptr as *mut f64, n)
}

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: madvise_thp <plain|madvise|reuse|madvise_reuse>");
        std::process::exit(2);
    });
    eprintln!("[madvise_thp] mode={mode} bytes={BYTES} iters={ITERS}");

    let n = BYTES / 8;
    let layout = Layout::from_size_align(BYTES, ALIGN).unwrap();

    // EMPIRICAL (documented in README): a glibc `aligned_alloc`-style
    // allocation with 64-B alignment (rstsr's path) is NOT page-aligned for
    // large sizes (memalign trim path), so madvise on it returns EINVAL.
    // The madvise modes therefore allocate with 2 MiB alignment — i.e. what
    // an rstsr allocator change would have to produce before THP hints can
    // apply at all.
    const HUGE_ALIGN: usize = 2 * 1024 * 1024;
    let layout_huge = Layout::from_size_align(BYTES, HUGE_ALIGN).unwrap();

    // Persistent buffer for reuse modes; allocated + touched before timing.
    let mut persistent: &mut [f64] = unsafe { as_f64_slice(alloc(layout_huge), n) };
    fill(persistent, 1.0);

    let ru0 = getrusage();
    let t0 = Instant::now();

    match mode.as_str() {
        // fresh 64-B-aligned alloc per iteration, fill, free (rstsr behavior)
        "plain" => {
            for _ in 0..ITERS {
                let ptr = unsafe { alloc(layout) };
                assert!(!ptr.is_null());
                let buf = unsafe { as_f64_slice(ptr, n) };
                fill(buf, 2.0);
                black_box(&*buf);
                unsafe { dealloc(ptr, layout) };
            }
        }
        // fresh 2-MiB-ALIGNED alloc + MADV_HUGEPAGE hint per iteration
        "madvise" => {
            for _ in 0..ITERS {
                let ptr = unsafe { alloc(layout_huge) };
                assert!(!ptr.is_null());
                let ret = unsafe { libc::madvise(ptr as *mut libc::c_void, BYTES, libc::MADV_HUGEPAGE) };
                assert_eq!(ret, 0, "madvise failed (addr not page-aligned?)");
                let buf = unsafe { as_f64_slice(ptr, n) };
                fill(buf, 2.0);
                black_box(&*buf);
                unsafe { dealloc(ptr, layout_huge) };
            }
        }
        // one persistent buffer, filled per iteration (floor)
        "reuse" => {
            for _ in 0..ITERS {
                fill(black_box(&mut persistent), 2.0);
                black_box(&*persistent);
            }
        }
        // persistent 2-MiB-aligned buffer re-hinted per iteration
        "madvise_reuse" => {
            for _ in 0..ITERS {
                let ptr = persistent.as_mut_ptr() as *mut u8;
                let ret = unsafe { libc::madvise(ptr as *mut libc::c_void, BYTES, libc::MADV_HUGEPAGE) };
                assert_eq!(ret, 0, "madvise failed (addr not page-aligned?)");
                fill(black_box(&mut persistent), 2.0);
                black_box(&*persistent);
            }
        }
        other => {
            eprintln!("unknown mode: {other}");
            std::process::exit(2);
        }
    }

    let elapsed: Duration = t0.elapsed();
    let d: Rusage = getrusage().delta(&ru0);
    report_accounting(&format!("thp_{mode}"), ITERS, elapsed.as_secs_f64(), &d);
    unsafe { dealloc(persistent.as_mut_ptr() as *mut u8, layout_huge) };
}

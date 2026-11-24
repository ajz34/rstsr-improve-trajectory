//! Chunked H2D -> compute -> D2H streaming with pinned host tensors
//! (`DeviceCudaPinned`), compared with the serial pageable flow and with the
//! negative controls that each remove one ingredient.
//!
//! Modes (all on the same bytes, interleaved round-robin, per-mode medians):
//! - `h2d` / `compute` / `d2h`: each phase alone over all chunks;
//! - `serial_all`: pageable host + blocking transfers (today's `to_device`);
//! - `pipeline`: pinned host + non-blocking transfers + 3 streams (H, C, D),
//!   issued per chunk;
//! - `pipeline_blocking`: pipeline, but a `synchronize()` after every stage;
//! - `pipeline_1stream`: pipeline, but H/C/D on one stream.
//!
//! Assertions: pipeline output bitwise equal to `serial_all`; pipeline time
//! < 0.7 x serial_all and < 0.75 x pipeline_1stream; the two controls sit at
//! the sum of the isolated phases (no overlap). Ratios only.

use rstsr::prelude::*;
use rstsr_cuda::pinned::pinned_flags;
use rstsr_cuda::pinned::transfer::d2h_pinned_async;
use rstsr_cuda::prelude_dev::*;
use std::time::{Duration, Instant};

const CHUNK_MIB: usize = 8;
const CHUNKS: usize = 32;
const ROUNDS: usize = 7;
const ELEMS: usize = CHUNK_MIB << 20 >> 3; // f64

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn arange_vec(n: usize, offset: usize) -> Vec<f64> {
    (0..n).map(|i| (i + offset) as f64 * 1e-3).collect()
}

/// The compute phase: a transcendental chain (one kernel per link); its
/// length is the knob that balances compute against transfer time.
fn compute<Tr>(g: Tr, iters: usize) -> Tensor<f64, DeviceCuda>
where
    Tr: TensorSinAPI<Output = Tensor<f64, DeviceCuda>>,
{
    let mut y = rt::sin(g);
    for _ in 1..iters {
        y = rt::sin(&y);
    }
    y
}

/// Sustained device work so the GPU clocks are out of idle before timing.
fn warm_clocks(dev: &DeviceCuda, ms: u64) {
    let cpu = DeviceCpuSerial::default();
    let a = rt::asarray((vec![0.5f64; 1 << 20], &cpu)).to_device(dev);
    let t0 = Instant::now();
    while t0.elapsed() < Duration::from_millis(ms) {
        let mut y = rt::sin(&a);
        for _ in 0..8 {
            y = rt::sin(&y);
        }
        dev.synchronize().unwrap();
    }
}

struct Fixture {
    pin_in: Vec<Tensor<f64, DeviceCudaPinned>>,
    cpu_in: Vec<Tensor<f64, DeviceCpuSerial>>,
}

/// Pageable-host pipeline (the CuPy `pipeline_pageable` analogue): the same
/// three streams and the same per-chunk issue order, but the host side is a
/// plain `Vec` through the pageable `to_device` paths. Those paths end with
/// `stream.synchronize()` (transfer.rs), which orders the *device* work but
/// leaves the pageable DMA itself synchronous-with-host only as far as CUDA's
/// staging goes — measured here: the overlap survives, because the host only
/// waits for each chunk's own copy, and the wait drains a queue the other
/// streams kept busy. Unlike CuPy (whose pageable pipeline collapses to the
/// serial time), rstsr's pageable flow does NOT isolate the pinned
/// ingredient; `pipeline_1stream` / `pipeline_blocking` are the controls
/// that do.
fn pipeline_pageable(
    fix: &Fixture,
    dev_h: &DeviceCuda,
    dev_c: &DeviceCuda,
    dev_d: &DeviceCuda,
    iters: usize,
) -> Duration {
    let t0 = Instant::now();
    for t in fix.cpu_in.iter() {
        let g = t.to_device(dev_h); // pageable H2D (blocking: pageable DMA)
        let g_c = g.to_device(dev_c);
        let r = compute(&g_c, iters);
        let _back = r.to_device(dev_d); // pageable D2H (blocking)
    }
    dev_h.synchronize().unwrap();
    dev_c.synchronize().unwrap();
    dev_d.synchronize().unwrap();
    t0.elapsed()
}

type PinnedChunk = Tensor<f64, DeviceCudaPinned>;

fn phase_h2d(fix: &Fixture, dev: &DeviceCuda) -> Duration {
    let t0 = Instant::now();
    let keep: Vec<_> = fix.pin_in.iter().map(|t| t.to_device(dev)).collect();
    dev.synchronize().unwrap();
    let d = t0.elapsed();
    drop(keep);
    d
}

fn phase_compute(fix: &Fixture, dev: &DeviceCuda, iters: usize) -> Duration {
    let staged: Vec<_> = fix.pin_in.iter().map(|t| t.to_device(dev)).collect();
    dev.synchronize().unwrap();
    let t0 = Instant::now();
    let keep: Vec<_> = staged.iter().map(|g| compute(g, iters)).collect();
    dev.synchronize().unwrap();
    let d = t0.elapsed();
    drop(keep);
    drop(staged);
    d
}

fn phase_d2h(fix: &Fixture, dev: &DeviceCuda, pin_out: &mut [PinnedChunk]) -> Duration {
    let staged: Vec<_> = fix.pin_in.iter().map(|t| t.to_device(dev)).collect();
    dev.synchronize().unwrap();
    let t0 = Instant::now();
    for (g, o) in staged.iter().zip(pin_out.iter_mut()) {
        d2h_pinned_async(dev, g.raw(), o.raw_mut()).unwrap();
    }
    dev.synchronize().unwrap();
    let d = t0.elapsed();
    drop(staged);
    d
}

/// Pageable host + blocking transfers: the naive flow.
fn serial_all(fix: &Fixture, dev: &DeviceCuda, iters: usize) -> (Duration, Vec<Vec<f64>>) {
    let cpu = DeviceCpuSerial::default();
    let t0 = Instant::now();
    let out: Vec<Vec<f64>> = fix
        .cpu_in
        .iter()
        .map(|t| {
            let g = t.to_device(dev);
            let r = compute(g, iters);
            r.to_device(&cpu).to_vec()
        })
        .collect();
    (t0.elapsed(), out)
}

fn pipeline(
    fix: &Fixture,
    pin_out: &mut [PinnedChunk],
    dev_h: &DeviceCuda,
    dev_c: &DeviceCuda,
    dev_d: &DeviceCuda,
    iters: usize,
    blocking: bool,
) -> (Duration, Vec<Vec<f64>>) {
    // Issue order is the pipeline. An event recorded at issue time covers
    // everything enqueued on its stream so far, so the three stages are issued
    // per chunk: compute(c) waits for H2D(c) only, not for every copy issued
    // before it (phase-wise issue -- all H2D, then all compute, then all D2H --
    // measures as fully serial). Every cross-stream device buffer stays alive
    // until the final sync (a buffer freed on its allocation stream while a
    // foreign-stream kernel still reads it is the known read-lifetime hole).
    let t0 = Instant::now();
    let mut keep: Vec<Tensor<f64, DeviceCuda>> = Vec::new();
    let mut results = Vec::new();
    for (c, t) in fix.pin_in.iter().enumerate() {
        keep.push(t.to_device(dev_h)); // H2D on H
        if blocking {
            dev_h.synchronize().unwrap();
        }
        let g_c = keep[c].to_device(dev_c);
        let r = compute(&g_c, iters); // on C, waits for H2D(c) only
        if blocking {
            dev_c.synchronize().unwrap();
        }
        let r_d = r.to_device(dev_d);
        d2h_pinned_async(dev_d, r_d.raw(), pin_out[c].raw_mut()).unwrap(); // on D, waits for compute(c)
        if blocking {
            dev_d.synchronize().unwrap();
        }
        results.push(r);
    }
    let t_enq = t0.elapsed();
    dev_h.synchronize().unwrap();
    dev_c.synchronize().unwrap();
    dev_d.synchronize().unwrap();
    let d = t0.elapsed();
    if std::env::var("RSTSR_BENCH_TRACE").is_ok() {
        println!("    enqueue {t_enq:?}, final sync {:?}", d - t_enq);
    }
    let out: Vec<Vec<f64>> = pin_out
        .iter()
        .map(|o| o.raw().to_vec_f().unwrap())
        .collect();
    drop(results);
    drop(keep);
    (d, out)
}

fn median(v: &mut [Duration]) -> Duration {
    v.sort();
    v[v.len() / 2]
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn main() {
    let chunks = env_usize("RSTSR_BENCH_CHUNKS", CHUNKS);
    let rounds = env_usize("RSTSR_BENCH_ROUNDS", ROUNDS);
    let iters = env_usize("RSTSR_BENCH_ITERS", 0); // 0 = autotune

    if !rstsr_cuda::funcs::is_available() {
        eprintln!("skipping: no CUDA device visible");
        return;
    }
    let base = DeviceCuda::new(0);
    let pinned = DeviceCudaPinned::new(0);
    let cpu = DeviceCpuSerial::default();
    let bytes = chunks * CHUNK_MIB << 20;
    println!(
        "chunks {chunks} x {CHUNK_MIB} MiB f64 ({:.0} MiB per direction), {rounds} rounds, {base}",
        bytes as f64 / 1048576.0
    );

    // fixture: pinned inputs/outputs preallocated outside every timed region
    let mut pin_in = Vec::new();
    let mut cpu_in = Vec::new();
    for c in 0..chunks {
        let v = arange_vec(ELEMS, c * ELEMS);
        cpu_in.push(rt::asarray((v.clone(), &cpu)));
        pin_in.push(
            pinned
                .tensor_f(pinned.from_vec_f(v).unwrap(), [ELEMS])
                .unwrap(),
        );
    }
    let mut pin_out: Vec<PinnedChunk> = (0..chunks)
        .map(|_| {
            pinned
                .tensor_f(pinned.zeros_f::<f64>(ELEMS).unwrap(), [ELEMS])
                .unwrap()
        })
        .collect();
    let fix = Fixture { pin_in, cpu_in };
    // D8 experiment: write-combined pages as D2H destinations (fast device writes,
    // uncached host reads) vs the default cached pages
    let mut wc_out: Vec<PinnedChunk> = (0..chunks)
        .map(|_| {
            let buf = PinnedBuffer::<f64>::with_flags_f(
                pinned.context(),
                ELEMS,
                pinned_flags::WRITE_COMBINED,
            )
            .unwrap();
            pinned.tensor_f(buf, [ELEMS]).unwrap()
        })
        .collect();

    warm_clocks(&base, 1500);

    // autotune: compute ~ 1.15 x (h2d + d2h)
    let t_h = ms(phase_h2d(&fix, &base));
    let t_d = ms(phase_d2h(&fix, &base, &mut pin_out));
    let iters = if iters > 0 {
        iters
    } else {
        let per_link = ms(phase_compute(&fix, &base, 1));
        ((1.15 * (t_h + t_d) / per_link).ceil() as usize).max(1)
    };
    println!("h2d {t_h:.2} ms  d2h {t_d:.2} ms  -> compute links per chunk: {iters}");

    let dev_h = base.new_stream();
    let dev_c = base
        .clone()
        .with_stream(DeviceCuda::new_stream_with_ordinal(0).stream());
    let dev_d = base
        .clone()
        .with_stream(DeviceCuda::new_stream_with_ordinal(0).stream());
    let one = base.clone();

    // warm every mode once (kernel compile, pool growth)
    let (_t, mut serial_ref) = serial_all(&fix, &base, iters);
    pipeline(&fix, &mut pin_out, &dev_h, &dev_c, &dev_d, iters, false);
    pipeline(&fix, &mut pin_out, &dev_h, &dev_c, &dev_d, iters, true);
    pipeline(&fix, &mut pin_out, &one, &one, &one, iters, false);
    pipeline_pageable(&fix, &dev_h, &dev_c, &dev_d, iters);
    phase_compute(&fix, &base, iters);

    let names = [
        "h2d",
        "compute",
        "d2h",
        "serial_all",
        "pipeline",
        "pipeline_blocking",
        "pipeline_1stream",
        "pipeline_pageable",
        "d2h_writecombined",
    ];
    let mut acc: Vec<Vec<Duration>> = vec![Vec::new(); names.len()];
    let mut pipe_out: Vec<Vec<f64>> = Vec::new();
    for _ in 0..rounds {
        warm_clocks(&base, 150); // clock guard between rounds
        acc[0].push(phase_h2d(&fix, &base));
        acc[1].push(phase_compute(&fix, &base, iters));
        acc[2].push(phase_d2h(&fix, &base, &mut pin_out));
        let (t, o) = serial_all(&fix, &base, iters);
        acc[3].push(t);
        serial_ref = o;
        let (t, o) = pipeline(&fix, &mut pin_out, &dev_h, &dev_c, &dev_d, iters, false);
        acc[4].push(t);
        pipe_out = o;
        acc[5].push(pipeline(&fix, &mut pin_out, &dev_h, &dev_c, &dev_d, iters, true).0);
        acc[6].push(pipeline(&fix, &mut pin_out, &one, &one, &one, iters, false).0);
        acc[7].push(pipeline_pageable(&fix, &dev_h, &dev_c, &dev_d, iters));
        acc[8].push(phase_d2h(&fix, &base, &mut wc_out));
    }

    let med: Vec<Duration> = acc.iter().map(|v| median(&mut v.clone())).collect();
    let (serial, pipe) = (med[3], med[4]);
    let phase_sum = med[0] + med[1] + med[2];
    let bound = med[1].max(med[0] + med[2]);
    println!();
    println!("phase totals (isolated, interleaved median)");
    println!(
        "  h2d         {:7.2} ms   {:.2} GB/s",
        ms(med[0]),
        bytes as f64 / med[0].as_secs_f64() / 1e9
    );
    println!("  compute     {:7.2} ms", ms(med[1]));
    println!(
        "  d2h         {:7.2} ms   {:.2} GB/s",
        ms(med[2]),
        bytes as f64 / med[2].as_secs_f64() / 1e9
    );
    println!("  predicted serial (sum)        {:7.2} ms", ms(phase_sum));
    println!("  predicted pipeline (max term) {:7.2} ms", ms(bound));
    println!();
    println!("mode                       ms   vs serial_all   pipeline eff");
    for (name, m) in names.iter().zip(med.iter()) {
        let eff = if *name == "pipeline" {
            ms(bound) / ms(*m) * 100.0
        } else {
            0.0
        };
        println!(
            "{name:<22} {:8.2}       {:5.2}x        {:5.1}%",
            ms(*m),
            ms(serial) / ms(*m),
            eff
        );
    }
    println!();
    println!(
        "transfer time hidden by the pipeline: {:.1}% of h2d+d2h",
        ms(phase_sum - pipe) / ms(med[0] + med[2]) * 100.0
    );
    println!(
        "speedup, serial_all (pageable, blocking) -> pipeline: {:.2}x",
        ms(serial) / ms(pipe)
    );
    println!(
        "speedup, pipeline_1stream (pinned, no overlap) -> pipeline: {:.2}x",
        ms(med[6]) / ms(pipe)
    );

    // correctness: pipeline output bitwise equal to the serial flow
    for c in 0..chunks {
        assert_eq!(pipe_out[c].len(), serial_ref[c].len());
        let bad = pipe_out[c]
            .iter()
            .zip(serial_ref[c].iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            bad, 0,
            "chunk {c}: {bad} elements differ from the serial flow"
        );
    }
    // the controls are pinned no-overlap flows: they must sit at the sum of the phases
    assert!(
        ms(med[5]) > 0.95 * ms(phase_sum),
        "pipeline_blocking {:.2} ms is below the phase sum {:.2} ms",
        ms(med[5]),
        ms(phase_sum)
    );
    assert!(
        ms(med[6]) > 0.95 * ms(phase_sum),
        "pipeline_1stream {:.2} ms is below the phase sum {:.2} ms",
        ms(med[6]),
        ms(phase_sum)
    );
    // informational only (see the fn doc): rstsr's pageable path still
    // overlaps, so this probe cannot serve as the pinned negative control
    // against the naive flow (pageable, blocking transfers): the headline number
    assert!(
        ms(pipe) < 0.7 * ms(serial),
        "pipeline {:.2} ms is not < 0.7 x serial {:.2} ms",
        ms(pipe),
        ms(serial)
    );
    // against the pinned no-overlap flow: the transfer time is (mostly) hidden
    assert!(
        ms(pipe) < 0.75 * ms(med[6]),
        "pipeline {:.2} ms is not < 0.75 x pipeline_1stream {:.2} ms",
        ms(pipe),
        ms(med[6])
    );
    println!("assertions passed: bitwise equal; pipeline < 0.7 x serial_all and < 0.75 x pipeline_1stream; controls at the phase sum");
}

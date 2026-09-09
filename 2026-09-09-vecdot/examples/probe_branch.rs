//! Layout-branch probe: replicates the branch-selection logic of
//! `vecdot_naive_cpu_serial` (rstsr-native-impl/src/cpu_serial/vecdot.rs:27-68)
//! for every bench geometry, using the same public layout API, so we can state
//! file:line which branch serves which bench case at 386948be.
//!
//! Logic replicated (same order as the kernel):
//! 1. `flag_contig_s` = summed axes of a and b share a common contiguous
//!    prefix (`get_axes_composition` comp_c lists, izip-broken).
//! 2. `flag_contig_m` = remaining axes of a, b AND c share a common
//!    contiguous prefix.
//! 3. else general branch (per-output stride fold when las.ndim()==1).
//! Branch order in the kernel: contig_s FIRST, then contig_m, then general.

use rstsr_common::layout::broadcast::broadcast_layout;
use rstsr_common::layout::indexer::IndexerDynamicAPI;
use rstsr_common::layout::rearrangement::get_axes_composition;
use rstsr_core::prelude::*;

fn analyze(name: &str, la: &Layout<IxD>, lb: &Layout<IxD>, lc: &Layout<IxD>, axes_a: &[isize], axes_b: &[isize]) {
    let (las, lam) = la.dim_split_axes(axes_a).unwrap();
    let (lbs, lbm) = lb.dim_split_axes(axes_b).unwrap();

    // accelerate flags generation -- verbatim shape of the kernel's logic
    let mut asc = vec![];
    let (_, _, asc_a, _) = get_axes_composition(&las);
    let (_, _, asc_b, _) = get_axes_composition(&lbs);
    for (ic_a, ic_b) in asc_a.iter().zip(asc_b.iter()) {
        if ic_a == ic_b {
            asc.push(*ic_a);
        } else {
            break;
        }
    }
    let flag_contig_s = !asc.is_empty();

    let mut amc = vec![];
    let (_, _, amc_a, _) = get_axes_composition(&lam);
    let (_, _, amc_b, _) = get_axes_composition(&lbm);
    let (_, _, amc_c, _) = get_axes_composition(lc);
    for ((ic_a, ic_b), ic_c) in amc_a.iter().zip(amc_b.iter()).zip(amc_c.iter()) {
        if ic_a == ic_b && ic_b == ic_c {
            amc.push(*ic_a);
        } else {
            break;
        }
    }
    let flag_contig_m = !amc.is_empty();

    let branch = if flag_contig_s {
        "BRANCH-1 contig_s (unrolled_binary_reduce per output, write once)"
    } else if flag_contig_m {
        "BRANCH-2 contig_m (MaybeUninit RMW per element per contraction step)"
    } else {
        "BRANCH-3 general (per-element stride/index fold)"
    };
    let lasd_desc = if las.ndim() >= 1 && !las.shape().is_empty() {
        format!("stride {}", las.stride()[0])
    } else {
        format!("ndim {}", las.ndim())
    };
    let n_contig_s =
        if flag_contig_s { asc.iter().map(|&i| las.shape()[i]).product::<usize>() } else { 0 };
    println!(
        "{name:44} -> {branch}   [las: {lasd_desc}, n_contig_s={n_contig_s}] [amc len={}]",
        amc.len()
    );
}

fn main() {
    // Row-major contiguous fixtures on a serial device; only layouts matter.
    let dev = DeviceCpuSerial::default();
    let layout_of = |shape: [usize; 2]| -> Layout<IxD> {
        let t = rt::asarray((vec![0.0_f64; shape[0] * shape[1]], shape, &dev));
        t.layout().to_dim::<IxD>().unwrap()
    };
    let layout_t = |shape: [usize; 2]| -> Layout<IxD> {
        // transposed view: stored (shape[1], shape[0]) contiguous
        let t = rt::asarray((vec![0.0_f64; shape[0] * shape[1]], [shape[1], shape[0]], &dev));
        t.t().layout().to_dim::<IxD>().unwrap()
    };
    let layout_v = |n: usize| -> Layout<IxD> {
        let t = rt::asarray((vec![0.0_f64; n], [n], &dev));
        t.layout().to_dim::<IxD>().unwrap()
    };

    println!("=== vecdot bench geometries at 386948be (row-major default order) ===");
    // 1-D dot n: both 1-D, contract -1
    analyze("dot1d n=1e7 (1-D dot)", &layout_v(10_000_000), &layout_v(10_000_000), &layout_v(1), &[-1], &[-1]);
    // batched am1: (m,k).(m,k) contract -1
    analyze(
        "batched_am1 (4096,512).(4096,512) ax-1",
        &layout_of([4096, 512]),
        &layout_of([4096, 512]),
        &layout_v(4096),
        &[-1],
        &[-1],
    );
    analyze(
        "batched_am1 (8192,256).(8192,256) ax-1",
        &layout_of([8192, 256]),
        &layout_of([8192, 256]),
        &layout_v(8192),
        &[-1],
        &[-1],
    );
    // batched axis0: (k,m).(k,m) contract 0 -> output (m,)
    analyze(
        "batched_axis0 (512,4096).(512,4096) ax0",
        &layout_of([512, 4096]),
        &layout_of([512, 4096]),
        &layout_v(4096),
        &[0],
        &[0],
    );
    // strided: a (m,k) contig, b stored (k,m) viewed .t()
    analyze(
        "batched_strided (4096,512).(512,4096).t() ax-1",
        &layout_of([4096, 512]),
        &layout_t([4096, 512]),
        &layout_v(4096),
        &[-1],
        &[-1],
    );
    // odd
    analyze(
        "batched odd (1000,777).(1000,777) ax-1",
        &layout_of([1000, 777]),
        &layout_of([1000, 777]),
        &layout_v(1000),
        &[-1],
        &[-1],
    );
    // broadcast remaining: (m,k).(k,) -> 2-D vs 1-D
    analyze(
        "batched_am1 (1000,777).(777,) ax-1 (brem bcast)",
        &layout_of([1000, 777]),
        &layout_v(777),
        &layout_v(1000),
        &[-1],
        &[-1],
    );
    // broadcast summed: b (1,k) broadcast_to (m,k), contract -1
    {
        let t = rt::asarray((vec![0.0_f64; 777], [1, 777], &dev));
        let t_full = rt::asarray((vec![0.0_f64; 1000 * 777], [1000, 777], &dev));
        let (tb, _) = broadcast_layout(t.layout(), t_full.layout(), RowMajor).unwrap();
        analyze(
            "batched_am1 (1000,777).(1000,777)bcast ax-1",
            &layout_of([1000, 777]),
            &tb.to_dim::<IxD>().unwrap(),
            &layout_v(1000),
            &[-1],
            &[-1],
        );
    }
    // f-contiguous a (col-major), contract -1
    {
        let t = rt::asarray((vec![0.0_f64; 1000 * 777], [777, 1000], &dev));
        let taf = t.t(); // f-contiguous (1000,777) view
        analyze(
            "batched_am1 fcontig-a (1000,777).(1000,777) ax-1",
            &taf.layout().to_dim::<IxD>().unwrap(),
            &layout_of([1000, 777]),
            &layout_v(1000),
            &[-1],
            &[-1],
        );
    }
}

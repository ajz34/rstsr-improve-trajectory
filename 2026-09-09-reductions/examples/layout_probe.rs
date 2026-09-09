//! Layout-branch probe for `reduce_axes_cpu_{serial,rayon}` (phase-1
//! anatomy). Read-only: calls rstsr's PUBLIC layout machinery to determine,
//! for each bench-relevant case, which kernel branch fires and whether the
//! final order-fixup copy fires. No rstsr edits.
//!
//! Reproduces the decisions of `reduce_axes_cpu_serial`
//! (rstsr-native-impl/src/cpu_serial/reduction.rs:180-370):
//! - branch 1 (unrolled inner):  size_sc > 1  (contiguous part to be SUMMED)
//! - branch 2 (CHUNK=48 fold):   size_mc > 1  (contiguous part REMAINS)
//! - branch 3 (plain fold):      neither
//! - broadcast-remaining pass:   size_m0 > 1
//! - order-fixup full copy:      FlagOrder default != K  &&  lo_default != lo
//!
//! Run: cargo run --release --example layout_probe

use rstsr_common::layout::indexer::IndexerDynamicAPI;
use rstsr_common::layout::rearrangement::{
    get_axes_composition, layout_for_array_copy, translate_to_col_major_unary, translate_to_col_major_with_contig,
};
use rstsr_core::prelude::*;

fn probe_case(name: &str, la: &Layout<IxD>, axes: &[isize]) {
    println!(
        "=== {name}: input shape {:?} stride {:?} offset {}",
        la.shape().as_ref() as &[usize],
        la.stride().as_ref() as &[isize],
        la.offset()
    );

    let (ls, lm) = match la.dim_split_axes(axes) {
        Ok(v) => v,
        Err(e) => {
            println!("    dim_split_axes failed: {e}");
            return;
        },
    };
    println!(
        "    ls (summed):    shape {:?} stride {:?}",
        ls.shape().as_ref() as &[usize],
        ls.stride().as_ref() as &[isize]
    );
    println!(
        "    lm (remaining): shape {:?} stride {:?}",
        lm.shape().as_ref() as &[usize],
        lm.stride().as_ref() as &[isize]
    );

    let (as1, as0, asc, asd) = get_axes_composition(&ls);
    let (_am1, am0, amc, amd) = get_axes_composition(&lm);
    let size_s0 = as0.iter().map(|&i| lm.shape()[i]).product::<usize>();
    let size_sc = asc.iter().map(|&i| ls.shape()[i]).product::<usize>();
    let size_m0 = am0.iter().map(|&i| lm.shape()[i]).product::<usize>();
    let size_mc = amc.iter().map(|&i| lm.shape()[i]).product::<usize>();
    println!(
        "    ls comp: singleton(as1)={as1:?} broadcast(as0)={as0:?} contig(asc)={asc:?} discontig(asd)={asd:?}"
    );
    println!(
        "    lm comp: singleton(am1)={_am1:?} broadcast(am0)={am0:?} contig(amc)={amc:?} discontig(amd)={amd:?}"
    );
    println!("    size_sc={size_sc} size_mc={size_mc} size_s0={size_s0} size_m0={size_m0}");

    let branch = if size_sc > 1 {
        "BRANCH 1: contiguous-summed (unrolled_reduce inner)"
    } else if size_mc > 1 {
        "BRANCH 2: contiguous-remaining (CHUNK clone-fold)"
    } else {
        "BRANCH 3: plain fold"
    };
    println!("    -> {branch}");

    let lo = layout_for_array_copy(&lm, TensorIterOrder::K).unwrap();
    let lo_default = layout_for_array_copy(&lm, TensorIterOrder::default()).unwrap();
    let fixup_condition = TensorIterOrder::default() != TensorIterOrder::K && lo_default != lo;
    println!(
        "    lo (K):       shape {:?} stride {:?}",
        lo.shape().as_ref() as &[usize],
        lo.stride().as_ref() as &[isize]
    );
    println!(
        "    lo (default): shape {:?} stride {:?}",
        lo_default.shape().as_ref() as &[usize],
        lo_default.stride().as_ref() as &[isize]
    );
    println!(
        "    order-fixup copy fires: {fixup_condition} (FlagOrder default = {:?})",
        FlagOrder::default()
    );
    println!();
}

fn main() {
    println!("FlagOrder::default() = {:?}  TensorIterOrder::default() = {:?}", FlagOrder::default(), TensorIterOrder::default());
    println!();
    let dev = DeviceCpuSerial::default();

    // 2-D row-major, both axes (the benched headline cases)
    let a: Tensor<f64, _, _> = rt::asarray((vec![0.0; 2048 * 2048], [2048usize, 2048], &dev));
    probe_case("2048x2048 rm, axes=[0]", a.layout(), &[0]);
    probe_case("2048x2048 rm, axes=[-1]", a.layout(), &[-1]);

    // odd
    let b: Tensor<f64, _, _> = rt::asarray((vec![0.0; 1000 * 777], [1000usize, 777], &dev));
    probe_case("1000x777 rm, axes=[0]", b.layout(), &[0]);
    probe_case("1000x777 rm, axes=[-1]", b.layout(), &[-1]);

    // 3-D
    let c: Tensor<f64, _, _> = rt::asarray((vec![0.0; 8 * 9 * 10], [8usize, 9, 10], &dev));
    probe_case("8x9x10 rm, axes=[0]", c.layout(), &[0]);
    probe_case("8x9x10 rm, axes=[-1]", c.layout(), &[-1]);
    probe_case("8x9x10 rm, axes=[0,1]", c.layout(), &[0, 1]);

    // transposed view
    let at = a.t();
    probe_case("2048x2048 t-view, axes=[0]", at.layout(), &[0]);
    probe_case("2048x2048 t-view, axes=[-1]", at.layout(), &[-1]);

    // f-contiguous (col-major stored) input
    let af = a.to_contig(ColMajor);
    probe_case("2048x2048 f-contig, axes=[0]", af.layout(), &[0]);
    probe_case("2048x2048 f-contig, axes=[-1]", af.layout(), &[-1]);

    // broadcast input: [1, 2048] broadcast to [2048, 2048]
    let row: Tensor<f64, _, _> = rt::asarray((vec![0.0; 2048], [1usize, 2048], &dev));
    let rb = row.broadcast_to(vec![2048, 2048]);
    probe_case("broadcast [2048,2048] of [1,2048], axes=[0]", rb.layout(), &[0]);
    probe_case("broadcast [2048,2048] of [1,2048], axes=[-1]", rb.layout(), &[-1]);

    // full reduction goes through reduce_all (different function): document
    // its branch too (translate_to_col_major K + size_contig >= 32 ->
    // unrolled_reduce)
    let layout = translate_to_col_major_unary(a.layout(), TensorIterOrder::K).unwrap();
    let (_lc, size_contig) = translate_to_col_major_with_contig(&[&layout]);
    println!("=== reduce_all 2048x2048 rm: after translate K, size_contig={size_contig} (CONTIG_SWITCH=32)");
}

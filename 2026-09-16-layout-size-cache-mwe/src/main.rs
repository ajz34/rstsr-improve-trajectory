// Prints size_of of the layout variants (the "+8 B derived field" cost side
// of the §4.5 question). Run: cargo run --release

use size_cache_mwe::*;

fn main() {
    println!("LayoutVecR (recompute, IxD regime): {:3} B", std::mem::size_of::<LayoutVecR>());
    println!("LayoutVecC (cached,    IxD regime): {:3} B", std::mem::size_of::<LayoutVecC>());
    println!("LayoutArrR<2> (recompute):          {:3} B", std::mem::size_of::<LayoutArrR<2>>());
    println!("LayoutArrC<2> (cached):             {:3} B", std::mem::size_of::<LayoutArrC<2>>());
    println!("LayoutArrR<4> (recompute):          {:3} B", std::mem::size_of::<LayoutArrR<4>>());
    println!("LayoutArrC<4> (cached):             {:3} B", std::mem::size_of::<LayoutArrC<4>>());
    println!("Option<LayoutVecR>:                 {:3} B", std::mem::size_of::<Option<LayoutVecR>>());
    println!("Option<LayoutVecC>:                 {:3} B", std::mem::size_of::<Option<LayoutVecC>>());
    println!("Option<LayoutArrR<4>>:              {:3} B", std::mem::size_of::<Option<LayoutArrR<4>>>());
    println!("Option<LayoutArrC<4>>:              {:3} B", std::mem::size_of::<Option<LayoutArrC<4>>>());
    // reference args (dominant use in kernels) are unaffected:
    println!("&LayoutVecR / &LayoutVecC:          {:3} B / {:3} B",
        std::mem::size_of::<&LayoutVecR>(), std::mem::size_of::<&LayoutVecC>());
}

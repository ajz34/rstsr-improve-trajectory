use rstsr::prelude::*;

fn show<D: DimAPI>(name: &str, t: &Tensor<f64, DeviceCpu, D>) {
    println!("{name}: shape={:?} stride={:?} c_prefer={} f_prefer={} c_contig={} f_contig={}",
        t.shape(), t.stride(), t.c_prefer(), t.f_prefer(), t.c_contig(), t.f_contig());
}

fn main() {
    let dev = DeviceCpu::default();

    let l: Layout<Ix2> = [0, 18].f();
    println!("static [0,18].f()       stride {:?}", l.stride());
    let l: Layout<Ix2> = [18, 0].f();
    println!("static [18,0].f()       stride {:?}", l.stride());
    let l: Layout<Ix2> = [0, 18].c();
    println!("static [0,18].c()       stride {:?}", l.stride());
    let l: Layout<Ix2> = [18, 0].c();
    println!("static [18,0].c()       stride {:?}", l.stride());

    let l: Layout<IxD> = vec![0, 18].f();
    println!("dynamic vec![0,18].f()  stride {:?}", l.stride());
    let l: Layout<IxD> = vec![18, 0].f();
    println!("dynamic vec![18,0].f()  stride {:?}", l.stride());
    let l: Layout<IxD> = vec![0, 18].c();
    println!("dynamic vec![0,18].c()  stride {:?}", l.stride());
    let l: Layout<IxD> = vec![18, 0].c();
    println!("dynamic vec![18,0].c()  stride {:?}", l.stride());

    let a: Tensor<f64, _, Ix2> = unsafe { rt::empty(([0, 18].f(), &dev)) }.into_dim::<Ix2>();
    show("empty static f [0,18]", &a);
    let a: Tensor<f64, _, Ix2> = unsafe { rt::empty(([18, 0].f(), &dev)) }.into_dim::<Ix2>();
    show("empty static f [18,0]", &a);
    let a: Tensor<f64, _, IxD> = unsafe { rt::empty((vec![0, 18], &dev)) };
    show("empty dynamic default(C) [0,18]", &a);
    let a: Tensor<f64, _, IxD> = unsafe { rt::empty((vec![18, 0], &dev)) };
    show("empty dynamic default(C) [18,0]", &a);
    let a: Tensor<f64, DeviceCpu, IxD> = rt::zeros(vec![0, 18]);
    show("zeros dynamic default(C) [0,18]", &a);
    let a: Tensor<f64, _, Ix2> = rt::zeros([0usize, 18].f()).into_dim::<Ix2>();
    show("zeros static f [0,18]", &a);

    let buf: Vec<f64> = Vec::new();
    let b = rt::asarray((&buf[..], [0, 18].f(), &dev)).into_reverse_axes().into_dim::<Ix2>();
    println!("asarray [0,18].f() reverse: shape={:?} stride={:?} ld_col={:?} ld_row={:?}",
        b.shape(), b.stride(), b.ld_col(), b.ld_row());
    let b2 = rt::asarray((&buf[..], [18, 0].f(), &dev)).into_dim::<Ix2>();
    println!("asarray [18,0].f() direct:  shape={:?} stride={:?} ld_col={:?} ld_row={:?}",
        b2.shape(), b2.stride(), b2.ld_col(), b2.ld_row());

    let c: Tensor<f64, DeviceCpu, Ix2> = rt::zeros(vec![0, 18]).into_reverse_axes().into_dim::<Ix2>();
    println!("zeros dyn [0,18] reverse:   shape={:?} stride={:?} ld_col={:?} ld_row={:?}",
        c.shape(), c.stride(), c.ld_col(), c.ld_row());
    let c: Tensor<f64, DeviceCpu, Ix2> = rt::zeros(vec![18, 0]).into_dim::<Ix2>();
    println!("zeros dyn [18,0] direct:    shape={:?} stride={:?} ld_col={:?} ld_row={:?}",
        c.shape(), c.stride(), c.ld_col(), c.ld_row());

    let d = rt::asarray((&buf[..], [0, 18].f(), &dev)).into_reverse_axes();
    let d2 = d.to_contig_f(ColMajor).unwrap();
    println!("reverse then to_contig_f(F): shape={:?} stride={:?}", d2.shape(), d2.stride());
    let d3: Tensor<f64, DeviceCpu, IxD> = rt::zeros(vec![0, 18]).into_reverse_axes();
    let d4 = d3.to_contig_f(ColMajor).unwrap();
    println!("dyn reverse then to_contig_f(F): shape={:?} stride={:?}", d4.shape(), d4.stride());
}

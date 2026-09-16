//! MWE: cached vs recomputed `Layout::size()`.
//!
//! Mirrors rstsr `Layout<D>` (aa24643): `{ shape, stride, offset }` where
//! shape/stride are inline arrays or heap Vecs. The "C" variants add a cached
//! `size` field (the §4.5 proposal); the "R" variants recompute the product
//! per call (current rstsr behavior).

/// Heap-backed layout, size recomputed (rstsr `IxD` regime, current behavior).
/// `size_of` = 56 B.
#[derive(Clone)]
pub struct LayoutVecR {
    pub shape: Vec<usize>,
    pub stride: Vec<isize>,
    pub offset: usize,
}

impl LayoutVecR {
    pub fn new(shape: Vec<usize>, stride: Vec<isize>, offset: usize) -> Self {
        Self { shape, stride, offset }
    }
    /// Current rstsr: `self.shape().as_ref().iter().product()`.
    #[inline]
    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Heap-backed layout with cached size (§4.5 proposal). `size_of` = 64 B.
#[derive(Clone)]
pub struct LayoutVecC {
    pub shape: Vec<usize>,
    pub stride: Vec<isize>,
    pub offset: usize,
    pub size: usize,
}

impl LayoutVecC {
    pub fn new(shape: Vec<usize>, stride: Vec<isize>, offset: usize) -> Self {
        // cache maintenance: the product is paid here instead of at each size()
        let size: usize = shape.iter().product();
        Self { shape, stride, offset, size }
    }
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }
}

/// Inline-array layout, size recomputed (rstsr `Dim<[usize; N]>` regime).
#[derive(Clone)]
pub struct LayoutArrR<const N: usize> {
    pub shape: [usize; N],
    pub stride: [isize; N],
    pub offset: usize,
}

impl<const N: usize> LayoutArrR<N> {
    pub fn new(shape: [usize; N], stride: [isize; N], offset: usize) -> Self {
        Self { shape, stride, offset }
    }
    #[inline]
    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Inline-array layout with cached size.
#[derive(Clone)]
pub struct LayoutArrC<const N: usize> {
    pub shape: [usize; N],
    pub stride: [isize; N],
    pub offset: usize,
    pub size: usize,
}

impl<const N: usize> LayoutArrC<N> {
    pub fn new(shape: [usize; N], stride: [isize; N], offset: usize) -> Self {
        let size: usize = shape.iter().product();
        Self { shape, stride, offset, size }
    }
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }
}

/// Iterator-state analog of `IterAxesView`: clones the whole layout per step.
#[derive(Clone)]
pub struct AxisStep<L> {
    pub layout: L,
    pub index: [usize; 4],
}

impl<L: Clone + ShapeRef> AxisStep<L> {
    /// One iteration of `IterAxesView::next`: clone self, advance index,
    /// touch the layout (stride read), return the advanced state.
    #[inline]
    pub fn advanced(mut self) -> Self {
        self.index[0] += 1;
        if self.index[0] == self.layout_stride0_len() {
            self.index[0] = 0;
            self.index[1] += 1;
        }
        self
    }
    #[inline]
    fn layout_stride0_len(&self) -> usize {
        // touch shape memory so the clone+read pattern is honest
        *self.layout.shape_ref().first().unwrap_or(&1)
    }
}

pub trait ShapeRef {
    fn shape_ref(&self) -> &[usize];
}

/// The one method under test, uniform across all four variants.
pub trait LayoutSize {
    fn size(&self) -> usize;
}

impl LayoutSize for LayoutVecR {
    #[inline]
    fn size(&self) -> usize {
        self.shape.iter().product()
    }
}
impl LayoutSize for LayoutVecC {
    #[inline]
    fn size(&self) -> usize {
        self.size
    }
}
impl<const N: usize> LayoutSize for LayoutArrR<N> {
    #[inline]
    fn size(&self) -> usize {
        self.shape.iter().product()
    }
}
impl<const N: usize> LayoutSize for LayoutArrC<N> {
    #[inline]
    fn size(&self) -> usize {
        self.size
    }
}

impl ShapeRef for LayoutVecR {
    fn shape_ref(&self) -> &[usize] {
        &self.shape
    }
}
impl ShapeRef for LayoutVecC {
    fn shape_ref(&self) -> &[usize] {
        &self.shape
    }
}
impl<const N: usize> ShapeRef for LayoutArrR<N> {
    fn shape_ref(&self) -> &[usize] {
        &self.shape
    }
}
impl<const N: usize> ShapeRef for LayoutArrC<N> {
    fn shape_ref(&self) -> &[usize] {
        &self.shape
    }
}

/// Elementwise add over `ntasks` chunks; one `size()` call per task
/// (the actual call pattern in rstsr rayon/serial kernels: size() is
/// per-invocation, never per-element).
pub fn kernel_per_task<L: LayoutSize + ShapeRef>(l: &L, a: &[f64], b: &mut [f64], ntasks: usize) {
    let s = l.size();
    let chunk = s.div_ceil(ntasks);
    for t in 0..ntasks {
        // modeled per-task dispatch cost: each task re-derives bounds from layout
        let s_task = l.size();
        let lo = (t * chunk).min(s_task);
        let hi = ((t + 1) * chunk).min(s_task);
        let mut acc = l.shape_ref()[0] as f64; // keep layout memory in the loop
        for i in lo..hi {
            acc += a[i];
            b[i] = acc;
        }
    }
}

/// Walk a slice; call `size()` every `every` elements (frequency sweep).
/// Returns a checksum so the compiler must keep all calls.
pub fn walk_with_size_freq<L: LayoutSize>(l: &L, a: &[f64], every: usize) -> f64 {
    let mut acc = 0.0f64;
    for (k, &x) in a.iter().enumerate() {
        if k % every == 0 {
            acc += l.size() as f64;
        }
        acc = acc.mul_add(x, 1.0);
    }
    acc
}

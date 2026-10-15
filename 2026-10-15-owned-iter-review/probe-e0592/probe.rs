// E0592 probe: can a *generic* TensorBase block define `iter` while the
// view-specialized block (DataRef storage) also defines it? (mirrors rstsr's
// `impl<'a,T,B,D> TensorView<'a,T,B,D>` vs a hypothetical generic block.)
struct Base<R> { _r: core::marker::PhantomData<R> }
struct Owned;
struct Ref<'a>(*const u8, core::marker::PhantomData<&'a u8>);

type AOwned = Base<Owned>;
type ARef<'a> = Base<Ref<'a>>;

impl<'a> ARef<'a> {
    fn iter(&self) -> core::slice::Iter<'a, u8> { todo!() } // view block: inner-lifetime iter
}
impl<R> Base<R> {
    fn iter(&self) -> core::slice::Iter<'_, u8> { todo!() } // generic block: borrow-tied iter
}
fn main() {}

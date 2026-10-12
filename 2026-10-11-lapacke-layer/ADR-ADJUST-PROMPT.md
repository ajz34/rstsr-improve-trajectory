Now we will discussing the ADR contents edits.
My comprehension is not always correct, and you are going to make validations and testify your self and my self. Also, ADR should follow common writing style, and you should follow common ADR style, not adding defensive things.

With the current changes on functions like gesdd, you may now gain what we actually do. You are now going to fully review and rewrite the adr-0009, and perhaps skill.

You are not only going to adjust scope, but all other details. However, scope adjustment is a most important issue.

**Scope adjustment**

We'd narrow down the scope of LAPACKE translation layer to certain files like `rstsr-blas-traits/driver_impl/lapack/svd/gesdd.rs`. These files are truely direct translation, corresponds to the non-translated `rstsr-blas-traits/driver_impl/lapacke/svd/gesdd.rs`: the former file should work exactly the same to the latter. Only because many BLAS vendors do not export LAPACKE in its shared library distributions, that we need to make this intermediate layer of handling row-order and column-order lapack computation via such kind of interface.

The LAPACKE translation layer (of lapack), or direct lapacke computation, will then interface to a layer called BLAS/LAPACK interface layer. Example is `rstsr-blas-traits/src/lapack_svd/gesdd.rs`.

You may see LAPACKE translation layer is true translation. In most cases, it must follow C code very closely. We as rust certainly have some differences to C code, but the mapping relations (like transpose function usage, lwork cast usage, error code handling) is crystal clear in most cases. What's different may be that we can use some of macro duplication on f32/f64 and c32/c64, respectively, and we can use trait function in some cases.

LAPACKE translation layer is connected to BLAS/LAPACK interface layer by driver API. Note that usually (BUT NOT ALL), complex can use and propose exactly same trait function signature to its real counterpart. If not, then this is a difficult problem that may need human interfer.

You may create a new ADR on BLAS/LAPACK layer (which involves some order handling), which may be more tricky that, though some rules should be applied (especially order rules that help input mutable/owned tensor to reuse memory buffer, similarly like `overwrite` argument in scipy), they need some non-trivial adjustments. This is a more complicated layer.

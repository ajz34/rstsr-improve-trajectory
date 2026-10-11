Q1: I thought of this way.

This can be implemented in future or now (possibly future). But this is resolvable.

Take syev as example:

```rust
pub trait SYEVDriverAPI<T>
where
    T: BlasFloat,
{
    unsafe fn driver_syev(
        order: FlagOrder,
        jobz: char,
        uplo: FlagUpLo,
        n: usize,
        a: *mut T,
        lda: usize,
        w: *mut T::Real,
    ) -> blas_int;
}
```

We may change to

```rust
pub trait SYEVDriverAPI<T>
where
    T: BlasFloat,
{
    unsafe fn driver_syev(
        order: FlagOrder,
        jobz: char,
        uplo: FlagUpLo,
        n: usize,
        a: *mut T,
        lda: usize,
        w: *mut T::Real,
    ) -> blas_int {
        // Default implementation

        let nan_check = if let Some(nc) {
            nc
        } else {
            #[cfg(nan_check)]
            true
            #[cfg(not(nan_check))]
            false
        }

        driver_adv_syev(
            None, nan_check, order, jobz, uplo, n, a, lda, w,
        )
    }

    unsafe fn driver_adv_syev(
        nthreads: Option<usize>,
        nan_check: bool,
        order: FlagOrder,
        jobz: char,
        uplo: FlagUpLo,
        n: usize,
        a: *mut T,
        lda: usize,
        w: *mut T::Real,
    ) -> blas_int;
}
```

And then move implementation from `driver_syev` to `driver_adv_syev`.
This is a proposal that you see if this is sounding. You are not supposed to immediately implement this.

Q2: Okay. Notice make new branch to each repo.

Q3: Okay.

Q4: Yes and we allow local override.

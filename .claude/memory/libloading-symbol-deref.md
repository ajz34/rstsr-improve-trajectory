---
name: libloading-symbol-deref
description: libloading 0.9 Symbol<T> deref yields the dlsym ADDRESS reinterpreted as T ("additional reference level"), not the data at it - for a data symbol use T = pointer-to-data and dereference once more; passing the address where the pointed-to value is expected silently corrupts the argument (cuTENSOR compute descriptors answered NOT_INITIALIZED at operation creation); a minimal C dlopen repro isolated it after pointer/context/RTLD theories were excluded
metadata:
  type: reference
---

Found 2026-11-28 while building the rstsr-cuda cuTENSOR wrapper (`2026-11-28-cuda-cutensor-wrap`).

libloading 0.9 `Symbol<T>` deref (os/unix/mod.rs:444-452, "Additional reference level for a dereference
on `deref` return value"): `*sym` = **the dlsym result itself** reinterpreted as `T`. For functions that
is the callable (the normal use). For a *data* global, `*sym` is the address of the variable, and the
value needs one more dereference:

```rust
let sym = lib.get::<*mut core::ffi::c_void>(b"SYMBOL\0")?;   // *sym == dlsym address
let value = unsafe { *( *sym as *const DataType ) };          // read the global's content
```

Corruption mode: passing `*sym` where the pointed-to value is expected — every type check passes (both
are pointer-sized), and the library fails downstream. cuTENSOR answered `CUTENSOR_STATUS_NOT_INITIALIZED`
at `cutensorCreateContraction` (the first call consuming the compute descriptor) while
`cutensorCreate`/descriptor creation succeeded; the earlier B1 unit only asserted non-null.

Debug path that worked (after wrong theories: thread context, RTLD_LOCAL/GLOBAL, cudart pre-load,
linked-vs-dlopen — all excluded by experiment): a **minimal C dlopen+dlsym repro** (`dl_test.c` in
`probe/versions/`) with the same call sequence — C failed identically, which shifted suspicion from the
environment to the argument values themselves; printing dlsym's address vs the dereferenced value
showed the mismatch.

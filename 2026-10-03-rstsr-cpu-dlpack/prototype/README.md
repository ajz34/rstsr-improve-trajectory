# Prototype: Rust tests + Python end-to-end harness

The crate itself lives in the rstsr checkout on branch `261003/rstsr-cpu-dlpack`
(`crates-interop/rstsr-cpu-dlpack`); `../../../rstsr` is a path dependency of `demo-ffi`.
`rstsr-cpu-dlpack/` here is an **archival snapshot** of the crate at the moment it moved —
do not develop in it.

## Rust (the crate + its tests)

```bash
cd ../../../rstsr
cargo test -p rstsr-cpu-dlpack                      # 18 tests + doctest
cargo +nightly miri test -p rstsr-cpu-dlpack        # same suite under miri
```

## Python end-to-end (real CPython + NumPy capsules)

```bash
cd prototype
cargo build -p rstsr-cpu-dlpack-demo-ffi            # builds target/debug/librstsr_cpu_dlpack_demo_ffi.so
cd python
conda run -n torch python test_e2e.py               # 8 cases, 64 checks; e2e-output.txt is a capture
```

`demo-ffi` is a prototype host only: it owns a handle table of rstsr tensors and exports a tiny C
ABI (`rstsr_demo_*`). Its most interesting piece is `dlpack_capsule_destructor`: the `PyCapsule`
destructor is written in Rust (calling the export's DLPack deleter when the capsule was never
consumed), which is what a pyo3-based host would do with pyo3's capsule API. `python/rstsr_dlpack.py`
is the reference Python-side holder the design talks about.

## Example: tensors and basic-indexed views, both directions

```bash
cd prototype/python
conda run -n torch python example_slicing.py        # prints the trace; example-slicing-output.txt is a capture
```

`example_slicing.py` is a narrated, runnable walk-through of the protocol with slices:

- **NumPy -> rstsr**: a whole array, a strided slice (`a[1:, ::2]`) and a reversed slice
  (`a[::-1, ::-1]`, negative strides) are handed to the host through real capsules; the script
  prints the `(shape, strides, offset)` the Rust side reconstructs, the (unchanged) element-zero
  address, and — for the negative strides — the span re-basing below element zero.
- **rstsr -> NumPy**: a Rust-built `arange(12).reshape(3, 4)` is moved into the shared repr,
  basic-indexed on the Rust side (`t[1:, ::2]`, `t[:, ::-1]`), and the views are exported
  zero-copy; NumPy's pointers and strides are compared against the NumPy-made views of the first
  part, and `np.shares_memory` against the whole-tensor export.

Every step compares buffer addresses, so the zero-copy claim is checkable, and mixed
offset/stride combinations cross unchanged. The host entry points it adds to `demo-ffi`
(`rstsr_demo_arange2d_f64`, `rstsr_demo_export_slice`, `rstsr_demo_layout`, `rstsr_demo_buffer_ptr`)
are in `demo-ffi/src/lib.rs`.

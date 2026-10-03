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

#!/usr/bin/env python
"""Example: DLPack between NumPy and rstsr - whole tensors and basic-indexed
(strided, offset) views, both directions, zero copy.

Run: conda run -n torch python example_slicing.py

- Direction A (NumPy -> rstsr): take a NumPy array, a strided slice of it, and
  a reversed (negative-stride) slice; hand each to the Rust host through
  `__dlpack__` + capsule, and print the layout the Rust side reconstructs.
- Direction B (rstsr -> NumPy): build a tensor on the Rust side, basic-index it
  there (NumPy slice conventions), export the view zero-copy, and import it in
  NumPy; the strided views of A and B are then compared value-by-value.

The Rust host is the `demo-ffi` cdylib wrapping `rstsr-cpu-dlpack`;
`rstsr_dlpack.py` is the Python-side capsule glue.
"""
import ctypes
import gc
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rstsr_demo
from rstsr_dlpack import DlpackExporter, dlpack_from_object

lib = rstsr_demo.load()
SLICE_NONE = -(2**63)  # `isize::MIN`: the host's "no bound" sentinel


def expect(cond, msg):
    if not cond:
        raise AssertionError(msg)


def elems(strides_bytes, itemsize=8):
    return tuple(s // itemsize for s in strides_bytes)


def import_into_rstsr(arr):
    """NumPy -> rstsr: export a capsule from `arr`, import it in the host."""
    return dlpack_from_object(arr, lambda p, legacy: rstsr_demo.import_handle(lib, p, legacy))


def slice_params(spec):
    """Flat `(start, stop, step)` isize triplets; `None` means no bound."""
    flat = (ctypes.c_ssize_t * (3 * len(spec)))()
    for axis, item in enumerate(spec):
        for k, v in enumerate((None, None, None) if item is None else item):
            flat[3 * axis + k] = SLICE_NONE if v is None else v
    return flat


def export_slice(shared, spec):
    """A holder whose `__dlpack__` exports the basic-indexed view, zero-copy."""
    params = slice_params(spec)
    return DlpackExporter(
        lambda: lib.rstsr_demo_export_slice(shared, params, len(spec)),
        destructor=lib.rstsr_demo_capsule_destructor(),
    )


def report_rstsr(handle):
    """Print `(shape, strides, offset)` and the buffer addresses the host sees."""
    shape, strides, offset = rstsr_demo.layout(lib, handle)
    data = lib.rstsr_demo_data_ptr(handle)
    base = lib.rstsr_demo_buffer_ptr(handle)
    print(f"      rstsr      shape={shape} strides(elem)={strides} offset={offset}")
    print(f"                 element-zero 0x{data:x}  buffer base 0x{base:x}")
    return shape, strides, offset


def main():
    print("=" * 76)
    print("DLPack: NumPy <-> rstsr - whole tensors and basic-indexed views (zero copy)")
    print("=" * 76)
    print("host:", lib.rstsr_demo_version().decode())
    print("python:", sys.version.split()[0], " numpy:", np.__version__)

    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    print("\nsource: a = np.arange(12.).reshape(3, 4)")
    print(f"        numpy      shape={a.shape} strides(elem)={elems(a.strides)}  data 0x{a.ctypes.data:x}")

    print("\n[A] NumPy -> rstsr (import; the Rust tensor keeps the array alive)")
    print("[A1] whole array")
    h = import_into_rstsr(a)
    report_rstsr(h)
    expect(lib.rstsr_demo_data_ptr(h) == a.ctypes.data, "A1: same buffer (zero copy)")
    expect(rstsr_demo.values(lib, h) == a.ravel().tolist(), "A1: values")
    print(f"      values     {a.tolist()}")
    lib.rstsr_demo_free(h)

    print("\n[A2] sliced view v = a[1:, ::2]  (row offset 1, column step 2)")
    v = a[1:, ::2]
    print(f"      numpy      shape={v.shape} strides(elem)={elems(v.strides)}  "
          f"data = a.data + {(v.ctypes.data - a.ctypes.data) // 8} elem")
    h = import_into_rstsr(v)
    shape, strides, offset = report_rstsr(h)
    expect(shape == list(v.shape) and tuple(strides) == elems(v.strides), "A2: layout crosses verbatim")
    expect(lib.rstsr_demo_data_ptr(h) == v.ctypes.data, "A2: same buffer (zero copy)")
    expect(rstsr_demo.values(lib, h) == v.ravel().tolist(), "A2: values")
    print(f"      values     {v.tolist()}")
    lib.rstsr_demo_free(h)

    print("\n[A3] reversed view r = a[::-1, ::-1]  (negative strides)")
    r = a[::-1, ::-1]
    print(f"      numpy      shape={r.shape} strides(elem)={elems(r.strides)}  "
          f"data = a.data + {(r.ctypes.data - a.ctypes.data) // 8} elem")
    h = import_into_rstsr(r)
    shape, strides, offset = report_rstsr(h)
    expect(shape == list(r.shape) and tuple(strides) == elems(r.strides), "A3: layout crosses verbatim")
    expect(lib.rstsr_demo_data_ptr(h) == r.ctypes.data, "A3: element zero at the same address")
    expect(rstsr_demo.values(lib, h) == r.ravel().tolist(), "A3: values")
    print(f"      values     {r.tolist()}")
    below = (lib.rstsr_demo_data_ptr(h) - lib.rstsr_demo_buffer_ptr(h)) // 8
    print(f"      note       the Rust span starts {below} elem below element zero "
          f"(layout offset {offset}): negative strides are kept, the address bound stays >= 0")
    lib.rstsr_demo_free(h)

    print("\n[B] rstsr -> NumPy (export; shared = one Arc buffer, repeatable)")
    print("[B0] source: arange(12).reshape(3, 4) built on the Rust side, moved into the shareable repr")
    owned = lib.rstsr_demo_arange2d_f64(3, 4)
    expect(bool(owned), "B0: arange2d: " + rstsr_demo.last_error(lib))
    shared = lib.rstsr_demo_to_shared(owned)
    expect(bool(shared), "B0: to_shared: " + rstsr_demo.last_error(lib))
    report_rstsr(shared)
    expect(rstsr_demo.values(lib, shared) == list(range(12)), "B0: values")

    print("\n[B1] whole tensor -> NumPy (read-only: the export carries the READ_ONLY flag)")
    whole_holder = DlpackExporter(
        lambda: lib.rstsr_demo_export(shared), destructor=lib.rstsr_demo_capsule_destructor()
    )
    w = np.from_dlpack(whole_holder)
    print(f"      numpy      shape={w.shape} strides(elem)={elems(w.strides)} writeable={w.flags.writeable}")
    expect(w.ctypes.data == lib.rstsr_demo_data_ptr(shared), "B1: same buffer (zero copy)")
    expect(w.tolist() == a.tolist(), "B1: values")

    print("\n[B2] view on the Rust side: t[1:, ::2]  (same spec as A2)")
    view_holder = export_slice(shared, [(1, None, None), (None, None, 2)])
    v2 = np.from_dlpack(view_holder)
    print(f"      numpy      shape={v2.shape} strides(elem)={elems(v2.strides)} writeable={v2.flags.writeable}")
    print(f"      offset     view ptr - whole ptr = {v2.ctypes.data - w.ctypes.data} bytes "
          f"({(v2.ctypes.data - w.ctypes.data) // 8} elem)")
    expect((v2.ctypes.data - w.ctypes.data) == 32, "B2: data points at the view's own offset")
    expect(v2.tolist() == v.tolist(), "B2: identical to the NumPy-made view of A2")
    expect(np.shares_memory(v2, w), "B2: one buffer behind two views")
    print(f"      values     {v2.tolist()}")

    print("\n[B3] view on the Rust side: t[:, ::-1]  (negative stride)")
    rev_holder = export_slice(shared, [None, (None, None, -1)])
    r2 = np.from_dlpack(rev_holder)
    print(f"      numpy      shape={r2.shape} strides(elem)={elems(r2.strides)}")
    expect((r2.ctypes.data - w.ctypes.data) == 24, "B3: data points at the last column")
    expect(r2.tolist() == a[:, ::-1].tolist(), "B3: values")
    print(f"      values     {r2.tolist()}")

    print("\n[wire format, as exercised above]")
    print("  * DLPack carries (data, dtype, shape, strides in elements, byte_offset);")
    print("    NumPy bakes basic-index offsets into `data` and emits byte_offset = 0.")
    print("  * NumPy -> rstsr: shape/strides cross verbatim (A2, A3) and the element-zero")
    print("    address is unchanged (zero copy). A negative-stride import re-bases the Rust")
    print("    span below element zero (A3) so every reachable address stays >= 0.")
    print("  * rstsr -> NumPy: the exporter sets data = buffer base + layout offset, so the")
    print("    imported pointer sits at the offset in elements x itemsize (B2: +4 elem,")
    print("    B3: +3 elem) and NumPy sees the same strides, converted to bytes.")
    print("  * No value was copied anywhere: every step above compared buffer addresses.")
    print("  * Shared exports are read-only in NumPy; imported tensors are read-only in")
    print("    Rust by type (the repr has no mutable accessor).")

    del w, v2, r2, whole_holder, view_holder, rev_holder
    gc.collect()
    lib.rstsr_demo_free(shared)

    print("\nOK: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

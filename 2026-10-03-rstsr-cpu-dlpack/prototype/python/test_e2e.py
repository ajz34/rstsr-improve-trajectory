#!/usr/bin/env python
"""End-to-end: rstsr-cpu-dlpack <-> NumPy through real CPython capsules.

Run: conda run -n torch python test_e2e.py

Every case is independent; failures are collected and reported at the end.
"""
import ctypes
import gc
import sys
import traceback
import weakref
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rstsr_demo
from dlpack_ctypes import HandmadeProducer, pointer_to
from rstsr_dlpack import DlpackExporter, dlpack_from_object, one_shot_export

lib = rstsr_demo.load()
producer = HandmadeProducer()
FAILURES = []


def case(name):
    print("\nCASE: %s" % name)


def ok(msg):
    print("OK: %s" % msg)


def run(name, fn):
    case(name)
    try:
        fn()
    except Exception:
        FAILURES.append(name)
        print("ERR: %s" % name)
        traceback.print_exc()


def check(condition, msg):
    if not condition:
        raise AssertionError(msg)
    ok(msg)


def new_shared_handle(n=5):
    owned = lib.rstsr_demo_arange_f64(n)
    assert owned, rstsr_demo.last_error(lib)
    shared = lib.rstsr_demo_to_shared(owned)
    assert shared, rstsr_demo.last_error(lib)
    return shared


def shared_holder(shared):
    return DlpackExporter(
        lambda: lib.rstsr_demo_export(shared),
        copy_fn=lambda: lib.rstsr_demo_export_copy(shared),
        destructor=lib.rstsr_demo_capsule_destructor(),
    )


def test_export_shared():
    shared = new_shared_handle(5)
    holder = shared_holder(shared)

    a = np.from_dlpack(holder)
    check(a.dtype == np.float64 and a.shape == (5,), "dtype/shape preserved")
    check(a.tolist() == [0, 1, 2, 3, 4], "values match the Rust tensor")
    check(not a.flags.writeable, "shared export is read-only in NumPy")
    check(a.ctypes.data == lib.rstsr_demo_data_ptr(shared), "zero copy (same pointer)")
    check(lib.rstsr_demo_sum_f64(shared) == 10.0, "Rust still reads while NumPy holds the array")

    b = np.from_dlpack(holder)  # a fresh export per __dlpack__ call
    check(b.ctypes.data == a.ctypes.data and b.tolist() == a.tolist(), "repeat __dlpack__ shares the buffer")
    del b
    gc.collect()
    check(lib.rstsr_demo_sum_f64(shared) == 10.0, "buffer intact after the second array dies")

    del a
    gc.collect()
    lib.rstsr_demo_free(shared)


def test_export_move_and_roundtrip():
    owned = lib.rstsr_demo_arange_f64(4)
    ptr = lib.rstsr_demo_export_move(owned)  # consumes the handle
    assert ptr, rstsr_demo.last_error(lib)
    holder = DlpackExporter(one_shot_export(ptr), destructor=lib.rstsr_demo_capsule_destructor())

    a = np.from_dlpack(holder)
    check(a.flags.writeable, "move export is writeable (sole ownership, IS_COPIED)")
    a[0] = 100.0

    h2 = dlpack_from_object(a, lambda p, legacy: rstsr_demo.import_handle(lib, p, legacy))
    check(rstsr_demo.values(lib, h2) == [100.0, 1.0, 2.0, 3.0], "Rust sees the NumPy mutation")
    check(lib.rstsr_demo_sum_f64(h2) == 106.0, "sum reflects the mutation")
    lib.rstsr_demo_free(h2)
    del a
    gc.collect()


def test_export_copy():
    shared = new_shared_handle(5)
    holder = shared_holder(shared)

    a = np.from_dlpack(holder, copy=True)
    check(a.flags.writeable, "copy=True export is writeable")
    check(a.ctypes.data != lib.rstsr_demo_data_ptr(shared), "copy=True shares nothing")
    a[0] = 42.0
    check(lib.rstsr_demo_sum_f64(shared) == 10.0, "writing the copy leaves the original untouched")
    del a
    gc.collect()
    lib.rstsr_demo_free(shared)


def test_import_layouts():
    base = np.arange(6, dtype=np.float64)
    cases = [
        (base, [0, 1, 2, 3, 4, 5]),
        (base[::2], [0, 2, 4]),
        (base[::-1], [5, 4, 3, 2, 1, 0]),
        (base[1:5], [1, 2, 3, 4]),
        (base.reshape(2, 3).T, [0, 3, 1, 4, 2, 5]),
        (np.arange(6, dtype=np.float32), [0, 1, 2, 3, 4, 5]),
        (np.arange(6, dtype=np.int64), [0, 1, 2, 3, 4, 5]),
        (np.arange(6, dtype=np.int32), [0, 1, 2, 3, 4, 5]),
    ]
    for arr, expected in cases:
        h = dlpack_from_object(arr, lambda p, legacy: rstsr_demo.import_handle(lib, p, legacy))
        check(lib.rstsr_demo_len(h) == arr.size, "%s: element count" % (arr.dtype,))
        check(lib.rstsr_demo_ndim(h) == arr.ndim, "%s: ndim" % (arr.dtype,))
        check(rstsr_demo.values(lib, h) == expected, "%s%r values" % (arr.dtype, arr.shape))
        check(lib.rstsr_demo_data_ptr(h) == arr.ctypes.data, "zero copy (same pointer)")
        lib.rstsr_demo_free(h)
        gc.collect()


def test_import_keeps_producer_alive():
    arr = np.arange(8, dtype=np.float64)
    ref = weakref.ref(arr)
    h = dlpack_from_object(arr, lambda p, legacy: rstsr_demo.import_handle(lib, p, legacy))
    del arr
    gc.collect()
    check(ref() is not None, "producer array kept alive by the imported tensor")
    check(lib.rstsr_demo_sum_f64(h) == 28.0, "values readable through the import")
    lib.rstsr_demo_free(h)
    gc.collect()
    check(ref() is None, "freeing the import released the producer (deleter ran)")


def test_legacy_capsule_import():
    buf = producer.pool(4)
    for i in range(4):
        buf[i] = 10.0 * (i + 1)
    managed = producer.make(legacy=True, data_ptr=ctypes.cast(buf, ctypes.c_void_p), shape=[4])

    from rstsr_dlpack import PyCapsule_New

    capsule = PyCapsule_New(pointer_to(managed), b"dltensor", None)

    class FakeProducer:
        def __dlpack__(self, **kwargs):
            return capsule

    h = dlpack_from_object(FakeProducer(), lambda p, legacy: rstsr_demo.import_handle(lib, p, legacy))
    check(rstsr_demo.values(lib, h) == [10.0, 20.0, 30.0, 40.0], "legacy (unversioned) tensor imported")
    lib.rstsr_demo_free(h)
    gc.collect()


def test_failure_injection():
    buf = producer.pool(8)
    bad = {
        "version 2.0": producer.make(data_ptr=buf, shape=[4], version=(2, 0)),
        "CUDA device": producer.make(data_ptr=buf, shape=[4], device=(2, 0)),
        "vector lanes": producer.make(data_ptr=buf, shape=[4], lanes=2),
        "sub-byte float": producer.make(data_ptr=buf, shape=[4], code=10, bits=8),
        "NULL data": producer.make(data_ptr=None, shape=[4]),
        "unaligned data": producer.make(data_ptr=buf, shape=[4], byte_offset=1),
        "negative shape": producer.make(data_ptr=buf, shape=[-4]),
    }
    for name, managed in bad.items():
        h = lib.rstsr_demo_import(pointer_to(managed), 0)
        check(not h, "%s rejected" % name)
        check(bool(rstsr_demo.last_error(lib)), "%s carries an error message: %s" % (name, rstsr_demo.last_error(lib)))


def test_unconsumed_capsules():
    shared = new_shared_handle(3)
    holder = shared_holder(shared)
    for _ in range(2000):
        capsule = holder.__dlpack__(max_version=(1, 0))
        del capsule  # never consumed: the Rust capsule destructor must free it
    gc.collect()
    check(lib.rstsr_demo_sum_f64(shared) == 3.0, "2000 unconsumed capsules freed without affecting the source")
    lib.rstsr_demo_free(shared)


def main():
    print("demo-ffi:", lib.rstsr_demo_version().decode())
    print("python:", sys.version.split()[0], "numpy:", np.__version__)
    run("shared export: zero-copy, read-only, repeatable", test_export_shared)
    run("move export + NumPy->Rust round trip", test_export_move_and_roundtrip)
    run("__dlpack__(copy=True) deep copy", test_export_copy)
    run("import layouts (strided / reversed / 2-D / dtypes)", test_import_layouts)
    run("import keeps the producer alive until freed", test_import_keeps_producer_alive)
    run("legacy (unversioned) capsule import", test_legacy_capsule_import)
    run("failure injection: malignant managed tensors", test_failure_injection)
    run("unconsumed capsules are freed", test_unconsumed_capsules)

    print()
    if FAILURES:
        print("FAILED: %d case(s): %s" % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("ALL CASES PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

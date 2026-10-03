"""ctypes bindings for the demo cdylib (``../demo-ffi``)."""
import ctypes
from pathlib import Path

DEFAULT_LIB = Path(__file__).resolve().parent.parent / "target" / "debug" / "librstsr_cpu_dlpack_demo_ffi.so"


def load(lib_path=None):
    lib = ctypes.CDLL(str(lib_path or DEFAULT_LIB))
    lib.rstsr_demo_version.restype = ctypes.c_char_p
    lib.rstsr_demo_last_error.restype = ctypes.c_char_p
    lib.rstsr_demo_capsule_destructor.restype = ctypes.c_void_p

    lib.rstsr_demo_arange_f64.restype = ctypes.c_void_p
    lib.rstsr_demo_arange_f64.argtypes = [ctypes.c_size_t]
    lib.rstsr_demo_to_shared.restype = ctypes.c_void_p
    lib.rstsr_demo_to_shared.argtypes = [ctypes.c_void_p]
    lib.rstsr_demo_export.restype = ctypes.c_void_p
    lib.rstsr_demo_export.argtypes = [ctypes.c_void_p]
    lib.rstsr_demo_export_copy.restype = ctypes.c_void_p
    lib.rstsr_demo_export_copy.argtypes = [ctypes.c_void_p]
    lib.rstsr_demo_export_move.restype = ctypes.c_void_p
    lib.rstsr_demo_export_move.argtypes = [ctypes.c_void_p]
    lib.rstsr_demo_import.restype = ctypes.c_void_p
    lib.rstsr_demo_import.argtypes = [ctypes.c_void_p, ctypes.c_int]

    lib.rstsr_demo_data_ptr.restype = ctypes.c_size_t
    lib.rstsr_demo_data_ptr.argtypes = [ctypes.c_void_p]
    lib.rstsr_demo_len.restype = ctypes.c_size_t
    lib.rstsr_demo_len.argtypes = [ctypes.c_void_p]
    lib.rstsr_demo_ndim.restype = ctypes.c_size_t
    lib.rstsr_demo_ndim.argtypes = [ctypes.c_void_p]
    lib.rstsr_demo_sum_f64.restype = ctypes.c_double
    lib.rstsr_demo_sum_f64.argtypes = [ctypes.c_void_p]
    lib.rstsr_demo_dump_f64.restype = ctypes.c_size_t
    lib.rstsr_demo_dump_f64.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]

    lib.rstsr_demo_free.restype = None
    lib.rstsr_demo_free.argtypes = [ctypes.c_void_p]
    return lib


def last_error(lib):
    msg = lib.rstsr_demo_last_error()
    return msg.decode() if msg else ""


def values(lib, handle):
    """Logical values of a handle, row-major, as a list of floats."""
    n = lib.rstsr_demo_len(handle)
    if n == ctypes.c_size_t(-1).value:
        raise RuntimeError("rstsr_demo_len failed: " + last_error(lib))
    buf = (ctypes.c_double * max(1, n))()
    written = lib.rstsr_demo_dump_f64(handle, buf, n)
    if written != n:
        raise RuntimeError("rstsr_demo_dump_f64 failed: " + last_error(lib))
    return list(buf)[:n]


def import_handle(lib, ptr, legacy):
    """`import_fn(ptr, legacy_flag)` for `dlpack_from_object`."""
    handle = lib.rstsr_demo_import(ptr, legacy)
    if not handle:
        raise RuntimeError("rstsr_demo_import failed: " + last_error(lib))
    return handle

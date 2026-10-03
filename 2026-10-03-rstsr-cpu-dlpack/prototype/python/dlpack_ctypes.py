"""ctypes model of the DLPack C structs, plus a hand-made producer.

Used to build malignant/crafted managed tensors for the validation tests
(the same shape as `../../2026-10-03-rust-numpy-review/experiments/probe_numpy_dlpack.py`).
"""
import ctypes

KDLCPU = 1
KDLFLOAT = 2


class DLPackVersion(ctypes.Structure):
    _fields_ = [("major", ctypes.c_uint32), ("minor", ctypes.c_uint32)]


class DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class DLManagedTensor(ctypes.Structure):
    _fields_ = [("dl_tensor", DLTensor), ("manager_ctx", ctypes.c_void_p), ("deleter", ctypes.c_void_p)]


class DLManagedTensorVersioned(ctypes.Structure):
    _fields_ = [
        ("version", DLPackVersion),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.c_void_p),
        ("flags", ctypes.c_uint64),
        ("dl_tensor", DLTensor),
    ]


def pointer_to(managed):
    """Address of a managed-tensor struct as an int."""
    return ctypes.cast(ctypes.byref(managed), ctypes.c_void_p).value


class HandmadeProducer:
    """Builds managed tensors by hand; keeps every piece alive."""

    def __init__(self):
        self._keep = []

    def pool(self, n=16):
        buf = (ctypes.c_double * n)()
        self._keep.append(buf)
        return buf

    def make(self, *, legacy=False, data_ptr=None, shape=(), strides=None, byte_offset=0,
             code=KDLFLOAT, bits=64, lanes=1, device=(KDLCPU, 0), version=(1, 0), flags=0):
        shape_arr = (ctypes.c_int64 * max(1, len(shape)))(*shape) if shape else None
        self._keep.append(shape_arr)
        strides_arr = None
        if strides is not None:
            strides_arr = (ctypes.c_int64 * max(1, len(strides)))(*strides)
            self._keep.append(strides_arr)

        managed = DLManagedTensor() if legacy else DLManagedTensorVersioned()
        if not legacy:
            managed.version.major, managed.version.minor = version
            managed.flags = flags
        managed.deleter = None  # NULL: nothing for Rust to free
        self._keep.append(managed)

        tensor = managed.dl_tensor
        tensor.data = ctypes.cast(data_ptr, ctypes.c_void_p) if data_ptr is not None else None
        tensor.device.device_type, tensor.device.device_id = device
        tensor.ndim = len(shape)
        tensor.dtype.code, tensor.dtype.bits, tensor.dtype.lanes = code, bits, lanes
        tensor.shape = shape_arr
        tensor.strides = strides_arr
        tensor.byte_offset = byte_offset
        return managed

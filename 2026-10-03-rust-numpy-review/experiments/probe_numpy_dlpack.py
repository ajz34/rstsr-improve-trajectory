#!/usr/bin/env python
"""Empirical probes of NumPy's DLPack producer/consumer behavior.

Run: conda run -n torch python probe_numpy_dlpack.py

Every probe is independent (its own try/except) and prints CASE/OK/ERR lines so
that one failure never aborts the rest.  The second half builds DLPack capsules
by hand with ctypes to probe the consumer's validation rules.
"""
import ctypes
import gc
import sys

import numpy as np

print("=" * 78)
print("command: conda run -n torch python probe_numpy_dlpack.py")
print("python:", sys.version.replace("\n", " "))
print("numpy:", np.__version__, np.__file__)
try:
    import torch
    print("torch:", torch.__version__)
except Exception as exc:  # pragma: no cover
    torch = None
    print("torch: import failed:", type(exc).__name__, exc)
print("=" * 78)


def case(name):
    print("\nCASE: %s" % name)


def ok(msg):
    print("OK: %s" % msg)


def err(exc):
    print("ERR: %s: %s" % (type(exc).__name__, exc))


def show(label, fn):
    try:
        ok("%s -> %r" % (label, fn()))
    except Exception as exc:
        err(exc)


# --------------------------------------------------------------------------
# ctypes DLPack model
# --------------------------------------------------------------------------
PyCapsule_New = ctypes.pythonapi.PyCapsule_New
PyCapsule_New.restype = ctypes.py_object
PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]

PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
PyCapsule_GetPointer.restype = ctypes.c_void_p
PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

PyCapsule_GetName = ctypes.pythonapi.PyCapsule_GetName
PyCapsule_GetName.restype = ctypes.c_char_p
PyCapsule_GetName.argtypes = [ctypes.py_object]

PyCapsule_IsValid = ctypes.pythonapi.PyCapsule_IsValid
PyCapsule_IsValid.restype = ctypes.c_int
PyCapsule_IsValid.argtypes = [ctypes.py_object, ctypes.c_char_p]


class DLPackVersion(ctypes.Structure):
    _fields_ = [("major", ctypes.c_uint32), ("minor", ctypes.c_uint32)]


class DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8),
                ("lanes", ctypes.c_uint16)]


class DLTensor(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("device", DLDevice),
                ("ndim", ctypes.c_int32), ("dtype", DLDataType),
                ("shape", ctypes.POINTER(ctypes.c_int64)),
                ("strides", ctypes.POINTER(ctypes.c_int64)),
                ("byte_offset", ctypes.c_uint64)]


class DLManagedTensor(ctypes.Structure):
    _fields_ = [("dl_tensor", DLTensor), ("manager_ctx", ctypes.c_void_p),
                ("deleter", ctypes.c_void_p)]


class DLManagedTensorVersioned(ctypes.Structure):
    _fields_ = [("version", DLPackVersion), ("manager_ctx", ctypes.c_void_p),
                ("deleter", ctypes.c_void_p), ("flags", ctypes.c_uint64),
                ("dl_tensor", DLTensor)]


class Factory:
    """Builds hand-made DLPack capsules.

    Capsules get a NULL destructor on purpose: calling back into Python from a
    capsule destructor while NumPy has a Python exception pending is fragile
    (it aborts the interpreter), so deleter ownership is probed via refcounts
    and capsule renames from the main flow instead.
    """

    def __init__(self):
        self.keep = []

    def pool(self, n=64):
        """A live float64 buffer usable as scratch storage for fake tensors."""
        buf = np.zeros(n, dtype=np.float64)
        self.keep.append(buf)
        return buf

    def make(self, *, versioned=True, name=None, tag="t", shape=(4,),
             strides=None, data_ptr=None, pool=None, code=2, bits=64, lanes=1,
             device=(1, 0), byte_offset=0, version=(1, 0), flags=0, ndim=None):
        if pool is None:
            pool = self.pool(max(64, int(np.prod(shape)) * 4 + 8))
        if data_ptr is None:
            data_ptr = ctypes.c_void_p(pool.ctypes.data)
        if ndim is None:
            ndim = len(shape)
        shape_arr = (ctypes.c_int64 * max(1, len(shape)))(*shape) if shape else None
        self.keep.append(shape_arr)
        if strides is None:
            strides_arr = None
        else:
            strides_arr = (ctypes.c_int64 * max(1, len(strides)))(*strides)
            self.keep.append(strides_arr)

        if versioned:
            managed = DLManagedTensorVersioned()
            managed.version.major, managed.version.minor = version
            managed.flags = flags
            managed.deleter = None  # NULL: no Python callback during dealloc
            mgr_ptr = ctypes.cast(ctypes.byref(managed), ctypes.c_void_p)
            capsule_name = name or b"dltensor_versioned"
        else:
            managed = DLManagedTensor()
            managed.deleter = None
            mgr_ptr = ctypes.cast(ctypes.byref(managed), ctypes.c_void_p)
            capsule_name = name or b"dltensor"
        self.keep.append(managed)

        t = managed.dl_tensor
        t.data = ctypes.cast(data_ptr, ctypes.c_void_p)
        t.device.device_type, t.device.device_id = device
        t.ndim = ndim
        t.dtype.code, t.dtype.bits, t.dtype.lanes = code, bits, lanes
        t.shape = shape_arr
        t.strides = strides_arr
        t.byte_offset = byte_offset

        cap = PyCapsule_New(mgr_ptr, capsule_name, None)
        return cap


F = Factory()


class Exporter:
    """Minimal object satisfying the consumer side of the protocol."""

    def __init__(self, cap_maker, device=(1, 0)):
        self._make = cap_maker
        self.device = device

    def __dlpack__(self, **kwargs):
        return self._make()

    def __dlpack_device__(self):
        return self.device


def cap_name(cap):
    n = PyCapsule_GetName(cap)
    return n.decode() if n is not None else None


def managed_versioned(cap):
    ptr = PyCapsule_GetPointer(cap, b"dltensor_versioned")
    return DLManagedTensorVersioned.from_address(ptr)


def managed_unversioned(cap):
    ptr = PyCapsule_GetPointer(cap, b"dltensor")
    return DLManagedTensor.from_address(ptr)


print()
print("################ PART 1: ndarray.__dlpack__ (producer) ################")

case("np.from_dlpack(arr) round-trip and raw capsule")
a = np.arange(6, dtype=np.float64).reshape(2, 3)
show("np.from_dlpack(a)", lambda: (lambda y: (y.dtype.str, y.shape, y.tolist(), bool(y.flags.writeable)))(np.from_dlpack(a)))
show("np.from_dlpack(a.__dlpack__())  # raw capsule", lambda: np.from_dlpack(a.__dlpack__()))
cap = a.__dlpack__()
show("capsule type", lambda: type(cap))
show("bare capsule name (no max_version)", lambda: cap_name(cap))
show("np.from_dlpack(capsule) direct", lambda: np.from_dlpack(cap))
del cap

case("__dlpack__ default vs max_version: capsule flavor")
for mv in [None, (0, 0), (0, 8), (1, 0), (2, 0), (100, 3), (True, False)]:
    try:
        c = a.__dlpack__(max_version=mv)
    except Exception as exc:
        err(exc)
        continue
    nm = cap_name(c)
    extra = ""
    if nm == "dltensor_versioned":
        m = managed_versioned(c)
        extra = " version=%d.%d flags=%d" % (m.version.major, m.version.minor, m.flags)
    else:
        m = managed_unversioned(c)
        extra = " ndim=%d dtype=(code=%d,bits=%d,lanes=%d) dev=(%d,%d) byte_offset=%d" % (
            m.dl_tensor.ndim, m.dl_tensor.dtype.code, m.dl_tensor.dtype.bits,
            m.dl_tensor.dtype.lanes, m.dl_tensor.device.device_type,
            m.dl_tensor.device.device_id, m.dl_tensor.byte_offset)
    ok("max_version=%r -> %s%s" % (mv, nm, extra))
    del c

case("__dlpack__ max_version malformed")
for mv in [[1, 0], (1,), (1, 0, 0), "1.0", (1.0, 0), (-1, 0), (None, 0)]:
    try:
        c = a.__dlpack__(max_version=mv)
        ok("max_version=%r -> accepted, name=%s" % (mv, cap_name(c)))
        del c
    except Exception as exc:
        err(exc)

case("__dlpack__ stream")
x = np.arange(5)
show("stream=None", lambda: cap_name(x.__dlpack__(stream=None)))
show("stream=1", lambda: x.__dlpack__(stream=1))
show("stream=(1,2)", lambda: x.__dlpack__(stream=(1, 2)))

case("__dlpack__ dl_device")
arr0 = np.arange(5)
show("dl_device=None", lambda: cap_name(arr0.__dlpack__(dl_device=None)))
show("dl_device=(1, 0)", lambda: cap_name(arr0.__dlpack__(dl_device=(1, 0))))
show("dl_device=(10, 0)", lambda: arr0.__dlpack__(dl_device=(10, 0)))
show("dl_device=1 (not tuple)", lambda: arr0.__dlpack__(dl_device=1))
show("dl_device=(1,) ", lambda: arr0.__dlpack__(dl_device=(1,)))
show("dl_device='cpu'", lambda: arr0.__dlpack__(dl_device="cpu"))

case("__dlpack__ copy flag (capsule flags byte)")
c = arr0.__dlpack__(max_version=(1, 0), copy=True)
m = managed_versioned(c)
ok("copy=True  flags=%d (IS_COPIED bit=%d)" % (m.flags, m.flags & 2))
del c
c = arr0.__dlpack__(max_version=(1, 0), copy=False)
m = managed_versioned(c)
ok("copy=False flags=%d" % m.flags)
del c
c = arr0.__dlpack__(max_version=(1, 0), copy=None)
m = managed_versioned(c)
ok("copy=None  flags=%d" % m.flags)
del c
try:
    arr0.__dlpack__(copy=np.array([1, 2, 3]))
except Exception as exc:
    err(exc)

case("export: strides / byte_offset of a non-contiguous view")
base = np.arange(12, dtype=np.float64).reshape(3, 4)
view = base[::2, ::2]
c = view.__dlpack__(max_version=(1, 0))
m = managed_versioned(c)
ok("view shape=%s numpy strides=%s -> dl shape=%s strides=%s byte_offset=%d" % (
    view.shape, view.strides,
    [m.dl_tensor.shape[i] for i in range(m.dl_tensor.ndim)],
    [m.dl_tensor.strides[i] for i in range(m.dl_tensor.ndim)],
    m.dl_tensor.byte_offset))
del c
c = base.__dlpack__(max_version=(1, 0))
m = managed_versioned(c)
ok("C-contig 3x4 -> dl strides=%s" % [m.dl_tensor.strides[i] for i in range(m.dl_tensor.ndim)])
del c
c = np.arange(5.0)[::-1].__dlpack__(max_version=(1, 0))
m = managed_versioned(c)
ok("a[::-1] -> dl strides=%s (negative stride exported)" % [m.dl_tensor.strides[i] for i in range(m.dl_tensor.ndim)])
del c
c = np.broadcast_to(np.arange(3.0), (2, 3)).__dlpack__(max_version=(1, 0))
m = managed_versioned(c)
ok("broadcast 2x3 -> dl strides=%s" % [m.dl_tensor.strides[i] for i in range(m.dl_tensor.ndim)])
del c

case("export: size-1 dims with stride not a multiple of itemsize")
y = np.ndarray(dtype='f8', shape=(10, 5, 1), strides=(8, 80, 4),
               buffer=np.ones(1000, dtype=np.uint8), order='F')
c = y.__dlpack__(max_version=(1, 0))
m = managed_versioned(c)
ok("shape=(10,5,1) strides=(8,80,4) -> dl strides=%s" % [m.dl_tensor.strides[i] for i in range(3)])
del c
dt = np.dtype([('int', np.int32), ('char', np.int8)])
z = np.zeros((5,), dtype=dt)['int']
show("dtype=structured field view (itemsize=4, stride=5) export", lambda: z.__dlpack__(max_version=(1, 0)))

case("export: 0-d, size-0, F-order")
c = np.array(1.0).__dlpack__(max_version=(1, 0))
m = managed_versioned(c)
ok("0-d: ndim=%d shape ptr null? %s strides ptr null? %s" % (
    m.dl_tensor.ndim, not bool(m.dl_tensor.shape), not bool(m.dl_tensor.strides)))
del c
c = np.zeros((0, 3)).__dlpack__(max_version=(1, 0))
m = managed_versioned(c)
ok("size-0 (0,3): ndim=%d shape=%s strides=%s data ptr null? %s (numpy strides=%s)" % (
    m.dl_tensor.ndim, [m.dl_tensor.shape[i] for i in range(2)],
    [m.dl_tensor.strides[i] for i in range(2)],
    not bool(m.dl_tensor.data), np.zeros((0, 3)).strides))
del c
c = np.asfortranarray(np.arange(6.0).reshape(2, 3)).__dlpack__(max_version=(1, 0))
m = managed_versioned(c)
ok("F-order 2x3 -> strides=%s" % [m.dl_tensor.strides[i] for i in range(2)])
del c

case("export: dtypes")
for dt_ in ["float32", "float64", "float16", "bool", "int8", "uint8",
            "complex64", "complex128", "longdouble", "clongdouble",
            "datetime64[ns]", "timedelta64[s]", "U3", "S3", "V8", "object",
            "int64"]:
    try:
        arr = np.zeros(3, dtype=dt_)
    except Exception as exc:
        ok("dtype %s not available: %s" % (dt_, exc))
        continue
    try:
        c = arr.__dlpack__(max_version=(1, 0))
        m = managed_versioned(c)
        ok("dtype %-14s export -> code=%d bits=%d lanes=%d" % (
            dt_, m.dl_tensor.dtype.code, m.dl_tensor.dtype.bits, m.dl_tensor.dtype.lanes))
        del c
    except Exception as exc:
        print("ERR: dtype %-14s export -> %s: %s" % (dt_, type(exc).__name__, exc))
    try:
        np.from_dlpack(arr)
        ok("dtype %-14s np.from_dlpack(arr) -> ok" % dt_)
    except Exception as exc:
        print("ERR: dtype %-14s np.from_dlpack(arr) -> %s: %s"
              % (dt_, type(exc).__name__, exc))

case("export: byteswapped dtype")
bs = np.arange(5, dtype=np.dtype('=i8').newbyteorder())
show("byteswapped int64 export", lambda: bs.__dlpack__(max_version=(1, 0)))
show("byteswapped int64 from_dlpack", lambda: np.from_dlpack(bs))

case("export: writeable=False")
ro = np.arange(5)
ro.flags.writeable = False
show("ro.__dlpack__() (no max_version)", lambda: ro.__dlpack__())
show("ro.__dlpack__(max_version=(1,0))", lambda: cap_name(ro.__dlpack__(max_version=(1, 0))))
c = ro.__dlpack__(max_version=(1, 0))
m = managed_versioned(c)
ok("readonly exported flags=%d (READ_ONLY bit=%d)" % (m.flags, m.flags & 1))
del c
c = ro.__dlpack__(max_version=(1, 0), copy=True)
m = managed_versioned(c)
ok("readonly copy=True flags=%d data==orig? %s" % (m.flags, m.dl_tensor.data == ro.ctypes.data))
del c
c = ro.__dlpack__(max_version=(1, 0), copy=False)
ok("readonly copy=False -> ok name=%s" % cap_name(c))
del c
y = np.from_dlpack(ro)
ok("np.from_dlpack(readonly) -> writeable=%s" % bool(y.flags.writeable))

case("export keeps the array alive (refcount)")
q = np.arange(5)
startcount = sys.getrefcount(q)
c = q.__dlpack__(max_version=(1, 0))
ok("refcount after __dlpack__: start+%d" % (sys.getrefcount(q) - startcount))
del c
gc.collect()
ok("refcount after capsule del: start+%d" % (sys.getrefcount(q) - startcount))

case("__dlpack_device__")
show("np.arange(5).__dlpack_device__()", lambda: np.arange(5).__dlpack_device__())
show("float32 scalar __dlpack_device__()", lambda: np.float32(1).__dlpack_device__())

print()
print("################ PART 2: np.from_dlpack (consumer) ################")

case("from_dlpack: accepted input objects")
class OldStyle:
    def __dlpack__(self, stream=None):
        return np.arange(4).__dlpack__(stream=stream)
class OnlyDlpack:
    def __dlpack__(self, **kw):
        return np.arange(4).__dlpack__(max_version=kw.get("max_version"))

show("old-style __dlpack__(stream=None) only", lambda: np.from_dlpack(OldStyle()).tolist())
show("object without __dlpack_device__", lambda: np.from_dlpack(OnlyDlpack()).tolist())
show("np.from_dlpack(None)", lambda: np.from_dlpack(None))
show("np.from_dlpack(123)", lambda: np.from_dlpack(123))
show("exporter returning non-capsule", lambda: np.from_dlpack(Exporter(lambda: None)))
show("exporter returning int", lambda: np.from_dlpack(Exporter(lambda: 42)))

case("from_dlpack: exact kwargs numpy passes to obj.__dlpack__")
class Recorder:
    def __init__(self):
        self.calls = []

    def __dlpack__(self, *args, **kw):
        self.calls.append((args, kw))
        return np.arange(3).__dlpack__(**{k: v for k, v in kw.items()
                                         if k in ("max_version", "copy", "dl_device")})

    def __dlpack_device__(self):
        return (1, 0)

class RecorderOld:
    """Refuses kwargs (old-style exporter): numpy must fall back to a bare call."""

    def __init__(self):
        self.calls = []

    def __dlpack__(self, stream=None, **kwargs):
        self.calls.append((stream, kwargs))
        if kwargs:
            raise TypeError("__dlpack__() got an unexpected keyword argument")
        return np.arange(3).__dlpack__()

r = Recorder()
np.from_dlpack(r)
ok("plain np.from_dlpack(obj) -> args=%r kwargs=%r" % (r.calls[0][0], r.calls[0][1]))
r = Recorder()
try:
    np.from_dlpack(r, device="cpu", copy=True)
except Exception as exc:
    err(exc)
ok("np.from_dlpack(obj, device='cpu', copy=True) -> kwargs=%r"
   % (r.calls[0][1] if r.calls else None))
r_old = RecorderOld()
np.from_dlpack(r_old)
ok("old-style exporter: attempts recorded by exporter = %r"
   % (r_old.calls,))

case("from_dlpack: device= kwarg")
show("device=None", lambda: np.from_dlpack(np.arange(3), device=None).tolist())
show("device='cpu'", lambda: np.from_dlpack(np.arange(3), device='cpu').tolist())
show("device='gpu'", lambda: np.from_dlpack(np.arange(3), device='gpu'))
show("device=1", lambda: np.from_dlpack(np.arange(3), device=1))
show("device='CPU'", lambda: np.from_dlpack(np.arange(3), device='CPU'))

case("from_dlpack: copy= kwarg")
src = np.arange(5)
show("copy=None shares memory", lambda: np.may_share_memory(src, np.from_dlpack(src, copy=None)))
show("copy=False shares memory", lambda: np.may_share_memory(src, np.from_dlpack(src, copy=False)))
show("copy=True  shares memory", lambda: np.may_share_memory(src, np.from_dlpack(src, copy=True)))
show("copy='x'", lambda: np.from_dlpack(src, copy='x'))
show("copy=1 (int)", lambda: np.may_share_memory(src, np.from_dlpack(src, copy=1)))

case("capsule reuse: same capsule consumed twice")
c = np.arange(5).__dlpack__(max_version=(1, 0))
w = Exporter(lambda: c)
y1 = np.from_dlpack(w)
ok("first import -> %r, capsule name now %s" % (y1.tolist(), cap_name(c)))
try:
    y2 = np.from_dlpack(w)
    ok("second import succeeded (!) %r" % y2.tolist())
except Exception as exc:
    err(exc)

case("capsule reuse: unversioned capsule consumed twice")
c2 = np.arange(5).__dlpack__()
w2 = Exporter(lambda: c2)
y1 = np.from_dlpack(w2)
ok("first -> %r, name now %s" % (y1.tolist(), cap_name(c2)))
try:
    y2 = np.from_dlpack(w2)
    ok("second import succeeded (!) %r" % y2.tolist())
except Exception as exc:
    err(exc)

case("from_dlpack: writes are forbidden when importing an unversioned capsule")
x_old = np.arange(5)
y_old = np.from_dlpack(Exporter(lambda: x_old.__dlpack__()))
ok("unversioned import writeable=%s" % bool(y_old.flags.writeable))
try:
    y_old[0] = 100
    ok("write to unversioned-imported array succeeded (!)")
except Exception as exc:
    err(exc)

case("from_dlpack: versioned capsule without READ_ONLY -> writeable")
x_new = np.arange(5)
y_new = np.from_dlpack(Exporter(lambda: x_new.__dlpack__(max_version=(1, 0))))
ok("versioned import writeable=%s" % bool(y_new.flags.writeable))
y_new[0] = 100
ok("write ok -> %r" % y_new.tolist())

case("from_dlpack: version major/minor handling")
for ver in [(1, 0), (1, 99), (0, 9), (2, 0), (3, 5)]:
    def mk(v=ver):
        return F.make(versioned=True, tag="ver%d" % v[0], shape=(2,), code=2, bits=64, version=v)
    try:
        arr = np.from_dlpack(Exporter(mk))
        ok("capsule version %s -> imported %r" % (ver, arr.tolist()))
    except Exception as exc:
        err(exc)

case("from_dlpack: device acceptance")
for dev in [(1, 0), (2, 0), (3, 0), (10, 0), (11, 0), (13, 0), (14, 0), (0, 0)]:
    def mk(d=dev):
        return F.make(versioned=True, tag="dev%d" % d[0], shape=(2,), code=2, bits=64, device=d)
    try:
        arr = np.from_dlpack(Exporter(mk))
        ok("device %s -> imported; __dlpack_device__=%s" % (dev, arr.__dlpack_device__()))
    except Exception as exc:
        err(exc)

case("from_dlpack: device_id handling (device_type accepted, id ignored?)")
for dev in [(1, 5), (13, 7), (3, 2)]:
    def mk(d=dev):
        return F.make(versioned=True, tag="devid%d" % d[1], shape=(2,), code=2, bits=64, device=d)
    try:
        arr = np.from_dlpack(Exporter(mk))
        ok("device %s -> imported; reports %s" % (dev, arr.__dlpack_device__()))
    except Exception as exc:
        err(exc)

case("from_dlpack: __dlpack__ raising TypeError leads to a silent retry")
class Raises:
    def __init__(self):
        self.calls = 0

    def __dlpack__(self, *args, **kwargs):
        self.calls += 1
        raise TypeError("boom from exporter")

rr = Raises()
try:
    np.from_dlpack(rr)
except Exception as exc:
    err(exc)
ok("exporter __dlpack__ called %d time(s) before the error surfaced" % rr.calls)

case("from_dlpack: dtype code/bits mapping")
for (code, bits, label) in [(0, 8, "int8"), (0, 16, "int16"), (0, 32, "int32"),
                            (0, 64, "int64"), (0, 128, "int128"),
                            (1, 8, "uint8"), (1, 64, "uint64"), (1, 128, "uint128"),
                            (2, 16, "float16"), (2, 32, "float32"), (2, 64, "float64"),
                            (2, 8, "float8"), (2, 128, "float128"),
                            (4, 16, "bfloat16"), (5, 64, "complex64"), (5, 128, "complex128"),
                            (5, 32, "complex32"), (6, 8, "bool8"), (6, 1, "bool1"),
                            (3, 32, "opaque32"), (9, 8, "code9")]:
    def mk(c=code, b=bits):
        return F.make(versioned=True, tag="dt%d_%d" % (c, b), shape=(2,), code=c, bits=b)
    try:
        arr = np.from_dlpack(Exporter(mk))
        ok("code=%d bits=%3d (%-9s) -> dtype %s" % (code, bits, label, arr.dtype))
    except Exception as exc:
        err(exc)

case("from_dlpack: lanes != 1")
for lanes in [1, 2, 4]:
    def mk(l=lanes):
        return F.make(versioned=True, tag="lanes%d" % l, shape=(2,), code=2, bits=64, lanes=l)
    try:
        arr = np.from_dlpack(Exporter(mk))
        ok("lanes=%d -> dtype %s" % (lanes, arr.dtype))
    except Exception as exc:
        err(exc)

case("from_dlpack: ndim limit")
for nd in [0, 1, 32, 64, 65]:
    def mk(n=nd):
        return F.make(versioned=True, tag="nd%d" % n, shape=(1,) * n, strides=None, code=2, bits=64)
    try:
        arr = np.from_dlpack(Exporter(mk))
        ok("ndim=%d -> shape %s" % (nd, arr.shape))
    except Exception as exc:
        err(exc)

case("from_dlpack: strides == NULL / negative / zero")
pool = F.pool(64)
pool[:] = np.arange(64, dtype=np.float64)
buf = pool

def mk_null():
    return F.make(versioned=True, tag="nullstride", shape=(2, 3), strides=None, pool=buf)

arr = np.from_dlpack(Exporter(mk_null))
ok("strides==NULL (2,3) -> flags C_CONTIGUOUS=%s, arr=\n%s" % (bool(arr.flags.c_contiguous), arr))

def mk_neg():
    return F.make(versioned=True, tag="negstride", shape=(3,), strides=(-1,),
                  data_ptr=ctypes.c_void_p(buf.ctypes.data + 16), pool=buf, bits=64, code=2)

try:
    arr = np.from_dlpack(Exporter(mk_neg))
    ok("negative stride (-1) -> %r, itemsize=%d" % (arr.tolist(), arr.itemsize))
except Exception as exc:
    err(exc)

def mk_zero():
    return F.make(versioned=True, tag="zerostride", shape=(3,), strides=(0,), pool=buf, code=2, bits=64)

try:
    arr = np.from_dlpack(Exporter(mk_zero))
    ok("zero stride (broadcast) -> %r" % (arr.tolist(),))
    arr[0] = 999.0
    ok("write through broadcast view -> %r" % arr.tolist())
except Exception as exc:
    err(exc)

def mk_coexist():
    return F.make(versioned=True, tag="s1", shape=(2, 3), strides=(1, 2), pool=buf, code=2, bits=64)

arr = np.from_dlpack(Exporter(mk_coexist))
ok("F-ish strides (1,2) -> strides=%s" % (arr.strides,))

case("from_dlpack: byte_offset honored")
def mk_off():
    return F.make(versioned=True, tag="byteoff", shape=(3,), strides=(1,), pool=buf,
                  code=2, bits=64, byte_offset=16)

arr = np.from_dlpack(Exporter(mk_off))
ok("byte_offset=16 -> %r (pool[2:5])" % (arr.tolist(),))

case("from_dlpack: misaligned data pointer")
raw = ctypes.create_string_buffer(128)
F.keep.append(raw)
addr = ctypes.cast(raw, ctypes.c_void_p).value
def mk_mis():
    return F.make(versioned=True, name=b"dltensor_versioned", tag="misaligned", shape=(2,),
                  strides=(1,), data_ptr=ctypes.c_void_p(addr + 1), pool=buf, code=2, bits=64)
try:
    arr = np.from_dlpack(Exporter(mk_mis))
    ok("misaligned (+1 byte) float64 import -> OK, flags.aligned=%s, writeable=%s, dump=%r"
       % (bool(arr.flags.aligned), bool(arr.flags.writeable), arr))
except Exception as exc:
    err(exc)
arr_misaligned = np.frombuffer(bytearray(64), dtype='f8', count=4, offset=1)
try:
    c = arr_misaligned.__dlpack__(max_version=(1, 0))
    ok("numpy unaligned array export -> ok, aligned flag=%s" % bool(arr_misaligned.flags.aligned))
    del c
except Exception as exc:
    err("numpy unaligned export: %s" % exc)
try:
    np.from_dlpack(arr_misaligned)
    ok("numpy unaligned array from_dlpack -> ok")
except Exception as exc:
    err(exc)

case("from_dlpack: capsule names")
def mk_named(nm):
    return F.make(versioned=(nm != b"dltensor"), name=nm, tag="nm_%s" % nm.decode(),
                  shape=(2,), code=2, bits=64, pool=buf)

for nm in [b"dltensor", b"dltensor_versioned", b"used_dltensor",
           b"used_dltensor_versioned", b"torch", b"numpy_dltensor"]:
    try:
        arr = np.from_dlpack(Exporter(lambda n=nm: mk_named(n)))
        ok("capsule name %s -> imported %r" % (nm, arr.tolist()))
    except Exception as exc:
        err(exc)

case("from_dlpack: validation order (bad device + bad dtype)")
def mk_multi_bad():
    return F.make(versioned=True, tag="multi", shape=(2,), code=6, bits=1, device=(2, 0))
try:
    np.from_dlpack(Exporter(mk_multi_bad))
    ok("unexpectedly imported")
except Exception as exc:
    err(exc)

case("from_dlpack: capsule ownership (refcount + rename) on failure vs success")
cap_bad = F.make(versioned=True, tag="bad_dtype", shape=(2,), code=6, bits=1)
rc0 = sys.getrefcount(cap_bad)
try:
    np.from_dlpack(Exporter(lambda: cap_bad))
except Exception as exc:
    pass
ok("FAILED import: refcount delta=%d, capsule name still %s (numpy released it,"
   " did not rename -> the producer's capsule destructor would call the deleter)"
   % (sys.getrefcount(cap_bad) - rc0, cap_name(cap_bad)))

cap_good = F.make(versioned=True, tag="good", shape=(2,), code=2, bits=64, pool=F.pool(16))
rc0 = sys.getrefcount(cap_good)
arr = np.from_dlpack(Exporter(lambda: cap_good))
ok("SUCCESSFUL import: refcount delta=%d, capsule renamed to %s -> numpy owns it"
   % (sys.getrefcount(cap_good) - rc0, cap_name(cap_good)))
base = arr.base
ok("array base type=%s, base capsule name=%s" % (type(base).__name__, cap_name(base)))
ok("array writeable=%s" % bool(arr.flags.writeable))
del arr, cap_good, cap_bad
gc.collect()

case("register_dlpack_dtype: NumPy 2.5 user-dtype escape hatch")
try:
    np.dtypes.register_dlpack_dtype((7, 8), np.dtype("S1"))
    a = np.array([b"a", b"b"], dtype="S1")
    y = np.from_dlpack(a)
    ok("registered (code=7, bits=8) for S1 -> export+import ok, dtype=%s %r" % (y.dtype, y.tolist()))
except Exception as exc:
    err(exc)
try:
    b = np.array([b"abc", b"def"], dtype="S3")
    np.from_dlpack(b)
    ok("unregistered S3 still importable (!)")
except Exception as exc:
    err(exc)

print()
print("################ PART 3: cross-library (torch) ################")
if torch is not None:
    case("torch -> numpy -> torch")
    t = torch.arange(6, dtype=torch.float32)
    try:
        y = np.from_dlpack(t)
        ok("np.from_dlpack(torch.arange(6)) -> %s %s" % (y.dtype, y.tolist()))
        y[0] = 42.0
        ok("aliases torch storage? torch t[0]=%s" % t[0].item())
    except Exception as exc:
        err(exc)
    try:
        t2 = torch.from_dlpack(np.arange(4, dtype=np.int64))
        ok("torch.from_dlpack(np.arange(4)) -> %s %s" % (t2.dtype, t2.tolist()))
    except Exception as exc:
        err(exc)
    try:
        np.from_dlpack(torch.arange(10)[::2])
        ok("np.from_dlpack(torch strided view) -> ok")
    except Exception as exc:
        err(exc)
    try:
        arr = np.arange(5.0)
        arr.flags.writeable = False
        t3 = torch.from_dlpack(arr)
        ok("torch.from_dlpack(readonly numpy) -> %s" % (t3,))
    except Exception as exc:
        err(exc)
else:
    print("torch not available, skipping")

print()
print("=" * 78)
print("ALL PROBES DONE")
print("=" * 78)

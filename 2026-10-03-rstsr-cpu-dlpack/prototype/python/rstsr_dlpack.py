"""Reference Python-side DLPack capsule holder over `rstsr-cpu-dlpack`.

The crate speaks raw `DLManagedTensorVersioned` pointers; `PyCapsule` is a
CPython type it cannot create, so this file is the glue:

- **export**: `DlpackExporter` wraps a host callback that returns a fresh
  managed-tensor pointer per `__dlpack__` call. The capsule carries a
  *Rust* destructor (`rstsr_demo_capsule_destructor`), which frees an export
  the consumer never took.
- **import**: `dlpack_from_object` calls `obj.__dlpack__()`, renames the
  capsule to `used_dltensor*` (transferring the free obligation to the
  imported tensor) and hands the pointer to the host's import callback.
"""
import ctypes

PyCapsule_New = ctypes.pythonapi.PyCapsule_New
PyCapsule_New.restype = ctypes.py_object
PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]

PyCapsule_GetName = ctypes.pythonapi.PyCapsule_GetName
PyCapsule_GetName.restype = ctypes.c_char_p
PyCapsule_GetName.argtypes = [ctypes.py_object]

PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
PyCapsule_GetPointer.restype = ctypes.c_void_p
PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

PyCapsule_SetName = ctypes.pythonapi.PyCapsule_SetName
PyCapsule_SetName.restype = ctypes.c_int
PyCapsule_SetName.argtypes = [ctypes.py_object, ctypes.c_char_p]


def one_shot_export(ptr):
    """Wrap a single managed-tensor address in a one-call export callback."""
    state = {"ptr": ptr}

    def take():
        if state["ptr"] is None:
            raise RuntimeError("one-shot DLPack export already consumed")
        ptr, state["ptr"] = state["ptr"], None
        return ptr

    return take


class DlpackExporter:
    """Minimal object satisfying the consumer side of the DLPack protocol."""

    def __init__(self, export_fn, copy_fn=None, destructor=None):
        self._export_fn = export_fn
        # `copy=True` means "give me something I solely own"; the host may
        # answer with a fresh move-export over a deep copy.
        self._copy_fn = copy_fn
        self._destructor = ctypes.cast(destructor, ctypes.c_void_p) if destructor else None

    def __dlpack_device__(self):
        return (1, 0)  # kDLCPU, device 0

    def __dlpack__(self, stream=None, max_version=None, dl_device=None, copy=None, **kwargs):
        if max_version is not None and max_version[0] > 1:
            raise BufferError("rstsr-cpu-dlpack only produces DLPack major version 1")
        fn = (self._copy_fn or self._export_fn) if copy else self._export_fn
        ptr = fn()
        if not ptr:
            raise RuntimeError("rstsr host failed to create an export")
        return PyCapsule_New(ctypes.c_void_p(ptr), b"dltensor_versioned", self._destructor)


def dlpack_from_object(obj, import_fn):
    """Hand `obj`'s DLPack tensor to the host's import entry point.

    `import_fn(ptr, legacy_flag)` must return a host handle (or raise).
    """
    if not hasattr(obj, "__dlpack__"):
        raise TypeError("object does not implement __dlpack__")
    capsule = obj.__dlpack__(max_version=(1, 0))
    name = PyCapsule_GetName(capsule)
    if name is None:
        raise TypeError("__dlpack__ did not return a capsule")
    ptr = PyCapsule_GetPointer(capsule, name)
    if not ptr:
        raise RuntimeError("could not read the capsule pointer")
    legacy = name == b"dltensor"
    # Transfer the free obligation to the imported tensor: the capsule must
    # never free a tensor the host now owns.
    used = b"used_dltensor" if legacy else b"used_dltensor_versioned"
    if PyCapsule_SetName(capsule, used) != 0:
        raise RuntimeError("could not rename the capsule")
    return import_fn(ptr, legacy)

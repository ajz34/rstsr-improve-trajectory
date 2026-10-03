"""Probe: common Python usage patterns of DLPack (numpy <-> torch, CPU).

Environment used for the captured output: numpy 2.5.1, torch 2.14.0+rocm7.2
(CPU tensors only). Captures: zero-copy sharing, write propagation, read-only
handling, strided views, capsule defaults/call patterns, lifetime independence
and copy=True detachment.
"""
import ctypes
import gc
import inspect

import numpy as np
import torch

print("versions: numpy", np.__version__, "| torch", torch.__version__)

get_name = ctypes.pythonapi.PyCapsule_GetName
get_name.restype = ctypes.c_char_p
get_name.argtypes = [ctypes.py_object]


def capsule_name(obj, **kw):
    try:
        return get_name(obj.__dlpack__(**kw)).decode()
    except Exception as e:
        return f"<{type(e).__name__}: {e}>"


# --- zero-copy sharing and write propagation -----------------------------------
a = np.arange(6.0)
t = torch.from_dlpack(a)
print("np->torch zero-copy:", t.data_ptr() == a.ctypes.data)
a[0] = 100.0
print("write numpy visible in torch:", t[0].item() == 100.0)
t[1] = 200.0
print("write torch visible in numpy:", a[1] == 200.0)

# --- read-only handling ---------------------------------------------------------
b = np.arange(4.0)
b.flags.writeable = False
print("readonly numpy, default capsule:", capsule_name(b))
print("readonly numpy, versioned capsule:", capsule_name(b, max_version=(1, 0)))
try:
    tb = torch.from_dlpack(b)
    print("readonly numpy -> torch: imported")
    try:
        tb[0] = 9.0
        print("  torch write succeeded; numpy now sees", b[0])
    except Exception as e:
        print("  torch write blocked:", type(e).__name__, str(e)[:80])
except Exception as e:
    print("readonly numpy -> torch raised:", type(e).__name__, str(e)[:120])

# --- strided views ---------------------------------------------------------------
c = np.arange(24.0).reshape(4, 6)[:, ::2]
try:
    tc = torch.from_dlpack(c)
    print(
        "strided numpy -> torch: ok; stride", tuple(tc.stride()),
        "contiguous", tc.is_contiguous(),
        "values ok", tc[0, 0].item() == 0.0 and tc[0, 1].item() == 2.0,
    )
except Exception as e:
    print("strided numpy -> torch raised:", type(e).__name__, str(e)[:120])

x = torch.arange(5.0)
n = np.from_dlpack(x)
print("torch->np zero-copy:", n.ctypes.data == x.data_ptr())
x[0] = 42.0
print("write torch visible in numpy:", n[0] == 42.0)
n[1] = 43.0
print("write numpy visible in torch:", x[1].item() == 43.0)
print("np array from torch: writeable =", n.flags.writeable, "| owndata =", n.flags.owndata,
      "| base type =", type(n.base).__name__)

y = torch.arange(12.0).reshape(3, 4)[:, ::2]
try:
    n2 = np.from_dlpack(y)
    print("strided torch -> np: ok", n2.shape, n2.strides)
except Exception as e:
    print("strided torch -> np raised:", type(e).__name__, str(e)[:120])

# --- consumer signatures and producer capsule defaults ---------------------------
print("np.from_dlpack signature:", inspect.signature(np.from_dlpack))
print("torch.from_dlpack signature:", inspect.signature(torch.from_dlpack))
print("numpy default producer capsule:", capsule_name(a))
print("numpy versioned producer capsule:", capsule_name(a, max_version=(1, 0)))
print("numpy copy=True producer capsule:", capsule_name(a, copy=True))
print("torch default producer capsule:", capsule_name(x))
print("torch versioned producer capsule:", capsule_name(x, max_version=(1, 0)))


# --- what consumers actually pass to the producer --------------------------------
class Spy:
    def __dlpack_device__(self):
        print("  Spy.__dlpack_device__ called")
        return (1, 0)

    def __dlpack__(self, **kw):
        print("  Spy.__dlpack__ called with", kw)
        return np.arange(3.0).__dlpack__(**kw)


print("torch consuming Spy:")
try:
    ts = torch.from_dlpack(Spy())
    print("  torch ok:", ts.tolist())
except Exception as e:
    print("  torch failed:", type(e).__name__, str(e)[:120])
print("numpy consuming Spy:")
try:
    ns = np.from_dlpack(Spy())
    print("  numpy ok:", ns.tolist())
except Exception as e:
    print("  numpy failed:", type(e).__name__, str(e)[:120])

try:
    import jax

    print("jax", jax.__version__)
except Exception as e:
    print("jax:", type(e).__name__, str(e)[:100])

# --- lifetime independence and copy=True detachment ------------------------------
x = torch.arange(4.0)
n = np.from_dlpack(x)
ptr = n.ctypes.data
del x
gc.collect()
print("numpy view survives dead torch producer:", n.tolist(), "| same pointer:", n.ctypes.data == ptr)

a = np.arange(4.0)
t = torch.from_dlpack(a)
del a
gc.collect()
print("torch view survives dead numpy producer:", t.tolist())

a2 = np.arange(4.0)
t2 = torch.from_dlpack(a2, copy=True)
a2[0] = 99.0
print("torch copy=True: detached:", t2[0].item() == 0.0, "| shares:", t2.data_ptr() == a2.ctypes.data)

x2 = torch.arange(4.0)
n2 = np.from_dlpack(x2, copy=True)
x2[0] = 99.0
print("numpy copy=True: detached:", n2[0] == 0.0, "| owndata:", n2.flags.owndata,
      "| shares:", n2.ctypes.data == x2.data_ptr())

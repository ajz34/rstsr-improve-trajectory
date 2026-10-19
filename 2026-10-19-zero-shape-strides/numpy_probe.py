"""Probe NumPy 2.5.x zero-size stride behaviour (reference for issue #161)."""
import numpy as np


def show(label, a):
    print(f"{label:42s} shape={a.shape!s:12s} strides={a.strides!s:12s} "
          f"C={a.flags.c_contiguous} F={a.flags.f_contiguous}")


print("numpy", np.__version__)
print("\n-- freshly allocated zero-size arrays (all strides 0) --")
for shp in [(0, 18), (18, 0), (0, 0), (0, 3, 4), (3, 0, 4), (3, 4, 0), (0,)]:
    show(f"np.empty({shp})", np.empty(shp))
    show(f"np.empty({shp}, order='F')", np.empty(shp, order="F"))

print("\n-- views keep / derive strides; flags stay both-contiguous --")
a = np.empty((0, 3))
show("a", a)
show("a.T", a.T)
show("a.reshape(3, 0)", a.reshape(3, 0))
show("a.reshape(-1)", a.reshape(-1))
show("a.ravel()", a.ravel())
show("a.squeeze()", a.squeeze())
base = np.empty((18, 5))
v = base[:, 0:0]
show("np.empty((18,5))[:, 0:0]", v)
baseF = np.empty((5, 18), order="F")
show("np.empty((5,18), order='F')[0:0, :].T", baseF[0:0, :].T)

print("\n-- numpy.linalg on empty operands succeeds (no BLAS error) --")
A = np.eye(18)
for label, f, args in [
    ("solve(eye(18), zeros((18,0)))", np.linalg.solve, (A, np.zeros((18, 0)))),
    ("solve(eye(18), C-view [:, 0:0])", np.linalg.solve, (A, base[:, 0:0])),
    ("inv(zeros((0,0)))", np.linalg.inv, (np.zeros((0, 0)),)),
    ("det(zeros((0,0)))", np.linalg.det, (np.zeros((0, 0)),)),
    ("matmul(eye(18), zeros((18,0)))", np.matmul, (A, np.zeros((18, 0)))),
]:
    r = f(*args)
    print(f"{label:42s} -> shape={getattr(r, 'shape', r)}")

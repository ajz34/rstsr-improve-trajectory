# Generates the numpy-derived constants used by
# rstsr-linalg-traits/tests/test_faer_edge/edge_f64.rs (numpy 2.x / scipy).
# Run: conda activate torch && python gen_expected_constants.py
import numpy as np
np.set_printoptions(precision=17)
rng = np.random.default_rng(20260914)

A = rng.standard_normal((5, 5)) + 5 * np.eye(5)
B = rng.standard_normal((5, 3))
b = rng.standard_normal(5)
x = np.linalg.solve(A, B)
x_vec = np.linalg.solve(A, b)
invA = np.linalg.inv(A)
detA = np.linalg.det(A)
S = np.array([[4.0, 1.0, 0.5, 0.0], [1.0, 3.0, 0.2, 0.3], [0.5, 0.2, 2.0, 0.1], [0.0, 0.3, 0.1, 1.5]])
w, v = np.linalg.eigh(S)
C = S + 2 * np.eye(4)
cl = np.linalg.cholesky(C)
M = rng.standard_normal((4, 6))
sv = np.linalg.svd(M, compute_uv=False)
R = np.array([[3.0, 1.0, 2.0, 0.5], [1.0, 2.0, 1.0, 1.0], [2.0, 1.0, 4.0, 0.5], [0.5, 1.0, 0.5, 1.0]])
Rrank2 = np.outer(R[:, 0], R[:, 0]) / 4 + np.outer(R[:, 1], R[:, 1]) / 9
pr = np.linalg.pinv(Rrank2)
L4 = np.tril(rng.standard_normal((4, 4))) + 4 * np.eye(4)
from scipy.linalg import solve_triangular
Xt = solve_triangular(L4, B[:4], lower=True)
from scipy.linalg import eigh as sc_eigh
gw, gv = sc_eigh(S, C)

def show(name, arr):
    arr = np.asarray(arr)
    print(f"pub const {name}: {arr.dtype}[..] = [", end="")
    print(", ".join(repr(float(v)) for v in arr.ravel().tolist()), end="];\n")

for name, arr in [("A", A), ("B", B), ("b", b), ("X", x), ("X_VEC", x_vec), ("INV_A", invA),
                  ("S", S), ("EIGH_W", w), ("C_MAT", C), ("CHOL_L", cl), ("M", M), ("SVD_S", sv),
                  ("RANK2", Rrank2), ("PINV_R2", pr), ("L4", L4), ("TRI_X", Xt), ("GEN_W", gw)]:
    show(name, arr)
print("DET_A", repr(float(detA)))

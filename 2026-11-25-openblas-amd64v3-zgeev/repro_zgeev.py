#!/usr/bin/env python3
"""Reproducer: zgeev returns eigenvectors that do not satisfy A v = lambda v.

Usage:
    python3 repro_zgeev.py [n] [library]

`library` defaults to the system libopenblas.so.0.  Only ctypes is used, so the
result does not depend on numpy/scipy or on whichever BLAS they bundle.

Exit status: 0 when both eigenvector blocks satisfy the defining relation,
1 otherwise.
"""

import ctypes
import ctypes.util
import math
import sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 80
LIB = sys.argv[2] if len(sys.argv) > 2 else (ctypes.util.find_library("openblas") or "libopenblas.so.0")


def deterministic_matrix(n):
    """Random-looking complex matrix with a fixed generator (no numpy)."""
    state = 0x243F6A8885A308D3
    mask = (1 << 64) - 1
    rows = [[0j] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            state = (state * 6364136223846793005 + 1442695040888963407) & mask
            re = (state >> 11) / float(1 << 53) - 0.5
            state = (state * 6364136223846793005 + 1442695040888963407) & mask
            im = (state >> 33) / float(1 << 31) - 0.5
            rows[i][j] = complex(re, im)
    return rows


def as_fortran(rows):
    """LAPACK complex*16 column-major buffer: interleaved (re, im) doubles."""
    n = len(rows)
    buf = (ctypes.c_double * (2 * n * n))()
    for j in range(n):
        for i in range(n):
            z = rows[i][j]
            buf[2 * (i + j * n)] = z.real
            buf[2 * (i + j * n) + 1] = z.imag
    return buf


def column(buf, n, j):
    return [complex(buf[2 * (i + j * n)], buf[2 * (i + j * n) + 1]) for i in range(n)]


def relation_residual(rows, w, vecs, left):
    """max_j || M v_j - mu_j v_j|| / ||v_j||, with mu = conj(w_j) for left."""
    n = len(rows)
    worst = 0.0
    for j in range(n):
        v = vecs[j]
        norm = math.sqrt(sum(abs(z) ** 2 for z in v))
        out = []
        for i in range(n):
            if left:
                acc = sum(rows[k][i].conjugate() * v[k] for k in range(n))
            else:
                acc = sum(rows[i][k] * v[k] for k in range(n))
            out.append(acc)
        mu = w[j].conjugate() if left else w[j]
        res = math.sqrt(sum(abs(out[i] - mu * v[i]) ** 2 for i in range(n)))
        worst = max(worst, res / norm)
    return worst


def main():
    n = N
    rows = deterministic_matrix(n)
    lib = ctypes.CDLL(LIB)
    lib.zgeev_.restype = None

    def zgeev(job_l, job_r, a, w, vl, vr, work, lwork, rwork, info):
        lib.zgeev_(ctypes.byref(job_l), ctypes.byref(job_r), ctypes.byref(ctypes.c_int(n)),
                   a, ctypes.byref(ctypes.c_int(n)), w, vl, ctypes.byref(ctypes.c_int(n)),
                   vr, ctypes.byref(ctypes.c_int(n)), work, ctypes.byref(ctypes.c_int(lwork)),
                   rwork, ctypes.byref(info))

    job_v, job_n = ctypes.c_char(b"V"), ctypes.c_char(b"N")
    w = (ctypes.c_double * (2 * n))()
    vl = (ctypes.c_double * (2 * n * n))()
    vr = (ctypes.c_double * (2 * n * n))()
    rwork = (ctypes.c_double * (2 * n))()
    query = (ctypes.c_double * 2)()
    info = ctypes.c_int(0)

    zgeev(job_v, job_v, as_fortran(rows), w, vl, vr, query, -1, rwork, info)
    lwork = int(ctypes.cast(query, ctypes.POINTER(ctypes.c_double))[0])
    work = (ctypes.c_double * (2 * lwork))()
    zgeev(job_v, job_v, as_fortran(rows), w, vl, vr, work, lwork, rwork, info)

    if info.value != 0:
        print(f"zgeev failed with info={info.value}")
        return 1

    spectrum = [complex(w[2 * j], w[2 * j + 1]) for j in range(n)]
    left = [column(vl, n, j) for j in range(n)]
    right = [column(vr, n, j) for j in range(n)]
    res_l = relation_residual(rows, spectrum, left, left=True)
    res_r = relation_residual(rows, spectrum, right, left=False)
    print(f"n={n} library={LIB}")
    print(f"  left  relation residual: {res_l:.3e}")
    print(f"  right relation residual: {res_r:.3e}")
    ok = res_l < 1e-8 and res_r < 1e-8
    print("  " + ("OK: eigenvectors satisfy the relation" if ok else "BROKEN: eigenvectors do not satisfy the relation"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

# Grill R2 — QR decomposition on the BLAS devices

## Answers to R1 (agent transcription; answers were given on screen 2026-11-24)

- **Q1, Q4, Q5, Q6, Q9, Q10, Q13, Q14** — recommendation accepted.
- **Q2** — deferred: "this keyword can be confusing and difficult to be
  implemented".
- **Q3** — deferred: "we may introduce a new function if column-pivoting is
  needed in the future".
- **Q7** — amended: "if not introducing much difficult code, I suggest you
  still introduce `TensorMut` overload. I acknowledged that it is mostly
  meaningless for QR." → carried to R2 Q15.
- **Q8** — accepted, "just follow numpy/scipy behavior. If they have different
  behavior, you can raise another question." → checked, see §A (no
  difference found).
- **Q11** — accepted with a check: "make sure if python script will generate
  QR values and you can test the fingerprint" → see §B, R2 Q16.
- **Q12** — rejected: "I prefer not edit unless contradiction occurs." The
  skill's §3a table stays untouched; the gap is recorded here only.

## §A Fact check for Q8 (probe `/tmp/qr_probe.py`, numpy 2.5.2 / scipy in env `torch`)

- numpy `qr(..., 'complete')` uses a dedicated `qr_complete` gufunc **only when
  `m > n`**; it zero-pads the `(m, n)` factor to `(m, m)` before `orgqr` with
  `k = m`. Otherwise `qr_reduced` with `k = min(m, n)`. `R = triu(h[:k, :])`.
- scipy `mode='full'` agrees with numpy on all four cells
  (tall/wide × reduced/complete): after making `diag(R)` real-positive the
  two differ by ≤ 5.3e-14 (fp agreement to 1e-13), and the sign choices are
  identical.
- Zero-size shapes agree between the two and match Q9:
  `(3, 0)` → reduced `Q (3, 0)`, `R (0, 0)`; complete `Q = I₃`, `R (3, 0)`.
  `(0, 4)`, `(0, 0)` → all-empty in both modes.

No behavioral question arises from this; the plan follows the common behavior.

## §B Fact check for Q11

`blas_tests_impl/linalg_func/func_validation_{f64,c64}.py` are the oracle
scripts: they load `rstsr-test-manifest/resources/{a,b}-f64.npy`, compute with
NumPy/SciPy, and `assert np.isclose(fingerprint(..), <literal>)`; the same
literal is pasted into the Rust test. Every linalg entry has a section
(`# ### svd` …). A `# ### qr` section fits the pattern with no script change.

Caveat found: a raw `fingerprint(Q)` is **not vendor-portable** (Q columns are
fixed only up to a unit factor; numpy's lapack_lite and OpenBLAS may pick
different signs), and the extra columns `k..m` of a complete Q are an
arbitrary orthonormal completion. So the fingerprint must be taken on a
canonical form — see Q16.

---

## Round 2

❓ **Q15 - `TensorMut` overload semantics**: add, for each device,
`QrAPI<DeviceBLAS> for (TensorMut<'a, T, B, D>, QrMode)` and for the bare
`TensorMut` (reduced), returning `QrResult<Tensor, Tensor>` like the
immutable forms. The input is **consumed as scratch and left destroyed** (its
buffer holds the Householder vectors after `GEQRF`); nothing is written back.
Cost: one extra row in each `duplicate_item` table plus the same write-up as
the SVD entry's docstring.

➡️ Yes. It is ~10 lines, mirrors `cholesky(a.view_mut())` call syntax, and
saves one full copy for callers who own the matrix.

---

❓ **Q16 - Fingerprint form in the conformance tests**: extend both validator
scripts with a `# ### qr` section and assert, in Rust, on the
**phase-canonicalized** results:

- `Qc = Q[:, :k] * d`, `Rc = triu(R)[:k, :] / d[:, None]` with
  `d = diag(R) / |diag(R)|` (real-positive diagonal);
- `fingerprint(Qc)`, `fingerprint(Rc)` for tall `1024×512` and wide
  `512×1024`, reduced and complete (`Rc` is identical across the two modes —
  asserted too);
- never fingerprint complete-Q columns `k..m`; they are checked by
  reconstruction only.

Both `func_validation_f64.py` and `func_validation_c64.py` (complex phase,
`abs(diag)` divisor).

➡️ Yes.

---

❓ **Q17 - Buffer plan in `ref_impl_qr_simple_f`**:

- `h = overwritable_convert(a)` → `GEQRF` in place → `tau`.
- **Reduced with `m ≥ n`**: `k = n`, `h` already has shape `(m, k)`; copy
  `R = triu(h)` into a fresh `(k, n)` tensor first, then run `ORGQR` **in
  place on `h`** (Q overwrites it; no extra allocation).
- **Every other case** (`m < n`, or complete): allocate `q = zeros((m, k'))`
  in the call order, copy `h[:, :min(n, k')]` into it, run `ORGQR` on `q`;
  `R = triu(h)` top `k` rows (complete: zero-padded rows `k..m`).

➡️ Yes. One extra buffer only where the shapes force it.

---

❓ **Q18 - Error and dimension contract**: `ndim < 2` → `InvalidLayout`
(from `StackLayout::new`, same message as the other stacked entries); no
squareness or rank check; `info > 0` never occurs for `geqrf`/`orgqr`
(mapped to a generic message anyway).

➡️ Yes.

---

❓ **Q19 - Close the grill**: with Q15–Q18 accepted the frontier is empty.
Next step: write `DECISIONS.md` here, then implement on rstsr branch
`261124/blas-qr` (no commits; PR only on your instruction), run
`tests_driver_impl` + `tests_linalg_func` on `rstsr-openblas` under both
`lapack` and `lapacke`, and re-run the two validator scripts.

➡️ Yes — confirm we have a shared understanding and I may start.

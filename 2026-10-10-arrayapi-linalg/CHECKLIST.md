# linalg convergence checklist — rstsr_faer.api × array-api-tests (2026-10-10)

The last remaining block of the array-API convergence. With `fft` declared out
of scope for this version, the *only* thing standing between rstsr_faer.api and
the NumPy band is the **`linalg` extension** (the array-API `xp.linalg`
namespace, its 23 member functions, and the four top-level linear-algebra
functions). This directory is the work plan for it; the frozen status read-out
and the other (now-drained) buckets live in
`../2026-10-07-arrayapi-convergence/`; register ids (G-nnn) refer to
`../2026-10-04-rstsr-faer-py/GAP-REGISTER.md`.

**Current hinge: 37 fails + 49 skips = 86 tests.** Suite `array-api-tests@6c0b59f`,
API version `2025.12`, module `rstsr_faer.api` at rstsr `main` `9db98e8`
(1249 / 51 / 82 of 1382 overall).

## 0. The red map this checklist must clear

**Fails (37).** All from `G-023` — the namespace and its top-level members are
simply absent today:

| group | nodes | what it checks |
|---|---|---|
| `test_has_names[linalg-*]` | 23 | each of the 23 `linalg` member names exists |
| `test_has_names[linear_algebra-{matmul,matrix_transpose,tensordot,vecdot}]` | 4 | the four top-level names exist |
| `test_has_names[array_method-__matmul__]` | 1 | the array object has `__matmul__` |
| `test_linalg.py::{test_matmul,test_matrix_transpose,test_tensordot,test_vecdot}` | 4 | top-level value tests |
| `test_signatures.py::test_array_method_signature[__matmul__]` | 1 | `__matmul__` signature |
| `test_signatures.py::test_func_signature[{matmul,matrix_transpose,tensordot,vecdot}]` | 4 | top-level signatures |

**Skips (49) that light up the moment `xp.linalg` exists** — every test carrying
`@pytest.mark.xp_extension("linalg")` stops skipping once
`xp_has_ext("linalg")` is true:

- `test_linalg.py` — 26 nodes: `test_{cholesky,cross,det,diagonal,eigh,eigvalsh,
  eig,eigvals,inv,linalg_matmul,matrix_norm,matrix_power,matrix_rank,
  linalg_matrix_transpose,outer,pinv,qr,slogdet,solve,svd,svdvals,
  linalg_tensordot,trace,linalg_vecdot}` (plus the `eig`/`eigvals` nodes gated
  on `min_version("2025.12")` — our pinned API version, so they run).
- `test_signatures.py::test_extension_func_signature[linalg.*]` — 23 nodes.

> Note: adding `xp.linalg` turns these **on**, so a partial namespace that
> exists but is missing `eig`/`eigvals`/`qr`/… converts skips into *new fails*.
> Land the surface only once its members are real (see the ordering in §4).

## 1. Rust-side inventory — what exists today

Rust linalg lives in **`rstsr-linalg-traits`** (`LinalgAPI` family, one
`func`/`func_f` per op) with per-device impls under `faer_impl/` and
`blas_impl/`; the tensor-level linear algebra (`matmul`, `vecdot`,
`matrix_transpose`, `diagonal`) is in **`rstsr-core/src/tensor/`**. DeviceFaer
(the shim's device) implements the faer column.

`✓` present · `◐` partial · `✗` absent.

| array-API name | rstsr entry today | status |
|---|---|---|
| `matmul` | `rt::matmul` (tensor tier) | ✓ |
| `matrix_transpose` | `rt::matrix_transpose` | ✓ |
| `vecdot` | `rt::vecdot` | ✓ |
| `diagonal` | `rt::diagonal` (indexing) | ✓ |
| `cholesky` | `CholeskyAPI` (faer) | ✓ |
| `det` | `DetAPI` (faer) | ✓ |
| `eigh` | `EighAPI` (faer) | ✓ |
| `eigvalsh` | `EigvalshAPI` (faer) | ✓ |
| `inv` | `InvAPI` (faer) | ✓ |
| `pinv` | `PinvAPI` (faer) | ✓ |
| `solve` | `SolveGeneralAPI` (faer) | ✓ |
| `svd` | `SVDAPI` (faer) | ✓ |
| `svdvals` | `SVDvalsAPI` (faer) | ✓ |
| `tensordot` | — | ✗ tensor-tier new |
| `outer` | — (reshape + broadcast mul) | ✗ new |
| `cross` | — (3-vector cross) | ✗ new |
| `trace` | — (diagonal + sum) | ✗ new (trivial) |
| `matrix_power` | — (repeated `matmul`) | ✗ new |
| `matrix_rank` | — (via `svdvals`) | ✗ new |
| `vector_norm` | `l2_norm` family (ord=2 only) | ◐ general `ord` missing |
| `matrix_norm` | `l2_norm` family (ord=2 only) | ◐ general `ord` missing |
| `qr` | — | ✗ **G-004** (no faer *or* blas impl) |
| `slogdet` | `SLogDetAPI` (blas only) | ◐ faer missing — **G-005** |
| `eig` | — (general, non-symmetric) | ✗ new (2025.12; complex output) |
| `eigvals` | — (general, non-symmetric) | ✗ new (2025.12) |

Also relevant: `solve_symmetric` exists on blas but not on faer
(**G-005**) — array-API `solve` itself maps to `solve_general`, so this is
tangential unless the shim routes symmetric inputs there.

## 2. Rust-side work (register + request; never fix agent-side)

Each item is a `GAP-REGISTER` entry. Ordered by failures-per-item — the four
small tensor-tier ops gate the largest, cheapest surface count.

- [ ] **R1. `tensordot`** (tensor tier, `tensor/linalg/`): the one top-level
      member with no kernel — a multi-axis contraction (general `axes` int or
      pair-of-lists). Its absence alone costs 1 `has_names` + 1
      `test_signatures` + 1 `test_linalg` node.
- [ ] **R2. `qr`** (**G-004**): neither `faer_impl` nor `blas_impl` has a QR
      trait. Needs a `QRAPI` + both device columns (BLAS: `geqrf`/`orgqr`; faer:
      `qr::no_pivoting` / `qr::with_pivoting`). `test_qr` grades mode
      (`reduced`/`complete`/`r`).
- [ ] **R3. `slogdet` on faer** (**G-005**): present on blas, absent on the
      DeviceFaer column the shim uses. Either port the blas routine or derive
      from `det` + sign (`lu` factorisation). `test_slogdet` grades the
      `(sign, logabsdet)` pair.
- [ ] **R4. `eig` / `eigvals`** (2025.12): general (non-symmetric) eigenproblem,
      complex output (`eig` returns a namedtuple `(eigenvalues, eigenvectors)`).
      None of `eigh`/`eigvalsh` covers it. Two activated nodes; needed or they
      fail once `linalg` exists.
- [ ] **R5. `vector_norm` / `matrix_norm` general `ord`**: today only the
      `l2_norm` family exists. `test_vector_norm` / `test_matrix_norm` grade
      every `ord` (`inf`, `-inf`, `0`, `1`, `2`, `-1`, `-2`, `fro`, `nuc`, …).
- [ ] **R6. small tensor-tier ops** `outer`, `cross`, `trace`, `matrix_power`,
      `matrix_rank`: all derivable (`outer` = broadcast-mul + reshape; `trace` =
      `diagonal`+sum over the last axis; `matrix_rank` = count of `svdvals >
      tol`; `matrix_power` = repeated `matmul` with `n` sign handling). Low
      risk, but each is a `test_linalg` node plus a `has_names` node.

## 3. Shim-side work (`crates-interop/rstsr-faer-py`)

Wrapper-only once the kernels exist. The shim exposes **no** linalg today.

- [ ] **S1. `xp.linalg` namespace**: a `linalg` submodule (or object) exposing
      the 23 array-API names, each marshalling through the corresponding
      rust entry. Register it so `hasattr(xp, "linalg")` is true **only when
      every member is present** (see §4).
- [ ] **S2. top-level members**: `matmul`, `matrix_transpose`, `vecdot`,
      `tensordot` on `xp` itself (all four also appear under `linalg`).
- [ ] **S3. `__matmul__`** on the array class (the `@` operator), matching
      `matmul`'s semantics.
- [ ] **S4. return structures**: array-API returns specific containers the
      tests introspect — `eigh` → namedtuple `(eigenvalues, eigenvectors)`,
      `eig` → same, `svd` → `(U, S, Vh)`, `qr` → `(Q, R)`, `slogdet` →
      `(sign, logabsdet)`, `pinv` → array (rank is not returned), `svdvals`/
      `eigvalsh` → arrays. Map rstsr's `*Result` structs (`SVDResult`,
      `EighResult`, `SLogDetResult`, `PinvResult`) to these.
- [ ] **S5. dtype/stack semantics**: `eig`/`eigvals` promote real input to the
      complex dtype (`dh.complex_dtype_for`); batched (stacked) inputs must
      iterate the leading axes as the rest of the surface does.

## 4. Ordering (avoid converting skips into fails)

The suite's marker flips 49 skips → *run* the instant `xp.linalg` exists. So:

1. Land **R1–R6** (kernels) first — nothing is exposed until they exist.
2. Then **S1–S5** in one step, so the namespace appears fully-formed.
3. Verify per §5 before recording any number.

Cheapest first slice if a partial landing is wanted: the pure-shim trio
(`matmul`, `matrix_transpose`, `vecdot`) + `__matmul__` (S2/S3) flips 4
top-level names + 1 signature + 1 method node + 2 `test_linalg` nodes with
**no** rust change — but it does not make `xp.linalg` exist, so it moves 8
nodes, not the 86.

## 5. Verification

Reuse the rstsr-faer-py-tests harness (skill `rstsr-faer-py-tests`):

```bash
cd <scratch>   # not inside a git checkout
FRESH=1 NO_EXPLAIN=1 CHUNKED=1 MODULE=rstsr_faer.api \
  /path/to/skills/rstsr-faer-py-tests/scripts/run.sh
```

- Build release at any opt-level (never the dev profile); record the wheel
  provenance as in `../2026-10-07-arrayapi-convergence/README.md`.
- Expect: `1249 / 51 / 82` → `1249 + 37 = 1286 / 14 / 82` when the full 86-node
  block lands (the residual 14 fails are `fft`), **provided** the 49 activated
  skips all pass; 0 regressions, node set test-for-test stable vs. the previous
  stamp.
- Per-item counts to confirm against §0 as work lands (the suite is
  hypothesis-based; expect ±2 wobble — see the sibling checklist's §D).

## 6. Decisions to grill (before coding)

- [ ] **D1. Implement vs. scope-out** — the owner's call (this is B1 of the
      sibling checklist). Scope-out would instead add the 37+49 `linalg` ids to
      a skips file (harness `SKIPS_FILE=`), leaving the red map at the 14 `fft`
      fails; but `linalg` is the far larger prize (86 tests) and most kernels
      already exist.
- [ ] **D2. `ord` coverage for norms** — full array-API `ord` table vs. the
      subset rstsr can express; determines R5's scope.
- [ ] **D3. `linalg` as namespace object vs. submodule** — affects import-time
      cost and how `__array_namespace__` reports it.
- [ ] **D4. Where the symmetric/triangular solvers surface** — array-API
      `solve` is general; decide whether `solve_symmetric`/`solve_triangular`
      stay rust-internal or get shim names (not required by the suite).

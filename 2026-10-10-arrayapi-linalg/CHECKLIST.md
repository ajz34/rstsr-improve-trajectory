# linalg convergence checklist — rstsr_faer.api × array-api-tests (2026-10-10)

The last remaining block of the array-API convergence. With `fft` declared out
of scope for this version, the *only* thing standing between rstsr_faer.api and
the NumPy band is the **`linalg` extension** (the array-API `xp.linalg`
namespace, its 23 member functions, and the four top-level linear-algebra
functions). This directory is the work plan for it; the frozen status read-out
and the other (now-drained) buckets live in
`../2026-10-07-arrayapi-convergence/`; register ids (G-nnn) refer to
`../2026-10-04-rstsr-faer-py/GAP-REGISTER.md`.

**Current: 1287 / 62 / 33 of 1382** (suite `array-api-tests@6c0b59f`, API
`2025.12`, module `rstsr_faer.api`). The shim-side init landed (uncommitted,
branch `261010/faer-py-linalg-init` off rstsr `main` `9db98e8`) — see
**Progress**; it exposed 13 of the 23 members and flipped the 49 `linalg` skips
into graded tests. The 62 failures are exactly **`linalg` 48 + `fft` 14**; the
33 skips are 28 `fft` + the 5 backend-independent `test_remainder`.

## Progress (2026-10-10): shim-side init

Branch `261010/faer-py-linalg-init` (off rstsr `main` `9db98e8`, **not yet
committed**). Purely shim-side — thin pass-throughs, no algorithm and no
batching in the wrapper:

- New `crates-interop/rstsr-faer-py/src/linalg.rs` (13 `#[pyfunction]`s) +
  registration in `src/lib.rs`; the `rstsr` dep gains the `linalg` feature.
- `python/rstsr_faer/api.py`: `_LinalgNamespace` exposing **13 of the 23**
  members (`cholesky, det, diagonal, eigh, eigvalsh, inv, matmul,
  matrix_transpose, pinv, solve, svd, svdvals, vecdot`), the top-level
  `matmul`/`matrix_transpose`/`vecdot`, and `Array.__matmul__`.
- Measured: **1249 / 51 / 82 → 1287 / 62 / 33**. Exposing the partial
  namespace flipped the 49 `linalg` skips → run (16 pass / 33 fail); **0
  regressions** outside the linalg/fft name family.
- **Dominant finding (new — the pre-init plan missed it):** rstsr's faer
  linalg entries are **2-D only** (`rstsr_assert_eq!(ndim, 2)`), so with a
  pass-through wrapper every *stacked* input (`ndim > 2`) raises. The
  array-API value tests are hypothesis-generated and hit stacks on the first
  example, so of the 13 exposed members only `diagonal` and `matrix_transpose`
  pass their value tests today. **Batched (stacked) linalg is a rust-side
  gap** — it is now the single biggest lever.
- Second finding: `matmul`/`vecdot` are same-dtype only (mixed pairs need the
  G-009 promotion table), so their value tests fail on the first mixed pair.

## 0. The red map this checklist must clear

**Current failures (62) = `linalg` 48 + `fft` 14.**

`linalg` 48:

| group | nodes | what it checks |
|---|---|---|
| `test_has_names[linalg-*]` | 10 | the 10 members still absent from `xp.linalg` |
| `test_has_names[linear_algebra-tensordot]` | 1 | top-level `tensordot` absent |
| `test_signatures.py::test_extension_func_signature[linalg.*]` | 10 | the same 10 members |
| `test_signatures.py::test_func_signature[tensordot]` | 1 | top-level `tensordot` signature |
| `test_linalg.py` (name absent) | 12 | `test_{cross,eig,eigvals,matrix_norm,matrix_power,matrix_rank,outer,qr,slogdet,trace,vector_norm,tensordot}` |
| `test_linalg.py` (exposed, value-failing) | 13 | `test_{cholesky,det,eigh,eigvalsh,inv,linalg_matmul,linalg_vecdot,matmul,pinv,solve,svd,svdvals,vecdot}` |

The 10 absent `linalg` members: `cross, matrix_norm, matrix_power,
matrix_rank, outer, qr, slogdet, tensordot, trace, vector_norm`.

`test_linalg.py` **passing** today (4): `test_diagonal`,
`test_linalg_matrix_transpose`, `test_matrix_transpose`, `test_vecdot_conj`.

**Skips (33)** — 28 `fft` (out of scope) + 5 `test_remainder` (upstream
`@pytest.mark.skip("flaky")`). No `linalg` skips remain.

## 1. Rust-side inventory — what exists today

Rust linalg lives in **`rstsr-linalg-traits`** (`LinalgAPI` family, one
`func`/`func_f` per op) with per-device impls under `faer_impl/` and
`blas_impl/`; the tensor-level linear algebra (`matmul`, `vecdot`,
`matrix_transpose`, `diagonal`) is in **`rstsr-core/src/tensor/`**. DeviceFaer
(the shim's device) implements the faer column.

`✓` present · `◐` partial · `✗` absent. "shim" = exposed by the init branch.

| array-API name | rstsr entry today | rust | shim |
|---|---|---|---|
| `matmul` | `rt::matmul` (tensor tier) | ✓ same-dtype | ✓ |
| `matrix_transpose` | `rt::matrix_transpose` | ✓ | ✓ |
| `vecdot` | `rt::vecdot` | ✓ same-dtype | ✓ |
| `diagonal` | `rt::diagonal` (indexing) | ✓ | ✓ |
| `cholesky` | `CholeskyAPI` (faer) | ◐ 2-D only | ✓ |
| `det` | `DetAPI` (faer) | ◐ 2-D only | ✓ |
| `eigh` | `EighAPI` (faer) | ◐ 2-D only | ✓ |
| `eigvalsh` | `EigvalshAPI` (faer) | ◐ 2-D only | ✓ |
| `inv` | `InvAPI` (faer) | ◐ 2-D only | ✓ |
| `pinv` | `PinvAPI` (faer) | ◐ 2-D only | ✓ |
| `solve` | `SolveGeneralAPI` (faer) | ◐ 2-D only | ✓ |
| `svd` | `SVDAPI` (faer) | ◐ 2-D only | ✓ |
| `svdvals` | `SVDvalsAPI` (faer) | ◐ 2-D only | ✓ |
| `tensordot` | — | ✗ tensor-tier new | ✗ |
| `outer` | — (reshape + broadcast mul) | ✗ new | ✗ |
| `cross` | — (3-vector cross) | ✗ new | ✗ |
| `trace` | — (diagonal + sum) | ✗ new (trivial) | ✗ |
| `matrix_power` | — (repeated `matmul`) | ✗ new | ✗ |
| `matrix_rank` | — (via `svdvals`) | ✗ new | ✗ |
| `vector_norm` | `l2_norm` family (ord=2 only) | ◐ general `ord` missing | ✗ |
| `matrix_norm` | `l2_norm` family (ord=2 only) | ◐ general `ord` missing | ✗ |
| `qr` | — | ✗ **G-004** | ✗ |
| `slogdet` | `SLogDetAPI` (blas only) | ◐ faer missing — **G-005** | ✗ |
| `eig` | — (general, non-symmetric) | ✗ new (tested; not in 2025.12 `__all__`) | ✗ |
| `eigvals` | — (general, non-symmetric) | ✗ new (ibid.) | ✗ |

Also relevant: `solve_symmetric` exists on blas but not on faer (**G-005**) —
array-API `solve` maps to `solve_general`, so this is tangential.

## 2. Rust-side work (register + request; never fix agent-side)

Ordered by failures-per-item. **R1 is now the dominant lever** — it alone
gates 13 of the 20 graded members the shim already exposes.

- [ ] **R1. Batched (stacked) linalg** — every faer factorization today asserts
      `ndim == 2`. The array-API contract is "a matrix *or a stack of matrices*"
      (`(..., M, M)`), and the value tests exercise stacks immediately. Give
      the faer linalg entries (and the tensor-tier `solve`/`pinv` route) a
      leading-axis loop — either iterate the batch in the device op or map over
      `iter_axes` and stack. Cheapest, largest payoff: unblocks `cholesky, det,
      eigh, eigvalsh, inv, pinv, solve, svd, svdvals` at once.
- [ ] **R2. mixed-dtype `matmul`/`vecdot`** (G-009) — promote the operands to
      their common dtype (the `DTypePromoteAPI` lattice already exists) before
      the kernel, so `u8 @ u16` works. Gates `test_matmul` / `test_vecdot` /
      `test_linalg_vecdot` beyond the batching issue.
- [ ] **R3. `tensordot`** (tensor tier, `tensor/linalg/`) — the one top-level
      member with no kernel (`axes` int or pair-of-lists). Its absence costs
      1 `has_names` + 1 signature + 2 `test_linalg` nodes.
- [ ] **R4. small derivable ops** `outer, cross, trace, matrix_power,
      matrix_rank` — all expressible from existing pieces (`outer` = broadcast
      mul + reshape; `trace` = `diagonal` + sum; `matrix_rank` = count of
      `svdvals > tol`; `matrix_power` = repeated `matmul` with `n` sign
      handling). Low risk; each is a `test_linalg` + a `has_names` node.
- [ ] **R5. norms general `ord`** — today only the `l2_norm` family exists.
      `test_vector_norm` / `test_matrix_norm` grade every `ord` (`inf`, `-inf`,
      `0`, `1`, `2`, `-1`, `-2`, `fro`, `nuc`, …).
- [ ] **R6. `qr`** (**G-004**): neither `faer_impl` nor `blas_impl` has a QR
      trait. Needs a `QRAPI` + both device columns (`geqrf`/`orgqr`; faer
      `qr::no_pivoting`). `test_qr` grades mode (`reduced`/`complete`/`r`).
- [ ] **R7. `slogdet` on faer** (**G-005**): present on blas, absent on the
      DeviceFaer column the shim uses (port the blas routine or derive from
      `det` + sign). `test_slogdet` grades the `(sign, logabsdet)` pair.
- [ ] **R8. `eig` / `eigvals`**: general (non-symmetric) eigenproblem, complex
      output. Not in the 2025.12 `linalg.__all__`, but `test_linalg.py` carries
      `test_eig`/`test_eigvals` gated on `min_version("2025.12")` = our pin, so
      they run and fail. Lowest priority (names not required for the surface).

## 3. Shim-side work (`crates-interop/rstsr-faer-py`) — done on the branch

Wrapper-only. S1–S4 landed on `261010/faer-py-linalg-init` (uncommitted):

- [x] **S1. `xp.linalg` namespace** — a `_LinalgNamespace` object exposing the
      13 available names; missing members stay absent (rendered as gaps, never
      stubbed). `api.py` §linalg.
- [x] **S2. top-level members** — `matmul`, `matrix_transpose`, `vecdot` on
      `xp`; `tensordot` absent (rust gap R3).
- [x] **S3. `__matmul__`** on `Array` (the `@` operator), delegating to
      `matmul`.
- [x] **S4. return structures** — `eigh` → namedtuple `(eigenvalues,
      eigenvectors)`, `svd` → `(U, S, Vh)`, `det`/`svdvals`/`eigvalsh` →
      arrays, `pinv` → array (rank dropped).
- [ ] **S5. dtype/stack semantics** — *not* done: the shim is a pass-through, so
      there is no batching (that is R1) and `matmul`/`vecdot` do not promote
      (that is R2). `eig`/`eigvals` complex promotion is moot until R8.

## 4. Ordering

The namespace is now exposed (partial), so the skip→run flip has already
happened. Going forward:

1. **R1 (batched linalg) first** — it is the largest single move and needs no
   new shim code; the 13 exposed-but-failing value tests start passing as soon
   as the rust entries accept stacks.
2. Then R2–R8 as members land; each new member is a shim one-liner plus a
   `_LinalgNamespace` entry (add the name only when its rust entry is real, so a
   new gap never becomes a fresh failure).
3. Re-run per §5 after each landed item.

If instead the namespace should be **held** until the surface is more complete
(see D1), drop the `linalg` object and keep only the top-level trio + `@` —
that reverts the 49 skips and the `linalg` name nodes to their pre-init state.

## 5. Verification

Reuse the rstsr-faer-py-tests harness (skill `rstsr-faer-py-tests`):

```bash
cd <scratch>   # not inside a git checkout
FRESH=1 NO_EXPLAIN=1 CHUNKED=1 MODULE=rstsr_faer.api \
  /path/to/skills/rstsr-faer-py-tests/scripts/run.sh
```

- Build release at any opt-level (never the dev profile); record the wheel
  provenance as in `../2026-10-07-arrayapi-convergence/README.md`.
- **Recorded today:** `1249 / 51 / 82` (pre-init) → **`1287 / 62 / 33`**
  (init branch). The `+38` passed / `-49` skipped is the 49 activated `linalg`
  tests (16 pass / 33 fail); expect **0 regressions** in the node set outside
  the linalg/fft name family.
- Expected after **R1** lands: `test_{cholesky,det,eigh,eigvalsh,inv,pinv,
  solve,svd,svdvals}` move to passed (≈ +9), the `matrix_norm`-family blockers
  remain. Confirm per-item counts against §0 as work lands (the suite is
  hypothesis-based; expect ±2 wobble).

## 6. Decisions

- [x] **D1. Implement vs. scope-out** — resolved: implement. The shim-side
      init is on the branch; the remaining `linalg` work is rust-side.
- [ ] **D2. Namespace now vs. gated** — the init exposes a *partial*
      `xp.linalg`, which converts the 49 `linalg` skips into graded tests
      (16 pass / 33 fail today). Keep it exposed and drive the failures down
      via R1–R8, or gate `xp.linalg` until the surface is more complete? (The
      owner's call; the branch makes either a one-line change.)
- [ ] **D3. Batching location** — implement stacked linalg inside the faer
      device ops (R1), or as a tensor-tier batch loop over the 2-D kernels?
      Affects every linalg entry's signature and the blas column too.
- [ ] **D4. `ord` coverage for norms** — full array-API `ord` table vs. the
      subset rstsr can express; determines R5's scope.
- [ ] **D5. Where the symmetric/triangular solvers surface** — array-API
      `solve` is general; decide whether `solve_symmetric`/`solve_triangular`
      stay rust-internal or get shim names (not required by the suite).

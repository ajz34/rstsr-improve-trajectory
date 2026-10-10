# linalg convergence checklist — rstsr_faer.api × array-api-tests (2026-10-10)

The last remaining block of the array-API convergence. With `fft` declared out
of scope for this version, the *only* thing standing between rstsr_faer.api and
the NumPy band is the **`linalg` extension** (the array-API `xp.linalg`
namespace, its 23 member functions, and the four top-level linear-algebra
functions). This directory is the work plan for it; the frozen status read-out
and the other (now-drained) buckets live in
`../2026-10-07-arrayapi-convergence/`; register ids (G-nnn) refer to
`../2026-10-04-rstsr-faer-py/GAP-REGISTER.md`.

**Current: 1296 / 53 / 33 of 1382** (suite `array-api-tests@6c0b59f`, API
`2025.12`, module `rstsr_faer.api`). The shim-side init is merged (`#133`,
`d7056ba`); it exposed 13 of the 23 members and flipped the 49 `linalg` skips
into graded tests. The remaining failures are `linalg` plus 14 out-of-scope
`fft` nodes; the 33 skips are 28 `fft` + the 5 backend-independent
`test_remainder`. `tensordot` (R3) and batched linalg (R1) landed after the
init — see **Progress**; `1287 / 62 / 33` → `1294 / 55 / 33` (R3) →
**`1296 / 53 / 33`** (R1).

## Progress (2026-10-10): shim-side init

Branch `261010/faer-py-linalg-init` (off rstsr `main` `9db98e8`; since merged
as `#133`, `d7056ba`). Purely shim-side — thin pass-throughs, no algorithm and
no batching in the wrapper:

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

## Progress (2026-10-10): tensordot (R3)

Landed rust-side + shim-side (branch `261010/tensordot` off `d7056ba`, PR
RESTGroup/rstsr#135). `rt::tensordot` is a new tensor-tier entry
(`tensor/linalg/tensordot.rs`) with a device op on `DeviceCpuSerial` and the
rayon auto-impl, and a view-only GEMM fast path; the shim exposes it as a
top-level `tensordot` and a `_LinalgNamespace` member.

- **4 nodes flipped to pass:** `test_has_names[linear_algebra-tensordot]`,
  `test_has_names[linalg-tensordot]`, `test_func_signature[tensordot]`,
  `test_extension_func_signature[linalg.tensordot]`.
- `test_tensordot` and `test_linalg_tensordot` now **run** (were name-absent)
  and fail **only** on R2/G-009 — the mixed-dtype promotion gap shared with
  `matmul`/`vecdot`, not a tensordot defect.
- Measured: **`1287 / 62 / 33` → `1294 / 55 / 33`**, 0 regressions.

## Progress (2026-10-10): batched linalg (R1)

Branch `261010/linalg-nbatch` (off rstsr `main` `50e7b6e`; PR RESTGroup/rstsr#136).
The n-dimensional generalization of the faer linalg column — the "dominant
lever" from the init finding (stacked inputs are the array-API contract).

- All 11 single-operand faer entries (`cholesky, det, eigh` standard **and**
  generalized, `eigvalsh, inv, pinv, slogdet, svd, svdvals`) plus the two solves
  (`solve_general, solve_triangular`) now accept `(..., M, N)` / `(..., M)` /
  `(..., M, K)`: each existing 2-D body became a `*_ix2` kernel and a `*_nd`
  wrapper walks the batch. The walk peels the **leading** axis under row-major
  and the **trailing** axis under column-major (the device order rule), so the
  2-D kernels keep their matrix orientation untouched. **Do not** reach for
  `reverse_axes` (the old slogdet trick) — it transposes the matrix and is only
  safe for transpose-invariants such as `det`.
- Shared machinery in `rstsr-linalg-traits/src/linalg_util.rs`: a batch walk
  (`map_batch_matrices`, plus a square-checking `map_batch_square_matrices`) and
  a broadcast-aware `map_batch_solve_into_output` / `map_batch_solve_inplace`.
  The helpers are `pub` so the BLAS column can reuse them later.
- `Out` types moved `D` / `T` → `IxD` (breaking; mirrors the `slogdet`
  precedent): a 2-D `det` now returns a 0-d tensor. The shim returns the stacked
  `det` / `slogdet`.
- Mutability: the in-place `solve` variants (`TensorMut` / owned `b`) solve
  straight into `b`'s buffer (no copy, output buffer is `b`); the allocating
  path copies the broadcast `b` into the output once and walks the same in-place
  kernel. Column-major stacked `solve` errors (the broadcast would need
  left-alignment); 2-D col-major works.
- Two `/code-review high` rounds. The second caught a **real regression** the
  retrofit had introduced: every square entry had been routed through a
  non-square-checking shape split, so non-square `det`/`cholesky`/`inv`/`eigh`/
  `eigvalsh`/`slogdet` *panicked* inside faer — now `InvalidLayout`. (A flagged
  finding that the nd solve kernels "dropped" `clone_to_mut` was a **false
  positive**: the callee's `clone_to_mut` writes back as a side effect.)
- **Measured: `1294 / 55 / 33` → `1296 / 53 / 33`**, 0 regressions. Flipped
  nodes: `test_linalg::{cholesky, det, eigh, eigvalsh, inv, svdvals}` + the 3
  `slogdet` nodes. The residual `linalg` value failures are *not* batching:
  `solve` (and `matmul`/`vecdot`/`tensordot`) = R2/G-009 mixed dtype;
  `pinv`/`svd` = faer SVD `NoConvergence` on degenerate huge-magnitude matrices
  (hits 2-D too).

## 0. The red map this checklist must clear

**Init-time failures (62) = `linalg` 48 + `fft` 14** — the groups below. R3 has
since removed the two `tensordot` has_names/signature rows and turned
`test_tensordot` / `test_linalg_tensordot` into G-009 failures (see
**Progress**); re-derive the group columns on the next full run (the suite is
hypothesis-based; expect ±2 wobble).

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
| `cholesky` | `CholeskyAPI` (faer) | ✓ batched (R1) | ✓ |
| `det` | `DetAPI` (faer) | ✓ batched (R1) | ✓ |
| `eigh` | `EighAPI` (faer) | ✓ batched (R1) | ✓ |
| `eigvalsh` | `EigvalshAPI` (faer) | ✓ batched (R1) | ✓ |
| `inv` | `InvAPI` (faer) | ✓ batched (R1) | ✓ |
| `pinv` | `PinvAPI` (faer) | ✓ batched (R1) | ✓ |
| `solve` | `SolveGeneralAPI` (faer) | ✓ batched (R1) | ✓ |
| `svd` | `SVDAPI` (faer) | ✓ batched (R1) | ✓ |
| `svdvals` | `SVDvalsAPI` (faer) | ✓ batched (R1) | ✓ |
| `tensordot` | `rt::tensordot` (tensor tier) | ✓ same-dtype (R3) | ✓ |
| `outer` | — (reshape + broadcast mul) | ✗ new | ✗ |
| `cross` | — (3-vector cross) | ✗ new | ✗ |
| `trace` | — (diagonal + sum) | ✗ new (trivial) | ✗ |
| `matrix_power` | — (repeated `matmul`) | ✗ new | ✗ |
| `matrix_rank` | — (via `svdvals`) | ✗ new | ✗ |
| `vector_norm` | `l2_norm` family (ord=2 only) | ◐ general `ord` missing | ✗ |
| `matrix_norm` | `l2_norm` family (ord=2 only) | ◐ general `ord` missing | ✗ |
| `qr` | — | ✗ **G-004** | ✗ |
| `slogdet` | `SLogDetAPI` (faer + blas) | ✓ batched (R1); faer landed in #133 | ✓ |
| `eig` | — (general, non-symmetric) | ✗ new (tested; not in 2025.12 `__all__`) | ✗ |
| `eigvals` | — (general, non-symmetric) | ✗ new (ibid.) | ✗ |

Also relevant: `solve_symmetric` exists on blas but not on faer (**G-005**) —
array-API `solve` maps to `solve_general`, so this is tangential.

## 2. Rust-side work (register + request; never fix agent-side)

Ordered by failures-per-item. **R1 is now the dominant lever** — it alone
gates 13 of the 20 graded members the shim already exposes.

- [x] **R1. Batched (stacked) linalg** — every faer factorization today asserts
      `ndim == 2`. The array-API contract is "a matrix *or a stack of matrices*"
      (`(..., M, M)`), and the value tests exercise stacks immediately. Give
      the faer linalg entries (and the tensor-tier `solve`/`pinv` route) a
      leading-axis loop — either iterate the batch in the device op or map over
      `iter_axes` and stack. Cheapest, largest payoff: unblocks `cholesky, det,
      eigh, eigvalsh, inv, pinv, solve, svd, svdvals` at once. **Landed** — PR
      RESTGroup/rstsr#136 (branch `261010/linalg-nbatch`), see **Progress**.
- [ ] **R2. mixed-dtype `matmul`/`vecdot`** (G-009) — promote the operands to
      their common dtype (the `DTypePromoteAPI` lattice already exists) before
      the kernel, so `u8 @ u16` works. Gates `test_matmul` / `test_vecdot` /
      `test_linalg_vecdot` beyond the batching issue.
- [x] **R3. `tensordot`** (tensor tier, `tensor/linalg/`) — **landed** (PR
      RESTGroup/rstsr#135): `rt::tensordot` (`axes` int or pair-of-lists) with
      serial + rayon device kernels and a view-only GEMM fast path; the shim
      exposes it top-level and on `_LinalgNamespace`. Its 4 `has_names` +
      signature nodes pass; the 2 `test_linalg` value nodes now fail only on
      R2/G-009.
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
      available names (13 at init, 14 with R3's `tensordot`); missing members
      stay absent (rendered as gaps, never stubbed). `api.py` §linalg.
- [x] **S2. top-level members** — `matmul`, `matrix_transpose`, `vecdot`, and
      `tensordot` (R3, landed — also added to `_LinalgNamespace`) on `xp`.
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
2. Then R2, R4–R8 as members land; each new member is a shim one-liner plus a
   `_LinalgNamespace` entry (add the name only when its rust entry is real, so a
   new gap never becomes a fresh failure). R3 (`tensordot`) is already done.
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
- **Recorded (tensordot / R3):** `1287 / 62 / 33` → **`1294 / 55 / 33`**. The 4
  `tensordot` has_names/signature nodes flip to pass; the 2 `test_linalg` value
  nodes now run and fail only on G-009; 0 regressions.
- **Recorded (batched linalg / R1):** `1294 / 55 / 33` → **`1296 / 53 / 33`**,
  0 regressions. Flipped to pass: `test_linalg::{cholesky, det, eigh, eigvalsh,
  inv, svdvals}` + the 3 `slogdet` nodes (has_names, signature, value). `pinv`
  and `svd` did **not** flip — their remaining failure is faer SVD
  `NoConvergence` on degenerate huge-magnitude matrices, a 2-D robustness limit,
  not batching; `solve` stays red on R2/G-009. The `matrix_norm`-family and the
  absent-member nodes (`cross, eig, eigvals, matrix_norm, matrix_power,
  matrix_rank, outer, qr, trace, vector_norm`) remain. (The suite is
  hypothesis-based; expect ±2 wobble.)

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

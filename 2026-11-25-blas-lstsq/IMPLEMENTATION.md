# lstsq for the BLAS devices — implementation report

Branch `261125/blas-lstsq` off `ed1f62d` in `/home/a/rstsr_pack/rstsr`,
committed as `182e38d`. Implements DECISIONS D1–D16 / PLAN B0–B6 of this
directory. The 2026-11-26 review follow-ups are in the last two sections.

## What landed

### `rstsr-blas-traits` (driver layer)

- `src/lapack_lstsq/{mod.rs, gelsd.rs}`: `GELSDDriverAPI<T>` and the `GELSD_`
  builder. `internal_run` returns `(x, s, rank, Option<residuals>)`; residuals
  are `None` when `M <= N` or `rank < min(M, N)` (the entry layer renders the
  SciPy NaN), and are read from the destroyed `b` workspace rows `n..m` with raw
  strides (no extra GEMM).
- Thunks `driver_impl/lapack/lstsq/{mod.rs, gelsd.rs}` (shared into all five
  device crates by the existing symlink sets): real/complex `duplicate_item`
  blocks, `LWORK`/`LIWORK`/`LRWORK` queries, `-1010/-1011` mapping, NaN/Inf
  info codes `-4/-5/-7/-10`.
- Thunks `driver_impl/lapacke/lstsq/{mod.rs, gelsd.rs}` (openblas/mkl only):
  one-liner forwarding to `LAPACKE_{s,d,c,z}gelsd`.
- `src/lib.rs`, `src/prelude.rs`, `src/trait_def.rs`: module export and
  `LapackDriverAPI<T>: GELSDDriverAPI<T>`.

### `rstsr-linalg-traits` (entry layer)

- `src/stack.rs`: the `b`-shape half of `solve_plan` is factored into a shared
  `split_rhs` helper, and `LstsqPlan` + `lstsq_plan` are added (rectangular
  `(M, N)`, vector or matrix `b`, broadcast batch; both matrix-axis orders).
  `solve_plan` keeps its square-only check.
- `src/blas_impl/lstsq.rs` (compiled per device crate through
  `src/linalg_auto_impl/lstsq.rs` symlinks): `blas_impl_lstsq_f` walks the
  stack, consumes `a` in place when it is owned/mutable, needs no broadcast
  and already has the call order (the factorization destroys it; an off-order
  `a` is factored from a work buffer and left untouched), broadcasts `a`/`b`
  to the plan, and batches `x`, `s`, per-slice `rank` and `residuals`; 32
  `LstsqAPI` impls (16 input-kind pairs × without/with `rcond`) all return
  dynamic-dim `LstsqResult`.
- `src/traits_def.rs` / `src/prelude.rs`: `LstsqAPI`, `lstsq`, `lstsq_f`,
  `LstsqResult` + `From` both ways.

### Tests and validators

- Driver: `blas_tests_impl/driver_impl/f64/gelsd.rs` —
  `test_dgelsd` (4 fingerprints: tall 1024×512 K=1/K=3, wide 512×1024 K=1,
  rank-deficient with `rank = 511`), `test_dgelsd_layout` (16 combinations:
  call order × a/b order × padded `lda/ldb` against the compact column-major
  reference, plus fully non-contiguous inputs, a padded in-place `TensorMut`
  and the destroyed-`a` parity of the row-major bridge), `test_dgelsd_relations`
  (tall/square/wide: `A x ≈ b`, `Aᵀ(b − Ax) ≈ 0`, residuals equal Σ(`b − Ax`)²,
  `s` agrees with `DGESDD`), `test_dgelsd_rank` (`rcond` sweep on graded
  singular values incl. the LAPACK `<= 0`/`>= 1` → eps clamp), `test_dgelsd_nan`
  (NaN `a` must error, NaN `rcond` rejected before the call),
  `test_dgelsd_zero_size` (`m == 0`, `n == 0` synthesized residuals, `(0, 0)`,
  `k == 0` pad-and-trim).
- Function: `linalg_func/func_f64/lstsq.rs` — `test_lstsq` (4 fingerprint
  cases, explicit `rcond`, rank-deficient NaN residuals for vector and matrix
  `b`), `test_lstsq_nd` (batch consistency against per-slice 2-D calls,
  batched vector `b`, owned-input consumption), `test_lstsq_order`
  (row-major vs column-major device give identical values, in-place
  column-major `a`), `test_lstsq_inplace` (mutable, owned, off-order, padded
  and broadcast routes in both device orders: results equal the read-only
  call, `b`/off-order/broadcast `a` stay untouched, destroyed `a` of the
  row-major bridge equals a direct column-major call), `test_lstsq_overloads`
  (all 16 input-kind pairs, 2-arg/3-arg/`_f`), `test_lstsq_errors` (1-D `a`,
  row mismatch, NaN `rcond`, short `b`, `b` rank and batch rules),
  `test_lstsq_rcond`, `test_lstsq_batch_ranks`, `test_lstsq_zero_size`
  (`m == 0`, `n == 0`, `(0, 0)`, `K == 0`, empty batch, batched empty-row and
  empty-column systems, both orders); `func_c64/lstsq.rs` — 3 fingerprint
  cases (`c64` is `Complex<f64>` here).
- Zero-size lstsq cases live in the shared `linalg_func` body since `04f7a12`;
  the OpenBLAS-only `tests/zero_size/lapack.rs` harness is back to its
  `ed1f62d` state.
- Python validators: GELSD sections in `driver_validation_f64.py`,
  `func_validation_f64.py`, `func_validation_c64.py` (same fingerprints).

### Docs

`blas_impl_lstsq_f` docstring; `docs/cheatsheet_linalg.md` prose bullet (no
table row — the cheatsheet rows are faer-backed) and `LstsqResult` in the
multi-output list; `docs/array_api_standard_extension.md` Notes bullet.

## Deviations from the plan

- No `ref_impl_lstsq_f` wrapper: the driver is already a single `GELSD` call,
  so the stack walk calls the builder directly (like `svd`/`det`, which also do
  not wrap `with_blas_num_threads`). Superseded by `04f7a12`: GELSD now runs
  under the device BLAS thread policy through `ref_impl_lstsq_f`.
- `lapacke` was compile-verified only on 2026-11-25 (the system OpenBLAS is
  built without LAPACKE, MKL was not on the linker path). Verified since
  2026-11-26 — see the follow-up section.
- Residuals for `rank < min(M, N)`: the *driver* returns `None`, the *entry*
  fills NaN (SciPy semantics). The driver validator prints NaN for that case
  even though LAPACK leaves numbers in `B(n+1:m)`.
- The documented `info = -4` ("NaN/Inf in `A`") path is not reachable on this
  OpenBLAS build: `DLANGE('Max')` swallows NaN on x86 (MAXSS), so a NaN `a`
  fails later with a convergence error (`info = 3`). The mapping is kept
  defensively; `test_dgelsd_nan` asserts "an error", and the NaN-`rcond`
  rejection (which is ours) asserts the exact message. Upstream fact found on
  2026-11-26: the check is in netlib master only (`3d90a787f`, 2026-11-06,
  after 3.12.1, in no release), so released and vendor libraries never report
  it.

## Driver bug found by the tests

For `M < N`, `DGELSD` first zeroes `B(M+1:N, 1:nrhs)` (`DLASET('F', ...)`) and
writes the length-`N` solution into `B(1:N, :)`. A RowMajor bridge that copies
back only `M` rows returns a truncated (wrong) solution — same `s`, different
`x`. The thunk now copies `max(M, N)` rows for both the in and the back
transpose. This was the gap notice for the `lapacke-translation` skill; the
skill was updated on 2026-11-26 (gelsd row in the workspace table, `b` bridge
sizing, zero-size synthesis, NaN info mapping, failed-call write-back).

## Review follow-up (2026-11-26)

Two-axis `/code-review` of `182e38d` (standards + spec). Fixes on branch
`261126/lstsq-review-fixes` (off `182e38d`, not pushed):

- `04f7a12` — behaviour: `(0, 0)` / `(m, 0)` corners return instead of
  erroring (residuals per D9), a `b` too short for `a`'s rank is an
  `InvalidLayout` error instead of a `usize` underflow (shared `split_rhs`,
  fixes `solve_general` too), all 16 input-kind pairs, GELSD under the device
  BLAS thread policy. Messages: no duplicated "Lapack" prefix (all builders),
  explicit non-convergence message. Docs: real residual/rank policy, `rcond`
  clamp, NaN handling; svd/svdvals docstrings lead with the read-only
  guarantee. Tests: zero-size cases moved to the shared body, `test_lstsq_rcond`,
  `test_lstsq_batch_ranks`, 16-pair overload coverage, short-`b` errors.
- `dc2274f` — owner reverted the narrowed raw-slice spans of `04f7a12`
  (`(rows − 1)·ld + cols`): LAPACKE touches the same elements, and the plain
  `rows * ld` spans keep the thunk a direct translation without span helpers.
  Separate revert commit, no amend.
- `2dc1be0` — in-place consumption (D8) and the LAPACKE write-back (D11):
  owned/mutable `a` is consumed without a copy when its batch needs no
  broadcast and its layout is the call order; the row-major bridge transposes
  the destroyed `a` back like `LAPACKE_dgelsd_work`. The owner accepted one
  divergence: the write-back is skipped when `info != 0` (LAPACKE writes back
  unconditionally) — recorded in the `lapacke-translation` skill (§3, §6).

Owner decisions along the way: LAPACKE fidelity is the criterion ("direct
translation from lapacke"); simple code over span helpers; separate commits.

Cost of the write-back (`2dc1be0`), a rough probe rather than a benchmark:
release profile, system OpenBLAS 0.3.32, `lapack` config, default BLAS
threading, view input through the row-major bridge, best of 5 × 50 calls,
9950X3D, a few % run-to-run noise:

| shape | with write-back | without | overhead |
|--|--|--|--|
| 64 × 32 | 37.8 µs | 36.8 µs | +3% |
| 64 × 4096 | 8.07 ms | 7.75 ms | +4% |
| 1024 × 512 | 28.0 ms | 26.7 ms | +5% |
| 4096 × 64 | 5.08 ms | 4.17 ms | +22% |

Column-major calls and the `lapacke` config are unchanged. For a mutable
input the copy that is saved roughly pays for the write-back, so immutable
views carry the overhead.

## Verification (after `2dc1be0`, 2026-11-26)

`RSTSR_DEV=1`, `--test-threads=1`; `RUST_BACKTRACE=0`.

| suite | OpenBLAS 0.3.32 `lapack` | OpenBLAS 0.3.34 `lapacke` | MKL 2026.1 `lapack` | MKL 2026.1 `lapacke` |
|--|--|--|--|--|
| `tests_linalg_func` | 104/104 | 104/104 | lstsq + solve 15/15 | lstsq + solve 15/15 |
| `tests_driver_impl` | 89/89 | 89/89 | GELSD 6/6 | GELSD 5/6 |
| `tests_zero_size` | 26/26 | 26/26 | — | — |

- MKL `lapacke` GELSD: `test_dgelsd_relations` fails in its `DGESDD
  (compute_uv = false)` oracle ("illegal argument in argument 11"); the same
  test passes under MKL `lapack`. Pre-existing, not lstsq code.
- `cargo test -p rstsr-blas-traits` / `-p rstsr-linalg-traits` (3 / 4 unit
  tests), `cargo test -p rstsr --features linalg` (cheat-sheet suites
  1 + 12 + 13) pass.
- `cargo clippy --all-targets -- -D warnings` clean for `rstsr-blas-traits`,
  `rstsr-linalg-traits`, `rstsr-openblas`, `rstsr-mkl`; `cargo fmt --check`
  clean; `cargo doc -p rstsr-linalg-traits` and `-p rstsr --features linalg`
  add no warnings (one pre-existing redundant link at `traits_def.rs` pinv).
- `cargo check --all-targets` clean for BLIS, AOCL, KML and OpenBLAS `ilp64`.
- Not re-run on 2026-11-26: the Python validators and the MKL suites outside
  lstsq/solve/GELSD.

The 2026-11-25 baseline numbers are in the history of this file (`f4539a4`).

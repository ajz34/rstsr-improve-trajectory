# Grill R1 — QR decomposition on the BLAS devices

rstsr `c9fcc20` (`main`), rstsr-ffi `83efe50`; no rstsr file edited.
Recommended answers are marked ➡️; "yes" accepts the recommendation.

---

## Factual baseline (agent-authored, verified 2026-11-24)

### 1. What exists today

- **Entry point** `QrAPI` (`rstsr-linalg-traits/src/traits_def.rs`, `#region qr`):
  `QrResult<Q, R>` (↔ `(Q, R)`), and
  `#[non_exhaustive] enum QrMode { #[default] Reduced, Complete }` with
  `TryFrom<&str>` (`"reduced"`/`"complete"`). Only the **faer** device
  implements it (`faer_impl/qr.rs`): `Tr` → reduced, `(Tr, QrMode)`,
  `(Tr, &str)`; n-dim batched; complete-mode zero-size `Q` is the identity.
- **BLAS devices have no `qr`** at all. Their linalg entries follow one
  three-layer pattern (SVD is the closest precedent — rectangular, two
  output matrices, stacked in #168):
  1. driver tier: `rstsr-blas-traits/src/lapack_svd/{gesvd,gesdd}.rs`
     (`XxxDriverAPI<T>` + `Builder` struct) and the shared vendor tree
     `rstsr-blas-traits/driver_impl/{lapack,lapacke}/svd/` (symlinked by all
     five device crates; `lapacke` is openblas+mkl only, mutually exclusive
     feature);
  2. composition: `ref_impl_svd_simple_f` (`rstsr-linalg-traits/src/ref_impl_blas.rs`);
  3. entry: `rstsr-linalg-traits/src/blas_impl/svd.rs` (`blas_impl_svd_f`
     stack walk + `QrAPI`-style impls), reached from every device crate by a
     **symlink** `crates-device/*/src/linalg_auto_impl/svd.rs`.
- `LapackDriverAPI<T>` (`rstsr-blas-traits/src/trait_def.rs:70`) lists every
  driver trait explicitly; a new one is a breaking change for external
  implementors (same as GEMV in #167).

### 2. NumPy / SciPy signatures

- `numpy.linalg.qr(a, mode='reduced')`: modes `'reduced'` → Q (…,M,K),
  R (…,K,N); `'complete'` → Q (…,M,M), R (…,M,N); **`'r'`** → R only
  (…,K,N); **`'raw'`** → `h` (…,N,M), `tau` (…,K). Requires `ndim ≥ 2`.
  LAPACK: `geqrf` + `orgqr`/`ungqr`. Results are unique only up to a unit
  factor per column (no positive-diagonal convention).
- `scipy.linalg.qr(a, overwrite_a, lwork, mode='full', pivoting=False)`:
  `'full'` = NumPy complete, `'economic'` = reduced, `'r'`, `'raw'`;
  `pivoting=True` → `geqp3`, returns `(Q, R, P)`.
- array-API `linalg.qr(x, mode='reduced'|'complete')` — exactly the two
  modes `QrMode` already carries.

### 3. FFI and vendor coverage (no FFI change needed)

`[sdcz]geqrf_`, `[sd]orgqr_`, `[cz]ungqr_` are bound in **all five** crates:
openblas/kml/blis via the shared `rstsr-lapack-ffi` sources, mkl and aocl in
their own trees; `LAPACKE_[sdcz]geqrf`, `LAPACKE_[sd]orgqr`,
`LAPACKE_[cz]ungqr` exist for the `lapacke` feature. `geqp3` is bound too.
The system OpenBLAS exports all of them (`nm` checked).

### 4. Policy that applies

- Skills `lapacke-translation` (input/output/zero-size/error policy,
  vendor file shape, test matrix) and `test-device` (symlinked conformance
  bodies `driver_impl/f64`, `linalg_func/func_{f64,c64}`).
- **Gap**: `lapacke-translation` §3a workspace table has no row for
  `geqrf`/`orgqr`/`ungqr` (all are plain `lwork` queries; no `iwork`/`rwork`).
  Per its gap-notice rule this must be recorded, not improvised.
- Workspace-policy row needed: `geqrf`/`orgqr`/`ungqr` → query (rare for
  `rwork`-free routines).

---

## Round 1 — the frontier

❓ **Q1 - Mode scope of this campaign**: `rt::linalg::qr` on the BLAS devices
implements exactly the two modes `QrMode::Reduced` / `QrMode::Complete`
(NumPy `'reduced'`/`'complete'`, SciPy `'economic'`/`'full'`, the whole
array-API surface), i.e. the same `QrAPI` impl set the faer device has:
`Tr`, `(Tr, QrMode)`, `(Tr, &str)`, always returning `QrResult<Q, R>`.

➡️ Yes. These two share one return type, so `QrMode` stays as it is and the
BLAS devices become a drop-in for the existing entry.

---

❓ **Q2 - NumPy `'r'` and `'raw'` modes**: defer both to a follow-up
campaign (not in this PR).

➡️ Yes. Both change the return arity: `'r'` returns one tensor, `'raw'`
returns `(h, tau)` of different shapes/dtypes-of-meaning. Under
`LinalgAPI<Inp>::Out` the output type is fixed **per argument type**, so they
cannot ride on `(Tr, QrMode)` without either a sum-type result or new
argument marker types — a signature decision worth its own round. The raw
data is still reachable: the `GEQRF` builder from Q4 is public.

---

❓ **Q3 - Column-pivoting (SciPy `pivoting=True`, `geqp3`)**: out of scope
here, same follow-up as Q2.

➡️ Yes. It adds a third output `P` and a separate driver; no consumer asked.

---

❓ **Q4 - Driver tier shape**: two small builders in a new family
`rstsr-blas-traits/src/lapack_qr/`, composed above the driver tier — no
monolithic "QR driver" trait:

- `GEQRF_` (`GEQRFDriverAPI<T>`): `a: TensorReference<Ix2>` →
  `(TensorMutable2 (h: R in the upper triangle, reflectors below), tau: Ix1)`.
- `ORGQR_` (`ORGQRDriverAPI<T>`): `a: h`, `tau`, optional `k` (columns of Q)
  → `TensorMutable2` holding Q (reduced = first `k = min(m,n)` columns,
  complete = all `m`).
- Composition `ref_impl_qr_simple_f` in `ref_impl_blas.rs` (like
  `ref_impl_inv_f` = GETRF + GETRI): allocate `q` in the call order, copy the
  first `k` columns of `h` into it (complete mode leaves columns `k..m`
  zero-padded? no — see Q5), call `ORGQR`, then extract `triu(h)`.
- Vendor thunks in the shared tree
  `driver_impl/{lapack,lapacke}/qr/{geqrf,orgqr}.rs`; both traits registered
  in `LapackDriverAPI<T>` and the preludes.

➡️ Yes. Mirrors the LAPACK split and keeps each thunk a one-routine
translation (the shape the skill and ADR-0009 prescribe).

---

❓ **Q5 - `orgqr` vs `ungqr` (real vs complex)**: one trait
`ORGQRDriverAPI<T>` whose vendor impl maps real → `orgqr`, complex → `ungqr`
(the same `duplicate_item` real/complex split every other routine uses; no
separate `UNGQRDriverAPI`).

➡️ Yes. Callers never see the difference; two traits would double the
registration for nothing.

---

❓ **Q6 - Stacked (n-dim) input**: the BLAS `qr` accepts `(..., M, N)`
stacks and walks them with `StackLayout` exactly like `blas_impl_svd_f`
(#168): matrix axes last under `RowMajor`, first under `ColMajor`; outputs
keep the batch shape (`D` for Q and R).

➡️ Yes. Matches faer and the #168 contract.

---

❓ **Q7 - Input ownership**: `qr` never writes back to its input. An owned or
`&mut` input is consumed as scratch (destroyed) like SVD; an immutable input
is copied. No `TensorMut` overload that returns the input (meaningless for
QR). `overwritable_convert_with_order` once for the whole stack.

➡️ Yes. SVD precedent; avoids one full copy for owned inputs.

---

❓ **Q8 - Complete-mode Q with `m > n`**: `ORGQR` is called with `k = m`
columns on an `(m, n)` `h`; LAPACK requires `a` to be `(m, k)` — columns
`n..m` of the work buffer are zero-filled before the call (LAPACK documents
them as "not referenced" when `k > n`, but the row-major bridge copies the
whole buffer, so they must be initialized). The call-order buffer is
`zeros`-allocated rather than `empty` + explicit fill.

➡️ Yes. Correct for every vendor, and the extra zeroing is O(m·(m−n)) only
in complete mode with tall inputs.

---

❓ **Q9 - Zero-size synthesis**: never call the drivers when `m == 0` or
`n == 0` (or the batch is empty). Shapes follow NumPy/faer:
reduced → `Q (m, k)`, `R (k, n)`; complete → `Q = I_m` when `n == 0, m > 0`,
`Q (0, 0)` when `m == 0`; `R` zeros. Done in the builder in the call order.

➡️ Yes. ADR-0010 §3 and the faer fix in #147.

---

❓ **Q10 - Devices**: implement through the shared trees + the per-crate
symlink `linalg_auto_impl/qr.rs`, so all five devices get it; no
device-specific code. Locally only `rstsr-openblas` (`openmp`, default
`lapack` **and** `--features lapacke`) is run.

➡️ Yes. This is the prompt's contract; the vendor trees are already shared.

---

❓ **Q11 - Tests** (per `lapacke-translation` §4 and `test-device`):

- driver conformance, `blas_tests_impl/driver_impl/f64/`:
  `geqrf.rs` (layout matrix: f-contig, f-prefer `ld > dim`, c-contig,
  c-prefer, fully non-contiguous; `m<n`, `m>n`, `m==n`; in-place buffer
  identity; zero-size) and `orgqr.rs` (reduced/complete `k`, same layouts).
- `driver_impl/f64/c64/`? — the skill says c64 for complex-specific paths;
  here only the `ungqr` mapping is complex-specific, so `driver_impl`'s
  existing `f64` body plus a small c64 case in the linalg body is enough.
- linalg conformance `blas_tests_impl/linalg_func/func_f64/qr.rs` and
  `func_c64/qr.rs`: tall/wide/square × reduced/complete, 3-D stack,
  non-contiguous and Fortran-ordered inputs, zero-size, error case
  (`ndim < 2`); assertions are `A ≈ Q R`, orthonormality `QᴴQ = I`, `tril(R,
  -1) == 0`, shapes; fingerprints only after phase canonicalization
  (real-positive `diag(R)`, like the faer suite).
- symlinks added in all five crates' `tests/` trees.

➡️ Yes.

---

❓ **Q12 - Skill gap**: add the missing row(s) to `lapacke-translation` §3a
(`geqrf`, `orgqr`, `ungqr`: query; no `iwork`/`rwork`) in `rstsr-agents`
(trivial edit, direct on that repo's branch policy — it has no branch rule),
rather than only a gap notice in this file.

➡️ Yes. Otherwise the next agent hits the same gap.

---

❓ **Q13 - Docs**: no new documentation page. `cheatsheet_linalg.md` already
lists `rt::linalg::qr(&a)` and the complete form (faer-backed page, BLAS
devices excluded by its own preamble); `array_api_standard_extension.md`
already marks `qr` `Y`. Docstrings on the new builders follow
`api-doc-conventions`; changelog by release-plz. The rustdoc of `QrMode`
gets one sentence that BLAS devices implement the same modes.

➡️ Yes.

---

❓ **Q14 - Delivery**: one PR on rstsr branch `261124/blas-qr`
(`rstsr-blas-traits` + `rstsr-linalg-traits` + the five device crates), no
auto-commit; skill edit in `rstsr-agents` separately.

➡️ Yes.

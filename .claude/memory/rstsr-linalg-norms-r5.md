---
name: rstsr-linalg-norms-r5
description: "matrix_norm/vector_norm (array-API linalg R5): faer-only MatrixNormAPI/VectorNormAPI, one shared NormOrd (None resolves per-op), composition over core reductions + batched svdvals, empty input -> 0 under every order, conformance 1312/37/33 -> 1318/31/33"
metadata:
  type: project
---

2026-10-11, branch `261011/worktree` (off main `bbc6a19`, i.e. post-`#141`
matrix_power; PR RESTGroup/rstsr#142), commits `0359b9a` + `2c2e8f9`. **R5** of
the array-API linalg convergence ([[rstsr-ext-linalg-r2]] is R2,
[[rstsr-linalg-outer-r4]] is R4). Suite **`1312 / 37 / 33` → `1318 / 31 / 33`**,
0 regressions (flipped: the two name + two signature `matrix_norm`/`vector_norm`
nodes and `test_{matrix_norm,vector_norm}`). The whole linalg extension is now
down to 17 graded failures — see `2026-10-10-arrayapi-linalg/CHECKLIST.md` §0.

- **Faer device only, no BLAS column** (as the plan predicted: the `nuc` order
  needs the singular values). New traits `MatrixNormAPI` / `VectorNormAPI<Inp>`
  in `rstsr-linalg-traits/src/traits_def.rs` (the three `duplicate_item` tables +
  `prelude.rs`), kernel in the new `faer_impl/norm.rs`. Receivers `(Tr, A)` where
  `A: TryInto<MatrixNormArgs|VectorNormArgs>`, plus a bare-`Tr` no-arg impl.
- **One shared `NormOrd { None, Fro, Nuc, P(f64), Inf, NegInf }`** — `None` is
  `#[default]` and resolves *per operation* (`'fro'` for a matrix, `2` for a
  vector), so the same value names different orders. Each op rejects the other's
  orders with `InvalidValue` (a matrix takes only `P(±1)`/`P(±2)` numerically; a
  vector rejects `Fro`/`Nuc`). Overloads: `TryFrom` impls on the args structs for
  `()`, `Option<()>`, `bool`, `NormOrd`, `&str`, `isize`, `f64`, the 2/3-tuples
  — concrete impls, **no blanket `TryFrom<O>`** (coherence, same reason as
  `AxesIndex`). `FromStr` is also implemented (clippy
  `try_from_instead_of_from_str`).
- **Composition, no new kernels**: `matrix_norm` = core reductions — `'fro'` →
  the `l2_norm` reduction (it already takes the magnitude, so **no `abs` copy**);
  `1/-1` → abs → sum → max/min over rows; `inf/-inf` → abs → sum over columns →
  max/min; `2/-2/'nuc'` → the **batched `svdvals` entry** (R1) → max/min/sum.
  `vector_norm` reduces `|x|` (l2 for `p = 2`, plain sum for `p = 1`, nonzero
  count for `p = 0`, general `(Σ|x|^p)^(1/p)` otherwise, via `pow_f`). The matrix
  axes follow the **device order** (`batch_and_matrix_shape`, like
  `matrix_power`); output is dynamic `IxD` (rank depends on `keepdims`).
- **The `TR` generic is mandatory** (the trap that cost the most time): faer's
  `ComplexField::Real` and rstsr's `ExtNum::AbsOut` must be tied to one type —
  `T: ComplexField<Real = TR> + ExtNum<AbsOut = TR> + ComplexFloat<Real = TR>`,
  `TR: NormReal + DTypePromoteAPI<TR, Res = TR>` — or the faer SVD result and the
  core reduction output disagree and the bounds do not resolve. `NormReal` is a
  small public trait alias for the real-scalar bundle.
- **Empty input → `0` under EVERY order** — deliberate divergence from NumPy
  (which is inconsistent: it raises for the matrix `-2` and `-inf` and returns
  `inf` for the vector `-1`). The cause is not a norm choice: the faer `max`/`min`
  reductions reject *any* zero-size input (`la.size() == 0`), so both entries
  early-return `zeros_f` whenever the input has any zero dim — which also covers
  the empty **stack** dims the suite's `finite_matrices` draws. Stated in the
  module doc and locked by the empty-input test (extended to the min-based
  orders).
- **Shim**: `linalg_matrix_norm` / `linalg_vector_norm` `#[pyfunction]`s + the
  `_LinalgNamespace` entries + `api.py` wrappers. `ord` is marshalled by a new
  `py_to_ord` (`None`/`'fro'`/`'nuc'`/int/float → `NormOrd`); `axis` **reuses the
  Python layer's `_norm_axes`** (`None` / int / tuple → `Option<Vec<isize>>`), so
  the Rust entry takes `Option<Vec<isize>>` — no shim-local `PyAny` axis parse
  ([[rstsr-faer-py-wrapper-only]]: the shim stays marshal-only).
- **Review round (`/code-review high`, fixes `2c2e8f9`)**: validate the
  vector-norm axes **before** the empty-input short-circuit (otherwise a bad axis
  was swallowed on an empty input); drop the redundant `abs` on the `'fro'` and
  vector `p = 2` paths; add `FromStr`; route the shim axis through `_norm_axes`;
  extend the empty-input test to the min-based orders. Conformance re-run
  unchanged.
- **Gates**: faer tests **50/50** (NumPy's own `test_matrix_2x2` goldens:
  `[[1,3],[5,7]]` → `fro` √84, `nuc` 10, `1`/`inf` 10/12, `±2`
  9.123105625617661 / 0.8768943743823404; plus n-dim / col-major / empty);
  fmt/clippy clean.

See `2026-10-10-arrayapi-linalg/CHECKLIST.md` (R5). Related:
[[rstsr-linalg-matrix-power]], [[rstsr-linalg-outer-r4]],
[[arrayapi-convergence-harness]].

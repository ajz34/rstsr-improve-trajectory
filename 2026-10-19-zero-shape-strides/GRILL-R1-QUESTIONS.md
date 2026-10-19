# Grill R1 — zero-size stride policy and BLAS/LAPACK empty operands

Round 1, 2026-10-19. Frontier questions only; recommendations are the agent's.
Answers are user-owned (`GRILL-R1-ANSWERS.md` is not created by the agent).

## Q1 — `ld_row` / `ld_col` failure mode

When the layout is preferred but the leading-dimension contract fails
(`stride[1] < max(1, shape[0])` in `ld_col`, symmetric in `ld_row`), what should
the accessor do?

- **(a)** return `None` (keep the `Option<usize>` signature); `None` means
  "no usable leading dimension" — covering both "not preferred" and "invalid /
  undefined for this layout".
- **(b)** panic with a descriptive `rstsr_raise!` (an invalid preferred layout
  is a programming error).
- **(c)** add a fallible `ld_col_f() -> Result<usize>` and keep `ld_col()` as a
  panicking wrapper.

Today every call site is `ld(order).unwrap()` in `rstsr-blas-traits`.

**Recommendation:** (a). Keep the signature and the existing "order is wrong
⇒ None" meaning, extend it to "no legal ld". The `None` case is unreachable for
non-empty layouts after `check_strides`, and empty operands are early-returned,
so no call site should ever see it. Change touched call sites to
`rstsr_unwrap()` per the crate convention so a violation keeps a backtrace.

## Q2 — exact `ld_*` contract

Confirm the rule (including the branch boundary):

```text
ld_col: !f_prefer()                      -> None
        shape[1] <= 1                    -> Some(max(1, shape[0]))
        stride[1] < max(1, shape[0])     -> None
        otherwise                        -> Some(stride[1] as usize)

ld_row: !c_prefer()                      -> None
        shape[0] <= 1                    -> Some(max(1, shape[1]))
        stride[0] < max(1, shape[1])     -> None
        otherwise                        -> Some(stride[0] as usize)
```

Notes: in the `else` branch we return the raw stride when it is legal (no
upward clamping); the `<= 1` branch ignores the stride entirely; `ld(order)`
keeps dispatching to these two. The comments currently call `[1, n]` a
"col-vector" and `[m, 1]` a "row-vector" — they are swapped and will be fixed.

**Recommendation:** accept exactly as written above.

## Q3 — P2 trigger and scope

Proposed: in both `Ix<N>` and `IxD`, `stride_f_contig` / `stride_c_contig`
return all zeros as soon as **any shape axis is 0** (`self.iter().any(|d| d ==
0)`), before any multiplication. `stride_contig` inherits the behavior.
A 0-dim shape (`[]`, size 1) is unaffected. Safety comments on
`new_c_contig` / `new_f_contig` are updated to cover the empty case, and the
`shape.rs` unit tests gain `[0,18]`, `[18,0]`, `[0,0]`, 3-D mixed, static and
dynamic.

**Recommendation:** accept. The alternative trigger `shape_size() == 0` is
equivalent for ndim ≥ 1 but relies on the same product; `any(d == 0)` is the
direct statement of the rule.

## Q4 — early-return placement and validation order

Proposed placement, per wrapper:

1. read shapes from the input tensors;
2. run the existing shape / device / flag validations unchanged (so malformed
   shapes still error, including on empty operands);
3. early-return the mathematically empty result;
4. only then `overwritable_convert*`, `ld_*`, and the driver call.

The guard belongs in the semantic wrapper tier (`src/blas3/*.rs`,
`src/lapack_*/*.rs`), not in `driver_impl` (the drivers only receive raw
pointers and dimensions).

Special case: `TRSM::run` should guard **before** `flip_trans`, because with an
empty B the current path may still transpose/copy a non-empty A; `internal_run`
keeps its own guard for direct callers. The other wrappers have no expensive
pre-work for the empty cases, so `internal_run` is enough.

**Recommendation:** accept this placement and keep the validations before the
return.

## Q5 — `gemm` with `k == 0`

`m == 0 || n == 0` is an empty-C no-op. `k == 0` is **not**: `C := beta*C`.
The operand layouts are empty (e.g. transposed A is `[0, m]`), so `ld_*` may
legitimately return `None` under Q1/Q2.

Options:

- **(a)** still call the driver, with shape-derived legal values
  (`lda = max(1, a.nrow())`, `ldb = 1`, `ldc` from the non-empty C). BLAS does
  not dereference A/B when `k == 0`; it only scales C by `beta`.
- **(b)** implement the scaling in the wrapper with element-wise ops — needs a
  new device-op bound on the GEMM wrapper.
- **(c)** leave `k == 0` out of this wave (would break transposed empty A under
  P2).

**Recommendation:** (a). It keeps the wrapper bound-free and matches the BLAS
contract; document the shape-derived exception in a comment.

## Q6 — `gesv` / `sysv` with `nrhs == 0`

NumPy's `solve(singular A, empty RHS)` raises `LinAlgError` (it still factors
A), while `solve(eye(n), empty RHS)` returns the empty result. For rstsr:

- **(a)** keep calling the LAPACK driver when `n > 0, nrhs == 0` (factor A,
  preserve the singular-matrix error); only `n == 0` short-circuits.
- **(b)** early-return for `nrhs == 0` too (cheaper, but hides singularity).

With the Q2 contract, an empty B `[n, 0]` has `ldb = max(1, n)`, so (a) already
works without extra ld handling.

**Recommendation:** (a), and add an OpenBLAS test for singular A + empty RHS.

## Q7 — PR granularity and merge order

- **(a)** one PR covering P2 + P3 + tests + docs.
- **(b)** two PRs: **(1) guards first** (`261019/blas-empty-operands`: early
  returns + OpenBLAS tests; fixes #161 on its own), then **(2) policy**
  (`261019/zero-size-stride-policy`: P2 + `ld_*` contract + layout unit tests +
  docstrings).
- **(c)** three PRs, splitting the rstsr-book ADR out (see Q8).

Why order matters: the `ld_*` change introduces `None` for some empty layouts.
If it lands before the guards, any unguarded empty BLAS path would panic at
`.unwrap()`. Guards first makes the sequence safe; P2 then changes only the
numbers.

**Recommendation:** (b), with (c)'s book PR as an additional separate PR if
Q8 says yes.

## Q8 — documentation target

P1 says "documentation only". Candidate locations:

- `rstsr-core/src/docs/order_semantics.md` — a new "Zero-size tensors"
  section (normalization rule, unspecified strides, both-contiguous, the
  leading-dimension contract).
- docstrings on `stride_*_contig`, `Layout::new_*_contig`, `ld_row` / `ld_col`.
- a new ADR in `rstsr-book/dev/adr/` (next number 0011), separate repo/branch.

**Recommendation:** all three; the ADR is a separate rstsr-book PR. If you
want the rstsr-side wave to stay code-only, defer the ADR.

## Q9 — test placement and coverage

- Layout/stride unit tests: `rstsr-common` (device-agnostic, not OpenBLAS-only).
- BLAS/LAPACK boundary tests: **new directory**
  `crates-device/rstsr-openblas/tests/zero_size/` + `tests_zero_size.rs`, so the
  tests never enter the shared `blas_tests_impl` symlinks (OpenBLAS-only, as
  requested). The existing `tests/issues/` directory remains for future
  issue-numbered regressions only.
- Proposed coverage:
  - issue #161: `trsm`/`solve_triangular` with `[18,0]` and `[0,18]`, owned /
    mut / view reprs, static and dynamic shapes, both orders — no `xerbla`;
  - `gemm`: `m==0`, `n==0`, `k==0` with `beta != 1`;
  - `gesv` / `sysv`: `n==0`, `nrhs==0`, and singular A + empty RHS;
  - `potrf` `0×0`, `getrf`/`getri` with a zero axis, `syev(d)`/`sygv(d)` `0×0`;
  - SVD empty already covered by `tests/issues/issue_gesvd_empty.rs`.

**Recommendation:** accept the new directory and this coverage set; keep the
xerbla handler as the future task you named.

## Q10 — `overwritable_convert_with_order` empty caveat

The repr-dependent failure came from this helper: it returns an empty
owned/mut tensor unchanged because `f_prefer()` is true, while a view is
normalized by `into_contig_f`. After the guards, BLAS never sees the
difference. Options:

- **(a)** no behavior change; add a docstring note that "preferred" carries no
  stride information for size 0;
- **(b)** normalize empty layouts here too (call `change_contig_f` when
  `size() == 0`), for consistency;
- **(c)** both.

**Recommendation:** (a) — the guarantees needed by BLAS are supplied by the
guards and the `ld_*` contract; normalizing here would add a special case for a
case that no longer matters.

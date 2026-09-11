# Proposal: NaN-aware arg-reduction in rstsr-core (`nanargmin`/`nanargmax`)

2026-09-11, follow-up to review of the T6 argmax/argmin patch (rstsr @ 386948b
+ patch + closure-API restoration). Facts verified against NumPy 2.5.2 source
(`/home/a/Git-Others/numpy`, tag v2.5.2) and empirically (NumPy 2.5.1).

> **OUTCOME 2026-09-11 (implemented, uncommitted at time of writing):**
> `nanargmin`/`nanargmax` were implemented and shipped (rt:: nanarg{min,max}
> families + `ArgCmp::{NanMin,NanMax}` kernels) — free on NaN-free input
> (the seed loop exits at element 0, then the plain 8-lane scan runs
> unchanged). Open question §5.1 was answered by measurement: adopting
> NumPy's first-NaN-wins rule for PLAIN argmin/argmax was **rejected** for
> now. Making NaN win requires either an unordered-aware update
> (`!(x <= best)`, an extra parity flag check per element) or a separate
> NaN pre-pass — every Rust-source variant tried de-vectorizes or doubles
> the auto-vectorized 8-lane kernel. Measured (n=1e7 f64, native Zen5,
> kernel level): fused +5…+18%, separate pre-pass +100…340%, per-block
> check +55…80%; small compute-bound sizes up to +100% for the fused form.
> NumPy itself pays the same (its argmax is still ~4× slower than rstsr's
> at 1e7). Plain argmin/argmax therefore keep the original rstsr NaN
> semantics, documented as a deliberate NumPy divergence (see `ArgCmp`
> docs and the tensor::reduction module docs). If NumPy parity is ever
> required, the fused-scalar variant is the least-bad implementation and
> its cost is the price of the semantics.

## 1. Current state

**rstsr-core (patched tree)** — `rt::argmin/argmax` (all/axes, raveled/unraveled,
serial+rayon+faer): ties → first occurrence (row-major); mid-stream NaNs are
**silently ignored**; NaN at the first scanned position or all-NaN input →
index 0, no error; empty input → `InvalidLayout` error. No `nan*`-prefixed op
exists anywhere in rstsr-core today.

**NumPy** — `argmin/argmax`: **first NaN wins, at any position** (kernel breaks
at the first NaN; `argmax([1, nan, 3]) == 1`). `nanargmin/nanargmax`: NaNs
replaced by ∓inf then plain arg-reduce; all-NaN slice → `ValueError("All-NaN
slice encountered")` (checked per reduced slice); empty → ValueError; slices
containing only NaN and ±inf return the first index (documented untrustworthy);
±inf are ordinary values; ties → first occurrence; `axis=None|int`, `keepdims`,
`out`; no arg-family semantic changes across NumPy 2.x.

**Divergence finding (needs a decision regardless of this proposal):** the T6
experiment docs claim rstsr's patched NaN behavior "coincide[s] with numpy's
documented argmax/argmin NaN behavior" (`2026-09-09-argmax-argmin/README.md:91`,
`examples/correctness.rs:13`). This is only true for NaN-at-front and all-NaN
inputs; for mid-stream NaN, NumPy propagates and rstsr ignores
(`[1, nan, 3]`: NumPy → 1, rstsr → 2). The trajectory README statement should
be corrected; the rstsr-core side needs an explicit decision (§4).

## 2. Proposed API (rstsr-core)

Mirror the existing argmin/argmax surface exactly:

- Device traits: `OpNanArgMinAPI<T, D>` / `OpNanArgMaxAPI<T, D>` with
  `nanargmin_all`, `nanargmin_axes` (and `nanargmax_*`), `TOut = usize`.
  Implemented for `DeviceCpuSerial` and `DeviceRayonAutoImpl` (faer picks it
  up via the existing `rayon_auto_impl` symlink).
- Tensor level via the same `trait_reduction!` generation:
  `rt::nanargmin(_f)`, `rt::nanargmin_all(_f)`, `rt::nanargmin_axes(_f, axes)`
  and the `nanargmax` family + Tensor methods `t.nanargmin*()`. Unraveled
  twins deferred until a consumer asks.
- Axes argument type identical to argmin (`impl TryInto<AxesIndex<isize>>`,
  negative axes, `None` → 0-D).

## 3. Semantics

| input | proposed rstsr | NumPy nanarg* |
|---|---|---|
| NaNs mid-stream | ignored | ignored |
| NaN at first position | ignored (unlike plain argmin) | ignored |
| all-NaN slice | `Err(InvalidValue, "All-NaN slice encountered")` per reduced slice; panics in non-`_f` forms via `rstsr_unwrap` | `ValueError` per slice |
| ±inf | ordinary values | ordinary values |
| only NaN+inf in slice | first index (documented caveat) | same caveat |
| ties | first occurrence (row-major) | same |
| empty | existing `InvalidLayout` error | ValueError |

All-NaN detection follows the established idiom
`rstsr_raise!(InvalidValue, "All-NaN slice encountered")?` from the kernel.

## 4. Kernel plan + the cheap implementation note

The current 8-lane contiguous kernel already ignores mid-stream NaNs; the only
deltas for nanarg* are (1) seeding must skip a NaN head (find first non-NaN as
lane seed instead of `xs[0]`) and (2) all-NaN → error instead of index 0. So:
extend the native-impl kernels with NaN-skipping variants — either widen
`ArgCmp` (`Min`/`Max`/`MinSkipNan`/`MaxSkipNan`) or a sibling enum + `_cmp_`
twins of the four `reduce_*_arg_cmp_cpu_{serial,rayon}` entries. Strided
fallback: the existing closure fold with `f_comp` returning `None` for NaN
current values already implements skip semantics.

Expected cost: one extra pass-in-parallel-with-scan to locate the seed + a
final "no seed found" check; no change to the inner loop. Bench before/after
per the campaign's paired-baseline convention.

## 5. Open questions (owner decisions)

1. **Plain `argmin`/`argmax` NaN semantics**: keep rstsr's ignore-NaN and
   document the divergence from NumPy, or align with NumPy's first-NaN-wins
   (breaking change; kernel needs a NaN-detect compare — measurable cost; T6
   gate fixtures would need re-anchoring). NumPy-parity tests for the NaN table
   are currently un-transferred (`tests/tracking/numpy_coverage.csv:169,171`
   note "nan/datetime/complex table N/A") — the decision should be recorded
   there either way.
2. **All-NaN**: error (recommended, matches NumPy) vs index 0 vs a
   `maybe`-style sentinel return.
3. **Scope**: raveled all+axes only (recommended) or also unraveled twins from
   day one.
4. **Naming**: `nanargmin` (NumPy-compatible, recommended) vs `argmin_skipnan`
   or similar.

## 6. Test plan

- core_func parity: `test_nanargmax.rs` / `test_nanargmin.rs` mirroring NumPy
  `test_nanargmax.py` surface (axes/None/keepdims analogs within rstsr's API),
  incl. NaN@0, all-NaN error, NaN+inf slices, ±inf handling, ties, empty.
- Regression: all-NaN error message; NaN@0 does not poison (differs from plain
  argmin).
- Device matrix: cpu serial, rayon (faer via symlink), entry-binary tests per
  ADR-0002.

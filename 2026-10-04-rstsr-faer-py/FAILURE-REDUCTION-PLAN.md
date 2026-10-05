# Minimizing failures — rstsr_faer.api × array-api-tests

How to turn the current red map green in the shortest path, derived from the
failure corpus itself (not guesswork). State analyzed: chunked stamp
`20261004-205656` (NO_EXPLAIN, suite pin `6c0b59f`), wheel = branch
`261004/rstsr-faer-py` @ `a52bf4f` (S1 surface + S2 DLPack):

> **302 passed / 998 failed / 82 skipped of 1382** (NumPy baseline: 1335/42/5)

Every number below is reproducible:

```bash
cd 2026-10-04-rstsr-faer-py/harness
NO_EXPLAIN=1 MODULE=rstsr_faer.api CHUNKED=1 ./run.sh     # full red map, ~90 s
NO_EXPLAIN=1 MODULE=rstsr_faer.api ./run.sh array_api_tests/test_array_object.py
```

Per-spec-function grouping: `./summarize_report.py reports/<report>.json`.

## 1. Failure anatomy (what the 998 actually are)

Message-class census of all 998 failures (register: GAP-REGISTER.md v2):

| class | n | what it means |
|---|---|---|
| marshalling-reject (TypeError) | 522 | the function *exists* but the shim declines the input: missing operator dunders (`__abs__`, `__ifloordiv__`, …), missing fns reached as `func=None`, missing `()` / slice indexing |
| wrong-value | 253 | ran and returned a wrong result (mostly `has_names`/`signatures` assertion messages, i.e. missing surface counted here) |
| missing-attr | 136 | `"<fn> is not defined in rstsr_faer.api"` — surface absent |
| unexpected exception | 58 | real bugs (e.g. `finfo` complex rejection, G-032) |
| blocked on `xp.astype` | 26 | operator tests fail in *their own* verification because the namespace-level `astype` is absent (we only have `Array.astype`) |
| explicit declines | 3 | deliberate `NotImplementedError` |

Three structural facts drive the whole plan:

1. **`__getitem__` is load-bearing for the suite itself, not just for its own
   tests.** The suite's verification helpers index with tuples/slices
   (`assert_array_ndindex` does `out[out_idx]`). Today even a *correct*
   `reshape` "fails" because the helper crashes first (observed in
   test_manipulation). Fixing indexing unblocks adjudication everywhere.
   (`test_has_names` already fails `__setitem__`, `__abs__`, `__neg__`, … as
   *methods* — same operator family.)
2. **`test_has_names` / `test_signatures` grade 159 missing names
   unconditionally** — including extension names (`linalg-*`, `fft-*`);
   `--disable-extension` does NOT scope them out (only the extension test
   files). The name list is the backlog (Appendix A); nearly every elementwise
   entry maps to an existing `rt::` function.
3. **Wrong-value debt is invisible until the surface exists.** Enabling a fn
   converts TYPE failures into VALUE failures — expect the red count to move
   slowly at first while adjudication catches up. That is progress, not
   regression; judge by class census, not by raw count alone.

## 2. The working loop (per wave)

1. Implement a wave slice (Python: `cp python/rstsr_faer/*.py` into
   site-packages; Rust: `maturin build --release -i "$TEST_PY" -o /tmp/wheels`
   **from the crate dir** + pip reinstall, ~30 s).
2. Adjudicate per file: `NO_EXPLAIN=1 MODULE=rstsr_faer.api ./run.sh
   array_api_tests/<file>.py` (seconds).
3. Every new divergence → GAP-REGISTER entry (rust-side problems are never
   fixed agent-side — G-030/G-031 pattern).
4. Full map per wave: `NO_EXPLAIN=1 MODULE=rstsr_faer.api CHUNKED=1 ./run.sh`;
   confirm 19/19 chunk reports; `FRESH=1` only for canonical record numbers.
5. Record: `reports/SUMMARY-<wave>.md` + register update.

Discipline reminders: red runs always CHUNKED (a runaway costs one file, and a
dying chunk is a hard error); never diagnose with NO_EXPLAIN when the "Draw N"
blob is the evidence; `ulimit -v` for any repro near a known loop bug.

## 3. Waves (ordered by measured leverage)

### W0 — quick wins (minutes each; do immediately)
- `capabilities()["max dimensions"]` rename (G-033; also type `int | None`).
- Expose `permute_dims` (G-034) — keep `permute_axes` internal.
- `()` no-op indexing for any ndim (suite helpers use `x[()]`; numpy-legal).
- Namespace `astype` (wrap the existing `Array.astype`) — unlocks the
  astype-blocked class (26).
- `xp.newaxis` sentinel (G-025).

### W1 — full indexing (G-021) — **first big wave; everything else verifies through it**
Slices, tuples (ellipsis/None), bool masks, int-array advanced indexing,
`__setitem__` (+ in-place dunders it needs). Direct: array_object 8,
indexing 2; enabling: manipulation/creation verification paths (observed
crashing today). Acceptance: `test_array_object` green modulo registered
gaps; manipulation failures become value-level, not helper crashes.

### W2 — elementwise + operator batch (G-022/G-026) — biggest single unlock (~470 special-cases + 141 operators)
~50 unary/binary fns via dispatch macros over `rt::` (trig/exp family,
floor/ceil/trunc/round, sqrt/square/abs/sign/signbit, bitwise_*, log*,
maximum/minimum, clip, floor_divide, remainder, pow, atan2, hypot, copysign,
nextafter, logaddexp, positive, conj, real/imag, logical_*) + the operator dunder set
(`__neg__ __pos__ __abs__ __invert__ __pow__ __mod__ __lshift__ __rshift__
__and__ __or__ __xor__` + in-place twins) + weak cross-kind scalar promotion
in `_operand` (spec allows py-scalar with array when scalar kind ≤ array
kind; today the shim declines). Acceptance: special-cases TYPE class
collapses; remaining failures there become VALUE adjudications.

> **Landed 2026-10-05** (`799c4b3` + rust-side `positive` in `1546e5e`):
> red map **902/398/82** at stamps `20261005-020258` / `-094602` —
> special-cases 135/482 → 508/109, operators 15/140 → 100/55. Details and
> the residual census in `reports/SUMMARY-w2.md`; new register entries
> G-052…G-058.

### W3 — reductions / stats / searching / set / utility (G-028/G-029)
sum, prod, min, max, mean, std, var, cumulative_sum/prod (whole-array first
via reduction macros, then axes), argmax/argmin (**already exist rust-side**),
where, nonzero, count_nonzero, searchsorted, take/take_along_axis, unique_*,
isin, diff. Resolves the data-dependent-shapes claim (G-029) with reality.
Acceptance: statistical/searching/set/utility files green or registered.

### W4 — creation & manipulation complement (G-024)
linspace, eye, tril/triu, zeros/ones/full/empty_like, meshgrid,
broadcast_to/arrays/shapes, concat, stack, unstack, expand_dims, squeeze,
flip, moveaxis, tile, repeat, roll (roll/repeat = known rust gaps G-002/G-003
— register, don't hack). Acceptance: manipulation + creation green modulo
those entries.

### W5 — dtype-introspection decision point (G-027/G-008; needs a user decision)
can_cast / result_type / isdtype: rstsr has no token-level promotion query.
Options to grill at review: (a) shim-side static table mirroring
DTypePromoteAPI (value-exact, drifts with rstsr), (b) tiny rust-side helper
(proper, needs permission), (c) SKIPS_FILE scope-out (recorded, honest, costs
~16 tests). Also G-032 finfo-complex fix lands here if not earlier.

### W6 — linalg (G-023) + Array method dunders tail
`xp.matmul`, `matrix_transpose`, `vecdot`, `tensordot` + the `xp.linalg`
namespace (23 names: inv, svd/svdvals, qr, cholesky, eigh, eigvalsh, det,
slogdet, solve, matrix_rank, pinv, matrix_power, norms, diagonal, trace,
outer, cross). Faer provides most kernels; QR/slogdet are known rust-side
gaps (G-004/G-005). Largest single-effort wave; do after the cheap 80%.

### W7 — fft: open scoping decision (user)
No rstsr FFT exists. The test file already self-skips (14), but has_names
still grades the 14 fft names. Either implement (out of proportion for a
validation shim) or SKIPS_FILE the fft names with a register entry and keep
them in the fulfillment diff. Decide at review; default = scope out.

## 4. What "done" looks like

Order-of-magnitude trajectory if waves land as estimated (bands, not
promises): W0+1 → ~330–380 passed; +W2 → ~800–900 (red becomes
value-adjudication-bound); +W3/W4 → ~1100+; +W5/W6 → NumPy-baseline
neighborhood minus fft + registered gaps. The honest close is not 1382 green:
it is *every failure either fixed or carrying a register entry*, with the
S4 report diffing the fulfillment table against reality.

## Appendix A — the 159-name backlog (from test_has_names, stamp 20261004-205656)

- **linalg (23)**: cholesky cross det diagonal eigh eigvalsh inv matmul
  matrix_norm matrix_power matrix_rank matrix_transpose outer pinv qr
  slogdet solve svd svdvals tensordot trace vecdot vector_norm
- **fft (14)**: fft ifft fftn ifftn rfft irfft rfftn irfftn hfft ihfft
  fftfreq rfftfreq fftshift ifftshift
- **elementwise (52)**: acos acosh asin asinh atan atan2 atanh ceil clip
  conj copysign cos cosh exp expm1 floor floor_divide hypot imag log log1p
  log2 log10 logaddexp logical_and logical_not logical_or logical_xor
  maximum minimum nextafter positive pow real reciprocal remainder round
  sign signbit sin sinh square sqrt tan tanh trunc bitwise_and bitwise_or
  bitwise_xor bitwise_invert bitwise_left_shift bitwise_right_shift
- **manipulation (14)**: broadcast_arrays broadcast_shapes broadcast_to
  concat expand_dims flip moveaxis permute_dims repeat roll squeeze stack
  tile unstack
- **reductions/stats (11)**: max mean min prod std sum var cumulative_sum
  cumulative_prod argsort sort
- **searching/set (13)**: argmax argmin count_nonzero nonzero searchsorted
  where isin unique_all unique_counts unique_inverse unique_values take
  take_along_axis
- **dtype fns (4)**: astype can_cast isdtype result_type
- **creation (9)**: empty_like eye full_like linspace meshgrid ones_like
  tril triu zeros_like
- **utility (1)**: diff
- **array methods (14)**: __abs__ __and__ __floordiv__ __invert__
  __lshift__ __matmul__ __mod__ __neg__ __or__ __pos__ __pow__ __rshift__
  __setitem__ __xor__
- **top-level extras (4)**: matmul matrix_transpose tensordot vecdot
  (linalg-adjacent, non-namespaced)

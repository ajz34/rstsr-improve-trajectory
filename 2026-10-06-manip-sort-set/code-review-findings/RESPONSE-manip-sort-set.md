# Response to REVIEW-manip-sort-set.md

Fixes landed in rstsr branch `261006/manip-sort-set` @ **`4444fbe`**
(2026-10-07). Every finding was independently re-verified before acting —
two internal-review premises had already been wrong earlier in this wave,
so NumPy behavior claims below were probed first-hand (numpy 2.5.1 env +
`~/Git-Others/numpy` v2.5.2 sources) rather than trusted from the review
text.

Gates after the fix commit: 462 entry tests, 243 doctests, clippy 0, fmt
clean, rustdoc 0 warnings, all five `crates-device/*` compile. Conformance
suite 1059/241/82 — test-for-test identical to the pre-review run.

## Top-15 dispositions

| # | finding | verdict | action |
|---|---|---|---|
| 1 | `crates-device/*` E0433 `half` | **confirmed** (all 5 fail) | dropped the f16/bf16 arms from the symlinked `use_fast_path`; doc comment now states why (`4444fbe`). Device crates compile. |
| 2 | take_along_axis reads raw buffer | **confirmed** (transposed → wrong, broadcast → panic, offset → spurious error) | `resolved` built via `IndexedIterLayout` over the index layout; new test `test_transposed_and_broadcast_index_views` covers t()/broadcast_to/sliced views |
| 3 | naive unique_all undercounts | **confirmed** (reproduced: `[1+0j,1+0j,2+0j]` → counts `[1,1]`) | fresh-entry flag from the match arm replaces the ambiguous `slot == count - 1` test |
| 4 | unique_* fail on broadcast views | **confirmed** (reproduced `Err(ValueOutOfRange)`) | `inverse.truncate(la.size())` in both device impls (raw storage length was wrong) |
| 5 | isin NaN membership | **confirmed** (np: `isin([nan],[nan]) → [false]`) | NaN keys short-circuit to `false`; docstring corrected; `test_isin_nan_membership` re-pinned to NumPy semantics incl. complex keys and invert |
| 6 | isin NaN fast path wrong for complex | **confirmed** (moot after #5: all NaN-bearing keys → `false` without search) | covered by the #5 fix |
| 7 | searchsorted truncates float scalar | **confirmed** (spec "mixing" section: int-array × float-scalar is *unspecified — may promote or raise*; silent truncation is neither) | shim raises the G-009 TypeError for float scalar vs int array (same verdict as a float *array* already gets); registered in api.py comment. NumPy promotes; rstsr's single-dtype kernel would need an array-side cast to follow — noted as follow-up, not shimmed |
| 8 | take_along_axis rejects broadcast indices | **confirmed** (spec text + np probe: `[[3,0]]` against (3,4) → [[3,0],[7,4],[11,8]]) | validation now broadcast-checks non-axis dims; output shape = broadcast; kernel walks the broadcast rest shape with per-tensor dim-1 clamping; docstring "Returns" fixed; `test_broadcast_indices` pins np parity |
| 9 | complex ExtSortCmp deviates from NumPy | **confirmed**, and the review's own description needed decoding: the true rule (from `arraytypes.c.src` `C@TYPE@_compare`, cross-validated 3000 random np.sort runs) is: *every finite value orders before every NaN-bearing value; among NaN-bearing values, part-wise lexicographic with NaN-after-finite per part* | `ext_total_cmp` rewritten as the exact transliteration; rust-vs-numpy cross-check on 2000 identical random inputs: **0 mismatches**; trait docs + unit tests rewritten to the settled semantics |
| 10 | roll rejects shift tuple × len-1 axis list | **confirmed** (np: `np.roll(x,(1,2),axis=(0,))` rolls by 3 — `broadcast(shift,axis)` then per-axis sum, read from `numpy/_core/numeric.py`) | rust implements broadcast-sum: len-1 sides broadcast; equal lengths pair; a len-1 *axis list* accumulates all shifts onto that axis; non-broadcastable lengths raise |
| 11 | shim rejects diff n=0 | **confirmed** (np: copy; rust `diff_f` already handles) | shim allows `n >= 0` |
| 12 | negative ints → OverflowError | **confirmed** (np: `ValueError: negative dimensions are not allowed`) | shim-side ValueError guards for `repeat` (scalar + array), `tile`, and `searchsorted` sorter (negative sorter entries were unvalidated rust-side too) |
| 13 | take_along_axis docstring wrong dtype/negatives | **confirmed** (contradicts its own signature and `test_negative_indices`) | docstring now says `isize`, negatives count from the back |
| 14 | unique_values signed-zero doc wrong | **confirmed** (stable-sort + `==` dedupe keeps first-seen; `-0.0` first stays) | both docstring copies now say first-seen in both paths |
| 15 | repeat/roll flatten copies per element | **confirmed by reading** (real, but a perf issue) | **not fixed in this correctness round** — registered below with the rest of the efficiency family as follow-up work |

## Below-the-cut dispositions

Applied:

- `OpUniqueAPI` doc now states the real contract: the **device impl must**
  truncate each output raw Vec to its initialized prefix (latent-UB hazard
  removed from the trait doc).
- `tensor::exports` re-exports `nonzero::*`.
- `numpy_differences.md` unique entry corrected: rstsr keeps NaNs distinct,
  **NumPy collapses them** — the wrong "matching NumPy" claim is gone, and
  the NaN part of the rstsr behavior is explicitly marked as part of the
  registered deviation.
- `numpy_coverage.csv`: stale `not-applicable`/`todo` rows for
  `TestRoll::{test_roll1d,test_roll2d,test_roll_empty,test_roll_big_int}`
  and `TestTile::test_basic` removed (the `transferred` rows remain;
  genuinely-inapplicable rollaxis/uint-shift rows stay).
- 3 rustdoc warnings fixed (redundant explicit targets ×2, ambiguous
  `concat` link → plain `rt::concat` reference).
- argsort/sort docstrings: the self-contradictory "NumPy raises for complex
  sort" removed — NumPy 2.5 sorts complex lexicographically (verified);
  rstsr declines per the standard's real-valued scoping, with
  `sort_custom`/`argsort_custom` + `ExtSortCmp` as the NumPy-parity path.

Noted, not fixed now (perf/cleanup family — one coherent follow-up pass):

- `test_unique_signed_zero_merge` cannot fail on its claimed property
  (true for both encodings).
- isin per-x1-value serial loop (comment overstates parallelism);
  duplicated `use_fast_path` ladder across 7 crates.
- rayon sort per-line pairs buffer; `sort_by` → `sort_unstable_by`;
  `nonzero_f` buffer clone + hand-rolled stride math; take_along_axis
  extra resolved Vec; `SortArgs.stable` threaded but ignored; `AxisIndex`
  conversion-matrix duplication; faer-py `dispatch_unique_values!`
  re-rolling; missing doc_draft twins / row-col notices for new anchors;
  `test_rank_promotion_up` missing `specify_test!`; >4-line comment in
  test_tile.rs.
- diff lacks `core_func` tests (only doctests) — test-authoring follow-up
  (n>axis, prepend/append validation, empty-axis stop).

## Note on the review's suite observation

Correct: the conformance suite's isin/take_along_axis value checks are
`TODO`s upstream, and `crates-device/*` is outside the wave's gate loop —
that is exactly how findings 1–8 survived the gates. The wave's own gates
now include a `cargo check` of the five device crates (performed in this
round; all green).

## Incidents during the fix round (for the record)

- First suite re-run after the shim guards regressed 4 tests:
  the guards' `any(...)` resolved to the api module's own namespace
  `any` (builtins are shadowed by the array-api surface), not
  `builtins.any` — generator handed to `_handle`. Fixed to
  `builtins.any`; second run test-for-test identical to baseline.
- One doctest failure in the full parallel run was a transient tmpfs
  SIGBUS in `ld` (243/243 pass at `--test-threads=2`; reproducible only
  under high parallel link load).

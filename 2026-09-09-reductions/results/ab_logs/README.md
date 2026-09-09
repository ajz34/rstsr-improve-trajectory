# A/B logs — faer16 sum_axis0 2048² (patched vs original, interleaved)

Committed raw evidence for the gate cell that motivated the back-to-back
A/B methodology (README "Gates" paragraph). Criterion filter:
`large_2048x2048-faer16_f64/sum_axis0`, `--baseline portable` (the saved
phase-1 baseline = 402.5 µs), full criterion output tee'd.

Interleaved A-B-A on the same session (2026-09-09, after phase 2; each run
= separate `cargo build --release --benches` of the respective code state):

| log | code state | time mid | vs saved baseline |
|---|---|---|---|
| `ab_faer16_sum_axis0_ORIGINAL_run0.log` | original (386948be) | 294.59 µs | −29.6% |
| `ab_faer16_sum_axis0_PATCHED.log` | proposed.patch applied | 337.02 µs | −20.0% |
| `ab_faer16_sum_axis0_ORIGINAL.log` | original | 302.49 µs | −25.6% |
| `ab_faer16_sum_axis0_PATCHED_run2.log` | patched | 273.19 µs | −33.2% |

Reading (and why this cell motivated the methodology):

- All four runs sit 25–33% BELOW the phase-1 saved baseline — i.e. the
  saved-baseline `change`% for this cell is dominated by session drift,
  not by code. (T0 measured the same op+code at 444.7 µs; phase 1 at
  402.5 µs; both pre-date these runs.)
- The interleaved pattern O < P > O > P with ±11% swings WITHIN a code
  state across consecutive minutes means this rayon-parallel cell cannot
  resolve layout-level (±10%) differences at all. Patched vs original is
  a tie within within-state noise (patched mean 305 µs, original mean
  299 µs), consistent with the rayon kernel being byte-identical between
  the two states.
- Reproduce: apply/remove `proposed.patch` on a clean 386948be, then
  ```bash
  cargo build --release --benches
  CRITERION_HOME="$PWD/results/criterion_portable" \
    cargo bench --bench reduce -- \
    "large_2048x2048-faer16_f64/sum_axis0" --baseline portable
  ```
  alternating with `git -C ../rstsr apply|checkout` between runs.

The three earlier interactive-session pairs quoted in the README (serial
sum_axis1 / sum_all / 1e7 sum_all + this cell's first A/B) were run the
same way but without tee; they are superseded as evidence by these
committed logs.

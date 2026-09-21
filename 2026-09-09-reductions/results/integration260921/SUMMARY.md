# T2' reductions integration session — 2026-09-21

## FINAL STATE (2026-09-21, end of session): REVERTED

Owner reverted the edit after the variant-V decision: rstsr is usually used
multi-threaded, so a serial-only gain is not the main aim. Working tree
cleaned, branch `260921-reductions` deleted (never committed), master back
at `ca325a7`. Artifacts below kept as the record. Follow-ups flagged:
- sequential-order branch-2 walk (~2x headroom, FP-order semantics gate):
  see memory `rstsr-sum-axis0-sequential-order`;
- multi-threaded paths first: memory `rstsr-multithreaded-first-priority`.

## SUPERSEDED decision record: variant V — A only

Owner ruled "array-api compliance before efficiency" and questioned the
PartialOrd-based strict-compare closures (A2). A2 was STRIPPED: the two
rstsr-core wiring files are back at HEAD; the branch carries ONLY the
band-walk hunk (`rstsr-native-impl/src/cpu_serial/reduction.rs`, +99/−15).
Diff snapshot: `/tmp/t2v_candidate.patch` → also saved as
`proposed-integration260921-variantV.patch`.

- Gates re-run, all green: lib 129+3ign / 111+1ign (faer / faer-free),
  entry_row_cpu 302/302 both, clippy clean both, col lib 111+1ign.
- A/B (V1/R5/V2 interleaved, native): serial `sum_axis0` 1.27–1.48×,
  serial `min/max_axis0` 1.17–1.26×, `min/max_all` parity (0.995–1.03×,
  A2 removed as expected), serial canaries parity. Rayon cells show no
  stable signal either direction (identical binaries swing ±10–15%
  between passes — lottery). Watch-item: f32 serial `sum_axis0` read
  4–6% slower than R5, inside today's ±10% ref band (350–429 µs).
- Rationale + compliance audit: [ARRAYAPI-CHECK.md](ARRAYAPI-CHECK.md) —
  array-api 2024.12 REQUIRES NaN propagation for min/max (rstsr skips,
  pre-existing divergence, unchanged by A2); arg* NaN is unspecified by
  the standard, so rstsr's arg divergence is compliant. A future
  NaN-propagation fix would rewrite the A2 closures anyway.

Everything below documents the (superseded) full-patch integration run;
kept because it established the flip protocol, the isolation experiment,
and the A2 trade-off quantification the decision was based on.

---

## Full-patch run (A + A2, superseded)

Patch 4 (reductions: A band walk + A2 strict-compare min/max) applied to
`../rstsr` branch `260921-reductions`, base = master `ca325a7` (post PR #106).
Applied via `git apply --3way` (clean on all 3 files); one formatting blemish
in the original `proposed.patch` fixed during integration (`pub fn
reduce_axes_cpu_serial<...>(    a: &[TI],` — parameter jammed onto the
signature line; AST-identical, rustfmt-clean after fix). Final diff:
`proposed-integration260921.patch` (284 lines, 3 files).

## Gates (all green)

| config | lib | entry_row_cpu | clippy |
|---|---|---|---|
| default (faer+rayon) | 129 pass + 3 ign | 302/302 | clean |
| faer-free row-major | 111 pass + 1 ign | 302/302 | clean |
| col-major | 111 pass + 1 ign | (no col test binary on master) | compile ok |

## Paired A/B (target-cpu=native, per-side CRITERION_HOME, interleaved)

Full suite (45 filtered benches): C1/R1/C2/R4 logs. Ratios = candidate/ref
median per interleaved pass; machine drifted across passes (all C4 cells
elevated) — only same-pass ratios are meaningful.

### Wins reproduce

| cell | ratio (per-pass range) |
|---|---|
| serial min_all/max_all 1e7 | **8.0–8.4×** (9.92 → 1.18 ms) |
| faer16 min_all/max_all 1e7 | **2.4×** (360 → 150 µs) |
| serial sum_axis0 2048² / 512² | 1.40–1.52× |
| serial min/max_axis0 2048² | 1.35–1.54× |
| serial min/max_axis0 512² | 1.29–1.37× |
| serial odd 1000×777 sum_axis0 | 1.34–1.84× |
| serial f32 sum_axis0 2048² | 1.06–1.08× (pass-1 ref was an outlier) |

Canaries at parity (0.98–1.02×): sum_all, sum_axis1 (all sizes/devices),
small 64×64 sum_axis0 (below the 16k floor guard → falls back to the
byte-identical iterator walk, as designed).

### Stable regression found, isolated

faer16 (rayon) min/max_axis0 @ 2048²: candidate slower in **all 4 paired
passes**, ~10–25% (ref 205–293 µs, cand 232–348 µs). Isolated with variant W
(`t2ab_W5.log`: rayon wiring hunk reverted, serial wiring kept):

- regression disappears (201 µs ≈ ref) — caused by the A2 strict-compare
  wiring in `feature_rayon/auto_impl/reduction.rs`;
- so is the faer16 min/max_all 2.4× win (357 µs in W).

One wiring change, two effects: big win on 1-D full reduce, moderate loss on
large 2-D axis-0. Rayon odd/small cells are lottery (±50–100%, unstable,
consistent with the phase-2 rayon-twin revert lesson).

### Decision for the owner

- **As-patched** (current tree): keep both effects — min/max_all 2.4× (faer16)
  + 8.4× (serial), serial axis0 wins; accept documented ~10–25% faer16
  min/max_axis0 @ 2048². This is what the reviewed proposed.patch shipped.
- **Variant W**: additionally revert the rayon hunk (single file:
  `rstsr-core/src/feature_rayon/auto_impl/reduction.rs` to HEAD) — no rayon
  behavior change at all, forfeits the 2.4×.

## Protocol notes (flip lessons)

- `proposed.patch` no longer applies plain-forward on drifted master; flips
  must use `git apply -R` (down) / `--3way` (up, blob-merges against 386948be
  preimages). `--3way` STAGES the result — `git reset` afterwards, and verify
  with `git diff` vs the snapshot.
- First flip attempt silently ran in the wrong repo (cwd drift between
  sessions); R1 measured the candidate and was discarded. All subsequent
  flips used `git -C <abs-path>` + content fingerprints
  (`reduce_axes_band_walk` count 0/5, strict-compare line count, `min_all`
  value fingerprint 9.9 ms ref vs 1.18 ms cand) before trusting a pass.

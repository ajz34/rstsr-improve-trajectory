---
name: rstsr-reductions-integration
description: Patch 4 (T2' reductions) integration session 2026-09-21 — applied, validated, then REVERTED at owner request (serial-only gain, no commit); kept: array-api NaN audit (min/max propagation REQUIRED, rstsr skip diverges pre-existing), apply/--3way flip protocol, rayon lottery.
metadata:
  type: project
---

T2' reductions integration (2026-09-21, branch `260921-reductions` on master
ca325a7): **REVERTED same day at owner request** — variant V never committed;
branch deleted, rstsr master clean at ca325a7. Kept lessons below.
Reason: serial-only gain (owner: multithreading is the usual case, see
[[rstsr-multithreaded-first-priority]]); follow-up flagged in
[[rstsr-sum-axis0-sequential-order]].

- **FINAL = variant V (A band walk only, +99/−15, 1 file)**: owner ruled
  array-api compliance > efficiency and rejected the A2 PartialOrd
  strict-compare closures. A2 stripped (both rstsr-core wiring files at
  HEAD); serial sum_axis0 1.27–1.48× and min/max_axis0 1.17–1.26× kept,
  min/max_all parity (8.4×/2.4× forfeited), faer16 axis0 regression gone.
- **array-api 2024.12 audit** (results/integration260921/ARRAYAPI-CHECK.md):
  min/max NaN propagation is REQUIRED by the spec — rstsr's skip (via
  ExtReal float minnum) is a PRE-EXISTING divergence, incl. all-NaN → ±MAX
  seed; arg* NaN is UNSPECIFIED by the spec (rstsr's arg divergence is
  compliant; NumPy parity is the only gap). A future propagation fix =
  ext_real.rs + wiring closures + T2' gate fixtures; main suites have no
  min/max NaN tests today (clean to change); cost unmeasured (arg campaign
  saw +55..+100% for arg; value-min/max any-NaN OR-mask should be cheaper).
- Integration mechanics (apply/flip): plain forward `git apply` FAILS on
  drifted master — 386948be-era patches need `git apply -R` (down) and
  `--3way` (up); --3way STAGES (git reset + verify diff vs snapshot). cwd
  can point at the wrong repo between sessions: always `git -C <abs>` and
  fingerprint each pass by VALUE (min_all 9.9 ms ref vs 1.18 ms cand) — one
  flip silently measured the candidate and had to be discarded. Rayon
  faer16 cells are lottery (identical binaries swing ±10–15%); only
  same-pass paired ratios are meaningful.
- `proposed.patch` also carried a fmt blemish (`pub fn ...(
      a: &[TI],`
  jammed signature — compiles, fails rustfmt); fixed in tree.
- Gates for variant V: lib 129+3ign / 111+1ign (faer/faer-free),
  entry_row_cpu 302/302 both, clippy clean both, col lib 111+1ign.
- Evidence: `2026-09-09-reductions/results/integration260921/` (SUMMARY.md,
  ARRAYAPI-CHECK.md, 11 logs, both patch snapshots) + probe
  `examples/arrayapi_check.rs` in the T2' dir.

Related: [[rstsr-reductions-t2p]], [[rstsr-vecdot-t3]] (patch 5 still pending).

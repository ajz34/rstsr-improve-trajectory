# Grill R5 — BLAS-2 GEMV dispatch (post-P0)

Session 2026-11-23 (round 5, after P0). rstsr `062c970` (`main`, v0.10.4 — the
grilled `matmul.rs` dispatch is unchanged); harness in this directory. No rstsr
file edited. Recommended answers marked ➡️.

---

## R4 answers — agent-authored transcript (chat; no answers file exists)

- **Q1 = (b)** bucket: *"Allow copy on non-contiguous buffer, no copy on
  contiguous buffer."* → D28.
- **Q2 = (b)** bucket: *"minimal surface is better"* → D27 (no `incx`/`incy` in
  `driver_gemv`). Also asked to check scipy's f2py BLAS layer → findings recorded
  in `DECISIONS.md` §Scipy cross-check.
- **Q3 = okay** (P0 approved) → the harness above.

---

## P0 outcome (numbers in `README.md` / `results/`)

Hypothesis **confirmed for the OpenBLAS device**: all 12 f64 cells slower than
raw `cblas_dgemv` (1.04–11.8×), worst relative `1024² matvec` (11.8×), worst
absolute the 0.93 GB `256×452676 vecmat` (55.0 → 15.9 ms, **39 ms wasted per
call**); f32 spot 14.6×; the `m = 1` orientation loses 2.9–3.5× on skinny/DRAM
cells while `n = 1` loses 1.04–1.45×.

**Faer is mixed, not uniformly fine**: DRAM cells are at parity or faster than
the cblas anchor (0.73–1.00×), `65536×64 vecmat` is **6.3× faster than
`cblas_dgemv`**, but the transposed orientation still loses 1.4–2.4× at `1024²`,
`4096²`, `65536×64` and 1.69× (11 ms) at `256×452676 vecmat`.

**Batched rule-6 cell is 20–40× off its ideal** (11.0 ms openblas / 8.9 ms faer
vs ~0.3–0.6 ms) — not explained by gemv-vs-gemm alone; it is exactly the branch
D11 rewrites.

---

## Q1 — Faer scope after the P0 numbers

- (a) keep D4: this campaign patches the BLAS device only; record the Faer gaps
  as a follow-up task with these numbers;
- (b) extend the campaign with a Faer pass after P2 (the levers are layout
  canonicalisation before the faer call, or an rstsr-side matvec kernel — faer's
  public API exposes only `matmul`, not `matvec`);
- (c) patch Faer before the BLAS device.

➡️ (a) with a recorded follow-up. The BLAS gap is 3–12× (and 39 ms per call at
the largest cell); Faer's is 1.4–2.4× at small/medium cells and needs its own
investigation (why the transposed layout misses faer's matvec at
`1024²`/`4096²` while the DRAM cells hit it). Mixing the two makes the P2
regression attribution impossible.

---

## Q2 — The batched-cell anomaly

The batched cell is 20–40× off its ideal, in the branch D11 already rewrites
(`n_task > 1` → rayon over batches with one BLAS thread per task; no
`with_num_threads` toggling).

- (a) park it: out of scope, record it;
- (b) make it part of P2's acceptance: after the D11 change, re-measure this
  cell; if it is still far off, open a separate investigation;
- (c) investigate it now, before P1.

➡️ (b). P2 touches exactly that loop, so the cell becomes a free acceptance
datum; investigating the Rayon/OpenMP interaction now would delay the patch and
is likely a distinct issue.

---

## Q3 — Go-ahead for P1 (patch-1: `rstsr-blas-traits` BLAS-2)

patch-1 = `src/blas2/{mod,gemv}.rs` (`GEMVDriverAPI`, `GEMV_`, aliases),
`lib.rs`/`prelude.rs` registration, `GEMVDriverAPI` composed into
`BlasDriverAPI`, `driver_impl/cblas/blas2/gemv.rs`, and the container test module
in the openblas crate (`tests/blas2_gemv.rs`). It compiles and tests without
changing any device behaviour.

➡️ Yes. patch-2 (the five `matmul.rs` dispatch sites) follows once patch-1 is
reviewed, per D10.

---

## Remaining tree

Design is closed (R1–R4); P0 is done. R5 only decides the Faer scope (Q1), the
batched-anomaly handling (Q2), and the patch-1 start (Q3).

# 2026-11-18 — BLAS-2 GEMV dispatch for matmul (P0 baseline, P3 after)

Grill: `GRILL-INIT-PROMPT.md`, `GRILL-R1..R5-QUESTIONS.md`; closed decisions in
[`DECISIONS.md`](DECISIONS.md). Grill facts were gathered at `d5c9b86`
(`261023/astype`); **the P0 build used `062c970` (`main`, v0.10.4)**, whose
`matmul.rs` dispatch is byte-identical to the grilled one (only `(1,1,0)` and
`(2,2,2)` arms). **No rstsr file is modified by this directory** — benchmarks
and numbers live here only (rstsr receives code, not measurement harnesses).

## Question

How efficiently does the current rstsr matmul path handle row-vector /
column-vector products (`xᵀ @ B`, `A @ x`, and their degenerate 2-D forms) on
`DeviceOpenBLAS` and `DeviceFaer`, compared with raw `cblas_?gemv`?

The current dispatch fast-paths only `(1,1,0)` and `(2,2,2)`; the vector forms
are resolved by `layout_matmul_dyn_row_major_with_lc` into 2-D layouts with a
fabricated size-1 axis and therefore run as **cblas gemm with m=1 / n=1**.

## Measurement contract (D3–D7, D24–D25)

- `cargo bench` (stock release profile, stated in `Cargo.toml`).
- `RAYON_NUM_THREADS=16`; the BLAS runtime is pinned by the harness
  (`omp_set_num_threads` for the OpenMP-built system library, otherwise
  `openblas_set_num_threads`) and the rayon device thread count is asserted.
- rstsr rows: the public `%` operator (fresh output, `beta = 0`, i.e. the
  `matmul_uninit` path). Anchor rows: `cblas_?gemv` on the same buffer.
- Correctness gate (`examples/correctness.rs`) runs before timing and compares
  every timed cell with the cblas gemv anchor (`max rel < 1e-10`).
- Three criterion passes (`run1..run3`); report the median of the three per-cell
  means.
- f64 unless stated; one f32 spot at `1024²`.

## Cells

| matrix | matvec `a @ xq` | vecmat `xp @ a` | note |
|---|---|---|---|
| `1024 × 1024` | `y(1024)` | `y(1024)` | L2/L3-resident |
| `4096 × 4096` | `y(4096)` | `y(4096)` | L3-resident |
| `8192 × 8192` | `y(8192)` | `y(8192)` | DRAM-bound |
| `65536 × 64` | `y(65536)` | `y(64)` | skinny, large K |
| `256 × 452676` | `y(256)` | `y(452676)` | 0.93 GB |
| `452676 × 256` | `y(452676)` | `y(256)` | the physical transpose, 0.93 GB |

Extras: `matmul_from` with `beta = 1` and a caller output (`1024²`); batched
rule-6 `(64, 1024, 1024) @ x`; f32 spot (`1024²`, OpenBLAS).

## Environment

- CPU: AMD Ryzen 9 9950X3D (Zen 5), 16 cores / 32 threads; AVX-512.
- rustc 1.100.0 (2026-11-09), cargo 1.100.0; stock release profile (see `Cargo.toml`).
- rstsr `062c970` (`main`, v0.10.4); harness build with `RSTSR_DEV=1` (links
  the system libopenblas and `gomp`).
- BLAS: `OpenBLAS 0.3.32 NO_LAPACKE DYNAMIC_ARCH NO_AFFINITY USE_OPENMP
  Cooperlake MAX_THREADS=128`, `openblas_get_corename() = Cooperlake` — the
  distro build selects an Intel kernel on this Zen 5 CPU (caveat, D25).
- Threads: `RAYON_NUM_THREADS=16`, BLAS runtime pinned by the harness
  (`omp_set_num_threads(16)`, since the OpenMP-built library ignores
  `OPENBLAS_NUM_THREADS`).
- numpy anchor: numpy 2.5.4 with its bundled `scipy-openblas 0.3.34`,
  `DYNAMIC_ARCH SkylakeX`.

## Results

f64, 16 threads, criterion `time: [lo mid hi]` per cell, clean pass `run4`
(`results/bench-run4.txt`; `run5` = repeat for stability). `ratio = rstsr /
cblas gemv`.

| device | cell | orientation | rstsr | cblas gemv | ratio |
|---|---|---|---|---|---|
| openblas | 1024×1024 | matvec | 0.0679 ms | 0.0076 ms | **8.96×** |
| openblas | 1024×1024 | vecmat | 0.1133 ms | 0.0728 ms | 1.56× |
| openblas | 4096×4096 | matvec | 1.0565 ms | 0.7238 ms | 1.46× |
| openblas | 4096×4096 | vecmat | 1.7979 ms | 1.2469 ms | 1.44× |
| openblas | 8192×8192 | matvec | 10.3270 ms | 8.1679 ms | 1.26× |
| openblas | 8192×8192 | vecmat | 12.5160 ms | 8.5967 ms | 1.46× |
| openblas | 65536×64 | matvec | 0.0585 ms | 0.0459 ms | 1.27× |
| openblas | 65536×64 | vecmat | 0.7997 ms | 0.3308 ms | 2.42× |
| openblas | 256×452676 | matvec | 17.0460 ms | 14.8870 ms | 1.15× |
| openblas | 256×452676 | vecmat | **55.4870 ms** | 16.0100 ms | **3.47×** |
| openblas | 452676×256 | matvec | 17.8330 ms | 17.1960 ms | 1.04× |
| openblas | 452676×256 | vecmat | 27.4200 ms | 22.8400 ms | 1.20× |
| faer | 1024×1024 | matvec | 0.0167 ms | 0.0078 ms | 2.15× |
| faer | 1024×1024 | vecmat | 0.0313 ms | 0.0618 ms | 0.51× |
| faer | 4096×4096 | matvec | 1.1818 ms | 0.7241 ms | 1.63× |
| faer | 4096×4096 | vecmat | 0.9833 ms | 1.1064 ms | 0.89× |
| faer | 8192×8192 | matvec | 7.8957 ms | 8.1648 ms | 0.97× |
| faer | 8192×8192 | vecmat | 8.0993 ms | 8.6090 ms | 0.94× |
| faer | 65536×64 | matvec | 0.0631 ms | 0.0451 ms | 1.40× |
| faer | 65536×64 | vecmat | 0.1028 ms | 0.6794 ms | **0.15×** |
| faer | 256×452676 | matvec | 14.7400 ms | 14.9530 ms | 0.99× |
| faer | 256×452676 | vecmat | 30.6070 ms | 17.4010 ms | 1.76× |
| faer | 452676×256 | matvec | 16.5890 ms | 17.1880 ms | 0.97× |
| faer | 452676×256 | vecmat | 15.3470 ms | 20.9450 ms | 0.73× |

Regenerate with `python3 results/make_tables.py results/bench-runN.txt`.

**Stability.** `run1` (same code, quiet machine) agrees with `run4` on every
large cell (< 3 %), but small cells move: `1024² matvec` rstsr 86.3 → 67.9 µs
(−21 %), `65536×64 vecmat` anchor 300 → 331 µs (+10 %). `run2` is **excluded**
(it overlapped a compile, anchors moved up to 133 %); `run3` failed to compile
(concurrent edit). Rule for P3: one cell per quiet pass, no other build or
process running; treat ±10 % on small cells as noise.

### Extras and anchors (run4)

| item | openblas | faer |
|---|---|---|
| f32 1024² matvec (`%`) | 65.4 µs (run1) | — |
| f32 1024² matvec (cblas sgemv) | 4.47 µs (run1) | — |
| `matmul_from` 1024², beta=1, caller `out=` | 66.5 µs | 16.2 µs |
| batched rule-6 `(64,1024,1024) @ x` | 11.01 ms | 9.12 ms |
| batched rule-7 `(64,512,512) @ (64,512,512)` | **20.41 ms** | 86.98 ms |
| numpy 1024² matvec / vecmat | 8 µs / 71 µs | (same numpy) |
| numpy 65536×64 matvec / vecmat | 66 µs / 510 µs | |
| numpy 256×452676 matvec / vecmat | 14.9 ms / 15.8 ms | |

The batched rule-7 number is the **D29 regression datum**: D29 replaces the
current rayon-over-batch/one-BLAS-thread policy with a main-thread loop and
multi-threaded BLAS for every `n_task > 1`, which affects batched GEMM too. The
current 20.4 ms is what the universal rule must not exceed by more than noise;
P3 measures it.

## Observations

1. **OpenBLAS: the hypothesis holds.** Every f64 cell is slower than raw
   `cblas_dgemv` (1.04–8.96×). Worst cases: `1024² matvec` (8.96×, 60 µs), the
   skinny `65536×64 vecmat` (2.42×) and the 0.93 GB `256×452676 vecmat`
   (55.5 → 16.0 ms, **39.5 ms wasted per call**). f32 spot 14.6×.
2. **The `m = 1` degenerate gemm is the culprit at the large cells.** The
   row-vector orientation (`xp @ a`, which becomes `m = 1`) loses 1.2–3.5× on
   the skinny/DRAM cells, while the column-vector orientation (`a @ xq`,
   `n = 1`) loses 1.04–1.46× — as the C probe predicted.
3. **Faer: mixed.** DRAM-bound cells are at parity or faster than the cblas
   anchor (0.73–0.97×) and `65536×64 vecmat` is 6.6× faster than `cblas_dgemv`,
   so its internal matvec kernel is reached there; but `1024²/4096²/65536×64
   matvec` still lose 1.4–2.15× and the 0.93 GB vecmat loses 1.76× (13 ms).
   Out of scope by D/R5-Q1.
4. numpy (its own OpenBLAS 0.3.34) sits at the cblas-gemv level on every cell,
   so the anchor is realistic, not an outlier.

## Conclusion

The BLAS device path needs the BLAS-2 dispatch (P1/P2): measured headroom is
1.0–11.8× on f64 and 14.6× on the f32 spot, with 39 ms per call at the largest
cell. These numbers are the P2 acceptance target (D7: within ~10 % of the raw
`cblas_?gemv` on the same cell; no regression elsewhere). Faer's gap is smaller
and concentrated in the transposed orientation — scope decision in R5.

## P3 — after the dispatch (2026-11-23)

Measured tree: rstsr branch `261123/blas2-gemv` at `6599375` — patch-1
(`3c3999d`, `718864a`), patch-2 plain dispatch (`c93fdde`), and the follow-up
commit that makes the mixed-dtype fallback rayon-parallel (`6599375`).

The baseline of record is **not** `run4`: those absolute numbers are no longer
reproducible on this machine. The batched-gemm cell, re-measured three times at
`062c970` in this session, gives 24.6–26.2 ms against `run4`'s 20.41 ms, so the
A/B below uses an adjacent pair of full passes taken minutes apart on a quiet
machine (`results/bench-run5-old.txt` = `062c970`, `results/bench-run6.txt` =
`6599375`); a second post-patch pass (`results/bench-run5.txt`) agrees with the
latter. Full table: `results/tables-p3.md`
(`python3 results/make_p3_table.py`).

| cell | orientation | before (ms) | before/anchor | after (ms) | after/anchor | speedup |
|---|---|---|---|---|---|---|
| 1024² | matvec | 0.0690 | 9.42× | 0.0078 | 1.07× | 8.8× |
| 1024² | vecmat | 0.1162 | 1.78× | 0.0259 | 0.40× | 4.5× |
| 4096² | matvec | 1.0902 | 1.50× | 0.7277 | 1.00× | 1.5× |
| 4096² | vecmat | 1.7993 | 1.81× | 1.0510 | 0.96× | 1.7× |
| 8192² | matvec | 10.329 | 1.26× | 8.176 | 1.00× | 1.3× |
| 8192² | vecmat | 12.673 | 1.46× | 8.581 | 1.00× | 1.5× |
| 65536×64 | matvec | 0.0571 | 1.24× | 0.0463 | 1.01× | 1.2× |
| 65536×64 | vecmat | 0.8946 | 2.96× | 0.7225 | 1.00× | 1.2× |
| 256×452676 | matvec | 17.519 | 1.18× | 14.825 | 1.00× | 1.2× |
| 256×452676 | vecmat | 55.100 | 3.45× | 15.762 | 1.00× | 3.5× |
| 452676×256 | matvec | 17.880 | 1.04× | 17.124 | 1.00× | 1.0× |
| 452676×256 | vecmat | 26.765 | 1.22× | 21.486 | 0.92× | 1.2× |

f32 spot (`1024² matvec`): 65.41 µs → 5.07 µs (14.2× → 1.13× of the anchor,
12.9× faster). All twelve OpenBLAS f64 cells are now at **0.92–1.07×** of the
raw `cblas_dgemv` anchor, i.e. within the P2 acceptance target (≈10 %).

Extras (same passes):

| extra | before (ms) | after (ms) | speedup |
|---|---|---|---|
| `matmul_from` with `beta != 0`, 1024² | 0.068 | 0.007 | 9.50× |
| batched gemv `(64,1024,1024) @ x` | 11.084 | 8.356 | 1.33× |
| batched gemm `(64,512,512) @ (64,512,512)` | 22.323 | 22.319 | 1.00× |

Faer is unchanged (0.9–1.1× pass-to-pass, i.e. noise): out of scope, kept as the
control that the BLAS patch does not perturb the other device.

## Observations (P3)

1. **The 39 ms/call cell is gone**: `256×452676 vecmat` 55.10 → 15.76 ms, at the
   anchor's 1.00×. `1024² matvec` 8.8× faster, `f32` 12.9×.
2. **Sub-1.0 ratios are real, not measurement noise.** rstsr normalises the
   matrix so the BLAS call traverses it contiguously (`c`-prefer becomes a
   transposed view with the flag flipped), while the anchor keeps the strided
   orientation of the same buffer; on `vecmat` cells that makes rstsr *faster*
   than the anchor (0.40× at 1024², 0.92× at 452676×256).
3. **No regression elsewhere**: batched gemm is bit-identical in time
   (22.32 vs 22.32 ms), and the batched vector cell gains 1.33× from the
   main-thread/multi-threaded-BLAS rule. `run4`'s 20.41 ms batched datum is a
   machine-state artifact, not a code effect (see above).

## P3 conclusion

The dispatch meets its acceptance target on every measured OpenBLAS cell, and
Faer/other dtypes are untouched. Remaining, recorded in `DECISIONS.md`: the
gemm/syrk `k == 0` zero-fill divergence from `matmul_from`'s `beta * c`
contract, and any decision to port the same dispatch to `DeviceFaer`.

## Reproducing

`./reproduce.sh` (correctness gate → three criterion passes → numpy anchors).
Two environment notes for this machine:

- `RSTSR_DEV=1` is required so `rstsr-openblas`'s build script links the system
  `libopenblas` and `gomp`; without it the linker fails on `omp_*`.
- The sandboxed shell cannot write `~/.cargo`; builds here use a writable
  `CARGO_HOME` (e.g. `CARGO_HOME=$HOME/rstsr_pack-tmp/cargo`, a copy of the
  registry plus `config.toml`). On a normal machine the default registry is fine.

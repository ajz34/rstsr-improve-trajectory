# 2026-11-26 — CUDA matmul special forms: inner dot, gemv, syrk

Follow-up of `../2026-11-25-cuda-matmul-impl`. Started from rstsr `261021/cuda` @ `80e626d`; the feature is committed on that branch as `bc4c8d9`, `a7c65aa`, `196e348`, `dae8198`, `63b694c`, `67a0c79` (not pushed).

- `GRILL-INIT-PROMPT.md`: the request and the facts gathered before round 1.
- `GRILL-R1-QUESTIONS.md`: round 1 (forms vs drivers, inner dot kernel, gemv and syrk scope,
  measure-first gate). `GRILL-R2-QUESTIONS.md`: round 2, with the round-1 answers on top.
- `RESEARCH-cublas.md`: the cuBLAS survey (dot, gemv, syrk, batched gemm; what cudarc binds).
- `PLAN.md`: the draft plan and batches B0 (probe) to B4 (matrix, bench, docs).

B0 probe: `src/`, `results/`.
## B0 probe (2026-11-26)

Source `src/probe_forms.rs` (`cargo run --release --example probe_forms -- <dot|gemv|gemv-batch|syrk>`),
raw outputs in `results/b0_*.txt`, tables and conclusions in `B0-REPORT.md`. rstsr `80e626d`,
RTX 5070 Laptop GPU, CUDA 13.2 / cuBLAS 13.4, median of 5 trials.

| form | result | outcome |
|---|---|---|
| inner dot | today's `vecdot` runs on one block: 7 % of the copy ceiling; the same dot as a `1 x k @ k x 1` GEMM on the chain reaches 92–97 % | implement as a GEMM (no driver code) |
| gemv | chain = raw `?gemv` within 3 % on all 22 large cells, batched identical | dropped (D6 withdrawn) |
| syrk | 1.8–2.0x (`f64`), 1.2–3.0x (`f32`), 1.8–2.8x (`c32`) above a size threshold; `f32` loses below `n = 2048` | implement with an `f32` size gate (D11) |

## What is left

- Grill closed 2026-11-26 (`DECISIONS.md`, D1–D13).
- **B1 committed**: `bc4c8d9` (alignment fix), `a7c65aa` (inner dot as a GEMM). `B1-REPORT.md`, diff in
  `patches/`. All gates pass. Found and fixed: cuBLASLt reads 16-byte `c64` scales with
  aligned loads; a stack local is aligned only by chance.
- **B2 (`a @ a.t()` as `?syrk`) implemented, not committed**: `B2-REPORT.md`, diff in
  `patches/b2_syrk.patch`. All gates pass. Routing checked by trace.
- **B2 committed**: `196e348`.
- **B3 (matrix, bench, docs) implemented, not committed**: `B3-REPORT.md`, diff in
  `patches/b3_matrix_bench_docs.patch`. All gates pass. **The benchmark showed the B2 gate wrong
  for thin `f32` operands (form up to 2.3x slower).** Settled: no size gate; the new flag
  `MatMulUseSyrk { Allow, Scientific, Disallow }` (default `Scientific`) selects the types.
- **B3 committed** (two commits on `261021/cuda`, see `git log` of rstsr). All three batches of the forms feature are in rstsr.
- **Loose ends found by a final review**: `MatMulUseSyrk` was missing from `prelude_dev`, and
  three dot items of the plan's test list (mixed dtype on the naive driver, a large `k`, the
  rank-0 output next to `[1]` / `[1, 1]`) were not written. Both fixed in the working tree
  (`src/lib.rs`, `tests/test_matmul_forms.rs`; gates in `results/b4_*`); committed as `67a0c79`.
- Possible follow-ups (not planned): batched syrk (no cuBLAS entry), a single-kernel mirror fused into the form, rank-0 CUDA reductions on one block.
- Known and left as is by the owner: rank-0 CUDA reductions outside matmul (whole-tensor `sum`,
  norms) run on a single block, the same cause as the slow dot.
- Deferred, no interface cost: gapped (non-affine) batches through the pointer-array cuBLAS
  entries; batched syrk (no cuBLAS entry).

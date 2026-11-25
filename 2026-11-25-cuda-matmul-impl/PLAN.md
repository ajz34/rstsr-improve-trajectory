# Plan — CUDA matmul implementation

Batch plan for the decisions in `DECISIONS.md` (D1-D28). Work happens on the current head of
`261021/cuda` in the rstsr repo (D28): **no new branch**, and **no commit or push in the rstsr repo
without the owner's go per batch**. Records (reports, numbers, copies of sources) go in this
directory.

Each batch is self-contained enough to test on its own: the crate's existing suites must stay green
after every batch.

## Batches

| batch | content | acceptance | status |
|---|---|---|---|
| **B1** | Tiled-kernel check (D20): correctness vs CPU over the layout matrix, timing vs cuBLAS, verdict per refusal. No change to the kernel. | `TILED-CHECK.md` written | **done 2026-11-25** (sources in `src/` here only) |
| **B2** | `src/matmul/` skeleton and the move (D8/D19); naive driver (D11/D12/D23); flags refactor (D16/D17/D25). Behaviour-preserving for f32/f64. | existing suites green; `test_matmul_tiled*` ported from the env var to the flags; naive passes the B1 layout matrix for f32/f64/integers | **done 2026-11-25** (`B2-REPORT.md`; rstsr `c75aecd`) |
| **B3** | Compute modes with the cuBLASEx and classic drivers (D2/D3/D9/D18/D24): `ComputeType32F` consumed, `ComputeType64F` incl. fixed-point emulation through the classic handle, mode resolution, the static driver table. | host tests for mode resolution and the table; hardware test per `(triple, mode)` for f32/f64 | **done 2026-11-25** (`B3-REPORT.md`; rstsr `493a42f`) |
| **B4** | cuBLASLt driver (D6): `cublaslt` feature, per-ordinal handle + workspace, heuristic cache, descriptor construction. | host tests for descriptors; Lt first in the chain for f32/f64; Lt alone passes the hardware tests | **done 2026-11-25** (`B4-REPORT.md`; rstsr `554f956`; workspace per stream) |
| **B5** | Half and complex triples (D1), tensor-level entries (D15). `ext_matmul` (D14) withdrawn by the owner before the batch: mixed operands go through `matmul_from`/`matmul_with_output` on the naive kernel. | f16/bf16/c32/c64 hardware tests; `c_f32.matmul_from(&a_f16, &b_f16, ..)` and `a % b` (f16) both work; the `*_mode` entries against each other | **done 2026-11-25** (`B5-REPORT.md`; rstsr `85ed399`) |
| **B6** | Full test matrix (D26), `bench_matmul` before/after (D27), crate docs, numbers in this README. The bench is the crate's `harness = false` median-of-R style, not criterion. | README with the tables; crate-level doc updated | **done 2026-11-25** (`B6-REPORT.md`; rstsr `3508d3a`; the matrix found and fixed a `bf16` compile failure of the naive kernel) |
## B2 in detail (done)

Deviations from the list below are recorded in `B2-REPORT.md`: `tensor.rs`, `cublas_lt.rs` and
`cublas_ex.rs` are left to their batches instead of being created empty; `geom.rs` returns the
batch parts and leaves every refusal to the drivers (naive needs the per-axis strides); a
regression suite `tests/test_matmul_naive.rs` was added to the rstsr tree.

1. Create `src/matmul/` with the layout of D19; `mod.rs` holds docs, `mod` lines and re-exports only.
2. Move, unchanged in behaviour: `kernels/gemm.rs` geometry -> `geom.rs`; `kernels/cublas_gemm.rs`
   argument derivation -> `cublas_args.rs`; the typed cuBLAS launch -> `drivers/cublas.rs`; the
   tiled kernel -> `drivers/tiled.rs`. Delete `kernels/{gemm,cublas_gemm}.rs`.
3. `drivers/naive.rs`: the universal kernel (one thread per output, loop over `k`, `@TA@ @TB@ @TC@
   @TACC@`, full layout addressing). It takes every refusal the tiled kernel makes.
4. `drivers/mod.rs`: `GemmDriver` enum, the ordered candidate list (at this point: cuBLAS, tiled,
   naive), filtered by the enabled set; `inner.rs`: `matmul_with_mode_inner` (mode argument
   present but only `Default` resolved until B3) and the `DeviceMatMulAPI` impls.
5. `src/flags.rs`: `CudaRuntimeFlags` `#[non_exhaustive]` with `matmul_drivers`, `trace_matmul`,
   the new compute-type fields; `matmul/flags.rs`: `MatMulDrivers`. Remove `RSTSR_CUDA_MATMUL`,
   `RSTSR_CUDA_DEBUG_GEMM`, `RSTSR_CUDA_COMPUTE_TYPE_32F`. Add `DeviceCuda::with_flags`.
6. Port `tests/test_matmul_tiled.rs`, `tests/test_matmul_tiled_check.rs` and
   `examples/bench_matmul_check.rs` to the flags (driver forced by the enable set).
7. Run: `--lib`, all hardware suites, the naive layout matrix. Report in this directory.

## Out of scope (recorded, not planned)

- FP8 GEMM (separate decision; rstsr has no fp8 dtype).
- A hand-written tensor-core kernel (D10).
- Mixed-operand GEMM on the accelerated drivers (a "promote-then-GEMM" path); `ext_matmul` runs naive.
- Tiled-kernel improvements (D30, deferred by the owner): accept a unit stride of `-1` in `lane_axis`
  and the arbitrary-lane fallback (both verified correct, `TILED-CHECK.md` 2b); per-axis batch
  strides (untested). B2 moves the tiled kernel **unchanged**.
- `tensordot` (D29): deferred entirely, to be implemented with cuTENSOR. The existing interim impl
  is left untouched and only inherits what `DeviceMatMulAPI` gives it.

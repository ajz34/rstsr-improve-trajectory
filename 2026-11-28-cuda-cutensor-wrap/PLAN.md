# PLAN — a safe cuTENSOR wrapper in `rstsr-cuda` (`src/cutensor_wrap`)

Decisions: `DECISIONS.md` (D1-D23). Target: `rstsr` repo, crate `crates-device/rstsr-cuda`, branch
`261021/cuda` (at `937ce04`, checked out, clean). No commit in `rstsr` without the owner's instruction;
each batch ends with a report (what landed, test output, open points) and waits for "go" before the next.

## Local setup (once)

- `libcutensor.so` symlink dir on `LD_LIBRARY_PATH` for tests and benches (cudarc's candidate list never
  names the wheel's `libcutensor.so.2`): `/scratch/rstsr_pack/rstsr-improve-trajectory/2026-11-27-cuda-einsum-cutensor/probe/lib`
  (-> 2.7.0) already exists; `2026-11-28-cuda-cutensor-wrap/probe/versions/lib<V>` for the others.
- Commands (nvcc 13.2 is on the machine, so `CUDARC_CUDA_VERSION` resolves itself):
  `cargo test -p rstsr-cuda --lib`, `cargo test -p rstsr-cuda --test test_cutensor_wrap`,
  `cargo bench -p rstsr-cuda --bench bench_cutensor_wrap`; `cargo clippy -p rstsr-cuda --all-targets`.
- `/tmp` has a per-user quota (`TMPDIR=$HOME/.cache/tmp-cargo` if phantom failures appear).

## B1 - wiring and foundation

Files: `Cargo.toml`, `build.rs`, `src/lib.rs`, `src/registry.rs`, `src/stream.rs`, `src/flags.rs`,
new `src/cutensor_wrap/{mod.rs, handle.rs, error.rs, compute.rs, workspace.rs, dtype.rs}`.

1. `Cargo.toml`: add `"cutensor"` to the cudarc features.
2. `build.rs`: `cargo:rustc-check-cfg=cfg(cutensor_dynamic_loading)`; emit it unless
   `CARGO_FEATURE_DYNAMIC_LINKING` is set; unit test in the existing `tests` module.
3. `compute.rs` (D14): the nine accessors; symbol source per cfg; missing symbol -> `UnImplemented`
   (no panic); `OnceLock` per descriptor.
4. `error.rs` (D17): `Stage` enum, `status_error(stage, status, ctx) -> Error` with the variant table,
   `is_decline(&Error)`, `not_loadable() -> Error`; units for the table.
5. `handle.rs` (D5, D6): `CutensorHandle { ptr, lock: Mutex<()>, version }`, `new(&Arc<CudaStreamObj>)`
   (`bind_to_thread` -> `cutensorCreate` -> `cutensorGetVersion`), `Drop` -> `cutensorDestroy`,
   `version()`, `lock()`; `unsafe impl Send/Sync` with the SAFETY note (header remarks);
   `CudaOrdinalObj::cutensor()` = `OnceLock<Option<Arc<CutensorHandle>>>` + `catch_unwind` probe
   (`is_culib_present` under the cfg; linked -> always present).
6. `dtype.rs` (D11): `pub trait CutensorDType: CudaDType + Copy { type Scalar: CutensorScalar; const
   DATA_TYPE: sys::cudaDataType_t; }` + `CutensorScalar { const DATA_TYPE }` for f32 / f64 /
   `Complex<f32>` / `Complex<f64>`; impls for the six types; `AlignedScale<T>` storage; units.
7. `workspace.rs` (D15): `CutensorWorkspace` (owning `CudaBuffer<u8>`, `new`, `len`, `as_ptr`),
   `CudaRuntimeFlags::cutensor_workspace_bytes: usize` (default `128 << 20`) + the `static` default,
   `CudaStreamObj::cutensor_workspace()` lazy slot.
8. `lib.rs`: `pub mod cutensor_wrap;` + module list line.
Done when: `cargo test -p rstsr-cuda --lib` green (incl. the new units), `cargo build -p rstsr-cuda`
green, clippy clean.

## B2 - contraction end to end (committed as rstsr `8a84c9d`)

Files: `src/cutensor_wrap/{descriptor.rs, operation.rs, plan.rs, oneshot.rs (or in mod.rs)}`,
`tests/test_cutensor_wrap.rs`.

1. `descriptor.rs` (D8, D9, D12): `TensorDescriptor::new / with_alignment(handle, &TensorView<IxD>,
   is_output: bool)`; refusals (negative stride, stride-0 output, extent 0, `numModes <= 64`), rank-0
   dummy pointer, alignment rule; stores `(extents, strides, dtype, alignment)` for re-validation;
   units for the pure checks.
2. `operation.rs` (D8, D11): `OperationDescriptor::new_contraction(handle, &descA, modeA, opA, ..,
   compute)`: mode consistency (lengths, duplicates, D modes in A or B, one extent per label),
   `op_c == IDENTITY`, create, then the `SCALAR_TYPE` verification against `T::Scalar`; `Drop`.
3. `plan.rs` (D1, D6, D12, D15): `PlanPreference::new(handle, algo, jit)`; `Plan::new(handle, &op,
   &pref, ws_limit)` under the handle lock (estimate `WORKSPACE_DEFAULT` -> create plan), `Plan`
   stores `Arc<CutensorHandle>`, `required_workspace()`, the operand layout records;
   `Plan::contract(&self, device, a, b, c: Option<_>, d, alpha, beta, ws)` = re-validation (D1) ->
   ordering (D7) -> `cutensorContract` -> `order_output`.
4. `oneshot` (D1, D5, D7, D15, D16): `contract(device, a, mode_a, op_a, b, mode_b, op_b, c, d, mode_d,
   alpha, beta, compute, workspace)`; handle from the ordinal slot (`None` -> `UnImplemented`);
   `c == None` requires `beta == 0`; workspace resolution (D15).
5. Tests (`tests/test_cutensor_wrap.rs`, D21), contraction half: `require_cuda`-style guard + a
   libcutensor-loadable guard, both printing why they skip; print `cutensorGetVersion`; GEMM-shaped
   `ij,jk->ik` for `f32 f64 c32 c64 f16 bf16` vs the CPU `rt::matmul` oracle (f16/bf16 via f32/f64
   data) and vs the crate's cuBLAS chain; batched `bij,bjk->bik`; strided A / B / D views; `C == D`
   accumulate; `beta = 0` with `c = None`; `op_a = CONJ` on c64; rank-0 output `ij->` (wait: that is a
   reduction -- here: `i,i->` is a contraction of a vector with itself; keep rank-0 via `ij,jk->` is not
   possible; use `i,i->`), diagonal-by-view; refusals (negative stride, stride-0 output, extent 0,
   `op_c = CONJ`, duplicate mode, D mode missing, extent mismatch, `c` layout != D, `beta != 0` with
   `c = None`, static-dim passed after `to_dim`); `f32 x c32` -> `UnImplemented` at plan (decline);
   plan re-executed with a changed layout / lower alignment -> refused; execute on a second stream.
Done when: all of the above pass on 2.7.0 (and the file also runs against 2.8.0 via `LD_LIBRARY_PATH`);
`--lib` green; clippy clean.

## B3 (implemented, uncommitted) - the other five operations

Files: `operation.rs`, `plan.rs`, one-shots, tests.

1. `new_contraction_trinary` (+ `Plan::contract_trinary`, `contract_trinary`), `new_reduction`
   (`op_reduce`, `Plan::reduce`, `reduce`: `ij->j`, `ij->i`, `ijk->k`, full reduction to rank 0,
   `beta` accumulate), `new_permutation` (`Plan::permute`, `permute`: `ij->ji`, `ijk->kij`,
   broadcast by omitted extent-1 mode, identity as strided copy), `new_elementwise_binary`
   (`D = opAC(alpha opA(A), gamma opC(C))`, `Plan::elementwise_binary`, one-shot), `new_elementwise_trinary`
   (`D = opABC(opAB(alpha opA(A), beta opB(B)), gamma opC(C))`; `c = None` only with `gamma == 0`).
2. Per-kind mode consistency: D modes subset of A (reduce / permute / elementwise), of A or B (trinary
   contraction: E modes in A or B or C), one extent per label.
3. Tests per operation vs a CPU oracle (`rt::sum`, `transpose().to_contig()`, `rt::add` / `mul`,
   `a*b+c`), complex `CONJ` on reduce / permute, refusals, stride-0 input pass-through on 2.7.0
   (accepted) with a note that 2.4.1 declines.
Done when: all pass; `--lib` green; clippy clean.

## B4 - bench and docs

1. `benches/bench_cutensor_wrap.rs` (`harness = false`): one-shot vs held plan per operation at the
   probe shapes (f64/f32 2048^3, c64 1024^3, batched 64x(256,256), `abc,cbd->ad` non-GEMM), planning
   cost (first plan / repeat), cuTENSOR vs the cuBLAS drivers on GEMM shapes; prints the cuTENSOR
   version.
2. Facade (D19): `pub use rstsr_cuda::cutensor_wrap;` in `rstsr/src/cuda.rs` and inside `rt::cuda` in
   `rstsr/src/prelude.rs`; extend the facade's `test_cuda_prelude` with one path check.
3. Docs: module docstring (pipeline, cost model, safety policy D8-D12, no `_f` deviation, local
   symlink requirement, how to run against another version); `lib.rs` module list line, "What is
   implemented" row, "Running the tests" line.
4. Record the bench numbers in this directory (`README.md`), update the memory index, final report.
Done when: `cargo test -p rstsr-cuda` and `cargo test -p rstsr --features cuda --lib` green, bench
runs, docs build (`cargo doc -p rstsr-cuda --no-deps`).

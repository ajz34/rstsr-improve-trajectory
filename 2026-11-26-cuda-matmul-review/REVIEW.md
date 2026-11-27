# Review: rstsr-cuda matmul, `c75aecd..67a0c79` (`261021/cuda`)

Two-axis review (`code-review` skill: Standards, Spec) plus a correctness pass driven by
probes on the GPU. The rstsr tree was not modified. Everything below was read in the final
sources and, where marked **verified**, reproduced with `probe/tests/probe.rs`
(raw output in `results/`).

Severity: **critical** = silent wrong results or stale data on the default path; **high** =
wrong behaviour on a documented configuration; **medium** = a promise the code does not keep,
or a performance trap; **low** = robustness, documentation, leftovers.

Status 2026-11-26 (same day): **C1 and C2 are fixed**, uncommitted on `261021/cuda`
(9 files, +214/-23): both `args()` functions refuse `stride <= 0` with extent over 1; every
matmul driver orders `c` as an input before the enqueue and its allocation stream behind the
enqueue. Regression tests: unit tests in `cublas_args.rs` and `syrk.rs`, and
`broadcast_along_matmul_axis` + `inplace_cross_stream_is_ordered` in `tests/test_matmul_modes.rs`.
Suite green (109 lib + 8 hw binaries); probe rerun: 0 `FAIL`, T5 0 of 30 stale on all three
drivers, T9/T11 correct on the chain and refused when a cuBLAS driver is forced alone.

## Summary

| # | Sev. | Finding | Where | Verified |
|---|---|---|---|---|
| C1 | critical | An operand broadcast along a matmul axis (stride 0, extent > 1) gives silently wrong results on every cuBLAS driver and on the default chain (out-of-bounds reads) | `cublas_args.rs:81-87, 65-69`; `drivers/syrk.rs:44, 47`; promised at `matmul/mod.rs:55` | yes (T9, T11) |
| C2 | high | In-place matmul across two streams: the host read-back sees the state before the product, systematically (the matmul drivers order the output **before** the launch; the elementwise kernels do it after) | `drivers/cublas.rs:493`, `syrk.rs:106,170`, `tiled.rs:269`, `naive.rs:208` | yes (T5) |
| C3 | medium | **FIXED (261021/cuda, uncommitted)** The symmetric form ignores the compute mode: under `EmulatedFixedPoint`, `a @ a.t()` is 3.3x slower than `a @ b`; `Pedantic` is not applied; the documented `fixedpoint_mantissa_bits` error is bypassed | `drivers/syrk.rs:66-80`; `drivers/mod.rs:421-434, 471` | yes (T3) |
| C4 | medium | **FIXED (261021/cuda, uncommitted)** `build.rs` never recognises a pinned `CUDARC_CUDA_VERSION` (parses 6 characters; cudarc spells 5), so any pinned build — including the `=13000` recommended for the emulation — has neither cfg | `build.rs:36-45`; docs `mode.rs:52`, `Cargo.toml:39` | yes (`results/cfg_*`) |
| C5 | low | **FIXED (261021/cuda, uncommitted)** `[0] @ [0]` fails with `CUBLAS_STATUS_INVALID_VALUE` when only `cublasEx` or classic is enabled (`ld` = 0) | `cublas_args.rs:65-69` | yes (T4) |
| C6 | low | **FIXED (261021/cuda, uncommitted)** A leading dimension over `i32::MAX` is clamped instead of refused (silent wrong addressing) | `cublas_args.rs:68`, `syrk.rs:47` | by reading (needs > 8 GiB) |
| C7 | low | **FIXED (261021/cuda, uncommitted)** The two batch `cublasLtMatrixLayoutSetAttribute` statuses are dropped | `cublas_lt.rs:281-282` | by reading |
| T1 | medium | The suite's complex matmul data is purely real (`Complex::new(x, 0.0)`, `.re` compared): `cublasEx` and classic never run with imaginary parts | `tests/test_matmul_modes.rs` | yes (probe T1: 319 cells correct) |
| T2 | medium | The suite never runs the symmetric form on a non-base offset or `lda != k`: `syrk_views` copies to contiguous (its comment says otherwise); the layout-matrix "syrk" row is a copied transpose, i.e. a fall-through case | `tests/test_matmul_forms.rs:75-76`, `test_matmul_modes.rs:637` | yes (probe T2: 108 cells correct) |
| T3 | low | No test observes routing (`trace_matmul`); `syrk_applies` has no host test | `tests/` | — |
| A1 | medium | `matmul_with_output_mode` and `matmul_from_mode` have no `_f` twin; `matmul_with_mode` (method) takes `&TensorAny` while its two siblings take `impl TensorViewAPI` | `matmul/tensor.rs:30-66` | — |
| A2 | low | Seven docstrings state things the code does not do (list in the section) | several | checked in the sources |

What is correct: the driver table and chain order, slot swap and `OP_*` flags under both
orders, complex products with imaginary parts on every driver, batch and broadcast batches,
reversed batch axes, the inner dot (including `k = 0, 1`), the symmetric form on offsets,
both operand orientations, `k = 0`, `n = 1`, `k = 1`, `alpha = 0` and strided output views,
empty products, the per-call-mode entries, `AlignedScale`, the math-mode set/restore under
the handle lock, and the cuBLASLt cache key. The crate's suite is green (242 tests).

## C1 (critical, verified): broadcast along a matmul axis

`cublas_args::args` refuses a negative stride with extent > 1 but not a **zero** stride with
extent > 1, and `ld()` raises a zero leading dimension to the declared row count:

```rust
// cublas_args.rs:81-87                     // cublas_args.rs:65-69
for (stride, extent) in [(sam, m), (sak, k), (sbk, k), (sbn, n), (scm, m), (scn, n)] {
    if stride < 0 && extent > 1 { return None; }
}
let clamped = s.max(rows as isize);
```

The guard's comment says "an extent-1 axis may legitimately carry stride 0", and the doc of
`ld()` builds on it: "a broadcast axis (extent 1, stride 0) gives a leading dimension of 0,
which cuBLAS rejects; clamping to the declared row count is safe because that axis has only
index 0". Nothing enforces extent 1, so the clamp turns a rejected call into an accepted one
with wrong addressing. Take `b = asarray([1., 2., 3., 4.]).into_shape([1, 4]).broadcast_to([3, 4])`
(strides `[0, 1]`, so `sbk = 0`, `sbn = 1`): slot A takes the `sbn == 1` arm, `lda_raw = 0`,
`rows = n = 4`, and is declared with `lda = 4`; cuBLAS then reads `b[i + 4 j]` for `i < 4`,
`j < 3` (up to index 11 of a 4-element buffer) instead of `b[i]`.

Probe T9 (`results/probe.log`), four shapes (`b` broadcast along `k` or `n`, `a` along `m` or
`k`), `f64` and `c64`, both default orders:

| drivers | result |
|---|---|
| cuBLASLt, `cublasEx`, classic, default chain | all 64 cells wrong (6 to 12 of 8 to 12 elements) |
| tiled, naive | all 24 cells correct (they address by the true strides, but the chain never reaches them) |

Probe T11 (`results/probe.log`, first four `FAIL` lines): a column `[n, 1]` broadcast along `k`
(strides `[1, 0]`) times its own transpose satisfies `is_syrk`; `syrk::args` (line 47) builds
the same clamped `lda`, and all 1369 elements are wrong on classic alone and on the chain.

Origin: pre-existing in `cublas_args.rs` (unchanged since the fixed point, where only the
classic driver used it). The change under review extends it to the two new cuBLAS drivers
through `cublas::prepare`, repeats it in `syrk.rs`, and states at `matmul/mod.rs:55` that zero
strides are served.

Fix (two lines plus a test): refuse `stride == 0 && extent > 1` in both `args()` functions, and
add the four T9 shapes and the T11 shape to the layout matrix for each cuBLAS driver.

## C2 (high, verified): cross-stream in-place matmul reads stale

`order_output` records its event **when called** (`kernel.rs:607-609`) and its own doc says
"required **after** an op writes into an existing buffer: host reads only synchronize that
buffer's allocation stream". Every matmul driver calls it before anything is enqueued:
`prepare` (`cublas.rs:493`, shared by cuBLASLt, `cublasEx`, classic), `syrk.rs:106` and the
mirror kernel `syrk.rs:169-170`, and the pre-existing `tiled.rs:269`, `naive.rs:208`. The
allocation stream therefore never waits for the product, and `to_cpu_vec` (`transfer.rs:69-81`,
copy and sync on the allocation stream) reads the old contents.

Probe T5: output allocated on the default stream and re-tagged onto a second stream
(`into_device`), `rt::matmul_from(&mut c, &a, &b, 1.0, 1.0)` in a loop (n = 1536, 30
iterations), read back after each call:

| drivers | iterations whose read-back is stale |
|---|---|
| one stream (control) | 0 of 10 |
| default chain, cuBLASLt, tiled (two streams) | 30 of 30 each, all 2 359 296 elements one update behind |

Origin: the fixed point's classic driver (`cublas.rs:297-299` there) and the tiled and naive
drivers already did this; the new drivers copy them. The elementwise and advanced-indexing
kernels have it right (`kernels/elementwise.rs:497-498` and `kernels/adv_index.rs:270` call
`order_output` after `launch()`), so matmul is the exception, and the invariant is on record
(`cuda-write-target-ordering`: a re-tagged buffer showed stale reads without two-way ordering).
With `beta != 0` the product also reads `c` unordered (`prepare` never calls `order_input` for
`c`; only a still-pending allocation is covered, by `read_ptr_on`). Only the fresh-output
entries (`rt::matmul`, `matmul_uninit`) are safe: their output is allocated on the executing
stream.

Fix: move the three `order_output` calls (and the mirror kernel's) after the enqueue, and
`order_input(c)` when `read_c`; regression test = probe T5. The stream mix is the crate's
documented re-tag scenario (`set_stream` + `into_device`), so this is not exotic.

## C3 (medium, verified): the symmetric form ignores the compute mode

`syrk::run` dispatches on the plan's type only (`CublasPlan::F32(_)`, `F64(_)`) and calls
`?syrk_v2` with the handle in its default math mode, while `cublas::launch` sets and restores
the mode for `gemm`. `dispatch` returns as soon as the form runs, before the chain sees the
mode.

Probe T3 (n = k = 1024, `f64`, this GPU, mean of 5 launches):

| flags | `a @ b` | `a @ a.t()` | trace |
|---|---|---|---|
| `EmulatedFixedPoint`, `MatMulUseSyrk::Scientific` (default) | 1.044 ms | **3.449 ms** | `cublas (syrk) -> ran` |
| `EmulatedFixedPoint`, `Disallow` | 1.045 ms | 1.010 ms | `cublas -> ran` |
| `Default` or `Pedantic`, `Scientific` | 6.419 ms | 3.448 ms | `cublasLt`, `cublas (syrk)` |

With the fixed-point emulation requested, `a @ a.t()` runs native `Dsyrk` and is 3.3x slower
than the `a @ b` it "optimizes". `Pedantic` is likewise not applied on the form. And the
documented error of a non-zero `fixedpoint_mantissa_bits` (`cublas.rs:280-287`) is bypassed:
`a @ b` returns the error, `a @ a.t()` returns a result (probe T3, last two lines).

Fix: take the form only for a plan whose mode is `Default` (and `fixedpoint_mantissa_bits ==
0`), or set and restore the math mode inside the syrk call as `launch` does.

## C4 (medium, verified): `build.rs` never parses the pinned version

cudarc 0.19.10 spells the variable `format!("{major}0{minor}0")`: `"12060"`, `"13000"` — five
characters. `build.rs:38` requires `v.len() == 6`, so every value falls through to
`return None` (line 44), which skips the nvcc fallback too, and neither `cublas_math_mode`
nor `cublas_fixedpoint_math` is set. Verified with `cargo rustc -p rstsr-cuda --lib -- --print
cfg`: `results/cfg_CUDARC_CUDA_VERSION_13000.txt` has neither cfg;
`results/cfg_nvcc_13_2.txt` (variable unset, nvcc 13.2 detected) has both. A pin of `12080`
also builds, without the two cfgs (expected for that toolkit, wrong for the installed one).

The gating itself is right: cudarc compiles `cublasSetMathMode` only from `cuda-12090` and
`CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH` only from `cuda-13000` (`src/cublas/sys/mod.rs:5429`
and `:221`). The thresholds in `build.rs` mirror that. What is wrong is only the pin: with
`CUDARC_CUDA_VERSION=13000` — the value `mode.rs:52` gives for the emulation — the classic
driver refuses `Pedantic`, `FastTF32`, `Fast16F` and `EmulatedFixedPoint` with "this build's
cudarc bindings have no cublasSetMathMode (CUDA 12.9 or newer needed)", although the
bindings are there. The CI pin `12060` hides the bug (no cfg is the right answer for 12.6).

Fix: accept cudarc's spelling (`v.len() == 5`, major one digit, minor two), and add a line
to the CI matrix that pins `13000` and checks the emulation runs.

### C3 fixed (261021/cuda, uncommitted)

Implemented with a measurement first, because the review's two candidate fixes differ in one
observable: **cuBLAS's `?syrk` entries ignore the math modes that change the arithmetic.** A
`cublasSsyrk_v2` under `CUBLAS_TF32_TENSOR_OP_MATH` measures `5.7e-7` max rel err — plain `f32`
arithmetic, identical to `Default` (the same run's classic `gemm` under `FastTF32` measures the
TF32 scale, so the set/restore mechanism itself works). So "set and restore the mode around the
syrk" alone cannot honor a down-converting mode; the form must decline it.

- `cublas.rs`: the ask machinery (`Ask`, `ask_32f`, `ask_64f`, `bound`) is now `pub(super)`, and
  the set/restore dance of `launch` moved into `pub(super) with_math_mode(blas, math, call)` —
  one place that takes the handle lock, sets the mode, runs the call, restores.
- `syrk.rs`: `run` computes the same ask as `cublas::run` (mantissa bits included, so the
  documented error fires identically from the form), then takes the form only for
  `Ask::Leave` or `Ask::Math(Pedantic)` — the typed `?syrk` is exact arithmetic, so `Pedantic`
  rides along (measured 3.46 ms vs the pedantic `gemm`'s 6.44 ms; the form is kept). Any other
  mode declines with "the syrk entries do not apply a math mode that changes the arithmetic",
  and the chain serves the mode on a `gemm`.
- Probe T3 after (`results/t3_after_c3c4.txt`): EmulatedFixedPoint `a @ a.t()` now runs the
  emulated `gemm` at **1.013 ms** (was 3.449 ms — the 3.3x loss is gone); Default/Pedantic keep
  the form at 3.460 ms; `fixedpoint_mantissa_bits = 30` errors identically on `a @ b` and
  `a @ a.t()` (was: a silent result). Full probe unchanged otherwise: 497 ok, 0 FAIL, the same
  28 intentional forced-tiled-complex ERRs (`results/probe_after_c3c4.log`).
- Regression test `syrk_honors_the_compute_mode` (`tests/test_matmul_modes.rs`): the mantissa
  error parity, and forced-classic `a @ a.t()` under `FastTF32` measuring `6.5e-5` — TF32
  accuracy, where the pre-fix form measured `5.7e-7` (`Default` silently).

### C4 fixed (261021/cuda, uncommitted)

`build.rs` now parses cudarc's actual spelling (`format!("{major}0{minor}0")`: two-digit major,
literal zero, one-digit minor, literal zero — `12060` is 12.6, `13000` is 13.0), mirroring
cudarc 0.19.10's own whole-string match (`build.rs` of cudarc: `detect_version_from_env`). The
parse is unit-tested (`cudarc_spelling_parses`, `other_spellings_are_refused`); because `cargo
test` does not execute `build.rs` as a target, the build script is included into the crate as a
`#[cfg(test)] #[path = "../build.rs"] mod build_script` of `lib.rs` so the tests actually run
(111 lib tests).

Verification (`results/cfg_*_after_fix.txt`): `CUDARC_CUDA_VERSION=13000` now yields **both**
cfgs (was neither); `12060` yields neither (correct for 12.6 — this is what hid the bug);
unset (nvcc 13.2) yields both. The whole hardware suite passes on the `13000` pin, including
the classic math-mode arms that the pin used to refuse; `--print cfg` checks run for both pins
in CI: `tier3-cuda.yml` now has a matrix over `cudarc_cuda_version: [12060, 13000]` with a step
asserting the cfg set of each pin.

## C5 to C7 (low)

- C5 (verified): for `[0] @ [0]` the lifted `a` is the `1 x 0` slot B with strides `[0, 1]`,
  `ldb = ld(0, 0) = 0` (the re-check at `cublas_args.rs:118-120` covers `ldc == 0` only), and
  cuBLAS answers `INVALID_VALUE` ("GEMM_EX parameter number 12", "DGEMM parameter number
  10"). On the chain cuBLASLt takes it; with `cublasEx` or classic alone it is an error instead
  of `0.0`. `ld()` needs `.max(1)`.
- C6: `ld()` and the copy in `syrk.rs:47` clamp a stride over `i32::MAX` to `i32::MAX`
  although the module doc promises a refusal ("a dimension over `i32`"). Reachable with an
  extent-2 axis whose stride exceeds 2^31 elements (an `f16` operand of 8.6 GiB), not on this
  GPU; silent wrong addressing when it is.
- C7: `cublas_lt.rs:281-282` drop the status of the two batch attributes while every sibling
  `SetAttribute` goes through `refused(..)`; a failure would leave a batch-1 layout and an
  unwritten rest of the output. Not observed.

### C7 fixed (261021/cuda, uncommitted)

The two batch attribute sets go through `refused(..)` like every sibling: a failure is now a
decline (falling to the next driver) instead of a silently batch-1 layout. `build` also takes the
derived `CublasArgs` directly instead of the whole `GemmCall` (it only ever used `.args`), which
is what makes the new test constructible.

The failure itself is not triggerable through any public path — a valid attribute on a valid
layout fails only on a broken library — so there is no direct regression test for the check.
What the fix protects is instead pinned: `batch_attributes_read_back` (a hardware-guarded lib
test, skipped without a device) builds a batch-3 entry and reads `BATCH_COUNT` and
`STRIDED_BATCH_OFFSET` back out of all four layouts, expecting the `batch_strides` triple by
cuBLASLt's slots. A refactor that drops or mis-orders the sets (the precondition of the dropped
status mattering at all) fails there. Probe unchanged after: 499 ok, 0 FAIL, the 28 intentional
forced-tiled-complex ERRs (`results/probe_after_c7.log`).

### C5 and C6 fixed (261021/cuda, uncommitted)

One rule change in both derivations, one commit: **a declared leading dimension floors at 1, and
a stride that does not fit `i32` is refused rather than clamped.**

- `ld()` (`cublas_args.rs`) and the `ld` closure (`syrk.rs`) now end in `.max(1)`: cuBLAS rejects
  `ld = 0` even for an empty matrix (`ld >= max(1, rows)`), so the `[0] @ [0]` dot — whose lifted
  `1 x 0` operand declared `ldb = ld(0, 0) = 0` — now serves `0.0` on every driver set. The dead
  `ldc == 0` arm of the `args` re-check (its only live case was the same empty-output clamp) is
  removed with it.
- Both guard loops refuse `stride > i32::MAX as isize`: the clamp to `i32::MAX` was silent wrong
  addressing (the module doc's promised refusal). Unreachable on this GPU (an extent-2 axis over
  2^31 elements is an 8.6 GiB `f16` operand), so the regression tests are unit-level on the pure
  derivations: `empty_dots_floor_and_wide_strides_refuse` in both files (the wide-stride case
  must be `None`; an in-range stride of the same shape is declared as is).
- Hardware regression: `degenerate_dots` (`tests/test_matmul_modes.rs`) — `[0] @ [0]` and
  `[3] @ [3]` exact on every driver set. Verified against the pre-fix source: it fails there with
  exactly the documented `CUBLAS_STATUS_INVALID_VALUE` ("GEMM_EX parameter number 12").
- Probe T4 after (`results/probe_after_c5c6.log`): all six `k = 0` rows print `ok` (dot = 0),
  `499 ok` / `0 FAIL` / `28 ERR` (the intentional complex-tiled refusals; the two `k = 0` ERRs
  are gone). Full gate green.

## Test coverage

- T1 (verified): `matrix`, `modes_complex` and `inner_dot` build complex data with
  `Complex::new(x as f32, 0.0)` and compare `.re` only, so a conjugation error (`OP_C`
  instead of `OP_T`) would be invisible on `cublasEx` and classic. Only
  `test_matmul_forms.rs` uses imaginary parts (cuBLASLt through the chain, and `?syrk`).
  Probe T1 (c32, c64, f32, f64; 7 shapes; both orders; each driver alone and the chain; real
  and imaginary parts compared): 319 cells correct, the 28 "ERR" cells are the tiled driver's
  expected refusal of complex types.
- T2 (verified): `syrk_views` does `big.i(s![..n, ..k]).to_contig(order).to_owned()` — a
  copy, offset 0, `lda = k` — under a comment saying the offset is not the buffer base; the
  layout-matrix row named `syrk` is `square.t().to_owned()`, a transpose copied into another
  buffer, which is exactly a case the form must not take. Probe T2 (f64, c64, f32 under
  `Allow`; both orders; classic alone and the chain): offsets, both operand orientations,
  `k = 0`, `n = 1`, `k = 1`, `alpha = 0`, strided and transposed output views — 108 cells
  correct, 129 `cublas (syrk) -> ran` trace lines.
- T3: no test fails if the form silently stops being taken (a performance regression); the
  promised "trace names the routine" check was not written, and `syrk_applies` has no host unit
  test (unlike `syrk::args`).
- No test anywhere multiplies a broadcast operand on a cuBLAS driver (C1); `matmul_with_output_mode`
  had no test at all (probe T7: correct); the tiled driver is not in the type-triple matrix.

## API soundness

- A1: the public trait `TensorCudaMatMulAPI` has `matmul_with_output_mode` and
  `matmul_from_mode` as infallible methods with no `_f` twin, while their core models
  (`matmul_with_output`, `matmul_from`) have one; a caller cannot avoid a panic path. The
  third entry is a free `matmul_with_mode_f` whose infallible form is the method builder
  `.out::<TC>()` (no free `matmul_with_mode`), and that method takes
  `rhs: &'a TensorAny<RB, ..>` where its two siblings take `impl TensorViewAPI`.
- A2: docstrings at odds with the code:
  - `flags.rs:24, 26` "(read by `f32`-output calls only)" / "(read by `f64`-output calls
    only)": complex outputs and half operands with a real output read the same fields
    (`mode.rs` documents it correctly).
  - `tensor.rs:9` the `*_mode` entries "read nothing from the device flags": they still read
    `matmul_drivers`, `matmul_use_syrk`, `fixedpoint_mantissa_bits` and `trace_matmul`.
  - `mode.rs:16-17` "`Pedantic` is never downgraded and is an error": no error path exists;
    `drivers/mod.rs:283` says it is served by every triple.
  - `naive.rs:15-17` "refused by name before the chain": the rank limit is raised by the naive
    driver, the last of the chain (`matmul/mod.rs:8` "the chain is total").
  - `stream.rs:27` `launch` is "the only place that talks to cuBLAS": `cublas_ex.rs`,
    `cublas_lt.rs` and `syrk.rs` call cuBLAS too (all under the lock).
  - `lib.rs:125-126` "six argument shapes": eight since the dot and syrk rows.
  - `flags.rs:125-135` `MatMulUseSyrk::takes` answers true for every type but `f32` although
    the doc says `f64`, `c32`, `c64` only (the driver declines the rest).
- `CudaRuntimeFlags` is `#[non_exhaustive]`, so `CudaRuntimeFlags { matmul_drivers, ..
  Default::default() }` does not compile for users; `default()` plus field assignment is the
  only way (deliberate, D17, but worth knowing).

## Standards (sub-agent report, annotated)

Hard violations:

1. Benchmark numbers in docstrings — `rules/testing.md` rule 6 (numbers live in this
   repository): `src/matmul/mod.rs:48-50` ("1.3x to 3.6x on a RTX 5070 Laptop", "up to
   2.3x"), `src/matmul/flags.rs:116, 118`; `src/matmul/geom.rs:149` says "measured at
   the bandwidth ceiling". (The pre-existing `tiled.rs` header has the same kind of numbers.)
2. Bench placement — `rules/testing.md` names `crates-bench/` (future) as the home;
   `benches/bench_matmul.rs` and the `[[bench]]` entry keep it in the device crate, as the
   pre-existing `bench_elementwise` does. Judgement call until `crates-bench/` exists.
3. Co-author trailers placed **before** the body in the six commits of the forms wave
   (`bc4c8d9`, `a7c65aa`, `196e348`, `dae8198`, `63b694c`, `67a0c79`): `git
   interpret-trailers --parse` returns nothing for them, so GitHub will not show the
   co-authors. The five earlier commits are correct. Fixing means rewording unpushed history —
   the maintainer's call.
4. `macro_rules!` where the guide prefers `duplicate_item`: `impl_ex_scalar`
   (`cublas_ex.rs:115`, new), `impl_cublas_scalar` (`cublas.rs:98`, extended). `mode.rs`
   uses the crate's shape.
5. Fallible twins (see A1).
6. `if cond { rstsr_raise!(..)? }`: `cublas.rs:281`, `:381`, `cublas_lt.rs:358`. The guide
   prefers `rstsr_assert!` for "simple branch-condition"s; the last two format a runtime
   status into the message (judgement), the first is a plain flag check.
7. Hand-rolled index generation in tests (`test_matmul_forms.rs:125-140`, `:181-189`) where
   `code-concepts.md` asks for the layout iterators. Judgement call for plain `Vec` fixtures.
8. Plain `.unwrap()` in new in-crate unit tests (`cublas_lt.rs:496, 532, 628`, `syrk.rs:187-205`,
   `geom.rs:383-396`). `code-concepts.md:41-42` asks for `rstsr_unwrap()` inside the crate,
   written for the panic twins of fallible functions; the crate's existing unit tests use
   `unwrap()` too (judgement).

Judgement calls: duplicated code (`serve!` + `match plan` twice: `cublas_ex.rs:159-176` =
`cublas_lt.rs:394-411`; the scale preamble four times; `op` closures beside `cublas::op`;
`data()` in three files), `match plan` seven times, names (`bound`, two meanings of
`retyped`, `Buffers as Uniform`, `ACCELERATED = GemmDriver::ALL`), the 72-line module header
in `matmul/mod.rs` and the syrk condition stated in four places, `(TA, TB, TC, TBuf)` bounds
in eleven signatures. Leftovers: the debug block `test_matmul_modes.rs:300-324` (`eprintln!`
of numbers nobody asserts, `let _ = raw;`, a hard-coded `768`), `trace_matmul = true` left
on at `:248` and `:280`, stale shape counts (`lib.rs:125`, `test_matmul_modes.rs:712`).

Clean: MSRV 1.82 (nothing newer), English throughout, no decision IDs or references to this
repository, `//` comments within the length rule, branch and subject conventions, the ADR-0008
contract (`uninit_impl`, `beta = 0` never reads `c`, `assume_init_impl`; the form is gated on
`!read_c`), `TypeId` fast paths, the matmul-axes rule, the regression mandate.

Not reaching this crate: `rules/testing.md`'s taxonomy and entry-binary rules (CUDA is listed
as future); the new tests fit no category and the rule for that case is "stop and ask the
human". `api-doc-conventions` binds `rstsr-core` only.

## Spec fidelity (sub-agent report)

Matches the decisions, checked in code: driver table, chain order, `MatMulDrivers` filtering
and the error naming drivers and triple (impl D5/D13/D17/D18); mode resolution per triple
incl. `Fast*` to `Default` for `c32` and half to `f32`, `Acc16F` to `Default` for `bf16`,
`EmulatedFixedPoint` to `Default` for `c64`, `Pedantic` never downgraded (D9); the tensor
entries and the `.out::<TC>()` family check (D15 and `80e626d`); `#[non_exhaustive]` flags,
the three environment variables gone, `with_flags`, launch-time `device.flags()` (D16/D25);
naive-only mixed and integer triples (D1); no host FFI (D19); the syrk condition, batched
`a @ a.t()` and `a @ a.h()` on the gemm chain, classic disabled turns the form off, the trace
name (forms D1/D7/D8/D11 final/D12); the dot lifted in `resolve_geom`; `bench_matmul`.

(a) Missing or partial — 3:

1. forms `PLAN.md` "trace_matmul names the routine": no test observes routing (= T3).
2. forms `PLAN.md` "sub-matrix views `a[:n, :k]`" and "form rows in the layout matrix test":
   not what the tests run (= T2).
3. impl D26 "layout matrix for naive and tiled": tiled is only in the pre-existing 5-test
   suite.

(b) Scope creep — 0. The extras (`build.rs`, `matmul_with_mode_f`,
`MatMulDrivers::none/minus/Sub`, `AlignedScale`) are logged in the batch reports.

(c) Implemented but wrong — 3: `build.rs` (= C4), the symmetric form and the compute mode
(= C3, including the bypassed mantissa error and the "read nothing from the device flags"
doc), and the `CudaRuntimeFlags` family docs (= A2).

## Probe index

`probe/tests/probe.rs`, one `#[test]` per group, prints `ok` / `FAIL` / `ERR` / `ref` lines
(`ref` = a refusal the driver is entitled to make).

| test | what | result |
|---|---|---|
| `t1_complex_imaginary_parts` | c32, c64, f32, f64 with imaginary parts; 7 shapes; both orders; Lt, Ex, classic, tiled, naive, chain | 319 ok, 0 wrong |
| `t2_syrk_shapes` | the form on offsets, both orientations, `k = 0`, `n = 1`, `k = 1`, `alpha = 0`, strided and transposed output views | 108 ok |
| `t3_syrk_and_compute_mode` | routing and timing under `EmulatedFixedPoint` / `Disallow` / `Default` / `Pedantic`; the mantissa error | C3 |
| `t4_degenerate_dots` | `[0] @ [0]`, `[3] @ [3]` per driver | C5 |
| `t5_cross_stream_inplace` | control on one stream; chain, cuBLASLt, tiled on two streams | C2 |
| `t6_empty_and_k0` | empty batch axis; `k = 0` with `beta = 0, 1, 2`; fresh output | all correct |
| `t7_mode_entries` | `matmul_with_output_mode`, `matmul_from_mode`, `.out()`, per-call mode against the flag, mixed operands | all correct |
| `t9_broadcast_matmul_axis` | operands broadcast along a matmul axis; reversed batch axis | C1; reversed batch correct on all drivers |
| `t11_syrk_of_a_broadcast_operand` | the form on a broadcast operand | C1 |

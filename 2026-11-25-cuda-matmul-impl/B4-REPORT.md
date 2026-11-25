# B4 — the cuBLASLt driver

Committed in rstsr `261021/cuda` as `554f956`, on top of `493a42f` (B3). The
patch is `patches/b4-cublaslt-driver.patch` (10 files, +861/-88; one new file, `drivers/cublas_lt.rs`).

| | |
|---|---|
| rstsr | `261021/cuda` @ `493a42f` + B4 working tree |
| GPU | NVIDIA GeForce RTX 5070 Laptop 8 GiB, sm_120, CUDA 13.2 toolkit, cuBLASLt 13.4 runtime, cudarc 0.19.10 |
| bindings | two builds: `nvcc`-detected (13.2) and `CUDARC_CUDA_VERSION=12060` (the CI pin) |
| raw output | `results/b4_full_suite.txt`, `results/b4_full_suite_12060.txt`, `results/b4_modes_lt_release.txt`, `results/b4_driver_release.txt`, `results/b4_probe_lt_fixedpoint.txt`, `results/b4_clippy.txt`, `results/b4_check_doc.txt` |

## What was done

| decision | result |
|---|---|
| D6 (Lt resources) | `CublasLtHandle` per **ordinal** (`CudaOrdinalObj::cublaslt()`, lazy, `catch_unwind` probe like the cuBLAS handle): one `cublasLtHandle_t`, one `Mutex` held across every call (heuristic included), and the heuristic cache. The **workspace** is per **stream** (`CudaStreamObj::cublaslt_workspace`, 32 MiB, lazy): it is the kernels' scratch memory, so two streams of one ordinal never share it. With one default stream per ordinal (the registry) this is D6's layout; a user-made second stream pays its own 32 MiB. |
| D6 (heuristic cache) | Keyed by the **whole problem**: compute type, scale type, both transposes, the four matrix layouts (dtype, rows, cols, ld, batch count, batch stride), the workspace size, and the low 8 address bits of the four pointers (an algorithm states a per-matrix alignment requirement). The value holds the descriptors themselves (matmul desc, 4 layouts, preference) plus the checked algorithm, so a hit is lock, lookup, `cublasLtMatmul`. Bounded: dropped whole at 1024 entries. |
| D18 / D13 (chain) | Lt is the first candidate of the accelerated rows: `f32` (every 32F mode, `Fast16BF` included) and `f64` `Default`/`Pedantic`. `EmulatedFixedPoint`: Lt declines ("served by the classic driver"), the table is unchanged — see the probe below for why that is right. |
| D19 (no new FFI) | Only `cudarc::cublaslt::{sys, result}`-level calls through the `cublaslt` feature (added to the dependency; `libcublasLt` was already linked by `cublas`). The `cublas` and `cublaslt` modules carry two generated copies of the shared enums: a `retyped` copy by value, unit-tested to keep every value the drivers pass. |
| shared code | `cublas::with_gemm_call` split into `prepare` (argument derivation, ordering guards, folded pointers → `GemmCall`) and the handle lookup, so Lt uses the same rule table as the other two drivers without needing the classic handle. |

Descriptor construction (`build`): matmul desc `(compute, scale)`, `TRANSA`/`TRANSB`, four
column-major layouts from `CublasArgs` (`op(A)` is `n x k` of `geom.b`, `op(B)` `k x m` of `geom.a`, `C = D`
`n x m` with `ldc`; batch count and **element** strides as the strided-batched entries take them), preference
with a 32 MiB workspace ceiling; then `cublasLtMatmulAlgoGetHeuristic` (1 result) and `cublasLtMatmulAlgoCheck`
(the 8-argument form cudarc binds: layouts only, no addresses). Every failure is a refusal; `Entry`'s `Drop`
destroys what a half-built entry created.

## Verification

**Whole crate, `--features faer`, `--test-threads=1`**, both binding levels: **221 tests pass** (B3's 215 plus
5 host tests and 1 hardware test; two existing hardware tests gained Lt).

| target | tests | B4 |
|---|---|---|
| `--lib` | 97 | stored matrices of the three slots per orientation/batch (1), key equality and the fields that distinguish (1), pointers follow the slots (1), `retyped` enum values (1), emulation left to classic (1); `table_32f`/`table_64f` now start with `cublasLt` |
| `test_matmul_modes` | 5 | `modes_32f`: every 32F mode on Lt alone (**`Fast16BF` served**, 2.4e-4); `modes_64f`: `Default`/`Pedantic` on Lt; `EmulatedFixedPoint` on Lt alone = "no enabled driver is a candidate"; `cublas_batched_and_transposed` (renamed): batch, transposed right operand broadcast, transposed left operand, broadcast batch — on Lt, Ex and classic alone; `cublas_lt_misaligned_views` (new) |
| all other targets | 119 | unchanged |

`cublas_lt_misaligned_views`: owning tensors whose layouts start one `f64` into their buffers (addresses off
the 16-byte grid). **Lt runs them and is exact**, for the fresh output and for the in-place `beta` path
(`c = 2 a b + c`), like Ex and the chain: `cublasLtMatmul` accepts what this needs and the alignment class in
the key keeps the cached algorithm honest.

`cargo clippy --all-targets --features faer`: no warning in `rstsr-cuda`. `cargo check --all-features`,
`cargo doc --no-deps`: clean. Changed files rustfmt-formatted.

**Speed** (`src/bench_matmul_driver_check.rs`, release, 20 reps, steady state includes the output allocation,
`results/b4_driver_release.txt`):

| shape | type | Lt | Ex | classic |
|---|---|---|---|---|
| 2048³ | f32 | **1.178 ms, 14.6 TF/s** | 1.219, 14.1 | 1.202, 14.3 |
| 512³ | f32 | 0.033 ms | 0.032 | 0.032 |
| 65536x64x64 | f32 | 0.069 ms | 0.073 | 0.078 |
| 64³ | f32 | 0.014 ms | 0.013 | 0.014 |
| batch 256 x 256³ | f32 | 0.990 ms | 0.991 | 0.992 |
| 2048³ | f64 | 51.5 ms | 51.5 | 51.5 |
| batch 256 x 256³ | f64 | 24.6 ms | 24.6 | 24.6 |

Steady state is a wash except the large f32 square (+3%). The price is the **first call of each new
problem** (descriptors + heuristic, once per cache key per process): 58.7 ms on f32 2048³ (Ex: 2.7 ms),
81.7 vs 51.9 ms on f64 2048³, 3.6 vs 0.13 ms on 64³, 1.6 vs 1.5 ms on the batch. The very first Lt call
of a process also creates the handle and allocates the workspace.

Per mode on Lt (`results/b4_modes_lt_release.txt`, f32 2048³): `Default` 14.2 TF/s, `Pedantic` 14.0 (same
bits), `FastTF32` 19.3, `Fast16F` 20.2, `Fast16BF` 20.5 — the same TF32 path and 2.75e-4 as on Ex.

## Probe: the fixed-point emulation through cuBLASLt (`src/probe_cublaslt_fixedpoint.rs`)

B3's open point 6: the 13.x bindings carry `CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT` for Lt and an emulation
descriptor (`cublasLtEmulationDesc*`, `CUBLASLT_MATMUL_DESC_EMULATION_DESCRIPTOR`,
`CUBLASLT_EMULATION_DESC_FIXEDPOINT_MAX_MANTISSA_BIT_COUNT`), i.e. the entries the classic handle lacks
for `fixedpoint_mantissa_bits`. Direct cudarc calls, f64 2048³, against the CPU product:

| case | max abs err | TF/s |
|---|---|---|
| plain `COMPUTE_64F` | 0 | 0.33 |
| `COMPUTE_64F_EMULATED_FIXEDPOINT`, no descriptor | 0 | 0.34 |
| + dynamic mantissa, max bits 0 / 24 | 0 | 0.34 |
| + fixed mantissa (1), max bits 16 / 24 | 0 | 0.34 |
| + strategy `PERFORMANT` / `EAGER`, default bits | 0 | 0.34 |

cuBLASLt 13.4 **accepts** the compute type and the descriptor and **never emulates** on this GeForce: exact
native FP64 at the plain rate, under every setting. The classic math mode gives 2.55 TF/s at 3.8e-15. So on
this machine Lt would serve the mode as "more precision, no speed-up", silently — D18's pin to the classic
handle is right, and Lt declining it (as built) is the behaviour the tests assert. The probe's numbers are
one library on one GPU; on a data-centre part Lt may well emulate (the cudarc path exists), which is why
this is reported rather than coded around.

## Bugs and pitfalls on the way

- Three layouts, not four: `D` mirrors `C`, and `stored_matrices(..).zip([4 strides])` silently truncated to
  three. cuBLASLt reported `CUBLAS_STATUS_INVALID_VALUE`, and its logger (`CUBLASLT_LOG_LEVEL=5`) names the
  null `Ddesc` — the quickest diagnostic for any "INVALID_VALUE and nothing else" from Lt.
- The pointers went to the slots in geom order while the layouts followed `CublasArgs` (slot A holds
  `geom.b`): a plausible wrong product, 1.1e0 rel err. `slot_ptrs` is now one named function with a test.
- `cublasLtMatmulAlgoCheck` in cudarc has **8** arguments (the 13.x header form, no addresses); the 12-argument
  form of older headers is not bound.
- rustfmt on `drivers/cublas.rs` descends into the sibling modules (`cublas_ex`, `naive`, `tiled`) and
  reformats files this batch did not touch; reverted to keep the patch on the changed files.
- Reading a tensor back: `to_cpu_vec(storage())` is the **whole storage**; with a non-zero layout offset it
  starts at the buffer (identical on the CPU devices, so not a CUDA bug). The alignment test reads `c` through
  an elementwise result (`&c + 0.0`) instead. `into_shape` and `to_owned` of a view materialize offset 0;
  a shifted owning tensor needs `TensorBase::new(storage, layout)`.

## For the owner

1. **Workspace per stream, not per ordinal** (the one deviation from D6): a per-ordinal scratch buffer races
   between two streams of one ordinal running matmuls at the same time. Per-stream costs 32 MiB per stream
   that ever runs a matmul; the default-stream registry makes that one per ordinal.
2. **The heuristic cache is unbounded in kind, bounded in size** (1024 entries, then dropped whole): a
   workload cycling through more than 1024 distinct `(shape, alignment)` problems pays the heuristic for
   each again after each drop. Not hit by any test or bench here; a LRU would fix it if it ever shows.
3. **First-call latency**: a new shape costs tens of ms on large problems (heuristic). cuBLAS caches
   heuristics internally too; the per-key descriptors are what the cache here saves on top. If the first
   call ever matters (e.g. many small distinct shapes), the answer is a warm-up, not a code change.
4. **The emulation probe** above: I suggest closing B3's open point 6 as "Lt does not serve
   `EmulatedFixedPoint`; on this GPU it cannot" and leaving `fixedpoint_mantissa_bits` as it is (error on
   non-zero, classic only). If a data-centre GPU is available, the same probe there would settle whether
   Lt can serve it on that class of hardware.
5. `drivers/cublas.rs` grew `GemmCall`/`prepare`/`Served` (the shared preparation); the Lt driver is the
   only other consumer until B5.
6. In-tree test additions in `tests/test_matmul_modes.rs` (Lt rows in two tests, one test renamed and
   widened, one new test); `cublaslt` added to the cudarc features of the crate.

# cuTENSOR empirical probe — NVIDIA GeForce RTX 5070 Laptop GPU (sm_120), cuTENSOR 2.7.0

Date: 2026-11-27. Scope: C/C++ probe only (no Rust). Everything below was measured on this machine with the
pip-packaged cuTENSOR 2.7.0 (CUDA-13 build); the optional 2.4.1 comparison is in the last section.

## Environment (as printed by `probe version`)

```
[19] cutensorGetVersion() = 20700 (major.minor.patch = 2.7.0)
[19] cutensorGetCudartVersion() = 13010 (13.1)
[19] cudaRuntimeGetVersion() = 13020, cudaDriverGetVersion() = 13020
[19] GPU = "NVIDIA GeForce RTX 5070 Laptop GPU", compute capability 12.0, 36 SMs, 7.50 GiB, nominal SM clock 1425 MHz
[19] cuBLAS version = 130400
```

Driver 595.99.02, nvcc V13.2.78 (`/usr/local/cuda/bin/nvcc`). The 2.4.1 binary prints
`cutensorGetVersion() = 20401`, `cutensorGetCudartVersion() = 13010`.

## Build and run commands that worked

All paths below are relative to this directory (`2026-11-27-cuda-einsum-cutensor/probe/`).

```bash
# one-time: make -lcutensor resolvable and provide the SONAME the linker records
mkdir -p lib lib241 bin
ln -sf /home/a/miniconda3/envs/torch-gpu4pyscf/lib/python3.11/site-packages/cutensor/lib/libcutensor.so.2 lib/libcutensor.so
ln -sf /home/a/miniconda3/envs/torch-gpu4pyscf/lib/python3.11/site-packages/cutensor/lib/libcutensor.so.2 lib/libcutensor.so.2   # needed at run time (SONAME)
ln -sf /scratch/miniconda3-bak/pkgs/cutensor-2.4.1.4-h15eaa2f_0/lib/libcutensor.so.2 lib241/libcutensor.so
ln -sf /scratch/miniconda3-bak/pkgs/cutensor-2.4.1.4-h15eaa2f_0/lib/libcutensor.so.2 lib241/libcutensor.so.2

# build (2.7.0); build.sh = the nvcc line below
./build.sh /home/a/miniconda3/envs/torch-gpu4pyscf/lib/python3.11/site-packages/cutensor/include "$PWD/lib" "$PWD/bin/probe"
#   nvcc -O2 -std=c++17 -arch=sm_120 -Xcompiler -fopenmp -I<include> main.cu check_strides.cu check_conj.cu check_dtype.cu check_perf.cu \
#       -L<libdir> -L/usr/local/cuda/lib64 -lcutensor -lcublas -o <out>
# build (2.4.1)
./build.sh /scratch/miniconda3-bak/pkgs/cutensor-2.4.1.4-h15eaa2f_0/include "$PWD/lib241" "$PWD/bin/probe241"

# run (perf must run in a fresh process for the "first plan" numbers of check 20)
export LD_LIBRARY_PATH="$PWD/lib:/usr/local/cuda/lib64"        # 2.7.0
./bin/probe version ; ./bin/probe strides ; ./bin/probe conj ; ./bin/probe dtype ; ./bin/probe align-extra ; ./bin/probe perf
./bin/probe plan 64 64 2048 2048 2050 8 1024                     # check-20 follow-up: planning-cost order dependence
export LD_LIBRARY_PATH="$PWD/lib241:/usr/local/cuda/lib64"      # 2.4.1
./bin/probe241 version ; ./bin/probe241 strides ; ./bin/probe241 conj ; ./bin/probe241 dtype ; ./bin/probe241 perf
```

Notes on the recipe: the linker records the SONAME `libcutensor.so.2`, so a `libcutensor.so` symlink alone is
enough to link but **not** to run (`error while loading shared libraries: libcutensor.so.2`); the second symlink
fixes that. Both cuTENSOR versions depend only on `libcublasLt.so.13` (+ librt/pthread/dl/m/gcc_s/c), resolved from
`/usr/local/cuda/lib64`. `-Xcompiler -fopenmp` is only for the CPU reference (OpenMP outer loop). Linking needs
`-lcublas` for the cuBLAS comparison. Raw outputs: `bin/run_*.txt` (2.7.0) and `bin/run_*_241.txt` (2.4.1).

## Method

* Pipeline per case: `cutensorCreateTensorDescriptor` ×4 → `cutensorCreateContraction` → `cutensorCreatePlanPreference(ALGO_DEFAULT, JIT_MODE_NONE)`
  → `cutensorEstimateWorkspaceSize(WORKSPACE_DEFAULT)` → `cutensorCreatePlan(limit = 128 MiB)` → `cutensorContract` (128 MiB
  cudaMalloc'd workspace, same stream). Every status is recorded; the first failing stage and its code/string are
  printed; nothing aborts on library errors.
* Reference: straightforward host loop in double (`std::complex<double>` for complex data; float accumulation for
  f32/f16/bf16 cases), written into a copy of the output buffer's initial content; comparison over exactly the
  elements the output descriptor addresses. "maxabs/max|ref|" is the normalised error; verdict `ok` requires it to be
  ≤ 1e-12 (f64/c64), 1e-5 (f32/c32), 1e-2 (f16), 8e-2 (bf16). The output buffer is pre-filled with a sentinel and
  every element *not* addressed by D is checked byte-for-byte afterwards ("untouched-elems-preserved").
* Random data: uniform in [-1, 1] (complex: independent re/im), fixed seed 20261127, values with |x| < 0.05 shifted.
* Base GEMM shape for checks 1–17: m=37, k=53, n=41.
* The reference was cross-validated against a standalone minimal program (`debug_beta.cu`) after a harness bug
  (reference β dropped) produced spurious FAILs in a first run; all numbers below are from the corrected build.

## Results (cuTENSOR 2.7.0)

Verdict legend: `ok` = all statuses SUCCESS and result within tolerance; `FAIL` = SUCCESS but wrong result;
`ERR` = a non-success status (code + `cutensorGetErrorString`); "(probed)" marks cases that were run to *record* the status.

| # | Status | Max error / status | Note |
|---|--------|--------------------|------|
| 1.0 | ok | maxabs 3.1e-15 | baseline packed col-major, β=0, align 256 |
| 1.1 | ok | maxabs 3.6e-15 | A strides (109,2) align 8; B (1,58) align 16; D (82,1) [every 2nd row] align 16; β=0.5, separate C; untouched elements preserved |
| 1.2 | ok | maxabs 3.6e-15 | A (109,3), B (1,57), D (83,2): no stride a 16-byte multiple, align 8, β=0 |
| 1.3 | ok | maxabs 3.6e-15 | same layout as 1.1, **C aliases D** (in-place D = A·B + 0.5·D) |
| 1.4 | ok | maxabs 2.2e-15 | permuted modes modeA={k,m} modeB={n,k} modeD={n,m}, packed |
| 1.5 | ok | maxabs 3.6e-15 | A (2,74) B (2,212) D (2,148): no mode contiguous, β=0.5 |
| 2a | **ok (accepted)** | maxabs 4.4e-15 | A stride_k = 0 (extent 53): descriptor **accepted**, result equals the broadcast `a[m]·Σ_k b[k,n]` — docs say INVALID_VALUE |
| 2b | **ok (accepted)** | maxabs 1.8e-15 | B stride_n = 0: accepted, correct broadcast over n |
| 2c | **FAIL (accepted)** | maxabs 9.7 | D stride_n = 0 (broadcast output): **accepted at all stages**, result is garbage (write race) |
| 2d | **ok (accepted)** | maxabs 1.3e-15 | A stride_m = 0 on the batch mode: accepted, correct |
| 2e | ok (accepted) | maxabs 1.8e-14 | 256³ f64 with A stride_k = 0: accepted, correct (also when routed to cuBLAS-sized kernels) |
| 2f | ok (accepted) | maxabs 5.3e-15 | 256³ f64 with A stride_m = 0: accepted, correct |
| 2g | ok (accepted) | maxabs 1.1e-14 | batched 3×256³ with A stride_b = 0: accepted, correct |
| 3a | ok | 0.0 | outer product C[a,b] = A[a]·B[b], both rank-1, no contracted mode |
| 3b | ok | 0.0 | C[a,b] = A[a]·B[a,b] (A rank-1 shares batch mode), β=0.5 |
| 3c | ok | maxabs 2.7e-15 | matvec D[a] = Σ_k A[a,k]·B[k] (B has only a contracted mode), β=0.5 |
| 3d | ok | maxabs 1.8e-15 | D[a] = Σ_b A[a,b]·B[b], β=0.5 |
| 3e | ok | 0.0 | strided broadcast: A[a] stride 2 · B[a,b] → D (82,1), β=0.5 |
| 4a | ok | maxabs 2.7e-15 | A ext (1,53) strides (3,1), D ext (1,41) strides (1,1) — extent-1 mode with arbitrary non-zero stride is fine |
| 4b | **ok (accepted)** | maxabs 1.3e-15 | same with stride **0** on the extent-1 mode: accepted, correct |
| 4c | ok | maxabs 1.8e-15 | batch-of-1 mode in A (3,1), B (7,1,53), D (1,1) |
| 4d | ok | maxabs 4.0e-15 | extent-1 mode present only in A (contracted, extent 1) |
| 5a | ERR | descA status=7 CUTENSOR_STATUS_INVALID_VALUE | A strides (1,−37), base offset 1924 (A[:, ::-1]) |
| 5b | ERR | descC status=7 INVALID_VALUE | D strides (−1,37), base offset 36 (reversed output view); descC carries the same layout and is the first failing descriptor |
| 5c | ERR | descA status=7 INVALID_VALUE | A strides (−1,−37) |
| 6a | ERR | descA status=7 INVALID_VALUE | A extent (37,0) |
| 6b | ERR | descA status=7 INVALID_VALUE | A extent (0,53) (and D (0,41)) |
| 6c | ERR | descC status=7 INVALID_VALUE | D extent (37,0) (output mode 0, not contracted) |
| 7a | ok | maxabs 2.2e-15 | A and D base = cudaMalloc base + 8 bytes, alignmentRequirement = 8 |
| 7b | ok (silent) | maxabs 2.2e-15 | same 8-byte-aligned pointers, alignmentRequirement declared **256**: no stage rejects it, result correct |
| 7c | ok | maxabs 3.6e-15 | strides with a 3-element (24 B) factor: A (3,111) B (3,159) D (123,3), align 8, β=0.5 |
| 7d | ok (silent) | maxabs 3.6e-15 | same strides, alignmentRequirement declared **256**: accepted, result correct |
| 7e | ERR / ok | f64 align 4 → INVALID_VALUE; f64 align 8 → ok; c64 align 8 → INVALID_VALUE; c64 align 16 → ok; f32 align 2 → INVALID_VALUE; f32 align 4 → ok | **alignmentRequirement must be ≥ the element size** (checked at descriptor creation) |
| 7f | ok (silent) | maxabs 1.8e-14 | non-GEMM rank-3/rank-4 contraction (cuTENSOR's own kernels), all four base pointers at +8 bytes, align declared 8 / 16 / **256**: all accepted, all correct |
| 7g | ok (silent) | maxabs 1.6e-14 | non-GEMM, all strides 24-byte multiples, align declared 8 and **256**: accepted, correct |
| 8a | ok | 0.0 | D = s·B + 0.5·C with s an operand of numModes=0 (extent/stride arrays = nullptr): accepted, correct |
| 8b | ok | maxabs 1.6e-14 | rank-0 **output** D = Σ_{m,k} A[m,k]·B[k,m] (full reduction to a scalar): accepted, correct |
| 8c | ok | 0.0 | all four operands rank-0: D = a·b + 0.5·c |
| 8d | ok | 0.0 | rank-0 scalar × strided B (109,2) → strided D (107,2), β=0.5 |
| 9a | **ok (accepted)** | maxabs 1.8e-15 | modeA = {i,i}, A ext (53,53) strides (1,54): **accepted** by CreateContraction; result = diagonal with element offset **i·(1+54)** (sum of the two strides) |
| 9b | ok (accepted) | maxabs 2.7e-15 | modeA = {i,i,n} rank-3: accepted, same sum-of-strides rule |
| 9c | ok (accepted) | maxabs 1.8e-15 | strides (2,54) → offset 56·i: confirms the sum rule (not "first occurrence only") |
| 9d | ok (accepted) | maxabs 3.6e-15 | strides (3,5) → offset 8·i: confirms the sum rule |
| 10 | ok | maxabs 7.2e-15 (c64) / 4.0e-6 (c32, norm 2.6e-7) | opA=CONJ, opB=ID: D = conj(A)·B |
| 11 | ok | maxabs 8.4e-15 / 3.1e-6 (norm 2.4e-7) | opA=ID, opB=CONJ: D = A·conj(B) |
| 12 | ok | maxabs 1.2e-14 / 7.3e-6 (norm 3.2e-7) | opA=opB=CONJ with complex α=(1.5−0.5i), β=(0.5+0.25i) |
| 12b | ok | maxabs 7.1e-15 / 3.8e-6 | both CONJ with permuted modes modeA={k,m} modeB={n,k} modeD={n,m} |
| 13a | ERR | CreateContraction status=15 CUTENSOR_STATUS_NOT_SUPPORTED | opA=CONJ on CUDA_R_64F |
| 13b | ERR | CreateContraction status=15 NOT_SUPPORTED | opB=CONJ on CUDA_R_64F |
| 14a | **FAIL (silently wrong)** | vs conj(C): 1.498; vs plain C: 8.9e-15 (c64); c32: 1.496 vs 3.9e-6 | opC=CONJ, β=0.75: **accepted (SUCCESS)**, computes D = A·B + 0.75·C — the conjugation of C is silently ignored (confirms the 2.8.0 release note for 2.7.0) |
| 14b | ok | maxabs 7.2e-15 / 3.3e-6 | opC=CONJ with β=0: accepted, result = A·B (C not read) |
| 15a | ok | maxabs 8.9e-15 / 4.9e-6 | conj(A_strided (109,2)) · B_strided (1,58) → D (82,1), c64 align 16 |
| 15a-align8 | ERR | descA status=7 INVALID_VALUE | same c64 layout with alignmentRequirement = 8 (< 16-byte element) |
| 15b | ok | maxabs 8.9e-15 / 4.0e-6 | A_strided · conj(B_strided) |
| 15c | ok | maxabs 7.1e-15 / 4.8e-6 | conj(A_strided) · conj(B_strided) + 0.5·C_strided (C separate) |
| 15d | ok | maxabs 7.2e-15 / 3.8e-6 | conj(A{k,m})·conj(B{n,k}) → D{n,m}, every mode 2× interleaved, β=0.5 |
| 16a | ok | maxabs 8.5e-15 | α/β as cuDoubleComplex{1,0}/{0,0} (documented) |
| 16b | ok | maxabs 7.3e-15 | α/β as double[2]{1,0}/{0,0}: identical bytes, no way to detect |
| 16c | ok (silently complex) | maxabs 1.1e-14 | α as double[2]{1,0.25}: no error; the 16 bytes are read as the complex number (1+0.25i) — the reference with α=1+0.25i matches exactly |
| 16d | **FAIL (silently wrong)** | maxabs 13.6, result ≡ 0 | α as float[4]{1,0,0,0} (wrong type): no error; the bytes reinterpreted as two doubles give a denormal α ≈ 5e-315, so D = 0 |
| 16e | ERR | SetAttribute(SCALAR_TYPE=CUDA_R_64F) status=7 INVALID_VALUE | default SCALAR_TYPE of a c64 problem is CUDA_C_64F (Get works); it cannot be changed to a real type |
| 17a | ok | maxabs 4.5e-15 | R64 × C64 → C64 (A real, B complex), complex β=0.5; SCALAR_TYPE attribute = CUDA_C_64F |
| 17b | ok | maxabs 4.5e-15 | C64 × R64 → C64 |
| 17c | ok | maxabs 7.1e-15 | A real, opB=CONJ on complex B: D = A·conj(B) + 0.5·C |
| 17d | ok | maxabs 4.0e-15 | mixed real/complex with strided A (109,2), B (1,58), D (82,1) |
| 17e | ERR | CreatePlan status=15 NOT_SUPPORTED | R64 × C64 → **R64** (real output, not in docs table): CreateContraction accepts, planning fails |
| 17a–d (c32) | ERR | CreatePlan status=15 NOT_SUPPORTED | R32 × C32 → C32 and C32 × R32 → C32 (with/without CONJ, strided): CreateContraction **accepts**, plan creation fails — not in the docs table |
| 17f (c32) | ERR | CreatePlan status=15 NOT_SUPPORTED | same problem with float[2] scalars: irrelevant, the problem itself is unsupported |
| 18 f16 packed | ok | maxabs 3.9e-3, norm 5.0e-4 | CUDA_R_16F ×4, COMPUTE_DESC_32F, float α/β; float32 CPU reference |
| 18 f16 strided | ok | 0.0 | A (109,2) B (1,58) D (82,1), alignmentRequirement = 2 (true alignment for 2-byte elements), β=0.5 |
| 18 bf16 packed | ok | maxabs 6.25e-2, norm 5.7e-3 | CUDA_R_16BF ×4, COMPUTE_DESC_32F; error = bf16 output rounding (2^-8) |
| 18 bf16 strided | ok | maxabs 3.1e-2, norm 4.1e-3 | same strided layout, align 2, β=0.5 |
| 18 f32 packed | ok | maxabs 1.7e-6, norm 1.9e-7 | CUDA_R_32F ×4, COMPUTE_DESC_32F |
| 18 f32 strided | ok | maxabs 1.9e-6, norm 2.2e-7 | strided layout, align 8, β=0.5 |
| 18 f32 TF32 | ok | maxabs 2.4e-3, norm 3.3e-4 | COMPUTE_DESC_TF32: 3.3e-4 relative (10-bit mantissa products), as expected |
| 18 f32 3xTF32 | ok | maxabs 2.1e-6, norm 2.4e-7 | COMPUTE_DESC_3XTF32 |
| 18 f64 data, COMPUTE_DESC_32F | ok | maxabs 3.6e-15, norm 4.5e-16 | listed in the docs table; result is **f64-accurate** (no precision loss observed at this size), scalars stay double |
| 19 | ok | — | versions above |
| 20 | ok | — | see timing table |
| 21 | ok | — | see timing table; cuTENSOR and cuBLAS results are **bit-identical** for f64 and f32 (max diff 0.0) |
| 22.0 | ERR | CreateContraction status=7 INVALID_VALUE | D[m,n,p] = Σ A[m,k1,k2]·B[k2,n,k1] with p occurring only in D: rejected (every D mode must occur in A or B) |
| 22a | ok | maxabs 2.9e-13, norm 3.9e-15 | valid form B[k2,n,k1,p], arbitrary strides, 64/32/64/64/64 |
| 22b | ok | maxabs 2.9e-13, norm 4.3e-15 | same, packed col-major |
| 22c | ok | maxabs 2.7e-13, norm 3.8e-15 | p in A instead: A[m,k1,k2,p]·B[k2,n,k1], strided |
| 22d–g | not verified (timing only) | — | larger sizes, see timing table |
| 23a | ok | maxabs 5.3e-15 | rank-8 A,B,D (4 contracted + 4 batch modes each), packed, β=0.5 |
| 23b | ok | maxabs 1.1e-14 | rank-8, every operand a padded slice (non-packed strides), align 8 |
| 23c | ok | maxabs 3.6e-15 | rank-8 operands sharing batch modes c,d; rank-6 output |
| 23d | ok | maxabs 1.4e-14 | rank-12 operands (6 contracted modes), rank-12 output, extents 2/3, padded strides |
| 23e | ok | maxabs 7.1e-15 | rank-8, modes permuted differently in A, B, D, extents 2..5 |

### Findings that differ from the documented behaviour (2.7.0)

1. **Stride 0 is accepted** (check 2). The descriptor docs say "each stride must be larger than zero … CUTENSOR_STATUS_INVALID_VALUE".
   2.7.0 returns SUCCESS for stride-0 descriptors (inputs, outputs, extent-1 modes, GEMM-sized and batched), and input
   broadcasts are computed correctly. A stride-0 **output** is also accepted and yields garbage (write race, no error).
   2.4.1 rejected every stride-0 descriptor (NOT_SUPPORTED).
2. **A mode may appear more than once per tensor** (check 9). The docs say at most once; `cutensorCreateContraction`
   accepts `modeA = {i,i}` and the element offset is the **sum** of the strides given for that mode (true diagonal).
3. **opC ≠ IDENTITY is silently accepted and ignored** (check 14): with β ≠ 0 the result is `A·B + β·C`, not `β·conj(C)`.
4. **alignmentRequirement is not checked against the actual pointers or strides** (7b, 7d, 7f, 7g): over-declaring
   (256 for 8-byte-aligned pointers / 24-byte strides) is accepted at every stage and gives correct results, including
   on cuTENSOR's own (non-cuBLAS) kernels. The only enforced rule is alignment ≥ element size (7e).
5. **Scalar pointer type is never checked** (check 16): any 16 bytes are read as the complex/double scalar; wrong
   types give silently wrong results (zeros). `CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE` is read-only in practice
   (Set to a real type → INVALID_VALUE).
6. **Mixed 32-bit real/complex** (R32×C32→C32) passes `cutensorCreateContraction` but fails in `cutensorCreatePlan`
   with NOT_SUPPORTED; the same holds for R64×C64→R64. Only plan creation reports it.
7. Negative strides and extent 0 → INVALID_VALUE at descriptor creation (2.7.0); 2.4.1 reported NOT_SUPPORTED for both.
8. `COMPUTE_DESC_32F` on f64 data gave f64-accurate results here (no reduced precision at m,k,n = 37,53,41).
9. `wsRequired` for the GEMM-shaped problems is 0 B (f64, c64) / 1024 B (f32), and the GEMM results are bit-identical
   to cuBLAS: for GEMM-shaped f64/f32 contractions cuTENSOR 2.7.0 is a thin dispatcher onto the same cuBLAS kernels.

## Timing (checks 20–22)

GPU state during the runs: SM clock 337→1995 MHz under load (max 3090), 43–46 °C; laptop GPU, so absolute numbers
carry clock uncertainty; two rounds are reported. flops = 2·m·n·k (real) and 8·m·n·k (complex, real-arithmetic count).

### Check 20 — host-side planning cost (f64 GEMM-shaped, fresh process, ALGO_DEFAULT, JIT none)

| Sequence | total | breakdown |
|----------|-------|------------|
| first-ever plan in process, 8³ (incl. library init) | 0.157 ms | EstimateWorkspaceSize 0.059, CreatePlan 0.081 |
| 2048³ first | **28.4 ms** | 28.3 ms of it in `cutensorEstimateWorkspaceSize`; CreatePlan 0.08 ms |
| 2048³ repeated ×20 (identical problem, default plan cache) | **median 0.013 ms** (mean 0.016, min 0.012, max 0.048) | CreatePlan 0.005, CreateContraction 0.003, EstimateWorkspaceSize 0.006 |
| 64³ first | 0.036 ms | after the 2048³ call |
| 64³ repeated ×20 | median 0.012 ms | |
| 2050³ first (distinct problem, no cache hit) | 0.025 ms | |
| 2048³ repeated ×5 with CACHE_MODE_NONE | median 0.012 ms | plan cache makes no measurable difference for GEMM-shaped problems |

The 28 ms is a **one-time lazy initialisation of the cuBLAS path**, not per-problem cost: with the sequence
`plan 64 64 2048 2048 2050 8 1024` the 28.4 ms lands on the first 64³ call and everything after is 0.014–0.055 ms;
with `plan 2050 2048` it lands on 2050³; 8³ (wsEstimate 0 B) never triggers it, 64³ (wsEstimate 32 MiB) does.
Per-problem planning afterwards is ~20–50 µs, ~12–15 µs on a plan-cache hit.

Non-GEMM problem of check 22 (where the heuristic does real work): first plan 0.19–0.24 ms, repeated (cache hit)
median 0.016 ms.

### Check 21 — device time (cudaEvent, median of 10 after 3 warm-ups, 128 MiB workspace, same stream)

| Problem | cuTENSOR | cuBLAS | ratio cuBLAS-time / cuTENSOR-time | result diff |
|---------|----------|--------|-----------------------------------|-------------|
| f64 2048³, COMPUTE_DESC_64F vs `cublasDgemm` | 52.24 / 52.05 ms → 0.329 / 0.330 TFLOP/s | 52.18 / 52.21 ms → 0.329 / 0.329 | **0.999 / 1.003** (≈1.00) | 0.0 (bit-identical) |
| f32 2048³, COMPUTE_DESC_32F vs `cublasSgemm` (no TF32) | 1.184 / 1.235 ms → 14.51 / 13.91 TFLOP/s | 1.200 / 1.227 ms → 14.31 / 14.00 | **1.014 / 0.993** (≈1.00) | 0.0 (bit-identical) |
| c64 1024³, COMPUTE_DESC_64F vs `cublasZgemm` | 20.28 / 20.27 ms → 0.424 / 0.424 TFLOP/s | 25.27 / 25.25 ms → 0.340 / 0.340 | **1.246 / 1.246** | 3.3e-13 (rel 3.8e-15) |

cuBLAS with an explicit 128 MiB workspace (`cublasSetWorkspace`): f64 52.21 ms, f32 1.19 ms, c64 25.28 ms — no change.
wsEstimate(DEFAULT) = 32 MiB, wsRequired = 0 B (f64, c64) / 1024 B (f32).

### Check 22 — non-GEMM contraction D[m,n,p] = Σ_{k1,k2} A[m,k1,k2]·B[k2,n,k1,p] (f64, k1=32, k2=64)

| Case | layout | size (m=n=p) | correctness | device median | TFLOP/s | wsRequired | host plan first / repeat |
|------|--------|--------------|-------------|---------------|---------|------------|--------------------------|
| 22a | arbitrary strides | 64 | ok, 2.9e-13 | 3.075 ms | 0.349 | 32 MiB | 0.212 / 0.016 ms |
| 22b | packed | 64 | ok, 2.9e-13 | 3.445 ms | 0.312 | 32 MiB | 0.208 / 0.016 ms |
| 22c | arbitrary strides, p in A | 64 | ok, 2.7e-13 | 3.443 ms | 0.312 | 32 MiB | 0.193 / 0.017 ms |
| 22d | arbitrary strides | 128 | not verified | 27.93 ms | 0.308 | 34 MiB | 0.236 / 0.017 ms |
| 22e | packed | 128 | not verified | 28.06 ms | 0.306 | 34 MiB | 0.215 / 0.017 ms |
| 22f | arbitrary strides | 256 | not verified | 198.96 ms | 0.345 | 36 MiB | 0.196 / 0.017 ms |
| 22g | packed | 256 | not verified | 198.85 ms | 0.346 | 36 MiB | 0.201 / 0.016 ms |

The non-GEMM contraction runs at 0.31–0.35 TFLOP/s, i.e. at the same FP64 rate as the plain f64 GEMM (0.33 TFLOP/s),
and needs its full 32–36 MiB of workspace (unlike the GEMM cases).

## Optional comparison with cuTENSOR 2.4.1 (same binary, `lib241`)

Checks run: 1–9 (strides), 10–17 (conj), 18 (dtype), 20–23 (perf). Everything is identical to 2.7.0 except:

| Check | 2.7.0 | 2.4.1 |
|-------|-------|-------|
| 2 (stride 0, all sub-cases incl. 256³ and batched) | **accepted**, inputs broadcast correctly, output stride 0 → garbage | rejected at descriptor creation: **status 15 CUTENSOR_STATUS_NOT_SUPPORTED** |
| 4b (stride 0 on an extent-1 mode) | accepted, correct | rejected: NOT_SUPPORTED (15) |
| 5 (negative stride) | INVALID_VALUE (7) | NOT_SUPPORTED (15) |
| 6 (extent 0) | INVALID_VALUE (7) | NOT_SUPPORTED (15) |
| 14 (opC=CONJ, β≠0) | silently accepted, conj ignored | identical (silently wrong) |
| 9 (duplicate mode, sum-of-strides diagonal) | accepted | identical |
| 20 planning | one-time 28.4 ms, then 0.013 ms | one-time 29.9 ms, then 0.013 ms (wsEstimate 16 MiB instead of 32 MiB) |
| 21 device time | f64 52.2 ms, f32 1.18 ms, c64 20.3 ms | f64 51.46 ms, f32 1.18 ms, c64 20.08 ms; same bit-identical f64/f32, same 1.24× c64 advantage |

Neither version returns the documented INVALID_VALUE for stride 0.

## Files

* `probe_common.h` — framework: descriptors, status-tracked pipeline, type-erased buffers, OpenMP CPU reference, reporting.
* `main.cu` — driver (`version | strides | conj | dtype | perf | plan <sizes…> | align-extra`).
* `check_strides.cu` (checks 1–9, 7e–7g), `check_conj.cu` (10–17), `check_dtype.cu` (18), `check_perf.cu` (20–23).
* `debug_beta.cu` — standalone minimal β≠0 check used to validate the reference.
* `build.sh` — build recipe. `lib/`, `lib241/` — symlink farms. `bin/run_*.txt` — raw outputs.

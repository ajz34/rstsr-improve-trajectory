# cuTENSOR cross-version probe — stride-0 descriptors, opC=CONJ, duplicate mode, alignment, extent mismatch

Date: 2026-11-28. Machine: NVIDIA GeForce RTX 5070 Laptop GPU (sm_120), driver 595.99.02, nvcc V13.2.78, cudart 13.2 in the
binary. Purpose: find the cuTENSOR version thresholds for a runtime gate in the Rust wrapper.

## Method

* One standalone binary (`vs.cpp`, compiled against the **2.7.0 header**, linked once against `lib2.7.0/`), run once per version with
  `LD_LIBRARY_PATH=lib<V>:/usr/local/cuda/lib64` so the loader picks that version's `libcutensor.so.2` (`./build.sh`, `./run.sh`).
  All five libraries export every symbol the binary needs (checked with `nm -D`), i.e. the 2.x ABI is indeed stable across them.
* Pipeline per case, every status recorded, first failing stage + code reported: `cutensorCreateTensorDescriptor` ×4 →
  `cutensorCreateContraction` (or `cutensorCreateReduction` for S2a) → `cutensorCreatePlanPreference(ALGO_DEFAULT, JIT_MODE_NONE)`
  → `cutensorEstimateWorkspaceSize(WORKSPACE_DEFAULT)` → `cutensorCreatePlan(limit = 1 MiB)` → `cutensorContract` on the default
  stream (`nullptr`), `cudaDeviceSynchronize`. 256 MiB `cudaMalloc` workspace buffer (1 MiB limit passed to the plan; no case ever
  needed the retry). `CUTENSOR_COMPUTE_DESC_64F` throughout (f64 and c64 data). Extents i=4, j=3, k=5 (S3d additionally 37×53×41).
* Reference: host double / `std::complex<double>` loops; verdict `ok` requires max |err| ≤ 1e-12·max(1, max|ref|).
  Random data uniform in [-1, 1] (complex: independent re/im), fixed seed 20261128, same data in combined and isolated runs.
* Interpretation notes: S1's A is a j-vector described as modes [i,j] extents (4,j) strides (0,1) (the task text says
  "(1, k)"; the contraction `ij,jk->ik` has no k in A). S2 is run through the reduction API (`ij->j`, ADD) **and** through the
  contraction API (`ij,j->j` with B = 1). S7 is run as a mismatch on the contracted mode (j: A=3, B=4) and on a batch mode
  (k: B=3, D=4), because `ij,jk->ik` has no k in A.
* No case crashed in any version, so every log is a single all-in-one process (the per-case fallback in `run.sh` was not needed
  except for 2.0.1, where every process fails at `cutensorCreate`).

## Version strings (first line of each log)

| library | `cutensorGetVersion()` | `cutensorGetCudartVersion()` | status on this machine |
|---|---|---|---|
| 2.0.1 (pyscf env, pip) | 20001 (2.0.1) | not reachable | **loads, but `cutensorCreate` fails: status 14 CUTENSOR_STATUS_INTERNAL_ERROR** |
| 2.4.1 (conda pkg h15eaa2f_0) | 20401 (2.4.1) | 13010 (13.1) | works |
| 2.6.0 (uv cache) | 20600 (2.6.0) | 13010 (13.1) | works |
| 2.7.0 (torch-gpu4pyscf env, pip) | 20700 (2.7.0) | 13010 (13.1) | works |
| 2.8.0 (ruaa-dev env, pip) | 20800 (2.8.0) | 13040 (13.4) | works |

2.0.1 details: it needs `libcublasLt.so.12` (not `.11`; CUDA 13.2 does not ship it). Attempt 1 used the pyscf env's
12.1.0.26 (`run-2.0.1-attempt1-cublasLt-12.1.0.26.log`), attempt 2 the conda pkg's 12.9.1.4 (`run-2.0.1.log`); both load
(`ldd` resolves `libcutensor.so.2` and `libcublasLt.so.12`) and both fail at `cutensorCreate` with INTERNAL_ERROR, so no case
could be run. cuTENSOR 2.0.1 is therefore **unusable on this sm_120 / driver 595 machine** — its behaviour for S1–S7 is unknown.

## Results

`ERR(<stage> <code> <name>)` = first failing stage and status; "ignored" = opC=CONJ accepted but result = α·A·B + β·C;
"accepted: garbage" = every stage SUCCESS but the result is wrong (stride-0 output → write race).

| Case | 2.0.1 | 2.4.1 | 2.6.0 | 2.7.0 | 2.8.0 |
|---|---|---|---|---|---|
| **S1** stride-0 input A[i,j] ext(4,3) strides(0,1), `ij,jk->ik` f64 | n/a (no handle) | **ERR(descA 15 NOT_SUPPORTED)** | **ok** (maxabs 0.0) | **ok** (maxabs 0.0) | **ok** (maxabs 1.1e-16) |
| **S2** stride-0 on reduced mode i: reduce `ij->j` / contract `ij,j->j` (B=1), expect D = 4·A[0,j] | n/a | **ERR(descA 15 NOT_SUPPORTED)** / same | **ok** / **ok** | **ok** / **ok** | **ok** / **ok** |
| **S3** stride-0 output D (5 variants a–e, see below) | n/a | **ERR 15 NOT_SUPPORTED** in all five (a,b: descD; c,d,e: descC) | a,b: ERR(CreateContraction 15 NOT_SUPPORTED); **c,d: accepted → garbage**; **e: accepted → one writer won** (row i=0) | same as 2.6.0 | same as 2.6.0 |
| **S4** opC = CONJ, β = 0.3, C aliases pre-filled D, c64 | n/a | **ignored** (D = A·B + 0.3·C; diff vs conj ref 5.7e-01) | **ignored** (same) | **ignored** (same) | **ERR(CreateContraction 15 NOT_SUPPORTED)** |
| **S5** duplicate mode A[i,i] ext(4,4) strides(4,1) | n/a | **ok-diagonal** (offset = sum of strides, 5·i) | **ok-diagonal** | **ok-diagonal** | **ok-diagonal** |
| **S6** alignmentRequirement = 256 on a pointer at cudaMalloc + 8 B | n/a | **ok** (silent, correct) | **ok** (silent, correct) | **ok** (silent, correct) | **ok** (silent, correct) |
| **S7** extent mismatch: contracted j (A=3, B=4) / batch k (B=3, D=4) | n/a | **ERR(EstimateWorkspaceSize 14 INTERNAL_ERROR)** / same | same | same | same |

S3 variants (all `ij,jk->ik` f64, β = 0): (a) 4×3×5 D strides (0,1) into a 5-element buffer, C packed [the task's layout];
(b) 4×3×5 D strides (1,0) into a 4-element buffer, C packed; (c) as (b) but C carries the same stride-0 layout;
(d) 37×53×41 as (c) — yesterday's check 2c; (e) as (a) but C carries the same stride-0 layout.
In 2.6.0–2.8.0 the rejection depends only on C's layout, not on size or which mode is stride 0: C packed → NOT_SUPPORTED at
`cutensorCreateContraction`; C with the same stride-0 layout (the usual in-place / β≠0 shape) → **accepted, wrong result**
(c, d: garbage, race; e: the buffer holds row i=0, i.e. rows 1–3 silently lost).

## Conclusions

1. **Stride-0 inputs**: the lowest installed version that accepts them and computes them correctly is **2.6.0** (2.4.1 rejects at
   descriptor creation with NOT_SUPPORTED; 2.0.1 cannot be tested here). 2.5.x is not installed, so the true threshold lies
   somewhere in (2.4.1, 2.6.0]; a gate `version >= 20600` is safe, `>= 20500` is a guess.
2. **opC = CONJ**: the lowest installed version that rejects it is **2.8.0** (`cutensorCreateContraction` → NOT_SUPPORTED).
   2.4.1, 2.6.0 and 2.7.0 accept it and **silently ignore the conjugation** (D = α·A·B + β·C) — the wrapper must reject opC≠IDENTITY
   itself when β≠0 on versions < 2.8.0 (or always, since no version honours it).
3. **Other differences**: stride-0 **outputs** are never reliably rejected (2.6.0–2.8.0 accept them with garbage whenever C shares
   D's layout) → the wrapper must validate output strides itself. Extent mismatches are **not** validated by any version at
   `cutensorCreateContraction`; they surface as `CUTENSOR_STATUS_INTERNAL_ERROR` (status 14) in `cutensorEstimateWorkspaceSize`
   in 2.4.1–2.8.0 alike — not tightened in 2.8.0 → the wrapper must check extents itself. Duplicate modes (sum-of-strides
   diagonal), over-declared alignment and the stride-0-input-into-reduction behave identically in 2.4.1+ (only stride 0 differs
   between 2.4.1 and 2.6.0+). 2.8.0 is built against cudart 13.4 (others 13.1); it ran fine with the 13.2 runtime in the binary.

## Files

* `vs.cpp` — the probe (`./vs [all | S1 … S7]`); `build.sh` — nvcc line (`-ccbin /usr/bin/g++`: the shell's conda gcc lacks cc1plus);
  `run.sh` — per-version run, writes `run-<V>.log` (per-process case fallback on non-zero exit).
* `lib<V>/` — symlink farms (`libcutensor.so`, `libcutensor.so.2`; `lib2.0.1/` also `libcublasLt.so.12` → conda pkg 12.9.1.4).
  Never put the whole pyscf `lib/` into `LD_LIBRARY_PATH`: its `libsystemd`/`libtinfo` break coreutils.
* `run-<V>.log` — raw outputs; `run-2.0.1-attempt1-cublasLt-12.1.0.26.log` — first 2.0.1 attempt.

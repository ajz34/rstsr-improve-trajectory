---
name: cutensor-version-matrix
description: 2026-11-28 cuTENSOR cross-version probe (2.0.1/2.4.1/2.6.0/2.7.0/2.8.0 on the RTX 5070) - stride-0 inputs accepted+correct from 2.6.0 (2.4.1 rejects, 2.5.x unknown), opC=CONJ silently ignored through 2.7.0 and rejected at CreateContraction only in 2.8.0, stride-0 outputs and extent mismatches never validated by any version, 2.x ABI stable (one binary ran all five), 2.0.1 unusable on this machine (needs libcublasLt.so.12, then INTERNAL_ERROR at create); nvcc needs -ccbin /usr/bin/g++ under the conda shell; never put a whole conda env lib on LD_LIBRARY_PATH
metadata:
  type: reference
---

Probed 2026-11-28 (C++, `2026-11-28-cuda-cutensor-wrap/probe/versions/`: `vs.cpp`, `build.sh`, `run.sh`,
`REPORT.md`, `run-<V>.log`, `lib<V>/` symlink farms). One binary compiled against the 2.7.0 header, run per
version through `LD_LIBRARY_PATH=lib<V>:/usr/local/cuda/lib64` (the loader picks that dir's
`libcutensor.so.2`). Where the five versions live: 2.0.1 `/scratch/miniconda3-bak/envs/pyscf/...`, 2.4.1
`/scratch/miniconda3-bak/pkgs/cutensor-2.4.1.4-*/lib`, 2.6.0 `~/.cache/uv/archive-v0/hQ6viqq5PutHCWkI/cutensor/lib`,
2.7.0 pip wheel in `torch-gpu4pyscf`, 2.8.0 `/home/a/miniconda3/envs/ruaa-dev/...` (py3.13, cudart 13.4 build,
runs on the 13.2 runtime). pip cannot fetch more (`pip index versions nvidia-cutensor` finds nothing).

## Results

| case | 2.4.1 | 2.6.0 | 2.7.0 | 2.8.0 |
|---|---|---|---|---|
| stride-0 input (`A[i,j]` strides `(0,1)`) | NOT_SUPPORTED at descriptor | ok | ok | ok |
| stride-0 on a reduced mode | NOT_SUPPORTED | ok | ok | ok |
| stride-0 output | NOT_SUPPORTED | accepted -> garbage when C shares D's layout; NOT_SUPPORTED only when C is packed | same | same |
| `opC = CONJ` (c64, beta != 0) | silently ignored | ignored | ignored | NOT_SUPPORTED at CreateContraction |
| duplicate mode `[i,i]` | diagonal, accepted | same | same | same |
| alignment 256 declared on an 8-byte-offset pointer | accepted, correct | same | same | same |
| extent mismatch (contracted / batch) | INTERNAL_ERROR at EstimateWorkspaceSize | same | same | same (not tightened) |

- Stride-0 **inputs**: lowest verified-correct version 2.6.0; 2.5.x not installed, boundary in
  (2.4.1, 2.6.0]. The wrapper gate is `cutensorGetVersion() >= 20600` (value = major*10000 + minor*100 + patch).
- No version honours `opC != IDENTITY` -> refuse unconditionally. Stride-0 outputs and extent mismatches
  are never validated -> the wrapper validates them itself.
- **2.0.1 is unusable here**: needs `libcublasLt.so.12` (CUDA 13.2 ships none); with the pyscf env's
  12.1.0.26 or the conda 12.9.1.4 it loads but `cutensorCreate` returns INTERNAL_ERROR on sm_120 / driver 595.
- 2.x ABI stable: all five export every symbol the 2.7.0-header binary needs.

## Gotchas

- `nvcc` from the conda shell fails (conda gcc without cc1plus): pass `-ccbin /usr/bin/g++`.
- Never put a whole conda env `lib/` on `LD_LIBRARY_PATH` (the pyscf env's libsystemd/libtinfo break
  coreutils: `date`, `tr`, `timeout`); symlink the one library needed into the probe's `lib<V>/`.

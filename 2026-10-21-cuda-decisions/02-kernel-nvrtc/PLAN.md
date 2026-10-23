# PLAN — implement `02-kernel-nvrtc` (kernel layer + dtype policy)

Implements DECISIONS.md D1–D12 of this directory. Record lives here;
code lands in the rstsr repo on branch **`261021/cuda`** (same branch
as device-basic, owner directive — no PR, no auto-commit; manual
review per main-repo policy when the owner asks).

## P — Pre-flight probes (before any rstsr edits; lives in `pre-flight/` here)

Throwaway bin crate, cudarc 0.19 (`nvrtc`, `driver`, dynamic loading,
cuda-version-from-build-system or `cuda-13020` per probe 1):

- **P1 compile**: NVRTC compile of a prelude string + one cast kernel
  via `compile_ptx_with_opts`; arch string from
  `compute_capability()` — try `compute_120` then `sm_120`; confirm
  `--std=c++17` accepted; load via `load_module` + `load_function`.
- **P2 latency**: cold compile+load; second-run (warm in-memory);
  restarted-process (disk-cache-shaped) numbers recorded here.
- **P3 marshal**: builder launch with `&CudaSlice<u8>`, `[u8; 16]`,
  `u64`, `usize` args only — prove no `DeviceRepr` beyond built-ins.
- **P4 streams**: two streams on ordinal 0; write on stream A, event
  record → wait on stream B, read; verify ordering recipe of D8.
- **P5 disk cache**: hash key → PTX file write/read; payload-format
  version bump invalidates.

Result recorded in this file's status log; gates the arch string and
cache-key details in B1.

## B — Batches (rstsr repo, one branch; each batch compiles + tests green)

| # | Batch | Content (decision refs) |
|---|---|---|
| B1 | Kernel infra | `nvrtc.rs` module: source templates + prelude const (`rstsr_cuda_common.cuh`); compile via safe layer, runtime arch (D3); `KernelKey` + per-ordinal in-memory cache in `CudaOrdinalObj` (D4); disk PTX cache + env knobs (D4); launch helper over `stream.launch_builder` with u8-wise marshaling (D2); event record/wait cross-stream helper (D8). |
| B2 | `CudaDType` + layout tiers | Trait + 16 impls (D5); tier selection helper (T0/T1/T2 layout inspection, D9); `arange`/`linspace` device impls (naive sequence kernels acceptable as first wave, registered). |
| B3 | Copy/assign kernels | Strided copy kernel (T1/T2) replacing `assign`/`assign_uninit`/`assign_arbitary*` host round-trips (D1b, D11 — fix-forward, strided/offset tests vs CPU); cast-kernel generation per (TA,TC) pair for in-table pairs (D6); fill/full/ones via fill kernel; `to_contig` non-contig path via copy kernel. |
| B4 | Format gather | One batched D2H gather + single sync replacing per-element `d2h_scalar` loop (D8); display parity tests vs CPU (incl. truncation, strided views, 0-d, empty). |
| B5 | Elementwise kernels | Unary/binary/compare float kernels (f32/f64 first, D7) on the T0/T1/T2 ladder (D9); `where`/mask with bool conditions; naive-path register for anything left host-side. |
| B6 | Stream relaxation | `same_device` drops stream clause → ordinal-only + flags-as-stored + order (D8); op-level event ordering exercised in tests (relabeled tensors, mixed-stream inputs); `DeviceMismatch` message names both devices. |
| B7 | Async sweep + tests | Remove per-op `synchronize` from replaced paths (D8); crate-local copy-and-compare suite per family (D12); extend for strided/offset sources (D11). |

Dependency order: P → B1 → B2 → B3 → B4; B5 after B2; B6 after B1
(parallel with B3); B7 last.

## Tests (crate-local; no GPU CI → local/manual tier)

Reuse `require_cuda!` skip macro, `host_vec1` gather helper, CPU
reference device from `test_device_basic.rs`. Per family: CUDA result
→ D2H → compare vs `DeviceCpuSerial`. Plus:

1. Tier selection unit tests: given layouts (contig / row-slice /
   padded / broadcast / negative-stride), expected tier chosen.
2. Strided/offset assign vs CPU (the D11 regression set): swapaxes→
   to_contig, reshape(copy), tile, repeat, roll, `i(1)` row views.
3. Cast kernel matrix: every in-table (TA,TC) pair incl. bool↔T,
   primitive→complex, c32↔c64; half pairs after the dtype-traits PR.
4. Format parity incl. `{:e}`, truncation at MAX_PRINT, strided views.
5. Stream matrix: relabeled tensor ops (old-stream buffer, new-stream
   device); two explicit streams; event ordering correct under
   `funcs::synchronize` fences.
6. Cache: same kernel twice → single `CudaFunction` (ptr-eq);
   `RSTSR_CUDA_CACHE=off` honored; disk write/read round-trip.
7. `same_device`: same ordinal different streams → true (post-B6);
   different ordinals → false + error names both.
8. arange/linspace values vs CPU.

## Naive-path register (replaced later — do not fossilize)

| Path | Interim | Replacement |
|---|---|---|
| `arange`/`linspace` | sequence kernel or host-gen + H2D | device sequence kernels (if not in B2) |
| remaining host round-trips (i128/u128 ends) | D2H→CPU→H2D | none planned (policy: out of `CudaDType`) |
| complex math kernels | host round-trip until prelude lands | complex prelude round |
| half math kernels | float-promotion wrappers | native half kernels if measured |

## Out of scope (per D1 + deferred list)

Reductions, GEMM/BLAS, RNG, cuSOLVER, graphs, DLPack, NCCL, allocator
framework, `TypeId` dispatch for kernel selection, dim-squash
specialization, facade `rt::cuda` wiring (follows with device-basic
PR2 when owner opens PRs).

## Ride-along side-ledger

1. `rstsr-dtype-traits`: half↔{f32,f64} `DTypeCastAPI` impls + fix
   `defaults = ["half"]` misspelling — **own reviewed PR, lands before
   B3's half-cast kernels** (D6 sequencing).
2. `rstsr-dtype-traits/src/tmp.rs` untracked artifact — owner to
   remove (do not touch from agent side; main-repo rule).
3. Device-basic status log B6/allocator discrepancy — noted in
   DECISIONS.md; allocator round owns it.

## Delivery

Same branch `261021/cuda`, no PR until owner instructs. When owner
opens PRs: this work is reviewable after device-basic's PR; PR
write-up per `pr-writeup` skill; facade wiring (device-basic PR2)
follows independently.

## Risks / checks

- cudarc 0.19 patch churn → lockfile pins; probe re-run on bump.
- Arch spelling / PTX-vs-sm_120 acceptance decided by P1, not assumed.
- `compute_120` PTX + driver JIT: if load fails on driver 595.91.07,
  fall back to `sm_120` string (P1 covers both).
- Event-tracking interplay: cudarc's automatic slice-push waits may
  double-fire with our explicit helper under some stream configs —
  P4 exercises the exact combination before B1 bakes it in.
- First-call latency acceptable per CuPy precedent; numbers recorded
  (P2) so docs can state them.

---

## Status log

(empty — implementation not started)

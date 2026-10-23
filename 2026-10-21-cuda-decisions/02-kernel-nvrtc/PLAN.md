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

### 2026-10-23 — DeviceFaer <-> DeviceCuda conversions (faer_as_default interop)

`test_kernels.rs` used `rt::asarray(vec![...])` (landing on the
`DeviceCpu` alias) + `.to_device(&d)`; with `faer_as_default` unified
in (workspace default, what rust-analyzer builds), that alias is
`DeviceFaer` and rstsr-cuda only implemented `DeviceChangeAPI` against
`DeviceCpuSerial` — E0277. Fix follows the BLAS device crates'
reference shape:

- rstsr-cuda gains cargo feature `faer = ["rstsr-core/faer"]`
  (default off; device crates carry no default-on CPU flavor).
- `conversion.rs::faer_impls` (cfg `faer`): `DeviceFaer ->
  DeviceCuda` and `DeviceCuda -> DeviceFaer` `DeviceChangeAPI` impls
  (copy semantics; `into_device` delegates to `change_device`;
  CPU-side repr is `DataOwned<Vec<T>>`, GPU-side
  `DataOwned<CudaBuffer<T>>`).
- **Test-side hardening (second half of the fix)**: under workspace
  feature unification `rstsr-core/faer_as_default` is always on (the
  facade's default), but `rstsr-cuda/faer` stays off — so the alias
  `DeviceCpu` resolves to `DeviceFaer` while the conversion impls are
  cfg'd out. Test/bench bodies that must compile in both worlds pin
  the concrete device (`rt::asarray((v, &DeviceCpuSerial::default
  ()))`); the BLAS device crates survive unification the same way
  (their bodies never touch the alias).
- New test `faer_to_cuda_and_back` (cfg `faer`): round-trip + add vs
  the DeviceFaer reference, plus `into_device`.
- Warning sweep: removed the dead `max_min_exprs`, `h2d_into`/
  `h2d_into_uninit`, `KernelScratch.len`, `TensorArg::scratch_bytes`,
  duplicate import, unused `CudaFunction` import; clippy 0 warnings
  in both `-F faer` and default configs.

Verified: `cargo test -p rstsr-cuda` (default, DeviceCpuSerial) 11+9
green; `-F faer -F rstsr-core/faer_as_default` (the failing config)
11+10 green; `cargo check --workspace --tests` clean (the
rust-analyzer scenario); clippy clean; rstsr-core lib 198 pass
untouched.

### 2026-10-23 — B1–B7 IMPLEMENTED, tests green, perf gate met (uncommitted on `261021/cuda`)

All batches done in one pass on branch `261021/cuda` (working tree only;
**no commit**, per owner instruction). Tests: `test_device_basic` 11/11 +
`test_kernels` 9/9 green; clippy clean; rstsr-core untouched (lib tests
198 pass).

Deviations from plan:
- **Marshaling architecture changed mid-implementation**: layout
  shape/stride arrays are uploaded to a per-launch **device scratch**
  buffer (tiny blocking H2D) and passed as device pointers — the first
  cut passed packed host bytes as `const usz*` args, which is
  meaningless to the kernel. Elementwise kernels decompose the linear
  index by the **output** shape and scale by operand strides (stride 0 =
  broadcast axis) — the input-shape decomposition collapsed on broadcast
  operands (found by the `(4,1)+(1,8)` test, 9j-mod-32 signature).
- **Kernel identity bug class avoided by memoizing keys on a source
  hash** (make_key): the first memo keyed on the kernel NAME, which is
  shared across dtype specializations — would have returned the wrong
  compiled kernel for a second dtype.
- `full_impl` (AnyAPI) routes through the fill kernel via a TypeId
  byte-erased dispatch (`fill_bytes_erased`); the geometric-doubling
  naive path remains for non-CudaDType T.
- `arange` length = `ceil((end-start)/step)` in f64 (the first cut
  looped scalar additions: 200k-iter arange cost 82 µs host-side).
- copy/gather kernels copy in dtype-width words (per-byte loop was 7×
  on `to_contig`).
- `same_device` relaxation (D8) landed; stream clause dropped; test
  updated to assert same-ordinal/different-stream = same-device.
- `sequence`/`linspace` made `pub` (bench access); formatting gather
  routes through the batched kernel (`gather_to_host`, one D2H sync).
- Dead code: `KernelScratch`'s blocking-upload sync removed (async
  upload, same-stream ordering); debug shims/benches (bench_parts)
  added then removed; `bench_elementwise` kept as the perf gate.

Bench (RTX 5070 Laptop, 200 iters, 30 warmup; `conda run -n ruaa-dev
python bench_cupy.py` for the cupy column):

| scenario | rstsr | cupy | ratio |
|---|---|---|---|
| fill f64 16 MiB | 0.020 ms | 0.013 ms | 1.5× |
| add f64 contig 16 MiB | 0.110 ms | 0.118 ms | 0.93× |
| add f64 strided-B | 0.041 ms | 0.037 ms | 1.1× |
| sin f64 16 MiB | 0.224 ms | 0.225 ms | 1.00× |
| to_contig(T) 4 MiB | 0.054 ms | 0.027 ms | 2.0× |
| arange f64 2e5 | 0.007 ms | 0.005 ms | 1.4× |

All within the 2× gate; math kernels at parity (sin 1.00×, add 0.93×).
Launch-path costs decomposed (bench_parts, since removed): alloc 0.6 µs,
key+lookup 0.4 µs, enqueue 3 µs — the gaps were host-side algorithmics
(arange loop, per-byte copy, per-call key rebuild), now fixed.

### 2026-10-23 — pre-flight P1–P5 PASSED (`pre-flight/`)

- **P1**: NVRTC accepts `--gpu-architecture=compute_120` (first
  candidate; not even needed to try `sm_120`); `--std=c++17` accepted.
  Driver 595.91.07, CUDA toolkit 13.2 via cuda-version-from-build-system.
- **P2 latency**: cold NVRTC compile ≈ 11.5 ms (2-kernel program);
  module load + fn lookup ≈ 5.4 ms; warm compile+load (same source) ≈
  5.4 ms. First-use cost per specialization is in the ~17 ms class —
  fine for lazy per-use compilation; disk cache still worthwhile for
  process restarts.
- **P3 marshal**: `&CudaSlice<u8>` + `u64` + `[u8; 4]` through
  `launch_builder` — no cudarc dtype trait needed (D2 confirmed
  empirically); fill and f32→f64 cast through byte buffers verified by
  read-back.
- **P4 streams**: `result::event::create/record` +
  `stream::wait_event` (`CU_EVENT_WAIT_DEFAULT`) orders a write on the
  default stream before a read on a second non-blocking stream — the
  D8 ordering recipe works.
- **P5 disk cache**: fnv1a key over (source|arch|options|payload
  format v1) → `~/.cache/rstsr/cuda/k_<hash>.ptx` round-trips.

Notes for B1: `CudaModule` does not retain PTX text — disk cache must
store the source we generate (or recompile from source on miss; text
storage is the cheap correct shape). Event destroy/record/wait are
`unsafe` fns (wrap in the helper).

### 2026-10-23 — review round 2 (`ee70fe4`) + fixes `b44cbef`

`/code-review high` on the branch through `ee70fe4` produced 8 findings;
all were re-evaluated against the code and the consequential ones
reproduced on the GPU before acceptance. Result: 4 confirmed bugs, 1
confirmed race, 1 confirmed perf issue, 1 latent API issue, 2 partial
(maintainability/perf, left open). The re-evaluation also found a missed
bug: **equal-size cross-dtype assigns were raw byte copies**
(`i64→f64` gave denormals), because the memcpy gate checked element size
instead of dtype identity.

Fixed in `b44cbef` on `261023/cuda-review-fixes` (branched from
`ee70fe4`): per-side shape/order decomposition in the copy/cast kernels
(rank/shape-changing assigns, OOB reads); dtype+shape+order gating for
the linear copy (also fixes the equal-size cast bug); `order_input` +
new `order_output` for write targets (host reads synchronize only the
allocation stream) and for `dtod_full`'s source; `T: Copy` POD gating
across storage/creation/conversion/transfer and `CudaDType: Copy`
(closes the non-POD `String` UB); `to_device` copies without the extra
clone and relabels on the same ordinal; `get_index_ptr`'s
backend-dependent contract documented in `rstsr-core`.

Verification: `test_kernels` 18, `test_device_basic` 13, lib 3, clippy
clean, `--features faer` clean; five new regressions (reshape of strided
views row/col-major, equal-size casts, re-tagged-stream in-place).

Full write-up: `REVIEW-ee70fe4.md`. Open: the format-walker mirror (#6)
and the launch-path key rebuild (#7); the perf gate has **not** been
re-run on the fixed tree yet.

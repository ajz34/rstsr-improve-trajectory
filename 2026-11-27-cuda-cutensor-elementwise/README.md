# cuTENSOR elementwise unary / binary / ternary vs the rstsr CUDA hand kernel

Investigation for rstsr: how fast and how expressive is cuTENSOR 2.7.0's elementwise API compared
with `crates-device/rstsr-cuda/src/kernels/elementwise.rs`, and what (if any) dtype promotion does it
do? Standalone C++ probe plus a small Rust baseline crate; no rstsr source was modified.

- rstsr commit: `937ce04` (`261127/cuda-prelude-f-convention`), clean tree
- GPU: RTX 5070 Laptop 8 GiB, sm_120; driver 595.99.02; CUDA 13.2.78
- cuTENSOR 2.7.0 (pip `nvidia-cutensor`, libs symlinked into `probe/lib/`)
- Rust baseline built against the same rstsr checkout (`CARGO_TARGET_DIR=…/rstsr/target`)

## Verdict

- **Efficiency**: cuTENSOR is at parity with the hand kernel for contiguous f32/f64/c32/c64
  unary/binary work (within ±3 %), and clearly better in exactly two places: both-operand
  transposed views (1.8×) and a fused `a*b+c` (1.5×, one pass vs two). The hand kernel is faster
  for `exp` (1.3×), for odd/large shapes (1.1–1.2×), for 1M-element tensors that fit in L2 (8×),
  and for small tensors (1.2–1.4×, lower per-call overhead).
- **Functionality**: cuTENSOR covers far less than the hand kernel. It has no integer/bool, no
  complex `neg/abs/exp/sqrt/…` (only `identity/conj/add/mul`), no `where`/`clip`/comparisons, no
  negative strides, no zero-size tensors, and no `f16`/`bf16` *arithmetic* in rstsr terms — but it
  *does* support f16/bf16 elementwise, which the hand kernel does not.
- **Dtype promotion**: there is a small, fixed subset — same family (real or complex), one width
  step apart (16↔32, 32↔64), compute type = the wider one. No integer promotion, no
  real↔complex, no bf16↔f32. The binary entry accepts that mix in both directions; the trinary
  entry only accepts C/D *narrower* than A/B.
- **Recommendation**: keep the hand kernel as the default path. A cuTENSOR fast path is worth
  considering only for (a) large both-strided f32/f64 tensors, (b) fused `a*b+c`, (c) f16/bf16 —
  and only with descriptor + plan caching, because plan setup costs ~50 µs per distinct problem.

## 1. What the API can express

cuTENSOR has two elementwise entry points and **no unary entry point**:

```
binary : D = opAC( α·opA(A) , γ·opC(C) )
trinary: D = opABC( opAB( α·opA(A) , β·opB(B) ) , γ·opC(C) )
```

`descC` and `descD` must be identical; the handles may be the same buffer (in-place) or different
buffers (verified). Mapping used in this probe:

| rstsr op | cuTENSOR form |
| --- | --- |
| unary `f(a)` | binary, `opC=IDENTITY`, `γ=0`, C = a valid scratch |
| binary `a+b` with two free layouts | trinary, `γ=0` (the binary entry pins the second operand to the output layout) |
| binary `a+b` same layout | binary entry directly |
| ternary `a*b+c` | trinary, `opAB=MUL`, `opABC=ADD`, in-place `C=D` or out-of-place `C≠D` |
| `where`, `clip`, comparisons | **not expressible** (`clip` = 2 passes, `where` = 0) |

Other hard facts:

- `C = nullptr` fails at execute with `INVALID_VALUE` even when `γ = 0`; the pointer must be valid
  (its contents are never read).
- Rank-0 operands fail at execute (`INVALID_VALUE`). A rank-1 extent-1 tensor broadcasts correctly
  and is the way to pass a scalar.
- α/β/γ are host scalars typed by the storage: `R32F` for f16/bf16/f32, `R64F` for f64, `C32F` for
  c32, `C64F` for c64. `COMPUTE_DESC_16F`/`16BF` are accepted (scalar becomes R16F/R16BF) but the
  compute type must match the widest storage width, so they are rarely useful.
- Descriptors carry **no offset**: windowed/sliced tensors pass `base + offset*sizeof(T)`; this
  works with a 16-byte-misaligned base (no measurable penalty).
- In-place `A = D` works for unary; `C = D` works; `D = 2A + 3C` works.

## 2. Supported dtypes and operators

Storage types: `R16F`, `R16BF`, `R32F`, `R64F`, `C32F`, `C64F` — **no integers, no bool**
(`R32I` fails at descriptor creation with `INTERNAL_ERROR`; `R8U` creates a descriptor but the plan
is `NOT_SUPPORTED`).

Operator support (probe `g1`, all verified numerically):

| op | f16 / bf16 / f32 / f64 | c32 / c64 |
| --- | --- | --- |
| IDENTITY, ADD, MUL | yes | yes |
| CONJ | no (`INVALID_VALUE`) | yes |
| NEG, ABS, EXP, SQRT, RCP, CEIL, SIN | yes | **no** (`INVALID_VALUE`) |
| MAX, MIN | yes | **no** |
| integer/bool anything | no | – |

So for complex data cuTENSOR is essentially a copy/add/mul engine; every complex transcendental,
`abs`, `neg` must stay in the hand kernel. In exchange, f16/bf16 `add/mul/exp/sqrt/max/min` work,
which the hand kernel cannot do at all (`CudaArith`/`CudaReal` exclude half).

## 3. Views

Probe `g3` (37×53 f64, `C = A + B`, 1961 elements, sampled full verification):

| view | result |
| --- | --- |
| contiguous, `A.T`, `B.T`, both `.T`, window (offset + padded row stride), stride-2 slice | **OK** |
| broadcast by omitting a mode; broadcast by stride 0 | **OK** |
| transposed output, windowed output, stride-2 output | **OK** |
| rank-6, in-place `A = D`, misaligned base pointer | **OK** |
| negative stride (flip): `stride = -53` | `INVALID_VALUE` at descriptor creation |
| zero extent (`extent = 0`) | `INVALID_VALUE` at descriptor creation |

Consequence: flips and empty tensors need a copy or a special case before cuTENSOR can be used.
The hand kernel handles both natively.

## 4. The dtype-promotion question

**There is no general promotion.** The rules found (probe `g2b`, `g2c`, `g2d`; every cell verified
against a reference rounded through the actual storage types):

1. **Trinary entry**: `typeA` must equal `typeB`. Any mixed A/B (`f32`+`f64`, `f16`+`f16`→`f32`,
   `f32`+`f32`→`f32` output with `f64` compute, real+complex) → `NOT_SUPPORTED`.
2. **A/B vs C/D**: same family only (real↔real, complex↔complex; no real↔complex), and at most
   **one width step** apart: `f16↔f32`, `f32↔f64`, `c32↔c64`. `f16↔f64` is `NOT_SUPPORTED`.
3. **Direction matters by entry point**:
   - binary entry: **both** directions work (`A=f32, C/D=f64` and `A=f64, C/D=f32`), compute type =
     the wider; e.g. `d(f64) = a(f32) + c(f64)` verifies correctly.
   - trinary entry: only **narrowing** works (`A/B=f64, C/D=f32` OK; `A/B=f32, C/D=f64` →
     `NOT_SUPPORTED`).
4. **bf16 is its own island**: `bf16↔bf16` only; `bf16→f32` is `NOT_SUPPORTED`.
5. Compute type must be the **wider** of the two storage widths (`f32` storage with `64F` compute →
   `NOT_SUPPORTED`, and `f64` with `32F` → `NOT_SUPPORTED`).
6. **No integer promotion at all**: `i32→f32`, `f32→i32` both fail; integers are not a supported
   storage type.

Practical reading: cuTENSOR can do a mixed-precision binary op such as "f32 input into an f64
accumulator", and a mixed-precision store (compute f64, store f32), but it cannot mix the two
operands of the trinary form, cannot mix more than one step, cannot mix bf16 with anything, and
cannot touch integers. Any NumPy-style promotion table must be resolved by the caller before
descriptors are built (or by the hand kernel).

## 5. Efficiency

### 5.1 Method

- cuTENSOR: `probe/bench.cpp`, preallocated buffers, plan created once, 10 warm-up executes,
  median of 7 × 50 executes with CUDA events; a sampled correctness check on every cell (512 random
  logical positions, reference rounded through the storage dtypes) guards against timing a broken
  plan. 5 s power warm-up before the cells.
- rstsr: `bench/src/main.rs`, same cells, same shapes/strides, `*_with_output` (preallocated
  output, kernel + tier checks) for binary; `rt::exp` / `assign` (allocating) for unary;
  `mul → add` two-op sequence for `muladd` because rstsr has no fused multiply-add. Same warm-up
  and median-of-7 × 50 protocol.
- Neither side includes output allocation for the binary cells; the unary rstsr cells do include it
  (there is no `_with_output` unary API), which flatters cuTENSOR slightly — `exp` still wins for
  rstsr.
- Both processes ran on the same GPU with a 5 s power warm-up; the f64 2048² add anchor agrees
  within 1 % between them across runs.

### 5.2 f64, 2048×2048 (n = 4.19 M), ms, lower is better

| cell | cuTENSOR | rstsr | rstsr / cuTENSOR |
| --- | ---: | ---: | ---: |
| add (contiguous) | 0.3100 | 0.3024 | 0.98 |
| mul (contiguous) | 0.3090 | 0.3008 | 0.97 |
| copy / identity (2 accesses) | 0.2062 | 0.2036 | 0.99 |
| exp (transcendental, compute-bound) | 0.5071 | 0.3919 | **0.77** |
| add, A transposed | 0.3090 | 0.3703 | **1.20** |
| add, B transposed | 0.3076 | 0.3692 | **1.20** |
| add, both transposed | 0.3015 | 0.5434 | **1.80** |
| add, windowed (offset + padded stride) | 0.3103 | 0.3016 | 0.97 |
| add, stride-2 slice | 0.4994 | 0.4914 | 0.98 |
| add, broadcast `[2048,1]` | 0.2103 | 0.2024 | 0.96 |
| add, transposed output | 0.3000 | 0.3189 | 1.06 |
| `a*b+c` | **0.3976** (fused) | 0.6035 (2 kernels) | **1.52** |

### 5.3 dtype

| cell | cuTENSOR | rstsr | rstsr / cuTENSOR |
| --- | ---: | ---: | ---: |
| f32 add contiguous | 0.1372 | 0.1262 | 0.92 |
| f32 add A.T | 0.1350 | 0.1772 | 1.31 |
| f32 stride-2 | 0.2465 | 0.2304 | 0.93 |
| f32 `a*b+c` | 0.1508 (fused) | 0.2584 (2 ops) | 1.71 |
| c64 add contiguous | 0.6132 | 0.6052 | 0.99 |
| c64 add A.T | 0.6096 | 0.7249 | 1.19 |
| c32 add contiguous | 0.3061 | 0.3007 | 0.98 |
| f16 add contiguous | 0.1170 | no baseline | – |
| bf16 add contiguous | 0.0300 | no baseline | – |

### 5.4 size (f64 contiguous add)

| size | cuTENSOR | rstsr | rstsr / cuTENSOR |
| --- | ---: | ---: | ---: |
| 100×100 (10 k) | **0.00251** | 0.00299 | 1.19 |
| 1000×1001 (1.0 M) | 0.07333 | **0.00918** | **0.13** |
| 2048² (4.2 M) | 0.3100 | 0.3024 | 0.98 |
| 2047×2053 (4.2 M) | 0.3439 | 0.3018 | 0.88 |
| 4093×4099 (16.8 M) | 1.4474 | 1.2033 | 0.83 |
| 4096² (16.8 M) | 1.2265 | 1.2028 | 0.98 |

The 1000×1001 row is not noise: cuTENSOR's trinary kernel stays at HBM speed (327 GB/s) there while
the rstsr kernel reaches 2.6 TB/s because the 24 MB working set stays in L2. cuTENSOR's own copy
kernel at the same shape does reach 969 GB/s, so this is a property of its elementwise kernels, not
of the measurement. For sizes above L2 both converge to ~330 GB/s.

### 5.5 Overheads

| item | cuTENSOR | rstsr |
| --- | ---: | ---: |
| create descriptors + preference + estimate + plan | 44–72 µs | n/a |
| of which `cutensorCreatePlan` alone | 11–16 µs | n/a |
| second identical plan (internal cache) | 11–24 µs | n/a |
| execute call, host side | ~2.0 µs | ~3.0 µs (S-size wall per op) |
| first execute after a new plan | +0.06–0.5 ms | NVRTC compile on first call (cached on disk) |

`JIT_DEFAULT` vs `JIT_NONE` made no difference on these cells (0.3101 vs 0.3100 ms). The ~50 µs
setup is 20× the kernel time of a 100×100 op, so a cuTENSOR path must cache descriptors and plans
keyed by (dtype, extent/strides, operators, scalars type); execute alone is comparable to the hand
kernel's per-call cost.

## 6. What this means for rstsr

1. **Keep the hand kernel as the default.** It is at parity on the common contiguous cases, wins on
   `exp`, odd/large shapes, L2-resident tensors and small tensors, and it expresses everything
   (complex math, `where`, `clip`, comparisons, negative strides, empty tensors, integers, bool,
   mixed dtypes, rank-0).
2. **The two places where cuTENSOR is materially better** are both-strided inputs (1.8× on
   both-transposed f64) and the fused `a*b+c` (1.5×). Those are the only candidates for a fast path.
   `f16`/`bf16` is a capability gap rather than a speed gap — cuTENSOR is the only option today.
3. **A cuTENSOR path is only viable with plan caching.** 50 µs of setup per call would dominate
   every small op; with a cache the per-call cost is ~2 µs, i.e. the same as one hand-kernel launch.
4. **Do not plan on cuTENSOR for promotion.** Only the narrow rule set in §4 is supported; rstsr
   will still need its own promotion/`TypePromote` layer to pick the compute type and detect the
   unsupported combinations before calling.
5. **Negative strides and zero-size tensors must stay in the hand kernel** (or be materialized
   first); broadcast, windows, steps, transposed outputs and in-place are all fine.

## 7. Files

```
DECISIONS.md                  experiment decisions D1–D5 (from the grilling round)
GRILL-R1-QUESTIONS.md         the questions and the fact base
probe/common.h                descriptors, plans, staging, f16/bf16/complex encode-decode, reference
probe/facts.cpp               functionality probes, groups g1 (ops) g2 (promotion) g2b g2c g2d g3 (views) g4 (misc)
probe/bench.cpp               timing harness, writes probe/results/cutensor_bench.csv
probe/build.sh                nvcc build (sm_120), targets facts / bench / dbg
probe/results/g*.log          probe outputs; bench.log + cutensor_bench.csv timings
bench/src/main.rs             rstsr hand-kernel baseline, writes bench/results/baseline.log
results/compare.csv           merged cuTENSOR/rstsr timings (35 matched cells)
```

Reproduce:

```bash
cd probe && ./build.sh facts && ./build.sh bench
LD_LIBRARY_PATH="$PWD/lib:/usr/local/cuda/lib64" ./bin/facts
LD_LIBRARY_PATH="$PWD/lib:/usr/local/cuda/lib64" ./bin/bench
cd ../bench && CARGO_TARGET_DIR=/scratch/rstsr_pack/rstsr/target cargo build --release
LD_LIBRARY_PATH="/home/a/Software/OpenBLAS-0.3.34/lib:/home/a/Software/blis-2.0/lib" \
  /scratch/rstsr_pack/rstsr/target/release/baseline
```

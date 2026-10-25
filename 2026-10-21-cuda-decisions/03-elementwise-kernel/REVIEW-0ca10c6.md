# Review of `0ca10c6` — total diff of the two elementwise commits

- Review target: the combined diff **`b44cbef..0ca10c6`**, i.e. the last two
  commits on `261023/cuda-review-fixes`:
  `ee0518d` (planned elementwise kernels, narrow index tier, tiled transpose,
  template complex) and `0ca10c6` (fix of the previous review's findings).
- Requested scope: review the two commits as **one total diff**, not per
  commit; correctness first, efficiency next; no limit on the number of
  findings; API may be changed.
- Method: full line-by-line read of the diff, eight review angles (kernel C
  sources, launchers, tensor-tier semantics, removed behavior/stream ordering,
  tests/benches, efficiency, Rust/CUDA pitfalls, conventions), then an
  independent empirical re-check of every consequential finding on the
  installed GPU.
- Environment: RTX 5070 Laptop (sm_120, 8 GB), driver 595.91.07, NVRTC 13.x,
  `--std=c++17` (crate default options), Rust debug build (the crate's
  `-lineinfo` option is on there). `ptxas` 13.x from `/usr/local/cuda` was
  used for the target-architecture checks.
- **No rstsr code was changed by this review.** Severity tags as in
  `REVIEW-ee0518d.md`: **[BUG]**, **[RACE]**, **[PERF]**, **[API]**,
  **[RESOURCE]**, **[NIT]**, **[TEST]**.

The fix commit `0ca10c6` closes essentially every finding of
`REVIEW-ee0518d.md`: the ILP-4 loop/entry pairing, the scalar-bound dtype
reinterpretation (by restricting where/clip to same-dtype and documenting the
divergence), the fill recursion, both-scalar NaN handling, NVRTC-safe query
names, in-place `not`, the complex operator surface, the order-aware tiled
transpose, the C99 complex ports, `cpow` corners, real `expm1`/`log1p`, the
Wide-arm `ParamsN` zeroing, the `clip_ss_pn` double advance, and the lazy
source rendering in the planned path. The remaining problems are a smaller,
different set: two integer unaries that still cannot compile, an operand-swap
in the new integer-pow guard, a missing `elems_per_thread` in the two clip
narrow arms (int32 offset wrap above 2^29 elements), an over-strong PTX
instruction (`.NaN` needs sm_80), a family of NaN/zero parity mismatches, and
a set of capability/API holes. They are all reachable from ordinary tensor
calls, and all were reproduced or proved on the installed machine unless
marked otherwise.

---

## Part A — status of the previous review (`REVIEW-ee0518d.md`)

| # | Prior finding | Status in `0ca10c6` | Evidence |
|---|---|---|---|
| 1 | `scalar_assign` ILP-4 on a loop-less `_p` entry | **fixed** | Flat now uses `rstsr_scalar_assign_f` (RSTSR_STEP) + ept(4); Narrow `_pn` + ept(4) |
| 2 | `where_sc` ILP-4 on a loop-less `_p` entry | **fixed** | new `rstsr_where_sc_f`; ept only on the looping Flat/Narrow arms |
| 3 | clip scalar bounds reinterpreted as the value dtype | **fixed by restriction** | same-dtype `OpClipAPI<T,T,T>`; divergence documented in `lib.rs` |
| 4 | `where` scalar arm reinterpreted | **fixed by restriction** | same-dtype `OpWhereAPI<T,T>` |
| 5 | rank>4 fill recursion | **fixed** | `fill_generic_same`; `regression_fill_generic_rank5` passes |
| 6 | both-scalar clip drops NaN | **fixed** | `clip2s` NaN-aware stage selection; `regression_clip_paths` NaN arm passes |
| 7 | `isnanf`/`isinff`/`isfinitef` | **fixed** | unsuffixed overloads + complex `||`/`&&` forms |
| 8 | in-place `not` NVRTC error / no ordering | **fixed** | `unary_launch_inplace`; `regression_inplace_not` passes |
| 9 | complex header has no C++ operators | **fixed** | full operator surface in `rstsr_complex.cuh` |
| 10 | tiled transpose row-major-only; grid.y cap | **fixed** | `col_major` indexing; 65535-tile fallback; `regression_col_major_transpose` |
| 11 | integer clip NVRTC | **fixed** | integer `max`/`min` expressions; integer clip test |
| 12 | `bool_not_scratch` OOB | **fixed** | in-kernel `inv` flag, no scratch; strided-condition test |
| 13 | `cabs` overflow / hyperbolic NaN | **fixed** | `cabs = hypot`; C99 `csqrt`/`ccosh`/`csinh`/`ctanh`/`ctan` |
| 14 | `cpow(0,0)` | **fixed** | `y == 0 -> 1`; complex-edge test |
| 15 | `expm1`/`log1p` cancellation | **fixed** | real intrinsics; complex `cexpm1`/`clog1p` |
| 16 | integer `pow` accepts negative exponent | **partially fixed** | scalar exponent now rejected, but the **tensor** exponent still silently yields 1; the new guard checks the wrong operand for `pow(scalar, tensor)` — see B3/B7 |
| A1 | promoting binaries same-dtype only | still true, now documented (owner: promotion is a bonus) | `lib.rs:18-25` |
| A2 | Wide arms marshal `ParamsN` as zeroed | **fixed** | `plan_bytes` widens; `widen_pack_rematches_wide_plan` unit test |
| A3 | `clip_ss_pn` advances `lin` twice | **fixed** | single `lin += w` |
| A4 | per-launch source assembly + full-source hash | **fixed for the planned path** | `planned_function` renders only on an intern miss; non-planned launchers still hash per call (pre-existing) |
| A5 | narrow arms lack ept(4) | **fixed except clip** | binary/scalar assign and where_sc gained ept; the two clip narrow arms did **not** — see B4 |
| A6 | `fill_v` grid sized from `n` | **fixed** | `launch_n = n/2`; the tail loop is still dead for even `n` (C5) |
| A7 | dead code / stale comments | mostly fixed | `copy_gather2d`, `ipowu`, double ept, `pub mod kernels`, bool in `CudaReal` all gone; stale `operators.rs:6-10` comment and the `t_real` duplicate remain (C5) |
| A8 | transcendentals accept integers | **fixed** | `CudaTranscend` bound |
| A9 | `order_output` audit | unchanged | out-of-place planned launchers still omit it; latent under the current allocation discipline (C4) |
| T1 | `require_cuda!` missing in one test | **fixed** | every test in `test_ops_wave.rs` guards |
| T2 | coverage holes | **improved, still large** | see C6 |
| T3 | `probe_maxnan_f64_jit` can never fail | **half fixed** | `#[ignore]`d, still only prints; the reason it fails is now understood (B5) and the test should be deleted or turned into an assertion |
| T4 | no-op `[[test]]`; no CI | **half fixed** | `[[test]]` gone; still no workflow / policy entry for `rstsr-cuda` (C7) |
| T5 | bench harness issues | partially | `.max(1)` guards added; still no baseline/cupy columns and the allocation churn is inside every cell |

---

## Part B — new findings (ranked)

### B1. [BUG] `rt::round` on integer tensors cannot compile (NVRTC), CPU returns identity

- Site: `src/kernels/exprs.rs:153-165` (`round_even`), `src/operators.rs:419-441`
  (`OpRoundAPI` bound `T: CudaReal`, i.e. every integer).
- Cause: `round_even` has branches for complex and half, then emits
  `rint{}({x})` with `f(tc)` empty for integers → `rint(long long)`.
  `impl_rounding_unary!` (floor/ceil/trunc, `operators.rs:369-412`) has the
  integer-identity branch; `round` is the one that does not.
- Repro (RTX 5070):
  `rt::round_f(&Tensor<i64>)` → `DeviceError: NVRTC compile failed for
  kernel 'rstsr_unary_f' … more than one instance of overloaded function
  "rint" matches … argument types are: (const long long)`.
  CPU returns the input unchanged (`ExtNum::ext_round` is identity).
- Suggested fix: return `x` unchanged for integer CNAMEs in `round_even`
  (the same branch already exists in `impl_rounding_unary`).

### B2. [BUG] `rt::signbit` on integer tensors cannot compile, CPU works

- Site: `src/kernels/exprs.rs:187-200` (`signbit`), `src/operators.rs:744-760`
  (`OpSignBitAPI` bound `CudaReal`).
- Cause: the integer path falls through to `signbit(x)`, which NVRTC declares
  only for `float`/`double`/`long double`; overload resolution is ambiguous
  for every integer type.
- Repro: `rt::signbit_f(&Tensor<i64>)` → NVRTC "more than one instance of
  overloaded function \"signbit\" matches … (const long long)"; CPU returns
  `[false, false, true]`.
- Suggested fix: emit `((x) < 0)` for signed, `false` for unsigned CNAMEs
  (CPU `ExtReal::ext_signbit`).
- Related nit: the half branch emits `signbitf(...)`, which does not exist in
  NVRTC either. It is dead today (`CudaReal` excludes half), but it will be
  the first failure when half joins the trait.

### B3. [BUG] `pow(scalar, tensor)` validates the scalar *base* as if it were the exponent

- Site: `src/operators.rs:320` inside `impl_promoting_binary!`'s
  `op_mutc_numa_refb`; the expression is `rstsr_ipow(v, t[it])`, so the
  scalar `a` is the base.
- Repro: `rt::pow_f(-2i64, &Tensor([0,1,2,3]))` →
  `Err(InvalidValue: integer power requires a non-negative exponent)`; CPU
  returns `[1, -2, 4, -8]`. `pow(-1i64, &idx)` (a common parity idiom) fails
  the same way. Meanwhile the exponent (the tensor) is still not checked at
  all (B7).
- Suggested fix: drop the check from `op_mutc_numa_refb`; keep it only
  in `op_mutc_refa_numb` (scalar exponent). A tensor exponent needs a
  device-side check (see B7), not this host branch.

### B4. [BUG] Clip narrow arms omit `elems_per_thread(4)`: 4× redundant work, and int32 `lin` wraps above 2^29 elements → out-of-bounds device writes

- Sites: `src/kernels/elementwise.rs:1830-1863` (`clip_launch` narrow arm
  selecting `rstsr_clip_pn`) and `:1947-1962` (`clip2s_out_launch` narrow arm
  selecting `rstsr_clip_ss_pn`). Both launch the 4-iteration grid-stride
  entries with the default `elems_per_thread == 1`, so `grid = ceil(n/256)`
  while each thread iterates 4 times with `int lin` and `lin += w`
  (`w = gridDim.x*blockDim.x`; entries at `:1604-1620`, `:1640-1680`).
- Consequences:
  1. Every strided (non-flat) clip with `n < 2^31` runs four times through
     the loop; three quarters of the iterations are rejected by `lin < n`.
     This is the *common* clip shape: `rt::clip(&view.swapaxes(..), (lo, hi))`
     selects `rstsr_clip_ss_pn` (verified with the `launch_trace` feature at
     `n = 32`: `[rstsr-cuda] clip2s_out entry=rstsr_clip_ss_pn n=32`).
  2. For `n >= 2^29 + 1` the third increment wraps. With
     `w = 256*ceil(n/256)` the largest linear id is `4w - 1`; at
     `n = 2^29 + 1`, `w = 2^29 + 256` and `4w - 1 = 2^31 + 1023`. The top
     1024 threads compute a negative `lin`, which still satisfies `lin < n`;
     `rstsr_plan_offset_n` then returns a negative flat offset (for a
     contiguous output) or a garbage magic-divmod offset, and the kernel
     writes outside the allocation. `n = 2^29` is ~0.5 GiB for u8/i8 and
     ~2 GiB for f32, so a transposed u8/i8 matrix clip on this 8 GB card is
     enough. (The earlier iterations overflow at larger `n`; the clip arms
     are the only narrow kernels in the file that can reach the wrap because
     all other `_pn` arms set ept(4), which makes the maximum id exactly
     `INT_MAX` at `n = 2^31 - 1`.)
- The prior review's A5 asked for ept on the narrow arms; the clip arms were
  the ones left behind. The fix is to add `.elems_per_thread(4)` to both
  narrow branches (the `lin += w` advance is already correct after A3), which
  also removes the 4× overhead.

### B5. [BUG / portability] f32 `maximum`/`minimum`/`clip` require sm_80 because of `max.NaN.f32`

- Site: `src/kernel.rs:71-76` (`rstsr_maxnanf`/`rstsr_minnanf` inline PTX),
  used by `exprs::nan_max_min` for every f32 max/min and by both clip
  expressions.
- `max.NaN`/`min.NaN` were introduced with PTX ISA 7.0 / `sm_80`; `ptxas`
  rejects them for `sm_75`:
  `Modifier '.NaN' requires .target sm_80 or higher`.
  Confirmed by compiling the crate's prelude + `MATH_HELPER` +
  `o[0] = rstsr_maxnanf(o[1], o[2])` with NVRTC for `compute_75` and feeding
  the PTX to `ptxas -arch=sm_75`. Unused helpers are eliminated, so a
  trivial kernel still compiles for sm_75; only kernels that call the helper
  fail, i.e. exactly f32 `maximum`, `minimum`, `clip` (and anything built on
  them) return `DeviceError` on V100/T4/RTX-20xx-class devices.
- The f64 helpers use a portable select chain (`kernel.rs:63-70`), so this is
  an avoidable asymmetry. `max.NaN.f64` does not exist in PTX at all (the
  `#[ignore]`d probe fails with `CUDA_ERROR_INVALID_PTX`), so the f64
  fallback is required; the f32 path can use the same helper, or gate the
  asm on `__CUDA_ARCH__ >= 800` inside the header (NVRTC defines
  `__CUDA_ARCH__` for the target).
- Update the comment at `kernel.rs:59-62`: the limitation is "`.NaN` is
  f32-only and sm_80+", not "f32-only on this driver's JIT".

### B6. [BUG / parity] NaN payloads and signed-zero results of `maximum`/`minimum`/`clip`/`rem` differ from the CPU reference

Verified bit patterns on the GPU (CPU = `DeviceCpuSerial`):

| input | CUDA | CPU |
|---|---|---|
| `f32 max(NaN(0x7fc12345), 1.0)` | `0x7fffffff` | `0x7fc12345` |
| `f32 min(1.0, NaN(0x7fc12345))` | `0x7fffffff` | `0x7fc12345` |
| `f32 max(0.0, -0.0)` / `max(-0.0, 0.0)` | `+0` / `+0` | `-0` / `+0` |
| `f32 min(-0.0, 0.0)` / `min(0.0, -0.0)` | `-0` / `-0` | `+0` / `-0` |
| `f64 max(0.0, -0.0)` / `max(-0.0, 0.0)` | `+0` / `+0` | `-0` / `+0` |
| `f64 min(-0.0, 0.0)` / `min(0.0, -0.0)` | `-0` / `-0` | `+0` / `-0` |
| `f64 rem(inf, 2.0)` | `0x7fffffffffffffff` | `0x7ff8000000000000` |
| `f32 clip([NaN(0x7fc12345)], (0,1))` | `0x7fffffff` | `0x7fc12345` |

- Cause 1: PTX `max.NaN` returns a canonical NaN and does not preserve the
  operand payload; `rstsr_nanf`/`rstsr_nan` use `0x7fffffff` /
  `0x7fffffffffffffff`, whereas Rust uses `0x7fc00000` /
  `0x7ff8000000000000`.
- Cause 2: `fmax`/`fmin` return `+0`/`-0` for mixed zeros, while the CPU
  `ExtReal::ext_max`/`ext_min` (and NumPy 2.4.6) tie-break to the second
  operand, so `max(0.0, -0.0)` is `-0` there. The f64 select-chain helper
  therefore still differs from the CPU on one zero ordering.
- Value semantics (`isnan`, `==`) are unaffected; this matters for bitwise
  parity tests and for downstream payload-sensitive code. All wave tests use
  `==`/tolerance, so none of this is caught today.
- Suggested fix: use Rust's canonical NaN constants; use the same select
  chain for f32 (also fixes B5) or add a `signbit`-preserving zero rule; add
  a bitwise `maximum/minimum` case with NaN and ±0 to the wave tests.

### B7. [BUG] Integer `pow` silently returns 1 for a negative tensor exponent; exponent width diverges; `rstsr_ipow` has signed-overflow UB

- Sites: `src/kernel.rs:110-121` (`rstsr_ipow`), `src/kernels/exprs.rs:80-92`
  (`pow` dispatch), `src/operators.rs:302/320` (host checks),
  `src/lib.rs:18-25` (documented as a divergence).
- Evidence:
  - `rt::pow_f(&[2,3], &[-1,2])` → `Ok([1, 9])`; CPU returns
    `Err(InvalidValue: integer power requires a non-negative exponent)`.
  - `rt::pow_f(&[2], &[4294967296])` → CUDA `[0]`; CPU `[1]` (the CPU
    truncates the exponent to `u32`, `2^0 = 1`, the device computes
    `2^(2^32) mod 2^64`).
  - `rstsr_ipow(base, exp)` takes a 64-bit signed exponent; `u64` exponents
    above `i64::MAX` convert negative and the `while (exp > 0)` loop is
    skipped → 1. On 64-bit inputs the loop's final `base *= base` is a signed
    overflow (C++ UB) even when the returned result fits.
- The project already settled this semantics: negative integer exponents are
  rejected (`InvalidValue`) precisely because a silently-wrong value was
  rejected before (`rstsr-pow-signbit-decisions` memory; C7). Documenting a
  "returns 1" divergence in `lib.rs` therefore contradicts a settled
  decision, not just a stylistic preference. The tensor-exponent form is the
  rare path, so a host pre-scan of the exponent tensor (one D2H + sync only
  for integer `pow` with a tensor exponent) is acceptable and gives exact
  CPU parity; if that is rejected on async-design grounds, the divergence
  must at least be elevated from a `lib.rs` footnote to an explicit tested
  decision, and the width must still be aligned. Align the exponent with
  `ExtNum::ext_pow` (`u32`, not 64-bit signed) and use unsigned
  accumulation in `rstsr_ipow` to remove the signed-overflow UB.

### B8. [BUG / parity] Integer `/`, `%`, `floor_divide` by zero: CUDA returns values, CPU panics

- Sites: `operators.rs` (`OpDivAPI`/`OpDivAssignAPI` emit `(a / b)` with no
  guard), `exprs.rs:60-76` (`rem` guards zero → 0), `operators.rs:345-356`
  (`floor_divide` guards zero → 0).
- Verified:
  - `rt::div_f(&[7,-8,i32::MIN], &[0,0,-1])` → `Ok([-1, -1, -2147483648])`;
    CPU panics on the first element ("attempt to divide by zero").
  - `rt::rem_f` by zero → `[0,0,0]`; CPU panics.
  - `rt::floor_divide_f` by zero → `[0,0]`; CPU panics.
- The `exprs::rem` docstring claims "the CPU guard `0 % x == 0`", but that
  guard lives in the `ExtNum::ext_rem` trait method; the CPU *device* only
  routes floats through `ExtReal::ext_rem` and uses Rust's `%` for integers
  (`op_ternary_arithmetic.rs:114`), so the CPU call panics rather than
  returning 0. `-1` for `7/0` is neither the CPU (panic) nor NumPy (0 with a
  warning); it is PTX/hardware behavior, not a contract.
- Suggested: decide one contract (error or documented 0) for *all* integer
  division families, implement it uniformly (`rem`/`%=`/`floor_divide` guard
  to 0 today, `/` and `/=` do not), and fix the docstring. At minimum, route
  `/` and `/=` through the same guard and record the divergence in
  `numpy_differences.md` if it stays.

### B9. [API] bool bitwise ops are missing although both the module comment and the dtype docs say they exist

- Site: `src/kernels/ops_binary_extra.rs:63-78` claims "bool bitwise is
  served through the same C operators"; the bounds are `CudaInt`
  (`dtype.rs:113-127`), which excludes `bool`. `dtype.rs` says bool is
  "emitted as logical `&&/||` and `!`-based xor by the op layer", but no op
  layer does that.
- Consequence: `rt::bitand/bitor/bitxor(&bool_t, &bool_t)` and the compound
  assigns compile on `DeviceCpuSerial` (`bool: BitAnd/…`) and fail to
  resolve `OpBitAndAPI<bool, bool, bool, D>` for `DeviceCuda`.
- Fix: implement the three bitwise ops and the three compound assigns for
  bool (`(a & b)`, `(a | b)`, `(a != b)` — or `&&/||/!=`, which coincide for
  bool), or narrow the doc to state the gap. `&&`/`||` short-circuit is
  legal here but pointless; plain `&`/`|` on C `bool` already works.

### B10. [API/BUG] complex → bool cast is rejected while the CPU casts it

- Site: `src/kernels_ops.rs:217-232` (`cast_expr_c`): the
  `("rstsr_c32" | "rstsr_c64", _)` "complex-to-real" arm comes before the
  `(_, "bool")` arm, so `bool` targets are caught by the rejection.
- Verified: `bool_tensor.assign_f(&Tensor<Complex<f64>>)` → CUDA
  `Err(InvalidValue: complex-to-real cast has no kernel)`; the CPU call
  returns `Ok(())`. The CPU cast is `self != Complex::zero()`
  (`DTypeCastAPI<bool>` in `rstsr-dtype-traits/src/promotion.rs:160-165`),
  i.e. `re != 0 || im != 0` — the same "nonzero test" the shadowed
  `(_, "bool")` arm intends.
- Fix: move `(_, "bool")` above the complex arm and emit
  `((src).re != 0 || (src).im != 0)` for complex sources (or
  `src != rstsr_complex<T>(0,0)`). complex → other real types stays rejected
  as decided in D6.

### B11. [API] complex `==` / `!=` are missing although the CPU implements them

- Site: comparison impls `operators.rs:67-113` bound `CudaCmp`
  (`dtype.rs:96-112`), which deliberately excludes complex. The comment says
  "rstsr declines them at the tensor layer as CPU does", but the CPU
  `OpEqualAPI`/`OpNotEqualAPI` bound is `PartialEq`
  (`op_ternary_common.rs:80-81`), which `num::Complex` implements, so
  `rt::equal(&c64, &c64)` works on the CPU.
- Equality is well-defined for complex; only the ordering traits are not.
  Split the marker (`CudaEqCmp` incl. complex for `==`/`!=`, `CudaCmp` for
  `< <= > >=`) and fix the dtype.rs comment. `a == b` already exists on the
  C++ `rstsr_complex` template.

### B12. [API] Over-broad dtype bounds turn compile-time rejections into runtime errors

- `dtype.rs:7-9` states the policy: "unsupported dtypes fail by missing
  bound (compile error)"; D5 records the same. Several impls do not follow it:
  - `impl_query_unary!` (`operators.rs:685-729`) is bounded on `CudaDType`
    and raises `UnImplemented` at runtime for integer CNAMEs:
    `rt::is_nan_f(&Tensor<i32>)` → `Error::UnImplemented("query unary dtype
    int not covered")`, while the CPU rejects the same program at compile
    time (`T: ComplexFloat`). Verified.
  - `OpNotAPI` (`operators.rs:661-687`) is bounded on `CudaDType`; a direct
    device-trait call with `f64` reaches NVRTC and fails on `~double`. The
    public `rt::not` is protected by the tensor tier's `T: Not` bound, so
    this one is only reachable through the public device traits.
  - `OpRealAPI::op_muta` (`operators.rs:503-506`) silently returns `Ok(())`
    for complex (`TOut != T`), unlike `OpImagAPI::op_muta`, which raises
    `UnImplemented` for the same case. Unreachable through the owned-tensor
    in-place path (it requires `TOut = T`), but it is a landmine if the
    dispatch changes.
- Fix: bound query unaries on `CudaFloat | CudaComplexFrom` (half handled by
  float promotion), `OpNotAPI` on `CudaInt`-plus-bool, and mirror the
  `ABS_SAME_TYPE` guard in `OpRealAPI::op_muta`.

### B13. [RESOURCE/BUG-latent] `FN_INTERN` pins modules and contexts for the process lifetime

- Site: `src/kernels/elementwise.rs:328-376`. The process-global
  `HashMap<(usize ordinal, String), Arc<CudaFunction>>` is never invalidated.
  In cudarc 0.19, `CudaFunction` holds `Arc<CudaModule>`, which holds
  `Arc<CudaContext>`; every planned specialization therefore pins the
  ordinal's primary context and module forever. This defeats the
  `Weak<CudaOrdinalObj>` registry design (the registry becomes irrelevant
  for anything that ever ran a planned op), and after a primary-context
  reset the interned function still points at the retired context, so every
  later planned launch fails with `CUDA_ERROR_INVALID_HANDLE`.
- The per-ordinal kernel cache inside `CudaOrdinalObj` has the right
  lifetime; the intern map should too (key it by `Weak<CudaOrdinalObj>` or
  store the discriminator inside the ordinal object, and drop entries when
  the context retires). Alternatively keep the map but verify the context
  identity on hit and re-intern on mismatch.

### B14. [PERF] The planned hot path still allocates and hashes per launch

- `plan_operand` allocates a `Vec<(u64,u64)>` and boxes an iterator
  (`Box<dyn Iterator<Item=usize>>`) **per operand per launch**
  (`plan.rs:120-190`): 2–4 heap allocations plus per-axis work for every
  binary/unary/where/clip call.
- `planned_function` builds `disc = format!(...)` + `subs_disc` (a fresh
  `String`) + mutex lock per launch (`elementwise.rs:343-376`); the previous
  measurement `key+lookup 0.4 µs` was for a static pre-assembled key.
- `offset_ptr` calls `buf.device_ptr()` for every operand; cudarc's
  `device_ptr` performs event tracking (`cuStreamWaitEvent`/`cuEventRecord`)
  and is a driver call, replicated 2–4× per op.
- The non-planned launchers in `kernels_ops.rs` still assemble ~5 KB of
  source (plus the 27 KB complex header) and SipHash it on every call
  (`copy_dispatch_generic`, `fill_generic_same`, `binary_assign_generic`,
  `where*_generic`, clip generic, sequence/linspace).
- Suggestions: stack arrays/`SmallVec` and direct index iteration in
  `plan_operand`; a `&'static`-keyed or thread-local cache for the
  discriminator; cache the raw `CUdeviceptr` in `CudaBuffer` (this also
  removes the per-fetch event calls); extend the `OnceLock`-assembled /
  interned pattern to the remaining generic launchers and benchmark a
  small-tensor cell (the bench's current smallest cells are 2048²).

### B15. [PERF] `full`/`ones` of complex and half still take the host round-trip path

- `creation.rs:22-37` routes anything not in `is_fill_supported`
  (`kernels_ops.rs:893-912`) to `transfer::fill_naive` (H2D + O(log n) D2D
  doublings + full stream synchronize). `fill_launch`/`fill_generic_same`
  are generic over `CudaDType` and already handle `rstsr_c32/c64`,
  `__half`, `__nv_bfloat16`; the TypeId list and `fill_bytes_erased`'s
  `try_fill!` list are a third hand-maintained copy of the dtype table.
- This is pre-existing behavior, but the diff added the generic fill kernel
  that makes the parity fix trivial. Suggested: move the fill capability onto
  a creation-trait bound (the `DEFER` comment already names this) or widen
  the TypeId list, and delete one of the two duplicated lists.

### B16. [BUG/API] Mixed complex/real `rem` compiles at the tensor tier but cannot be generated for CUDA

- Site: `src/kernels/exprs.rs:60-77` (`rem` dispatches only on the left
  operand's complex-ness / the output type), `src/kernels/ops_binary_extra.rs:14-60`
  (`OpRemAPI` bounded `TA/TB/TC: CudaArith`), and the `OpRemAssignAPI` impl
  below it, which reuses the same builder.
- Reachable through `rt::rem`/`rt::rem_assign` because `num-complex`
  implements `Rem` for the mixed pairs:
  - `rt::rem_f(&Tensor<Complex<f64>>, &Tensor<f64>)` → CPU
    `[1+0i, -1+1i]`; CUDA NVRTC error
    `no instance of function template "rstsr_crem" matches ... (const rstsr_c64, const double)`.
  - `rt::rem_f(&Tensor<f64>, &Tensor<Complex<f64>>)` → CPU `[2+0i, 2+0i]`;
    CUDA NVRTC error `no suitable conversion function from "const rstsr_c64" to "double" exists`
    (`rstsr_frem(double, complex)`).
- Unlike `where`/`clip`/promoting binaries, this family still advertises the
  mixed form in its bounds, so it fails at launchtime instead of at the type
  level; `lib.rs` does not document it.
- Suggested fix: branch on the whole `(tx, ty, tc)` triple — when `tc` is
  complex, wrap the non-complex operand with `rstsr_make_complex` and use
  `rstsr_crem`; when `tc` is real and either operand is complex, reject
  (no such CPU case). Or restrict the impls to same-dtype and document, as
  the other mixed families do.

### B17. [PERF] The tiled transpose fast path only triggers for "wide" transposes

- `transpose2d_plan` (`elementwise.rs:919-953`) requires `cols >= rows` for
  the `OutContigInSwap` orientation; tall cases (e.g. `to_contig` of a
  `10000×100` transposed view) fall back to the planned gather kernel even
  though the kernel could handle them by swapping the grid axes (grid.x has
  the 2^31 limit, grid.y the 65535-tile limit). Not a bug; a missed
  optimization for one half of the 2-D shapes. Consider always mapping the
  larger dimension to `grid.x`.

---

## Part C — latent issues, cleanup, tests, documentation

### C1. [LATENT] The tiled transpose is not alias-safe
`copy_launch`/`copy_transpose_launch` do not check that source and
destination do not overlap (`elementwise.rs:1033`, `:955-1005`). Each block
stages only its own tile and writes immediately, so an in-place transpose
(`c` and `a` sharing an allocation with swapped layouts) corrupts the
tiles a later block has not read yet. The safe tensor API cannot construct
this today (borrowck forbids `&mut` + `&` on one tensor), but a raw/Cow repr
or a future `*_into` path can. A `if c_ptr == a_ptr` (or extents-overlap)
fallback to the element-wise gather is one line.

### C2. [LATENT] `plan_operand` classifies an all-size-1 layout as Flat without checking its size against `n`
`plan.rs:167-172`: `0 => OperandPlan::Flat` (all axes dropped) and
`1 if sq[0] == (n, 1) => Flat`. If any future caller passes a shape-`[1]`
(or rank-0) operand layout while the output has `n > 1` (rather than the
broadcast-padded shape with stride 0), the kernel reads `a[lin]` for
`lin in 0..n` — out of bounds. All current tensor-tier callers broadcast
first, so it is latent; a `debug_assert!(layout.size() == n)` (or an
explicit `n == 1` guard) documents and enforces the invariant.

### C3. [PERF/NIT] Clip launcher cleanup
`clip_launch` still deviates from the shared pattern:
`let _ = (&blob, &hib);` (`:1819`) is a no-op; `narrow` re-tests the
all-Flat predicate that the entry selection already computed (`:1820-1824`);
`zn_zeroed_n()`/`zeroed_plan_bytes_as_slice()` (`:1968-1981`) exist only
here while every other launcher uses `PlanBytes::zeroed()`; the 4-byte arm of
`scalar_bytes_nan` (`:1871-1885`) is unreachable because its only call is
under `TD::CNAME == "double"`. None of this is wrong; it is the kind of
divergence that let B4 slip through.

### C4. [LATENT] Out-of-place planned launchers do not publish on the output's allocation stream
`binary_launch` (377), `unary_launch` (459), `scalar_launch` (614),
`where_launch` (1511) and `where_launch_sc` (1142) call `order_input` but
not `order_output`; fill/copy/copy_transpose/assign/clip/unary-inplace all
do. This is safe today because every output comes from `uninit_impl` on the
exec stream (the invariant recorded in `cuda-write-target-ordering`), but it
is an asymmetry that will break the first `*_into`/user-provided-output
path. Deeper fix: give `FlatLaunch` typed in/out buffer slots and record the
events inside `launch()` so no call site has to remember.

### C5. [NIT] Dead code, stale comments, naming
- `operators.rs:6-10`: the comment `/// CUDA C call expression
  name<suffix>(arg).` sits above the arithmetic `duplicate_item` and
  describes nothing there.
- `t_real` is defined twice with identical bodies (`operators.rs:181`,
  `exprs.rs:202`); keep the `pub(crate)` one in `exprs`.
- 11 of the 12 `_pn` bodies declare `int w = gridDim.x * blockDim.x;` and
  then advance with `lin += gridDim.x * blockDim.x;`; every NVRTC compile
  emits `warning: variable "w" was declared but never referenced` (the only
  user is `rstsr_clip_ss_pn`, the same family as B4). Use `w` or delete it.
- `rstsr_fill_v`'s tail loop (`:735-736`) is dead: the host only selects the
  vector entry when `n % 2 == 0` (`:773`), so `tail == n`.
- `rstsr_unary_inplace_n` is the only narrow entry not named `_pn`;
  `binary_assign` has no `_f` entry, `clip_ss` has no `_p` entry, `clip` has
  no `_f` entry, and the module doc (`elementwise.rs:10-12`) claims "each
  program carries both the `_p` and `_f` entry points". Align the names or the
  doc — the clip/assign families were exactly where the ept bug hid.
- `OpRealAPI::op_muta` silent `Ok(())` for complex (B12).
- `probe_maxnan_f64_jit` (`kernel.rs:488-514`) is `#[ignore]`d and can never
  fail; now that the reason is known (no `.NaN.f64` in PTX), delete it or
  assert the expected `DeviceError`.
- NaN constants: `rstsr_nanf`/`rstsr_nan` use `0x7fffffff`/`0x7fffffffffffffff`
  rather than Rust's canonical patterns (B6).

### C6. [CLEANUP] Structural duplication that makes the launch layer error-prone
- 11 launchers re-expand the same Flat/Narrow/Wide argument-marshaling match
  (~330 lines, 52 `unwrap_or(&zero)` sites), and the Rust argument order has
  no compile-time link to the generated CUDA parameter list — a mis-slotted
  plan silently misinterprets kernel parameters. A single `PlanArgs` builder
  taking `&[(OperandPlan, ptr, base)]` plus scalar slots (and returning the
  entry name) would remove the class; `binary_assign_launch`'s Flat and Wide
  arms are already byte-identical copies.
- 34 entry functions are maintained as 13 hand-written templates; each
  family's `_pn` body is a token-for-token clone of its `_p` body. A
  width-token template would halve the count; the drift is visible in C5.
- `copy_cast`'s shape-unequal arm (`kernels_ops.rs:265-291`) is a verbatim
  duplicate of `copy_dispatch_generic` (`:976-1016`); delegate with the
  `in[io]` expression instead.
- `KernelKey` stores the full source (several KB) as a live `HashMap` key and
  joins options with `\u{1}`; `make_key`'s `extra` is always `&[]` and
  `compile_options` re-splits that string. `KernelCachePublic` +
  `type KernelCache = KernelCachePublic` name one type twice.
  `FlatLaunch` carries `n`/`elems_per_thread` that the tiled caller must pass
  as a dummy 0 — a separate `TiledLaunch` would be clearer.
- `CudaComplexFrom` and `CudaArith` have exactly the same member set, edited
  in lockstep; `dtype.rs`'s class predicates are re-derived by substring
  tests on `CNAME` in `exprs.rs`/`kernels_ops.rs` instead of one descriptor
  with `IS_FLOAT`/`IS_SIGNED_INT`/… consts.
- The unsafe scalar marshaling (`copy_nonoverlapping` from `&T`) is inlined
  at ~17 sites although `scalar_bytes_of` exists; the
  `&mut CudaBuffer<MaybeUninit<T>> -> &mut CudaBuffer<T>` cast is re-inlined
  at ~11 sites although `assignment::transmute_mut` is the documented single
  place for it; `gather_to_host` hand-rolls the scratch allocate+H2D that
  `KernelScratch::new` provides 40 lines below. Any change to the
  byte-marshaling contract currently has to be replicated at every site.

### C7. [TEST] Coverage and policy
- Untested paths that would have caught B1–B16: integer `round`/`signbit`
  (only f64 is tested), `pow` with a scalar *base*, tensor-exponent `pow`,
  `maximum`/`minimum` with NaN payload and ±0, float `rem` (only integer rem
  is tested), `copysign`/`nextafter`/`log_add_exp`/`reciprocal`
  (grep: zero hits), complex `asin`/`acos`/`atanh`, bool bitwise,
  complex→bool assign, mixed complex/real `rem`, `where` with a complex
  payload, f16/bf16 (zero hits in all tests, which is how D10 stayed
  hidden), and the Wide plan tier end to end (only unit-tested through
  `widen_pack`).
- Policy: the ten `regression_*` tests live in `tests/test_ops_wave.rs`, but
  `rules/testing.md` rule 1 puts backend regressions in the device crate's
  `tests/issues/` with an issue number. If these are pre-merge fixes of the
  branch's own bugs, say so in the file (or move them once an issue exists);
  the crate also has no `entry_row_cuda` / symlink set / `tests/README.md`,
  and no CI workflow or `rules/testing.md` entry — none of the 800+ new
  assertion lines gates anything. The benchmark harness is committed in the
  device crate although the policy designates `crates-bench/` (future) and
  ad-hoc numbers in this repository; the checked-in bench prints only rstsr
  medians, so the cupy/torch comparisons in `REPORT.md` cannot be reproduced
  from the harness.
- `probe_maxnan_f64_jit` should assert or go (see C5).

### C8. [DOC] Divergence documentation should list what actually differs
`lib.rs:15-25` currently lists where/clip/promoting-binaries same-dtype,
integer transcendental widening and the tensor-exponent pow. Missing:
NaN payload canonicalization and ±0 ties in `maximum`/`minimum`/clip
(B6); integer division-by-zero results (B8); bool bitwise / complex equality
gaps (B9/B11); complex division using Smith vs the CPU's naive `num-complex`
divide (CUDA matches NumPy/SciPy, CPU does not — worth a note and a CPU-side
follow-up); integer exponent width (B7); mixed complex/real rem (B16); `fill`
 dtype support (B15).
`dtype.rs`'s `CudaCmp` comment ("rstsr declines complex comparisons as the
CPU does") is wrong for `==`/`!=`; `exprs.rs:54`'s "CPU guard `0 % x == 0`"
is wrong for the CPU device; `kernel.rs:59-62`'s `.NaN` reason is wrong
(B5).

### C9. [STYLE/POLICY] Convention items from `rstsr-agents`
- `CudaBuffer::new` (`buffer.rs:31`) is a public fallible constructor without
  the `_f` suffix required by `rules/code-style-general-guide.md`; the same
  crate's `DeviceCuda::new_f`/`CudaBuffer::clone_f` follow it.
- The new op families use `macro_rules!` (`ops_binary_extra.rs:67,126,187,
  221,274`; `operators.rs:267,369,685`) where the rule prefers
  `duplicate::duplicate_item` (already used for the same shape of impl at
  `operators.rs:13-19`, `dtype.rs:27`).
- Comment/docstring policy: `elementwise.rs:40-48` (plan node doc),
  `plan.rs:27-31` (Granlund-Montgomery proof in an API docstring),
  `elementwise.rs:971-978` and `:1918-1923` (8- and 6-line explanatory
  blocks) exceed "concise, descriptive, not explanatory" (1-2 lines, max 4);
  keep the derivation in the PR/design notes, not in docstrings.
- `tests/test_*.rs` each redefine `host`/`host_vec1`/`host_nd` and
  `require_cuda!` (three copies with slightly different bounds); a
  `tests/common/mod.rs` would make the next bound fix single-site.
- `kernels_ops`/`kernels` were made `pub(crate)` in this diff (good); the
  rest of the module-surface cleanup (the `KernelCachePublic` alias, the
  unused `make_key` `extra` parameter) is in C6.

---

## Part D — pre-existing on the branch (outside the two commits), still worth a ticket

These were found during the review but the files are unchanged (or changed
only cosmetically) in `b44cbef..0ca10c6`; they are listed so the branch does
not ship with them unnoticed.

1. **[RACE, reproduced] Cross-stream input use-after-free.** `CudaBuffer::device_ptr`
   drops cudarc's `SyncOnDrop` guard inside the accessor and always passes
   the *allocation* stream, so the read event never covers the exec stream;
   a foreign-stream input can be freed/reused while a kernel still reads it
   (the write side is covered by `order_output`). Reproduced on the local
   GPU: retag one clone of the ordinal-0 device to stream S2, then run
   `rt::add(&a_on_S0, &rt::arange((0., 4_194_304., 1., &dev2)))`; the
   arange temporary is freed on S2 with no dependency on the S0 kernel, the
   next S2 allocation returns the same block, and the add output is wrong
   in 4,194,303 of 4,194,304 elements (`same block reused: true`); keeping
   the temporary alive gives zero errors. This is the documented
   `set_stream` workflow, not an exotic misuse. It predates the diff, but
   every new planned launcher calls `device_ptr` 2–4× per op, so the
   exposure is now much wider.
2. **[BUG/UB, reproduced] D2H staging.** `transfer.rs:80` runs
   `copy_nonoverlapping` from an empty `Vec`'s dangling pointer on a
   zero-element read; reproduced as a process abort in debug
   (`unsafe precondition(s) violated: ptr::copy_nonoverlapping requires
   that both pointer arguments are aligned and non-null`), release UB. All
   the new zero-size early returns avoid the kernels but not this read
   path. The same staging path reinterprets a `Vec<u8>` as `T`
   (`copy_nonoverlapping::<T>`, `ptr::read::<T>` in `transfer.rs:44/80`,
   `kernels_ops.rs:590`), which needs `align_of::<T>()` alignment that
   `Vec<u8>` does not guarantee — stage into `Vec<T>` (or use
   `read_unaligned`) for f64/u64/`Complex<f64>` tensors.
3. **[BUG] Cross-ordinal copies cannot work.** `dtod_full`/`Event::create`
   create events in the thread-current context and never enable peer access;
   `into_device` across ordinals fails with `CUDA_ERROR_INVALID_HANDLE`.
   Either implement it or reject cross-ordinal transfers explicitly.
4. **[PERF] `get_index` does one D2H + one stream sync per index element**
   (`device.rs:184`), and `rstsr-core`'s advanced indexing calls it per
   entry; a 1e6-index tensor costs 1e6 syncs although `gather_to_host`
   already batches this.
5. **[BUG/PARITY] `arange`/`linspace`.** The CUDA `arange` lacks the CPU's
   trailing-element pop, and the two disagree with each other and with
   NumPy 2.4.6 on some float ranges (`arange(0.5, 0.51, 0.01)`: CUDA 2,
   NumPy 2, CPU 1 — the CPU is the outlier here; `arange(0.0f32, 1.0,
   0.01f32)`: CUDA 101, CPU/NumPy 100). Integer ranges above 2^53 are also
   wrong: the CUDA path converts `start` and `end` to f64 separately and
   subtracts, so `arange(i64::MAX-5, i64::MAX, 1)` and
   `arange(1<<60, (1<<60)+10, 1)` return 0 elements (the CPU computes the
   span exactly in `isize` and returns 5/10); the empty result then feeds
   D2. Separately, NVRTC's default `--fmad=true` lets `start + step*i`
   become an FMA, giving 1-ulp differences from the CPU's
   multiply-then-add; if bitwise parity matters, pass `-fmad=false` (the
   D3 "strict defaults" decision).
6. **[ROBUSTNESS] Disk cache has no fallback.** `load_or_compile_ptx`
   returns cached PTX unconditionally; a PTX-ISA/toolchain mismatch
   (`cuda-12080` vs `cuda-13020` feature) makes every launch fail until the
   user deletes the cache, and `arch_slot` caches a failed capability query
   forever.
7. **[BUG/PARITY] float → integer casts are not the Rust-saturating casts
   the CPU reference performs.** `cast_expr_c` emits a plain C cast
   (`((int)(v))`, `((signed char)(v))`). Verified by assigning a float tensor
   into integer tensors:
   - `f64 -> i32`, input `[1e300, -1e300, +inf, -inf, NaN, 2.7, -2.7]`:
     CUDA `[INT_MAX, INT_MIN, INT_MAX, INT_MIN, INT_MIN, 2, -2]`;
     CPU `[INT_MAX, INT_MIN, INT_MAX, INT_MIN, 0, 2, -2]` (NaN -> 0).
   - `f64 -> i8`, same input: CUDA `[-1, 0, -1, 0, 0, 2, -2]` (the
     intermediate saturates to i32, then truncates to the narrow target);
     CPU `[127, -128, 127, -128, 0, 2, -2]` (saturates to the target type).
   The new shape-equal `copy_cast` fast path keeps the same expression, so
   it does not introduce the divergence but does route more casts through
   it. If CPU parity is required, emit saturating conversions
   (`__double2lrint_r` is not enough; use `max/min` clamping before the
   cast, and the NaN->0 rule explicitly).
8. **[LATENT/OOB] CUDA printing with `MIN_PRINT == 0` reads past the
   buffer.** `format.rs:28` mirrors the CPU print walker's *sentinel*
   advance (`p = shape - min_print`) as a visited index; with the public
   knob `rstsr_core::format::MIN_PRINT = 0` the sentinel equals the axis
   extent, and `gather_to_host` reads one element past every truncated
   prefix axis (the CPU walker itself never visits it). The print mirror
   should either share the walker or exclude the sentinel.
9. **[MULTI-GPU] Context binding gaps.** `DeviceCuda::set_stream_f`
   validates only the stream's cached ordinal, not its CUDA context;
   `transfer.rs`'s raw `memcpy_*_async`/`memset_*_async` calls do not bind
   the stream's context; `Event::create` uses the thread-current context.
   Single-GPU single-primary-context runs are unaffected, but any
   non-primary context or second ordinal breaks ordering/launches with
   `CUDA_ERROR_INVALID_HANDLE` rather than an error message.
10. **[BUG, merge blocker] Half-dtype kernels cannot compile: no NVRTC
    include path for `cuda_fp16.h`/`cuda_bf16.h`.** `dtype.rs:47-56` puts
    `#include <cuda_fp16.h>` / `#include <cuda_bf16.h>` in `INCLUDES`, but
    `compile_now` (`kernel.rs:223-231`) builds `CompileOptions` with
    `..Default::default()`, i.e. `include_paths: []`. Reproduced directly
    with the crate's options (`t.cu(1): catastrophic error: could not open
    source file "cuda_fp16.h" (no directories in search list)`) and
    end-to-end: `rt::greater(&f16, &f16)` panics with `NVRTC compile
    failed`. f16/bf16 tensors can be created and host-filled, but every
    half kernel (comparisons, and any future arith/transcend) fails; the
    crate has no test that compiles a half kernel. Fix: pass the toolkit
    include dir (or cudarc's bundled headers) in `CompileOptions`, and add
    an f16/bf16 comparison test. Pre-existing, but it makes the `CudaCmp`
    half impls and `dtype.rs`'s half claims non-functional.
11. **[CI, merge blocker] `--all-features` cannot build the crate, and the
    workspace clippy workflow uses it.** `Cargo.toml:43-44` exposes cudarc's
    mutually exclusive `dynamic-linking` and `static-linking`; cudarc's
    `build.rs:90` panics when both are on. Reproduced:
    `cargo check -p rstsr-cuda --all-features` → `Both "dynamic-linking"
    and "static-linking" features are active, this is a bug`.
    `.github/workflows/clippy.yml:26` runs
    `cargo clippy --all-targets --all-features -- -D warnings` at the
    workspace root, and `rstsr-cuda` is a workspace member
    (`Cargo.toml:13`), so clippy fails for every PR/push before compiling
    anything. Fix: make the two linking modes mutually exclusive in this
    crate's feature graph (or stop forwarding both to cudarc; e.g. keep one
    and expose the other behind a cfg that cannot be `--all-features`-ed),
    or exclude the crate from that workflow. Pre-existing in the branch,
    but it gates CI.

---

## Part E — verified-correct (negative space)

To scope the review honestly, these were checked and found sound:

- The plan layer: `AxisPlan`/`AxisPlanN` magic division (fuzzed up to 2^16
  and at large divisors), `narrow_pack` bounds (divisor < 2^31, stride
  magnitude, accumulated extent), signed-stride two's-complement encoding,
  `widen_pack` recomputation, flat/broadcast gates, rank-4 limit and
  row/col decomposition. The `PlanBytes`(136 B)/`PlanBytesN`(68 B)
  `repr(C)` mirrors match the device structs; a `const` size assertion
  would make that explicit.
- The 1-D launch geometry: every `_f`/`_pn` entry that sets ept(4) is a
  4-step grid-stride loop with `grid = ceil(n/(256*4))`, and the maximum
  narrow id is exactly `INT_MAX` at `n = 2^31 - 1`; no other family has the
  clip-arm gap (B4). Wide arms take the single-step `_p` entries and widen
  `ParamsN`.
- The tiled transpose in both orientations and both device orders
  (`regression_col_major_transpose`, `new_ops_transpose_copy_tiled`), the
  grid.y cap, and the out-of-range tile guards (all threads reach
  `__syncthreads`).
- `where` inversion (`a[ia] ^ inv`), the four-layout where family, the
  clip bound-kind dispatch (tensor/scalar/None, `(None,None)`, both-scalar
  lean path), and the NaN-aware `clip2s` f64 host check.
- `rem` float parity (floored remainder) for finite/inf/zero cases, integer
  truncated `%` (apart from B8's zero-divisor contract), `floor_divide`
  signs, float rem's inf/zero corner cases, the C++ complex operator surface,
  Smith division, `cabs = hypot`, the C99 `csqrt`/`cexp`/`clog`/`cpow`/
  `crem` corners, and `cexpm1`/`clog1p`.
- Cross-stream ordering on existing-buffer writes (fill/copy/assign/clip/
  in-place unary), the `MaybeUninit` transmutes, the include-splicing dedupe,
  and the generic rank>4 fallbacks (probed with a permuted 5-D tensor across
  binary/unary/scalar/where/clip/assign/rem/sin).
- Test results on the installed machine: 16/16 `test_ops_wave`, 18/18
  `test_kernels`, 13/13 `test_device_basic`, 18/18 lib tests; clippy clean;
  benches compile. (`cargo test -p rstsr-cuda` as a whole fails only on the
  untracked local debug probes `tests/tmp_audit*.rs`, which are not part of
  the diff; the `arange` failure in them is Part D.5's CPU-side issue.)

---

## Verdict

The planned elementwise layer is in good shape after `0ca10c6`: the walk,
the marshaling ABI, the tiled transpose, the complex header and the
where/clip families are correct on this hardware, and the previous review's
16 findings are addressed. What remains is not architectural: two integer
unaries still cannot compile (B1/B2), the new pow guard checks the base
instead of the exponent (B3), the clip narrow arms can write out of bounds
above 2^29 elements (B4), the f32 NaN helpers need sm_80 (B5), and a family
of dtype-capability and NaN/zero parity holes (B6–B12) is silent or
runtime-only. B3/B7 also contradict a settled project decision (negative
integer exponents must be rejected, not answered with 1). I would fix B1–B5
before the branch is proposed for merge, and B6–B12/B16 either in the same
wave or as explicit follow-up tickets. Three Part D items should also gate
the merge even though they predate these commits: the reproduced cross-stream
use-after-free (D1), the broken half-dtype kernel path (D10) and the
`--all-features` clippy failure (D11). Separately, D7 (float→int cast
saturation) is a sharp, easy-to-hit parity bug that deserves its own ticket
now that the cast fast path is in place.

---

## Appendix — how the reproductions were made

All GPU reproductions used a throwaway crate outside both repositories with
a path dependency on `crates-device/rstsr-cuda`, calling the `*_f` tensor
forms (`round_f`, `signbit_f`, `pow_f`, `rem_f`, `div_f`, `floor_divide_f`,
`is_nan_f`, `assign_f`) or the infallible forms inside `catch_unwind`; values
were compared against `DeviceCpuSerial` bit-for-bit for floats, including
the mixed `rem` `(c64, f64)`/`(f64, c64)` matrix. The clip entry selection
was observed with `cargo run --features rstsr-cuda/launch_trace`. The sm_80
check compiled the crate's own `CUDA_PRELUDE` + `MATH_HELPER` sources with
NVRTC for `compute_75` and ran `ptxas -arch=sm_75`. D10 was checked directly
with NVRTC (`#include <cuda_fp16.h>` under the crate's compile options) and
end-to-end by the finder's `rt::greater` probe; D11 with
`cargo check -p rstsr-cuda --all-features`; D1/D2 were reproduced by the
finder's cross-stream and host-read probes. The B4 overrun is proved by the
kernel arithmetic and the traced entry selection; it was not executed at
the 0.5–4 GiB scale it needs.

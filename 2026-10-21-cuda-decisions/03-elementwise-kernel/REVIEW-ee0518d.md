# Review of `ee0518d` — planned elementwise kernels, tiled transpose, template complex

Review of the last commit on `261023/cuda-review-fixes`:

- Review target: **`ee0518d`** ("rstsr-cuda: planned elementwise kernels,
  narrow index tier, tiled transpose, template complex"), diff against
  `b44cbef`.
- Method: full line-by-line read of the diff, six parallel review angles
  (kernel C sources, launch layer, tensor-tier semantics/API conventions,
  removed behavior + stream ordering, tests/benches, cleanup/efficiency),
  then an independent empirical re-check of every consequential finding.
- Environment: RTX 5070 Laptop, driver 595.91.07, NVRTC 13.x,
  `--std=c++17 --gpu-architecture=compute_80`; all reproductions used a
  throwaway harness outside both repositories.
- **No rstsr code was changed by this review.** Severity tags as in
  `REVIEW-6b56c18.md`: **[BUG]**, **[RACE]**, **[PERF]**, **[API]**,
  **[NIT]**, **[TEST]**.

The commit adds a large planned-kernel layer (`kernels/elementwise.rs`,
`kernels/plan.rs`), a narrow (u32) plan tier, tiled 2-D transpose, the
`rstsr_complex<T>` template header, and many new device ops. The planned
walk itself is sound (the magic division was fuzzed, the Flat/broadcast
gate holds), but the **launch layer mixes ILP-4 grids with non-looping
entries**, and **scalar bounds/operands bypass dtype promotion**. Both
classes produce silent wrong results in the most common (contiguous /
scalar) call shapes.

---

## Top findings (ranked)

| # | Tag | Site | Finding | Verdict |
|---|---|---|---|---|
| 1 | [BUG] | `elementwise.rs:1422` | `scalar_assign_launch` applies `elems_per_thread(4)` to the loop-less `rstsr_scalar_assign_p`: `x += 1.0` updates only ~n/4 elements | Confirmed on GPU |
| 2 | [BUG] | `elementwise.rs:1192` | `where_launch_sc` has the same ept/entry mismatch for the Flat and Wide arms: `where(cond, x, scalar)` writes ~n/4 | Confirmed on GPU |
| 3 | [BUG] | `elementwise.rs:1726-1735`, `ops_ternary.rs:96` | Clip marshals scalar bounds as `size_of::<TD>()` raw bytes with no promotion: `clip(f32, (0.0f64, 1.0f64))` returns all zeros | Confirmed on GPU |
| 4 | [BUG] | `ops_ternary.rs:51-56`, `:25` | `where` reinterprets a `TY` scalar as `TX`: `where(cond, x_f32, 1.0f64)` writes 0.0; `TOut = TX` diverges from CPU promotion | Confirmed on GPU |
| 5 | [BUG] | `elementwise.rs:769-771`, `kernels_ops.rs:313-317/937-952` | Fill's Generic fallback re-enters the planned path: rank>4 fill recurses until the process dies | Confirmed (process killed) |
| 6 | [BUG] | `elementwise.rs:1907-1914` | Both-scalar clip drops NaN inputs: `clip([NaN,..], (0.,1.))` yields `hi` instead of NaN | Confirmed on GPU |
| 7 | [BUG] | `operators.rs:687-693` | Query unaries emit `isnanf`/`isinff`/`isfinitef`, undefined in NVRTC: `is_nan/is_inf/is_finite` fail for f32/f16/bf16/c32 | Confirmed on GPU |
| 8 | [BUG] | `operators.rs:657-663` | In-place `not` passes an `a[io]` expression to the out-of-place launcher (kernel vars are `b`/`d`): NVRTC error; also no `order_output` and mirrored aliasing | Confirmed on GPU |
| 9 | [BUG] | `operators.rs:15-72/588-637`; `rstsr_complex.cuh` | The complex header defines no C++ operators, but arithmetic/square/ext_neg/compound assign emit `+ - * /`: all complex binary arithmetic fails at NVRTC | Confirmed on GPU |
| 10 | [BUG] | `elementwise.rs:931-1000` | `transpose2d_plan` is order-aware but the tiled kernel is row-major only: ColMajor non-square transpose is permuted (12798/12800 wrong); grid.y overflows above 2,097,120 rows | Confirmed on GPU |
| 11 | [BUG] | `elementwise.rs:1907-1914`, `:1856-1867` | Both-scalar clip emits float-only `isnan/fmax/fmin` for every `CudaReal`: integer clip fails at NVRTC | Confirmed on GPU |
| 12 | [BUG] | `elementwise.rs:1971-1983` | `bool_not_scratch` writes through the condition view's layout into an offset-less `n`-element buffer: device OOB write for sliced/strided conditions | Confirmed by walk analysis |
| 13 | [BUG] | `rstsr_complex.cuh:82/128-149/417-438` | `cabs` is `sqrt(re²+im²)` (CPU uses `hypot`): abs/log/sqrt/sign overflow, and sinh/ccosh/ctanh/ctan return NaN for large inputs | Confirmed on GPU |
| 14 | [BUG] | `rstsr_complex.cuh:442-447` | `cpow(0,0)` returns 0 (CPU/NumPy: 1); `cpow(0, y<0)` also returns 0 instead of inf | Confirmed on GPU |
| 15 | [BUG] | `operators.rs:211-219` | `expm1`/`log1p` are composed as `exp(x)-1`/`log(1+x)`: total cancellation for small x | Confirmed on GPU |
| 16 | [BUG] | `kernel.rs:110-121`, `exprs.rs:85-87` | Integer `pow` never validates a negative exponent: `pow(2i64,-1)` returns 1 where CPU returns `Err(InvalidValue)` | Confirmed on GPU |

The table is ordered by severity × reachability; the narrative below is in
the same order. Items 16 and the "Additional findings" section are outside
the 15-finding output contract but belong in the record.

---

## 1. [BUG] Scalar compound assign writes only a quarter of a contiguous tensor

`scalar_assign_launch` selects `rstsr_scalar_assign_p` for the Flat arm
(`elementwise.rs:1394-1404`) — a one-element-per-thread kernel with
`if (lin >= n) return;` (`SCALAR_ASSIGN_ENTRIES`, `:1262-1270`) — but
launches it through the Flat+Wide arm that sets `.elems_per_thread(4)`
(`:1419-1429`, ept at `:1422`). `FlatLaunch::launch` (`kernel.rs:342-348`)
then sizes the grid for 4 elements per thread, while each thread writes one.

Reproduced:

```rust
let mut x: Tensor<f64, _, _> = rt::zeros(([1000], &d));
rt::add_assign(&mut x, 1.0f64);
// updated = 256, first unchanged index = 256
```

The inverted shape of the bug is why the suite misses it: a *strided*
destination takes the Narrow arm, which happens not to set ept and is
correct; a contiguous destination is the broken one. The wave test uses a
10×10 tensor (`test_ops_wave.rs:100-104`), which fits in one block
(ceil(100/1024)·256 = 256 threads) and passes. Applies to every scalar
compound assign family built on this launcher (`+= -= *= /= %=`, bitwise,
shifts).

`binary_assign_launch`'s Wide arm (`:1335-1344`, ept at `:1338`,
`rstsr_binary_assign_p` non-looping) has the same defect; there the trigger
is a Wide plan (n ≥ 2^31 or a stride/offset walk beyond i32), so it is rare
but silent.

Fix: give the Flat/Wide arms a looping entry (`rstsr_scalar_assign_f` /
`rstsr_binary_assign_f` with `RSTSR_STEP`) or drop `elems_per_thread` from
those arms; the Narrow arm should then receive ept(4) for ILP.

## 2. [BUG] `where` with a scalar arm writes only a quarter of the output

Same pattern in `where_launch_sc`: the Flat arm (all operands Flat) and the
Wide arm select `rstsr_where_sc_p` (`:1162-1171`) — no loop
(`WHERE_SC_ENTRIES`, `:1098-1110`) — and both set `.elems_per_thread(4)`
(`:1189-1200`, ept at `:1192`).

Reproduced:

```rust
let cond: Tensor<bool, _, _> = rt::full(([1000], false, &d));
let one: Tensor<f64, _, _> = rt::ones(([1000], &d));
let w = rt::where(&cond, &one, 2.0f64);
// only 256 of 1000 elements are 2.0; the rest are uninitialized device memory
```

and, for the scalar-*middle* form (`where(cond, scalar, tensor)`), the
sliced-condition probe in the harness showed 744/1000 mismatches. The wave
test exercises only the tensor-tensor `where` (`test_ops_wave.rs:366-368`,
on transposed views), which goes through `where_launch` whose `_f` entry
does loop.

Fix: add a looping `rstsr_where_sc_f` entry, or remove ept from the
Flat/Wide arms. Note the Wide arm additionally marshals any `ParamsN`
operand through `plan_bytes(...).unwrap_or(&zero)` — a zeroed rank-0 plan
(see Additional findings).

## 3. [BUG] Clip reinterprets scalar bounds as the value dtype

The tensor tier does **not** promote clip bounds: `OpClipAPI` is
instantiated with `LO::TD`/`HI::TD` unchanged and the reference CPU device
promotes inside the op (`ClipRes<TA,TB,TC>`,
`rstsr-core/src/device_cpu_serial/operators/op_quaternary_common.rs:85-115`).
The CUDA impl, however, sets `type TOut = TA` (`ops_ternary.rs:96`) and the
both-scalar fast path copies `size_of::<TD>()` bytes out of a `TB`/`TC`
scalar (`elementwise.rs:1726-1735`), declaring the kernel params as `@TD@`
(`CLIP_SS_ENTRIES`, `:1594`). The comment "the tensor tier promotes scalars
to the same dtype" is false.

Reproduced (the most natural call on an f32 tensor, since float literals
default to f64):

```rust
let x: Tensor<f32, _, _> = rt::asarray((vec![1.0f32, 2.0, 3.0], &d));
rt::clip(&x, (0.0f64, 1.0f64));
// CUDA: [0.0, 0.0, 0.0]  (low 4 bytes of f64 1.0 are 0)
// CPU : [1.0, 1.0, 1.0]
```

The general path has the same marshaling (`:1784-1847`; scalar blob is
TD-sized while the kernel param is `@TB@`), and `clip_generic` repeats it
(`kernels_ops.rs:1271-1284`). When `TB > TD` the host copies TD bytes from a
smaller scalar (or reads past it when `TB < TD`) and sends the wrong number
of bytes for the kernel parameter, shifting `v_hi`/`n` in the param list.
The mixed tensor-bound path also computes in `TA` and returns `TA`, while
CPU returns the promoted `ClipRes` (e.g. `clip(&f32, &f64_lo)` returns f32
here, f64 on CPU).

Fix: follow the CPU contract — promote all three operands to
`ClipRes<TA,TB,TC>` at the tensor tier (or convert each bound with
`DTypeCastAPI` into the compute dtype before marshaling), and marshal using
the declared parameter type. Add tests with f32 values + f64 literals and
with integer literals.

## 4. [BUG] `where` reinterprets a scalar arm instead of converting it

`op_mutd_refa_refb_numc` does
`let b_typed: &TX = unsafe { &*(addr_of!(c) as *const TX) }`
(`ops_ternary.rs:51-56`) on the claim that the tensor tier pre-promotes the
scalar. It does not: `op_where.rs:132-163` passes the raw `TY` scalar. CPU
promotes with `<TX as DTypePromoteAPI<TY>>::promote_other`
(`op_quaternary_common.rs:24-28,59`).

Reproduced:

```rust
let x: Tensor<f32, _, _> = /* [10.0; 12] */;
rt::where(&cond, &x, 1.0f64);
// CUDA: [10, 0, 10, 0, ...] (reads the low 4 bytes of f64 1.0)
// CPU : [10, 1, 10, 1, ...]
```

`where(cond, &x_f64, 0i32)` reads 8 bytes from a 4-byte scalar (host UB /
garbage high half). `TOut = TX` (`:25`) also diverges from CPU's
`TX::Res` for mixed tensor arms: `where(cond, &i32, &0.5f64)` returns i32
with truncation here, f64 on CPU. The same CPU promotion is the reference
for `maximum`/`minimum`/`pow`/`clip`.

Fix: add the `TX: DTypePromoteAPI<TY>` bound and call `promote_other` (or
require `TY == TX` at the trait level), and set `TOut` to the promoted type.
The scalar-middle arm additionally goes through `bool_not_scratch` (item
12).

## 5. [BUG] Fill of a Generic-plan layout recurses forever

When `plan_operand` returns `Generic`, `fill_launch` calls
`kernels_ops::fill_dispatch_same` (`elementwise.rs:769-771`). That function
transmutes and calls `fill_dispatch`, whose same-`TypeId` branch calls
`fill_launch` again with the identical layout (`kernels_ops.rs:313-317`,
`:937-952`). The generic kernel is never reached.

Reproduced: filling a 5-D permuted owned tensor
(`arange(32).into_shape([2,2,2,2,2]).into_permute_dims((4,3,2,1,0)).fill(2.5)`)
prints the pre-fill line and the process is killed (unbounded recursion).
`plan_operand` returns Generic for any rank that stays above 4 after
merging, so rank-5 views, some broadcast combinations and any layout beyond
the pack are affected; this is the assignment path too.

Fix: route the Generic case to the true generic kernel (`fill_cast` with a
same-dtype cast expression, or a rank-generic fill entry) instead of
`fill_dispatch_same`; add a rank-5 fill/assign test.

## 6. [BUG] Both-scalar clip loses NaN inputs

The lean path's non-NaN branch emits
`val = (isnan(val) ? val : fmax(val, lo)); out = fmin(val, hi);`
(`elementwise.rs:1907-1914`). `fmin(NaN, hi)` returns `hi`, so an input NaN
is replaced by the upper bound. CPU clip is
`ext_max` then `ext_min` (`op_quaternary_common.rs:119-131`), both
NaN-propagating.

Reproduced:

```rust
rt::clip(&[f64::NAN, -1.0, 0.5, 2.0], (0.0f64, 1.0f64));
// CUDA: [1.0, 0.0, 0.5, 1.0]
// CPU : [NaN, 0.0, 0.5, 1.0]
```

The reasoning "non-NaN bounds cannot inject NaN, so only the x-guard is
needed" is right for the max stage but the guard must survive the second
stage.

Fix: re-apply `isnan(val) ? val :` around the min stage (or use
`minimum(TD, val, hi)` for the both-scalar path as well; the cost was the
reason for the lean path, so the guard is cheaper). Add a NaN-input case.

## 7. [BUG] `is_nan`/`is_inf`/`is_finite` fail for every dtype except f64/c64

`impl_query_unary!` builds `format!("{}f(b[ib])", $real)` for `"float"` and
`{}f(__half2float(...))` for half (`operators.rs:687-693`), i.e. `isnanf`,
`isinff`, `isfinitef`. CUDA/NVRTC defines only the overloads `isnan`,
`isinf`, `isfinite` for float; the `f`-suffixed spellings do not exist.

Reproduced:

```
f32 is_nan    -> DeviceError: NVRTC ... error: identifier "isnanf" is undefined
f32 is_inf    -> DeviceError (isinff)
f32 is_finite -> DeviceError (isfinitef)
c32 is_nan    -> DeviceError (isnanf on the .re component)
```

f16/bf16 take the same `isnanf` path by source inspection. Only f64 (and
c64 for is_nan) work — which is exactly what the wave test uses
(`test_ops_wave.rs` query test on f64).

Fix: for `float`, emit `isnan(x)`/`isinf(x)`/`isfinite(x)` (implicit
conversion handles float and double); for half use `__hisnan` /
`__nv_isnanbf` plus the half infinities/finiteness. Add f32/c32 cases.

## 8. [BUG] In-place `not` cannot compile, and has no write-side ordering

`OpNotAPI::op_muta` calls `unary_launch` with the expression `(!a[io])` /
`(~a[io])` (`operators.rs:657-663`), but `unary_launch` generates the
out-of-place kernel, whose variables are `b[ib]` and `out[oo]`
(`UNARY_ENTRIES`, `elementwise.rs:90-100`).

Reproduced (`rt::not(owned_bool_tensor)`):

```
error: identifier "a" is undefined
      out[oo] = (!a[io]);
error: identifier "io" is undefined
```

Three secondary problems in the same call: (a) `unary_launch` only calls
`order_input`; it assumes a fresh output on the exec stream, so this
in-place write never publishes an `order_output` event and a later host read
(which syncs only the allocation stream) can observe pre-not data when the
allocation and exec streams differ; (b) it creates `&mut a_mu` and `&a`
aliases of the same buffer; (c) `unary_launch_inplace` already exists with
exactly the right contract (`a[io]` expression, `order_output`) and is used
by the other in-place unaries.

Fix: call `unary_launch_inplace` with `(!a[io])` / `(~a[io])` and drop the
transmute/aliasing.

## 9. [BUG] Complex binary arithmetic, square, ext_neg and compound assign fail at NVRTC

`rstsr_complex.cuh` defines the `rstsr_complex<T>` struct and named helpers
(`rstsr_cadd`, `rstsr_cmul`, ...) but **no operator overloads**. The
arithmetic families still emit C++ operators:

- `OpAddAPI/OpSubAPI/OpMulAPI/OpDivAPI` → `(a[ia] + b[ib])` etc.
  (`operators.rs:15-72`);
- compound assigns → `(t[it] + v)` etc. (`ops_binary_extra.rs`);
- `OpSquareAPI` → `(b[ib] * b[ib])`, `OpExtNegAPI` → `(-b[ib])`
  (`operators.rs:588-637`).

Reproduced:

```
c64 add    : no operator "+" matches these operands: const rstsr_c64 + const rstsr_c64
c64 mul    : no operator "*" matches ... const rstsr_c64 * const rstsr_c64
c64 square : no operator "*" matches ... const rstsr_c64 * const rstsr_c64
c64 += s   : no operator "+" matches ... rstsr_c64 + rstsr_c64
```

Square/ext_neg are new APIs in this commit, so at least those are a new
regression; add/mul are pre-existing (the old struct also had no operators)
but the commit's complex-template work is the natural place to close the
hole. Every complex tensor arithmetic call is currently unusable.

Fix: define the C++ operators in the header (simplest; also fixes assign and
future expression sites), or route the Rust-side expressions through the
named helpers for complex CNAMEs.

## 10. [BUG] Tiled transpose is order-aware on detection, row-major-only in the kernel

`transpose2d_plan` correctly recognizes both `RowMajor` and `ColMajor`
patterns (`elementwise.rs:931-953`), but `copy_transpose_launch` passes only
`rows/cols` and the kernel indexes `in[r*cols + c]`, `out[r*rows + c]`
(row-major) regardless of the device order (`:880-1000`).

Reproduced on a `ColMajor` device, assigning a row-major-contiguous
`[128, 100]` tensor into a col-major destination via `to_contig(RowMajor)` /
`assign` (both directions; 64×64 is correct by symmetry):

```
128x100: bad = 12798 / 12800   cuda=[0, 1678, 3356, ...] cpu=[0, 128, 256, ...]
100x128: bad = 12798 / 12800   cuda=[0, 128, 256, ...]    cpu=[0, 100, 200, ...]
64x64  : bad = 0
```

The wave test only runs on a row-major device (and its col-major case is
3×4, below the 4096 threshold).

Also at `:995`: `grid.y = ceil(t_rows/32)`. `gridDim.y` is limited to 65535,
so a tiled transpose with more than 2,097,120 rows (e.g. 2.1M × 2) turns the
fast path into a launch error instead of falling back.

Fix: pass an order/stride descriptor to the kernel and compute the linear
index as `i + j*ld` for ColMajor; fall back to the planned copy when
`grid.y` would exceed the limit.

## 11. [BUG] Integer clip with scalar bounds fails at NVRTC

The both-scalar branch always emits the float expressions
`(isnan(val) ? val : fmax(val, lo))` / `fmin(val, hi)`
(`elementwise.rs:1907-1914`), while `OpClipAPI` admits every `CudaReal`
including i8..i64/u8..u64 (`ops_ternary.rs:91-93`). `scalar_bytes_nan`
(`:1856-1867`) only inspects exponent bits, so ordinary integer bound bits
select this path.

Reproduced:

```rust
rt::clip(&[1i64, 5, 9], (2i64, 5i64));
// CUDA: DeviceError — NVRTC "more than one instance of overloaded function
//       isnan/fmax/fmin matches the argument list"
// CPU : [2, 5, 5]
```

(For i8/i16 the width check routes to the NaN path only by accident; the
semantics are still unverified.) `scalar_bytes_nan` also misclassifies many
integer bit patterns as NaN/inf, so which broken path is taken depends on
the bound values.

Fix: choose the max/min expression by dtype (`exprs::maximum/minimum` are
dtype-aware and correct for ints), and gate the float-only fast path on
`TD::CNAME == "float" || "double"`. Add integer clip tests.

## 12. [BUG] `bool_not_scratch` writes out of bounds for offset/strided conditions

`bool_not_scratch` allocates `la.size()` bools and launches `unary_launch`
with `lc = la` (`elementwise.rs:1971-1983`). `unary_launch` folds
`la.offset()` into the base pointer and writes at every plan-relative walk
offset, so the writes span `[offset, offset + span)` inside a buffer of
`size` elements. This path is used by `where(cond, scalar, tensor)`
(`ops_ternary.rs:74-79`).

Failure: `rt::where(&cond.i(3..), 1.0, &y)` or any sliced/strided/flipped
condition — the inversion kernel writes up to `offset + n` bools into an
`n`-bool device allocation (and negative strides write before it). The
existing tests only use `where` with tensor arms, so the scratch is never
allocated.

Fix: allocate with the layout's full span (or, better, compact the mask:
allocate `n`, launch with an offset-0 `Layout` so the scratch is dense, and
keep `la` only for reads). Avoid the scratch entirely by adding a
condition-inverting variant of the scalar arm.

## 13. [BUG] `cabs`-based complex functions overflow; hyperbolic/trig edge cases return NaN

`rstsr_cabs` is `sqrt(re*re + im*im)` (`rstsr_complex.cuh:82-84`) while the
CPU reference is num-complex `norm()` = `hypot(re, im)`. `clog` uses it at
`:128-129`, `csqrt` uses it at `:139`, and `csign` divides by it at
`:455`. The `csinh`/`ccosh`/`ctanh`/`ctan` formulas (`:417-438`, `:164`)
have no large-argument handling.

Reproduced (f64):

| call | CUDA | CPU |
|---|---|---|
| `abs(1e200+1e200i)` | inf | 1.414213562373095e200 |
| `log(1e200+1e200i)` | inf + 0.785i | 460.8636 + 0.785i |
| `sqrt(1e200+1e200i)` | inf + 0i | 1.0987e100 + 4.5509e99i |
| `sign(1e200+1e200i)` | 0 + 0i | 0.7071 + 0.7071i |
| `sinh(-800+0i)` | NaN + NaN i | -inf + 0i |
| `tanh(-800+0i)` | NaN + NaN i | -1 + 0i |
| `tan(0+800i)` | 0 + NaN i | 0 + 1i |

The commit's complex parity tests only use moderate magnitudes, so none of
these paths is exercised. `rstsr-dtype-traits/src/c99_complex.rs` already
contains the C99 scaling/special-case logic (and is the CPU reference); the
CUDA header should mirror it (it already does for catrig/cacosh).

Fix: `rstsr_cabs` → `rstsr_rhypot(re, im)`; port the C99 special cases for
`csqrt`/`clog`/`csinh`/`ccosh`/`ctanh`/`ctan` (or use `expm1`-based forms
for the hyperbolic family). Add large-magnitude and boundary parity cases.

## 14. [BUG] `cpow(0, 0)` returns 0 instead of 1

`rstsr_cpow` special-cases `x == 0` by returning 0 for every exponent
(`rstsr_complex.cuh:442-447`). `std::pow(0,0)` and num-complex `powc(0,0)`
return 1; CPU/NumPy return 1. `cpow(0, -1)` should be inf, but also returns
0.

Reproduced: `pow(0+0i, 0+0i)` → CUDA `0+0i`, CPU `1+0i`.

Fix: return `1` when `y == 0`, and handle `x == 0` with negative real
exponent in the `exp(y log x)` limit (inf/NaN per num-complex). Add the
two special values to the complex parity test.

## 15. [BUG] `expm1`/`log1p` lose all precision for small arguments

`transc_expr` composes the real forms as `exp(x) - 1` and `log(1 + x)`
(`operators.rs:211-219`), although libdevice provides `expm1f/expm1` and
`log1pf/log1p` and the CPU device calls libm's versions.

Reproduced:

```
expm1(1e-16): CUDA 0.0   CPU 1e-16
log1p(1e-16): CUDA 0.0   CPU 1e-16
```

Both results are total cancellation, not a small ULP difference. The same
composition is used for `log2`/`log10` (acceptable precision, slower than
the native functions) — the `expm1`/`log1p` cases are the bug.

Fix: for real dtypes call the libdevice `expm1`/`log1p` directly; keep the
composition only for complex (where num-complex composes too).

## 16. [BUG] Integer `pow` silently accepts negative exponents

`exprs::pow` maps every integer dtype to `rstsr_ipow(x, y)`
(`exprs.rs:85-87`); `rstsr_ipow` documents "exp >= 0 guaranteed by the
tensor layer (CPU parity)" and returns 1 for `exp <= 0` (`kernel.rs:110-121`).
The tensor layer does not guarantee it: CPU `ext_pow` returns
`Err(InvalidValue)` for a negative integer exponent.

Reproduced:

```
pow(2i64, -1): CUDA Ok([1])   CPU Err(InvalidValue)
```

Fix: validate scalar exponents on the host before launch and reject or
define tensor-exponent behavior; at minimum remove the false comment and add
a regression test.

---

## Additional findings (below the 15-finding cut)

1. **[API] Promoting-binary families are same-dtype only.** The
   `impl_promoting_binary!` macro implements `OpAPI<T, T, D>` only
   (`operators.rs:248-336`), while CPU promotes via `DTypePromoteAPI`
   (`TOut = TX::Res`). `maximum/minimum/pow/atan2/copysign/nextafter/
   logaddexp` therefore fail to compile for mixed operand dtypes
   (`rt::maximum(&f32, &f64)`), and their device impls cannot see the
   promotion the operators.rs comment claims. The `T,T` restriction should
   become `TX,TY` with `TOut = TX::Res` (same shape as the CPU impls), or
   the tensor tier should pre-promote these families.
2. **[BUG] Wide arms marshal `ParamsN` plans as zeroed rank-0 plans.**
   `plan_variant` returns `Wide` for any mix of `Params` and
   `Flat | ParamsN`, but each Wide arm builds its arguments with
   `plan_bytes(...).unwrap_or(&zero)` (`elementwise.rs:423-446, 501-520,
   681-694, 1082-1090, 1202-1214, 1354-1364, 1430-1436, 1558-1570`).
   `plan_bytes` only handles `Params`, so a `ParamsN` operand on a Wide
   launch is read/written through a zeroed rank-0 (flat) plan — element 0
   for every thread. Reachable when one operand's stride magnitude or
   offset walk exceeds i32 while another operand still narrows (multi-GiB
   views); silent, wrong, and sometimes OOB.
3. **[BUG] `rstsr_clip_ss_pn` advances `lin` twice per iteration**
   (`elementwise.rs:1613-1616`: once inside the `if`, once after it).
   Currently masked because the entry is launched with a full grid
   (`gridDim*blockDim >= n`, so only rep 0 runs); it will skip every other
   chunk the moment `elems_per_thread` is added to that arm. Fix while
   touching the clip launchers.
4. **[PERF] The "interned" kernel lookup still builds and hashes the full
   source on every launch.** Every launcher calls
   `render(ENTRIES, &[...])` unconditionally and `planned_function`
   formats `format!("{name}|{}|{entries}", ...)` and clones the resulting
   string for the HashMap key (`elementwise.rs:345-362`); ~1-3 KB per
   launch plus a mutex. The doc comment says "the per-launch source
   assembly and full-source cache hash of `make_key` stay off the hot
   path", but the assembly happens at every call site and the formatted
   string is hashed by the map. For small tensors (the bench's 12-element
   `where_ts` class) this dominates. Consider a static
   `(family, dtype, expr)` key at the call sites, or cache the rendered
   source per specialization, and re-measure with a small-tensor cell.
5. **[PERF] `binary_assign`/`scalar_assign`/`where_sc` Narrow arms do not
   use ept(4)** (they pick `_pn` entries but no ept, so rep 0 covers
   everything): correct but no ILP. Once the Flat arms are fixed, set ept
   consistently for the narrow arms too.
6. **[PERF/NIT] `rstsr_fill_v`: grid is sized from `n` while the work is
   `n/2` pairs** (`elementwise.rs:805-812`, kernel `:721-735`), so half the
   threads idle; the tail loop `oo = nv4*2 + g` is dead for even `n`; and
   the comment says "32-byte stores" while the pair struct is 16 bytes for
   f64 (the perf report's measured 16-B pair result is unaffected).
7. **[NIT] Dead code and stale comments:** `rstsr_copy_gather2d`
   (`elementwise.rs:897-910`) is compiled but never launched, while
   `transpose2d_plan`'s comment promises a "gather form covers
   non-multiple-of-32 margins"; `impl_rounding_unary` has a dead half
   branch; `rstsr_ipowu` is unused; `pub mod kernels;` exposes a module
   whose items are all `pub(crate)`; the `abs` doc comment sits on
   `t_real` (`exprs.rs`); `clip2s_out_launch` calls `.elems_per_thread(4)`
   twice (`:1928-1929`); `OpSignBitAPI` bounds `CudaReal` (which includes
   bool) although CPU requires `ExtReal`.
8. **[API] Transcendentals accept integer dtypes.** The bound widened from
   `CudaFloat` to `CudaArith` (`operators.rs:166`) and the comment at
   `:127-128` says "other dtypes are rejected at source-gen", but
   `transc_expr` emits `sin(b[ib])` for ints; `rt::sin(&i32_tensor)` now
   compiles and returns a truncated integer, while CPU rejects it at the
   type level. Reject non-float CNAMEs with `UnImplemented`, or tighten the
   bound back to `CudaFloat | CudaComplexFrom`.
9. **[RACE] `order_output` audit:** all out-of-place planned launchers
   correctly omit it (outputs come from `uninit_impl` on the exec stream),
   and fill/copy/assign order both ways — except the in-place `not` (item
   8). `where_launch_sc`'s scratch mask is exec-stream-local, so no extra
   event is needed once the OOB is fixed.

## Tests and benches

1. **[TEST] `new_ops_transpose_copy_tiled` lacks `require_cuda!()`**
   (`tests/test_ops_wave.rs:381`) unlike every other test in the file, so a
   CPU-only `cargo test -p rstsr-cuda` panics in `DeviceCuda::new` instead
   of skipping. Its comment "above and below the 4096-element kernel
   threshold" is false (the smallest case is exactly 4096), and the 63×67 /
   2048×3500 T-read cases fall back to the planned copy because
   `transpose2d_plan` requires output cols ≥ rows; the tiled read kernel is
   only reached by 64×64 and 128×100.
2. **[TEST] Coverage holes that hide the findings above.** No test covers:
   `where` scalar arms (either side); clip with tensor bounds, `None`
   bounds, NaN bounds, integer bounds or mixed dtype bounds; scalar
   compound assigns beyond `+=`/`*=` on a 100-element tensor; f32/c32
   `is_nan`/`is_inf`/`is_finite`; complex `add`/`square`/`ext_neg`;
   ColMajor transpose above the threshold; small-argument `expm1`/`log1p`;
   integer `pow` with a negative exponent; rank-5 fill. The five new tests
   pass because the where/clip tests use transposed views (Narrow plans,
   the correct arms) and small sizes.
3. **[TEST] `probe_maxnan_f64_jit`** (`src/kernel.rs:493-514`) is a
   `#[test]` that prints OK/FAILED and can never fail; if the driver JIT
   rejection of `max.NaN.f64` matters it should assert, otherwise it is a
   probe, not a test.
4. **[TEST] `Cargo.toml:51-53` adds a no-op `[[test]]` block** (test files
   are auto-discovered; `required-features = []` is the default). More
   importantly, no workflow under `.github/workflows` runs `rstsr-cuda`
   tests, so none of the 428 new assertion lines gates anything; the
   testing-policy table in `rstsr-agents/rules/testing.md` has no entry for
   this crate.
5. **[BENCH] `bench_elementwise`**: the `where_ts` cell (0.0133 ms, 0.06×
   cupy) is measuring the under-write bug from item 2, not a fast kernel;
   the harness prints only rstsr medians (no baseline, no device/clock
   report, no check that the software warm-up reached boost), so the
   "17/19 ≤ 1.2×" claim cannot be reproduced from the checked-in harness;
   `RSTSR_BENCH_ITER=0`/`REPEAT=0` panic (`partial_cmp().unwrap()` /
   empty median). The per-iteration allocation churn is included in every
   cell, a different protocol from a stream-ordered baseline.

## Verdict

The planned walk itself is well designed and empirically correct (magic
division fuzzed across the 32-bit and 64-bit ranges; the
Flat/broadcast-extent gate fixed earlier is sound), and the benchmark
numbers at boost clocks are at parity with cupy/torch for the cells that
are actually correct. But the launch layer is not safe to ship as-is:
three launchers pair ILP-4 grids with loop-less entries, three scalar
paths reinterpret bytes instead of converting dtypes, and the fill generic
fallback does not terminate. All of these fire on ordinary contiguous
inputs with scalar arguments — the most common call shapes — and the test
suite happens to use strided views and ≤256-element tensors, so it stays
green. The complex template work is also incomplete: arithmetic does not
compile at all, and the math helpers diverge from the CPU on overflow and
special values. None of the fixes requires an API redesign except the
promotion contract (`TOut`/where/clip), which should follow the CPU
reference devices.

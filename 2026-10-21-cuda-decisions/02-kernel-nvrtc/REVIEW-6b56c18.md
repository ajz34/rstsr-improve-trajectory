# Line-for-line review — rstsr `6b56c18`

Review of commit **`6b56c18`** ("rstsr-cuda: NVRTC kernel layer for layout
ops and elementwise math") on branch `261021/cuda`, done 2026-10-23 as a
manual line-by-line pass (all 16 changed files read in full), against the
current `rstsr-agents` rules — in particular the new
`rules/code-style-general-guide.md` (added 2026-10-23 06:06 UTC, i.e.
before this commit).

- Repo state: `6b56c18` is HEAD; working tree clean; **no code changes
  made during the review** (review only, no amend, no follow-up commit in
  rstsr).
- Method: static read of every file plus three empirical probes on the
  GPU (RTX 5070 Laptop) run from a throwaway crate outside the repo.
- Environment: driver 595.91.07, NVRTC `compute_120`, `--std=c++17`,
  debug profile (`-lineinfo`).

Severity tags: **[BUG]** functional defect, **[RACE]** ordering /
concurrency, **[RULE]** violation of the new code-style rule, **[NIT]**
robustness / structure / docs.

---

## Critical — functional bugs

### 1. [BUG] `linspace` endpoint flag marshaled as 1 byte for an `int` param

`crates-device/rstsr-cuda/src/kernels_ops.rs:402`

```rust
.arg_bytes(&(endpoint as u8).to_ne_bytes());
```

The kernel declares `int endpoint` (`kernels_ops.rs:366`). The driver
copies the kernel's parameter size (4 bytes) from the given pointer; only
the low byte is ours, the other three are an out-of-bounds heap read and
garbage. `endpoint = false` therefore reads back as "nonzero" (true).

Empirically confirmed (probe):

```
endpoint=false: [ 0 0.25 0.5 0.75 1]
endpoint=true : [ 0 0.25 0.5 0.75 1]
expected false: [0, 0.2, 0.4, 0.6, 0.8]
```

The only linspace test (`arange_linspace_values`) uses the default
`endpoint = true`, so the suite never sees it.

Fix: `(endpoint as i32).to_ne_bytes()` (or `arg_u32`).

### 2. [BUG] Complex `linspace` generates invalid CUDA C

`kernels_ops.rs:362-379` (`linspace_source`), reachable through
`creation.rs:131` (`DeviceCreationComplexFloatAPI` for `T: CudaDType +
ComplexFloat` = `Complex<f32>`/`Complex<f64>`).

For `t = "rstsr_c64"` the generated source contains:

```c
rstsr_c64 frac = (rstsr_c64)((double) i / (double) denom);
rstsr_c64 span = (rstsr_c64)((double) end - (double) start);
```

C-style casts between the struct and `double` do not exist. Empirically
confirmed: `rt::linspace_f` on a `Complex<f64>` CUDA tensor fails at
first use with NVRTC `NVRTC_ERROR_COMPILATION` and six
"no suitable conversion function" errors.

Fix options: restrict the impl bound to `T: CudaFloat` (drop complex
linspace until a complex prelude lands), or emit component-wise complex
arithmetic.

### 3. [BUG] Printing any offset (sliced) view panics

`crates-device/rstsr-cuda/src/cpu.rs:69-71`, with
`crates-device/rstsr-cuda/src/format.rs:39`.

`staged` is sized `layout.size()` but indexed by
`touched_element_indices`, which returns **absolute** buffer indices
(offset included); the shared CPU walker (`print_vec_with_layout_dfs_inner`)
also indexes the vec absolutely. Any view with `offset >= 1` indexes past
the end.

Empirically confirmed: printing `a.i(1)` of a `[2,3]` CUDA tensor panics:

```
thread 'main' panicked at crates-device/rstsr-cuda/src/cpu.rs:71:
index out of bounds: the len is 3 but the index is 3
```

Pre-existing (the old `d2h_scalar` loop had the same shape), but this
commit rewrote the fetch path (batched gather) and the format tests only
cover offset-0 tensors. Fix: size the staging vec to
`max(touched index) + 1`, or rebase the layout to offset 0 and store
elements compactly (then the walker layout must be rebased too).

### 4. [RACE] Binary/unary kernels never order foreign-stream inputs

`kernels_ops.rs:581-672` (`launch_binary`), `kernels_ops.rs:632-672`
(`launch_unary`), also the scalar variants in `operators.rs:79-101`,
`140-162`.

`copy_same_dtype` (line 110), `copy_cast` (line 220) and `gather_to_host`
(line 443) call `order_input(a, exec)` — an event record on the input
buffer's allocation stream plus a wait on the execution stream. The
elementwise launchers do not, for any of their inputs.

Since this same commit makes `same_device` ignore the stream
(`device.rs:137-145`), `rt::add(&a, &b)` where `a`/`b` were produced on
another stream of the same ordinal is a genuine race: the kernel on the
exec stream may read them before the producer stream completes.

Consequences:
- the doc at `device.rs:29-31` ("ops order foreign-stream inputs with
  event record/wait") is untrue;
- `test_kernels.rs:206-217` ("a cross-stream op succeeds and orders
  correctly") cannot detect the bug — it passes on timing alone.

Fix: `order_input(a, exec)` + `order_input(b, exec)` in `launch_binary`,
`order_input(b, exec)` in `launch_unary`, guards held until after
`launch()` (as in the copy paths).

### 5. [RACE] `memcpy_fast` has the same ordering gap

`crates-device/rstsr-cuda/src/assignment.rs:16-30`.

The contiguous fast path enqueues `memcpy_dtod_async` on the destination
buffer's stream using `a.device_ptr()` — our raw accessor, which passes
the buffer's own stream to cudarc, so cudarc's safe-API cross-stream
event is a no-op. No event orders the source stream before the copy.

`mixed_stream_assign` (`test_kernels.rs:219-234`) uses two full
contiguous buffers, so it takes exactly this path — the test passes by
timing, not by ordering. Fix: same `order_input(a, exec)` treatment (and
consider ordering the destination's allocation stream as well for
relabeled/in-place buffers).

---

## Robustness / structure

### 6. [NIT] `make_key`'s memo is write-only

`kernel.rs:180-208`. The thread-local map is inserted into (line 205) and
never read; the function always rebuilds name/source/arch/options, then
clones the whole key (including the source `String`) into the map,
hashing the source a second time. Either read it back
(`guard.get(...)`) or delete the memo.

Note the deeper cost: callers rebuild the generated source *before*
`make_key` on every call (`assemble_source` at `kernels_ops.rs:111, 224,
293, 344, 388, 444, 596, 645`). The static sources (`copy_bytes_source`,
`gather_bytes_source`) should be `const` / `OnceLock` strings; this is
the very "per-call key rebuild" cost the bench work claimed to remove.

### 7. [NIT] `_Generic` in the prelude under `--std=c++17`

`kernel.rs:46`. `_Generic` is C11, invalid in C++; it is harmless only
because no kernel currently expands `rstsr_isnan`. Latent trap for the
next kernel author. Use overloaded `__device__` functions.

### 8. [NIT] `int` kernel params pushed as 8-byte values

e.g. `kernels_ops.rs:124, 238` — `arg_usize(lo.shape.len())` for
`int orank`. Works only because of little-endian truncation. Add
`arg_u32`/`arg_i32`. Related: the copy/cast kernels never use
`ishape`/`irank` (lines 78/147), yet still take and upload them — drop
both.

### 9. [NIT] Fill-set comment contradicts the implementation

`creation.rs:22-31` claims half and complex are in the byte-fill set;
`is_fill_supported` (`kernels_ops.rs:741-756`) excludes `f16`, `bf16`,
`Complex<f32>`, `Complex<f64>`. `rt::full` on those takes the
geometric-doubling host path. Fix the comment or extend the list.

### 10. [NIT] Stale docs

- `fill_dispatch` documents a `value_bytes` parameter that does not exist
  (`kernels_ops.rs:257`).
- `KernelScratch` says "H2D blocking" but the upload is async
  (`kernels_ops.rs:483` vs `499-505`).
- `sequence_source` says "i promoted through double" but casts directly
  to `T` (`kernels_ops.rs:330`).
- `bench_elementwise.rs:8`: "writes bench.out next to it" — it does not.

### 11. [NIT] `gather_to_host` bound and SAFETY comments

`kernels_ops.rs:433`: `T: Clone`, then byte-copies into `Vec::<T>` and
`set_len` — should be `T: Copy + 'static` (or a POD marker). Several
`unsafe` blocks (`kernels_ops.rs:449-457`, `471-474`, `300-302`,
`602-604`, `648-650`) lack the `SAFETY:` comments the crate uses
elsewhere (e.g. `assignment.rs:139`).

### 12. [NIT] Scalar binary ops synchronize the whole stream

`operators.rs:21-41`. `scalar_buffer` performs an async H2D followed by
`stream.synchronize()` on every scalar op (`a + 1.0`, `a < 2.0`), which
serializes all prior work on the stream and also allocates a 1-element
buffer per call. The bench's `add` scenario uses tensor+tensor, so the
2x gate never exercises this. Passing the scalar as a kernel argument
(as the fill kernel already does) removes the allocation and the sync.

### 13. [NIT] `op_muta` returns `UnImplemented`

`operators.rs:224-228`, comment "not expressible without aliasing
analysis". In-place unary **is** trivially expressible elementwise each
thread reads and writes its own element; no aliasing analysis is
needed. Today the tensor layer calls `op_muta` (`op_unary_common.rs:242`),
so `sin`-in-place compiles and then fails at runtime.

### 14. [NIT] Duplication

The scratch-pack / mark / split / launch sequence is copy-pasted in
`copy_same_dtype`, `copy_cast`, `launch_binary`, `launch_unary`; the
`TensorArg` packer is used by only two of them; `split_mark`'s second
tuple element is never read; `FlatLayout` has two adjacent `impl` blocks.

### 15. [NIT] Smaller items

- `kernel.rs:155`: `Box::leak` per compile for the arch string (bounded,
  but avoidable with a per-ordinal `OnceLock<String>`).
- `cpu.rs:62`: `eprintln!` inside the library formatting path.
- `transfer.rs:165-168`: leftover comment describing functions that were
  deleted.
- `kernels_ops.rs:338, 741, 761`: `sequence`, `is_fill_supported`,
  `fill_bytes_erased` are `pub` but only used in-crate (the bench uses
  `rt::arange`), while `linspace` is `pub(crate)` — narrow all three.
- `Cargo.toml`: `required-features = []` on the bench target is a no-op.
- `test_kernels.rs:20, 25, 37`: `.unwrap()` instead of `.rstsr_unwrap()`.
- `bench_elementwise.rs:20, 25`: constructs a fresh `DeviceCuda::new(0)`
  inside the timing helper; pass `&d`.

---

## New code-style rule compliance

The rule was committed (`1cb398b`, 2026-10-23 06:06 UTC) before this code
commit (12:18 UTC), so these are genuine violations.

### 16. [RULE] Temporary decision IDs / trajectory references in code

`rules/code-style-general-guide.md`: "Do not reference temporary
decisions ID … The contents in rstsr-improve-trajectory or other logging
of AI assistant should not involved."

Present at:

| file | lines | text |
|---|---|---|
| `kernel.rs` | 1, 14, 76, 100, 162, 213, 271 | "B1 infrastructure, D2–D4", "D9", "D3", "D4", "D3", "D2", "D8" |
| `kernels_ops.rs` | 2, 4, 182-184, 480 | "B2/B3", "D9", "policy D6", "B5" |
| `operators.rs` | 1, 226 | "B5", "the ladder round" |
| `assignment.rs` | 4 | "D6" |
| `test_kernels.rs` | 1, 206 | "Kernel-round tests", "D8" |
| `test_device_basic.rs` | 73 | "kernel-round decision D8" |
| `bench_elementwise.rs` | 3, 83 | "kernel round", "bench_cupy.py in the task dir" |

The bench line 83 is doubly offending: it points at an **untracked file
in this trajectory repo** ("Do not reference files that are at local
directory, or git untracked temporary files").

### 17. [RULE] `duplicate_item` preference

`dtype.rs:26-38`, `62-64`, `77-79` use `macro_rules!` for plain impl
tables; `kernels_ops.rs:772` (`fill_as!`) is a macro where a small
TypeId-dispatch helper would do. (The operator tables and the
`DeviceFormatAPI` table in `cpu.rs` already use `duplicate_item` — good.)

### 18. [RULE] Docstring placement / length

`kernel.rs:1-18` (18-line module doc) and `kernels_ops.rs:1-9` are the
"long module docstring" the rule asks to weaken; `creation.rs:22-31` is
a >4-line explanatory non-doc comment.

---

## Test and bench coverage

Gaps that would have caught the findings above:

- `endpoint = false` (finding 1);
- complex `linspace` (finding 2);
- printing an offset/sliced view (finding 3);
- negative strides (the kernels rely on unsigned modular arithmetic for
  this — it happens to be correct, but nothing tests it);
- scalar ops (`a + 1.0`) — the only place `scalar_buffer` is exercised
  (finding 12);
- in-place unary (finding 13);
- a cross-stream test that cannot pass without ordering, e.g. a large
  producer on stream A consumed immediately on stream B in a loop
  (findings 4–5).

Also: `test_kernels.rs:188, 191` use
`iter().zip(...).all(...)`, which silently truncates if the lengths
differ — assert the lengths first.

---

## What holds up

- Kernel-key hashing includes source + arch + options; the
  name-collision class is correctly avoided (`KernelKey::disk_hash`).
- Index decomposition by the **output** shape with stride-0 broadcasting
  is implemented consistently across all T0 generators.
- Scratch upload is enqueued on the launch stream, so its lifetime
  argument (free is stream-ordered after the kernel) is sound.
- `assign_impl`'s fast-path guards (contiguity, same order, full buffer,
  zero offsets, equal element sizes) are correct.
- The `faer` feature gate matches the BLAS-device-crate precedent, and
  pinning `DeviceCpuSerial` in shared test bodies is the right answer to
  workspace feature unification.
- TypeId-guarded re-typing (`transmute_mut`, `transmute_ref_cross`,
  `fill_bytes_erased`) follows the house fast-path idiom.

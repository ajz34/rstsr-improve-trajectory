# Can `rstsr-cuda` stream with a pinned host tensor?

Question: if rstsr had a *pinned* host tensor (different allocator, CPU buffer),
could H2D/D2H transfers overlap with device computation — assuming the streaming
and async infrastructure works or can work in principle?

Checked against `rstsr` branch `261021/cuda` at `6d17d04`.

## Short answer

**Yes — and the async half is already in the crate.** `rstsr-cuda` is async by
design: op entry points return immediately and *only* host reads synchronize
(`src/lib.rs:12-13`). The streaming scheduler machinery a pipeline needs —
per-value streams, cross-stream producer/consumer events, stream-ordered
allocation — already exists and is used by every kernel.

What is missing is narrower than "streaming support": it is the **host boundary**.
Pinned memory is the enabling condition, but it needs two companions:

1. a pinned host owner type (a `Vec<T>` cannot own `cuMemHostAlloc` memory);
2. non-syncing `h2d`/`d2h` entry points (`h2d_full`/`d2h_full` both end in
   `stream.synchronize()`, and the tensor-level `to_device`/`to_cpu` route
   through them).

Everything else — ordering the copy against the compute, keeping the free
ordered after the kernel, not stalling the host on allocation — is already
solved. Concretely: **if the copy can be enqueued without a sync, the GPU side
pipelines today.**

## What already works (code evidence)

| capability | where |
|---|---|
| Async by design: ops return after enqueue; only host reads sync | `src/lib.rs:12-13`; the whole crate contains 9 `synchronize()` calls: 6 in `transfer.rs` (the host-transfer fns), 1 `DeviceCuda::synchronize`, 1 `CudaStreamObj::synchronize`, 1 trait forward. **Zero in any kernel or operator.** |
| Per-device-value stream | `DeviceCuda { stream: Arc<CudaStreamObj> }`, `set_stream`/`set_stream_f` (`src/device.rs`) |
| Multi-stream tensors of one ordinal interoperate | `same_device` compares ordinal `Arc` + flags + order, **not the stream** — documented: "The execution stream is not compared: ops order foreign-stream inputs with event record/wait" (`src/device.rs`) |
| Cross-stream producer/consumer ordering | `order_input` / `order_output` + raw variants (`src/kernel.rs:535-575`), used by every kernel (`kernels_ops.rs:152-153,176`, `kernels/elementwise.rs`, `reduce*.rs`, `gemm.rs:437`, `adv_index.rs`), by `assignment.rs:21`, and by `dtod_full` |
| Stream-ordered allocation, no host stall | `CudaBuffer::new_f` → `cuMemAllocAsync` on the buffer's alloc stream; frees are stream-ordered (`src/buffer.rs`) |

That means a tensor produced on stream H and consumed on stream C is *legal and
correct* today: the consumer's `order_input` records an event on H and waits on
C, and `order_output` publishes the write back so the allocation stream's free
waits for the kernel. Nothing in a 3-stage pipeline needs a hand-written event.

## What is missing

1. **Pinned host storage — the real blocker.** Host tensors are
   `DataOwned<Vec<T>>` on `DeviceCpuSerial` (`src/conversion.rs: cpu_tensor`).
   Rust's `Vec` frees through the global allocator, so `cuMemHostAlloc` memory
   cannot be wrapped in one without UB. You need a distinct owner
   (`PinnedBuf<T>`, e.g. around cudarc 0.19.10's `PinnedHostSlice<T>`) with
   value semantics: `Clone` = deep copy, `AsRef<[T]>` / `as_mut_slice`, len.
2. **Non-syncing transfer entry points.** `h2d_full`, `d2h_full`, `h2d_scalar`,
   `d2h_scalar`, `dtod_full` and `fill_naive` each end with
   `stream.synchronize()` (`src/transfer.rs`) — these are the *host-boundary*
   paths. `dtod_full` is the cross-ordinal (multi-GPU) move only; the device
   `Clone` path does not go through it and is already async (next section). The
   async forms must only enqueue and hand back something that keeps the host
   buffer borrowed until a wait.
3. **Public stream creation.** `CudaStreamObj::new` is `pub(crate)`; the only
   public door is `CudaStreamObj::wrap(ordinal, Arc<CudaStream>)`, which needs a
   `cudarc` stream. A `DeviceCuda::new_stream()` (or a `with_stream` builder)
   would make the pipeline expressible from an ordinary crate.
4. **Where the pinned buffer lives in the type system** — two shapes:
   - *Minimal*: `PinnedBuf<T>` as a plain staging type + async copy functions;
     the user drives the loop. ~200 lines, no tensor integration — but the
     pipeline is then hand-written Rust, not tensor code.
   - *Integrated*: a `DeviceCpuPinned` host device with `Raw = PinnedBuf<T>` and
     only the two `DeviceChangeAPI<DeviceCuda>` edges implemented. Then
     `pinned.to_device(&dev_h2d)` / `t.to_cpu_pinned(&dev_d2h)` are tensor ops
     and the pipeline is written in rstsr's language. This needs **no CPU
     kernels**: the pinned tensor is a transfer endpoint only (fill it through
     `&mut [T]` before wrapping, read it through `AsRef<[T]>` after syncing).

## Recommended pipeline shape: three streams, not per-chunk streams

```
dev_h2d   (stream H) : the H2D copies, one per chunk
dev_comp  (stream C) : the compute ops, one per chunk
dev_d2h   (stream D) : the D2H copies
```

- `h2d(i)` on H allocates and copies on H; `comp(i)` on C waits on H
  automatically (`order_input` sees the buffer's alloc stream ≠ exec stream);
  `d2h(i)` on D waits on C. No explicit events are written by the caller.
- Per-chunk rotating streams (the CuPy shape in `bench_streaming.py`) are not
  needed here: this GPU serializes copies from different streams anyway
  (see `README.md`), and three dedicated streams reproduce the same
  `max(compute, h2d+d2h)` bound with fewer moving parts.

## Hazards — what silently kills the pipeline

- Any host read inside the loop: `to_scalar`, `to_vec`, `Display` (each is an
  explicit sync by design, `src/transfer.rs`).
- Register/naive-path fallbacks that round-trip through the host.
- `fill_naive` and the per-element scalar transfers sync per call.
- Pinned memory is slower for CPU access than pageable memory, and cudarc's
  `alloc_pinned` defaults to `CU_MEMHOSTALLOC_WRITECOMBINED` (good for D2H
  writes, very slow for CPU reads). If the host post-processes the output,
  copy pinned → pageable once, or allocate without the write-combined flag and
  measure.
- `cuMemAllocAsync` can block on pool exhaustion: preallocate one buffer set per
  in-flight chunk.
- The 3-stream arrangement is only as good as the ops: a `t.to_device(...)`
  re-tag inside the loop is free, a `clone()` is not free but is at least
  non-blocking (see below).

## Does device `clone()` synchronize? No — and it already does not

`CudaBuffer::clone_f` → `self.slice.try_clone()` → cudarc
`CudaStream::clone_dtod` → `alloc` + `memcpy_dtod_async` **on the buffer's own
allocation stream** (cudarc 0.19.10 `driver/safe/core.rs:1795`, `:1750-1766`),
returning with no host barrier. So D1's wording ("deep D2D copy on the owning
stream") is what the code does. The only syncing D2D path is
`transfer::dtod_full`, and it is used *only* for cross-ordinal (multi-GPU) moves
(`src/conversion.rs:228,245`) — not for `clone`.

Why clone must stay sync-free:

- Value semantics only need an independent buffer; a device buffer's contents are
  observed by device work, which is already stream-ordered.
- The ordering a sync would buy is provided elsewhere: the same allocation stream
  (copy before free), `order_input` (consumer waits for the copy), `order_output`
  (free waits for the consumer).
- A sync drains the whole stream, making `clone()` the one blocking op in an
  otherwise fully async op set — fatal inside a pipeline.
- CuPy and PyTorch both do it this way:
  `ndarray.copy()` → `astype` → `elementwise_copy` kernel on the current stream
  (`cupy/_core/core.pyx:863`, `:782`) with no sync; `torch.clone` = `empty_like`
  + `copy_` (`aten/src/ATen/native/TensorFactories.cpp:2252-2276`) →
  `copy_device_to_device` → `CUDACachingAllocator::memcpyAsync` on the current
  stream (`aten/src/ATen/native/cuda/Copy.cu:598,644`), with event barriers only
  when `src_device != dst_device`.

The one case where a sync looks tempting is the read-lifetime hole recorded in
[[cuda-write-target-ordering]]: a *foreign-stream* source has no `exec -> alloc`
event, so a temporary can be freed while still being read. It does not apply to
`clone` (the copy runs on the source's own allocation stream), and a sync would
hide the hole rather than fix it — the fix is the missing `exec -> alloc` event
for read-only foreign-stream inputs.

## Test shape

Mirror `bench_streaming.py`, on the same bytes:

- `serial` — pageable host + blocking transfer (today's `to_cpu`/`to_device`);
- `pipelined` — pinned host + async transfer + 3 streams;
- negative controls: pageable-but-async, pinned-but-blocking, pinned-but-one-stream.

Assert `pipelined < 0.7 × serial` *and* that the controls do not improve; report
the isolated `h2d` / `compute` / `d2h` times to show the bound is reached.

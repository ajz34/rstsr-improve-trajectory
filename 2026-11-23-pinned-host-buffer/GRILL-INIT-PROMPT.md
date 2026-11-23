# Grill: pinned host memory buffer tensor type (2026-11-23)

Initial prompt, recorded verbatim (agent-authored file; the answers files are user-owned).

> You are going to create pinned memory buffer tensor type.
>
> - I'm also not sure what's really required, and so you are going to also check how cupy and pytorch do on this.
> - I'm probably going to create a new data type, `DeviceAPI<Raw = PinnedBuffer<T>>` (the CPU tensor counterpart is `Vec<T>` if I remember correctly). This kind of tensor can use `host_view<B>(&device: B) -> TensorView<T, B, D>` and `host_view_mut<B> ... -> TensorMut ...`, where `B: DeviceAPI<Raw = Vec<T>>`, to cast as any CPU tensor view/view_mut; so that the pinned buffer can also involve into CPU computation or storage (we do not derive operation trait implementation to PinnedBuffer data, but they can convert to usual tensor view in zero-cost way).
> - This is going to related to better streaming overlapping (data transfer / comput, see rstsr-improve-trajectory). So probably we need to make sure in some bench test (not usual test) that overlapping does work with pinned buffer.
>
> You can spawn subagents if necessary. You may need design the API shape for this task. I may still have wrong thoughts, and you need to make fact checks on this.
> Should create a new directory in rstsr-improve-trajectory for grill session.

Context: `../2026-11-23-streaming-overlap/` (CuPy prototype, `RSTSR-FEASIBILITY.md`);
CUDA decisions in `../2026-10-21-cuda-decisions/`; rstsr branch `261021/cuda` at `6d17d04`.

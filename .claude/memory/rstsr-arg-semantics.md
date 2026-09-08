---
name: rstsr-arg-semantics
description: rstsr argmin/argmax exact semantics at 386948be — ties, NaN, empty, flat-index contract — and the kernel call chain incl. dead-code duplicate
metadata: type: project
---

rstsr argmin/argmax semantics at commit 386948be (locked by the T6 correctness
gate in `2026-09-09-argmax-argmin/examples/correctness.rs`, all verified
empirically on both devices):

- Ties: FIRST occurrence in row-major visit order wins (= lowest row-major
  flat index); the rayon fold/reduce combine resolves ties by lower global
  index explicitly.
- NaN: first row-major element seeds the accumulator unconditionally; NaN
  never replaces it afterwards. So NaN at the end/middle loses to the real
  extreme; NaN at the FRONT poisons (returns flat 0); all-NaN returns 0 (no
  error) — coincides with numpy's argmax/argmin NaN behavior.
- Empty: `_f` variants return Err(InvalidLayout) "empty sequence is not
  allowed for reduce_arg." (cpu_serial/reduction.rs:439, cpu_rayon:455); the
  infallible rt::argmax panics via rstsr_unwrap.
- Whole-tensor arg* on any-rank input returns the flat ROW-major index
  regardless of input layout/device order; 2-D whole == 1-D raveled.
- Trait bound is `T: Clone + PartialOrd` (no ExtNum).

Call chain (serial): tensor/reduction.rs:197 → device_cpu_serial/reduction.rs
(argmax_all :405) → cpu_serial/reduction.rs `reduce_all_arg_cpu_serial` →
`reduce_all_unraveled_arg_cpu_serial` (c-contig 8-lane fast path after T6
phase 2; strided fallback = original closure fold). Rayon: DeviceFaer's impls
live in `feature_rayon/auto_impl/reduction.rs` — `device_faer/rayon_auto_impl/*`
are SYMLINKS to it (the T0 code map's "dead duplicate" is actually the live
file's physical storage; diffs appear under feature_rayon/). Rayon contig
path: contiguous chunks with the serial 8-lane kernel, partials collected
in order and combined sequentially (deterministic).

Phase-2 kernel (proposed.patch, applied-post) NaN semantics: NaN anywhere
except global index 0 never wins, on BOTH devices — this intentionally
refines the pre-patch rayon behavior, where a NaN at a rayon fold-split could
suppress the true global max depending on pairing order (demonstrated:
n=1025, NaN@896, max@1024 → clean tree answered 294). The generic NaN test
`x == x` (false iff NaN; reflexive for all other PartialOrd types) is the
trick that makes the Option-free 8-lane lane-seeding safe: seed all lanes
with the (checked) first element; NEVER seed lanes from xs[0..8] — a NaN
seed at positions 1..8 blocks its lane (`x > NaN` false) and swallows a
later same-lane max (bug found by examples/probe_nan_corner.rs).

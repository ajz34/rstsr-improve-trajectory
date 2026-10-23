# GRILL R3 — `02-kernel-nvrtc`: streams revisited + loose ends

## Agent-authored background — R2 answers as received on screen (2026-01-23 [sic: 2026-10-23])

Transcribed from the owner's on-screen replies to GRILL-R2-QUESTIONS.md
(answers file remains user-owned):

- **R2-Q1–Q4: Okay.** Launcher = builder-as-mechanics u8-wise;
  options policy = strict defaults now, additive flags later; async ops
  with sync only at host reads; crate-local copy-and-compare tests.
- **R2-Q5: Still on same branch. No PR.** All CUDA work continues on
  `261021/cuda`; no PR is opened.
- **New directive A**: "maybe we have different streams for input and
  output (last Q8). Maybe we need move out same-stream test on
  equivalence of device?" — the owner proposes relaxing `same_device`'s
  strict stream equality so tensors on different streams of the same
  ordinal can interact. This reopens R1-Q8.
- **New directive B**: "this rstsr-cuda is not published. Any
  revert/API breaking change is acceptable." — the crate is
  unpublished; free rein on API changes, including revisiting
  device-basic decisions.

## Facts bearing on directive A

- Current `same_device` (device.rs:138-153): ordinal Arc-ptr-eq ∧
  stream `==` ∧ flags-as-stored ∧ order. Strict stream comparison was
  an explicit owner choice in 01-device-basic (R1 position "strict
  same-stream", Option A; mixed-stream = `DeviceMismatch`), motivated by
  loud mixed-mode errors.
- Same-ordinal `change_device` relabel is already zero-copy: the buffer
  keeps its allocation stream, the device value points at the new
  stream — so a kernel op on a relabeled tensor ALREADY encounters a
  foreign-allocation-stream input under today's semantics; the R1-Q8
  event-wait helper exists precisely for this.
- Streams user-visible from day one (`set_stream`); event tracking on
  (Drop safety). Multi-stream = the intended programming model
  (overlapping H2D/compute/D2H on side streams).
- Doc-set guidance: kernel caches are per-device not per-stream;
  BLAS handles are stream-bound (a second stream needs a second
  handle — relevant only when the BLAS round lands). Per-call stream =
  the device value's stream (D4: "device's stream owns launch").

## Questions

### R3-Q1 — `same_device` stream relaxation (directives A + B)

Split "device identity" from "execution stream" in `same_device`:

- `same_device` compares **ordinal only** (+ flags-as-stored + order as
  today): any two tensors on the same GPU are same-device, regardless
  of stream.
- Stream becomes an execution detail the **op** handles, not a gate the
  type system enforces: the op launches on the input device value's
  execution stream; every input buffer allocated on a different stream
  gets the event-wait (record on the buffer's stream, wait on the
  launch stream) — the R1-Q8 helper, now doing this work for ALL
  foreign-stream inputs, not just relabeled ones.
- Output allocation happens on the launch stream (as today).
- No API break needed (`same_device` signature unchanged); published
  status irrelevant since unpublished (directive B), but the change is
  additive in behavior: ops that used to error now succeed and order
  correctly.
- Alternative considered and rejected: keeping strict equality and
  requiring explicit re-tag for cross-stream use — that is exactly the
  01-device-basic design the owner is now revisiting.

➡️ Recommend yes: ordinal-only `same_device` + op-level event-wait
ordering. The type-level gate was doing runtime ordering's job; with
kernels landing and multi-stream tensors realistic, ordering belongs in
the op path. [Owner has effectively already chosen this; question
exists to confirm scope: `same_device` drops the stream clause
entirely?]

### R3-Q2 — `DeviceMismatch` error semantics after R3-Q1

After R3-Q1, `same_device` failure means only: different ordinals, or
flags mismatch, or order mismatch. Confirm that the *error message* for
cross-ordinal ops still names both devices (`cuda:0` vs `cuda:1`), and
that cross-ordinal peer access remains an explicit `change_device`/
`into_device` copy, never an implicit kernel-op fallback (peer-copy on
every op would silently serialize).

➡️ Recommend yes (explicit cross-ordinal copy only; error names both
devices).

### R3-Q3 — pre-flight validation list (locked before code)

Before implementation begins, a `pre-flight/` probe in this task dir
validates, on the local RTX 5070 (sm_120, CUDA 13.2 toolkit, cudarc
0.19):

1. NVRTC compile of the prelude + one cast kernel through the safe
   layer with `arch` from `compute_capability()`; verify accepted
   arch-string spelling(s) (`compute_120` vs `sm_120`) and that
   `--std=c++17` is accepted.
2. First-use latency measurement: compile+load of one kernel
   (cold, warm cache, disk-cache hit) — numbers recorded in this dir.
3. Builder u8-marshal probe: push `&CudaSlice<u8>`, `[u8; 16]`, u64,
   usize args through `launch_builder` + `LaunchArgs::launch`; verify
   no `DeviceRepr` beyond built-ins is needed.
4. Event-wait probe: two streams on one ordinal, event record/wait
   ordering a write→read across streams through `result::event`; verify
   correctness of the R3-Q1 ordering recipe.
5. Disk-cache layout probe: hash key, read/write, version invalidation
   on payload-format change.

➡️ Recommend yes — this list, run before the rstsr-cuda edits, closes
the doc03 §4.6 evidence gaps (compile latency, arch spelling, builder
marshal) for this machine.

### R3-Q4 — `CudaDType` name (small)

The R1-Q5 trait name `CudaDType` is free to change before code exists.
Options: `CudaDType` (parallel to `DTypeCastAPI`/`DTypePromoteAPI`
naming family), `CudaScalar` (doc01 §5's own suggestion), or
`CudaRepr` (mirrors cudarc's declined name, "CUDA representation").

➡️ Recommend `CudaDType` — reads as the device-side member of the
dtype-trait family, matches house `*API`-less trait naming for
capability markers (cf. `BlasScalar`, `ExtZero`).

### R3-Q5 — Round completion record

When this grill closes: DECISIONS.md in this dir (canonical record,
D-numbered), memory note update (supersede the "next grills likely"
line in cuda-device-basic-grill-closed), PLAN.md here with batches
B1..Bn mirroring the 01-device-basic plan shape (pre-flight P, batches,
tests, naive-path register for whatever remains naive, delivery), and a
side-ledger for the ride-along fixes (dtype-traits `defaults = ["half"]`
misspelling; tmp.rs cleanup; half↔float cast PR sequencing). No rstsr
commits without explicit owner instruction.

➡️ recommend yes (this is convention; question exists for the record).

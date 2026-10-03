# rust-numpy code literature review (rstsr ↔ NumPy interop)

- **Date**: 2026-10-03
- **Task**: comprehensively read rust-numpy (and the NumPy-DLPack / DLPack-v1.3 layers) to see what rstsr needs for a future *bi-directional* rstsr ↔ NumPy bridge — **review and consensus input only, no implementation**.
- **Status**: **review complete** (REVIEW + evidence + probes); **consensus in progress** — R1 answers received and processed (`ANSWERS-discussion-R1.md` → `RESPONSE-discussion-R1.md`), Q4 still open. Nothing was changed outside this directory; no implementation started.
- **Round 1 outcome**: Q1 (c), Q2 (c) + two-layer interchange, Q3 (a) `rstsr-cpu-*` named crate, Q7/Q10/Q12 accepted; Q4 re-opened; Q6's `device_faer` UB note verified correct-and-faer-specific (V1); a bridge storage repr needs **no** rstsr-core change (V2).
- **Pinned revisions** (everything in this directory is relative to these):
  - rust-numpy `da6bf5be05d4053cf0c51afa12efc018a984ca7f` (v0.29.0 + 8; 2026-08-28)
  - numpy `v2.5.2` (`48fecee545`) in `~/Git-Others/numpy`; installed numpy 2.5.1 (conda env `torch`) for probes
  - DLPack `v1.3` (`84d107b`; local checkout is `v1.3-4-g94485e2` — drift documented in notes)
  - `dlpack-ffi` 1.3.0 (`/home/a/rstsr_pack/dlpack-ffi`; see sibling task `../2026-10-03-dlpack-ffi/`)
  - rstsr `f179c46` (v0.9.0)
- **Touches nothing outside this directory** (reference checkouts read-only; no changes to any rstsr repo).

## Files

| File | What it is |
|---|---|
| [REVIEW.md](./REVIEW.md) | **main artifact** — the comprehensive review: rust-numpy architecture (§2–5), NumPy's DLPack support (§6), DLPack v1.3 contract (§7), rstsr-side gap analysis (§8), bridge options (§9–10) |
| [QUESTIONS-discussion-R1.md](./QUESTIONS-discussion-R1.md) | the consensus list: Q1–Q14 decision points with recommendations; Q1–Q4 are load-bearing |
| [ANSWERS-discussion-R1.md](./ANSWERS-discussion-R1.md) | maintainer's answers to R1 (2026-10-03): Q1c/Q2c (+two layers)/Q3a (`rstsr-cpu-*` naming)/Q7/Q10a/Q12; Q4 re-opened; Q6 asks for a check |
| [RESPONSE-discussion-R1.md](./RESPONSE-discussion-R1.md) | R1 response: per-question disposition, source-level verifications V1–V5, corrections C1–C5, minimal Q4 shapes, and the five points R2 must settle |
| [notes/numpy-dlpack.md](./notes/numpy-dlpack.md) | raw evidence: NumPy v2.5.2 DLPack producer/consumer source-read + empirical probes |
| [notes/dlpack-v13-spec.md](./notes/dlpack-v13-spec.md) | raw evidence: DLPack v1.3 contract (ownership, capsule protocol, versioning, layout) + `dlpack-ffi` surface |
| [notes/rstsr-state.md](./notes/rstsr-state.md) | raw evidence: rstsr bridge-relevant inventory (storage/layout/dtype/device) and gap list |
| [experiments/probe_numpy_dlpack.py](./experiments/probe_numpy_dlpack.py) | executable probe suite against the installed NumPy |
| [experiments/probe-output.txt](./experiments/probe-output.txt) | captured raw output of the above |

## Headline conclusions (details in REVIEW.md §0)

1. rust-numpy proves the NumPy-C-API route works and is mature, but it is a *large* maintenance surface (235+38 API slots, hand-kept struct layouts, per-version branches) and is NumPy-specific.
2. Its boundary is ownership-transfer out, borrow-only in; DLPack is the only mechanism that makes ownership transferable in **both** directions (the deleter travels with the tensor).
3. rust-numpy's cross-extension borrow-checking capsule is good prior art for aliasing discipline, but a v1 rstsr bridge likely does not need it.
4. NumPy 2.x speaks DLPack 1.0 (`max_version=(1,0)`) with a well-defined accept/reject table; the producer/consumer footguns are enumerated in §6 (e.g. NumPy may call `__dlpack__` twice; `max_version=None` yields a legacy, read-only-semantics capsule).
5. **`np.from_dlpack` requires a Python object with `__dlpack__`** — a bare capsule is rejected. The export side therefore needs one small Python-facing object (Q4 is the one question still open after R1; shapes proposed in `RESPONSE-discussion-R1.md` §4).
6. rstsr's numeric dtypes round-trip zero-copy except `i128/u128`/`bf16` (not consumable by stock NumPy); the gap in the other direction is datetimes/strings/object/structured and byte-swapped data (copy-fallback territory).
7. rstsr is closer than it looks: zero-copy *views* need only unsafe glue over existing `DataRef` machinery; *owning* foreign buffers needs a storage repr that carries the owner — and that repr can live in the bridge crate, **no rstsr-core change** (`RESPONSE-discussion-R1.md` V2; the in-repo UB note is faer-specific, V1).
8. **R1 settled every question except Q4** (see the disposition table in `RESPONSE-discussion-R1.md` §1); no implementation should start before the five R2 points in `RESPONSE-discussion-R1.md` §5 are answered.

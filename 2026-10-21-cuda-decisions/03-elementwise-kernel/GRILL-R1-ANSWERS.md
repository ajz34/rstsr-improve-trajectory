# GRILL R1 — answers (owner, 2026-10-23)

- **Q1 dtype coverage**: option 2 — full coverage including complex
  transcendentals (c32/c64) this round. Clarified: no device libm for
  complex exists; the prelude is hand-written from real libdevice calls,
  consulting cupy `complex.cuh` / pytorch `c10::complex` as references.
- **Q2 bench matrix**: option (a) — ~20 cells, each focus scenario at
  two working-set sizes (~8 MB L2-resident and ~48-72 MB DRAM-bound)
  plus a contig-add size ladder 1e3..1.6e7.
- **Q3 small sizes**: informational appendix, outside the 1.2x/2x
  gates — but the agent should still work on minimizing host-side
  overhead where reasonable (launch-path cost).
- **Q4 measurement**: median over **5 repeats (7 near gates)**, not 3.
  Owner will run sudo clock locking for the final tables:
  `sudo nvidia-smi -pm 1` + `sudo nvidia-smi -lgc <min>,<max>`
  (agent to confirm supported clock range first); reset with
  `sudo nvidia-smi -rgc`. Development runs proceed unlocked with the
  shared-GPU caveat documented.

Owner additions: rstsr-cuda APIs are fully changeable this round
(unpublished crate); no auto commit; stay on `261023/cuda-review-fixes`.

## Additional directives (post-R1, before implementation)

- Complex/hard formulas: hand-written or adapted from cupy/pytorch with
  explicit attribution comments; large headers as real `.cu`/`.cuh`
  files pulled in via `include_str!`, not Rust string templates.
- Prefer CUDA's own functions (fmax/fmin/rint/...) where CPU semantics
  allow; do not force Python-Array-API formulations in kernel source.
- Where dtype-change plumbing for some function is too difficult, a
  same-type-only implementation is acceptable for now.
- Sudo clock lock is applied by the owner (persistence on, SM pinned
  210 MHz; verified: cupy add 4e6 f64 = 0.438 ms, DRAM-saturated).
- **Implementation start waits for the owner's explicit instruction.**

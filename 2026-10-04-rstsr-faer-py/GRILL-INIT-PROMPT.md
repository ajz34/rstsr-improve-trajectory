# Grill init prompt — rstsr-faer-py (2026-10-04)

Session launched from the pack root via `/mattpocock-skills:grill-me`,
arguments verbatim:

> You are going to write a python package, rstsr-faer-py. Record this grill
> session in rstsr-improve-trajectory.
> - This is a blur requirement. You can challenge me, and I'm not always correct.
> - The main objective is not truely a python tensor package with rust backend.
>   The main objective is **testify python array-api** compliment.
> - This python package only merely intefers from/to rust/python. Interfacing
>   may be some trivial yet laborness work, but never implement any algorithms
>   at python side.
> - It is possible that our rstsr have not implemented all features that
>   python-array-api requires. We should be honest on this, and fix in future.
> - This package is going to be at repo `rstsr`, in some proper subfolder.

Follow-up corrections from the user (mid-R1 interrupt, 2026-10-04):

- `2026-10-04-rstsr-cpu-py03` → **rstsr-cpu-pyo3 is going to be abandoned.**
- We are NOT going to distribute a Python wrapper for all devices
  (efficiency of maintenance). **This task is validation-only.**
- Fact-finding requested by the user: current status of the rstsr repo, the
  rstsr-cpu-dlpack work, and the recent array-api comprehension notes.

Fact-finders dispatched (background subagents, results land in R2):

1. rstsr repo status: devices, dtype coverage per device, op inventory
   (creation/elementwise/reductions/manipulation/sorting/indexing), faer
   linalg surface, existing python/pyo3/dlpack integration, CI, placement
   candidates for a python subfolder.
2. rstsr-improve-trajectory: 2026-10-03-rstsr-cpu-dlpack,
   2026-10-04-arrayapi-compliance-notes, 2026-10-02-arrayapi-nan-compliance,
   2026-10-04-rstsr-cpu-pyo3 (abandonment context), 2026-10-03-rust-numpy-review,
   plus in-repo agent memory.
3. ~/Git-Others/array-api-tests checkout: version, invocation, numpy
   dependency, test inventory, the contract it imposes on `xp`.

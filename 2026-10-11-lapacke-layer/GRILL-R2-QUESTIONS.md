# Grill R2 — LAPACKE translation layer (session 2026-10-11)

## R1 answers (agent-transcribed from the on-screen session; user-owned original in the conversation)

- **Q1 placement**: OK — skill (operational recipe) + ADR (rationale).
- **Q2 form**: Recipe preferred, but rationale must be included for cases the
  recipe does not cover; when the agent hits an uncovered case it must STOP
  and notify the user that the skill/ADR lacks coverage and needs updating
  (rather than improvising silently).
- **Q3 scope**: OK as recommended — guideline + SVD workspace/protocol retrofit
  as the committed scope; remaining deviations become a checklist executed
  routine-by-routine on a rstsr branch behind review. User adds: the session
  should also make actual code modifications to validate the thinking
  (prototype-level validation is expected, not just prose).
- **Q4 fidelity / input policy** (settled): non-f-prefer inputs (c-contig,
  c-prefer, fully non-contiguous) are ALWAYS cloned; f-prefer inputs pass by
  reference (zero-copy, derived ld); routines that overwrite the input clone
  (or consume) regardless of contiguity. Difference vs LAPACKE: LAPACKE only
  accepts c-prefer/f-prefer (contiguous, ld == dim by construction); we
  additionally handle fully non-contiguous via the clone path.
- **Q5/Q6 tests**: no such test policy exists yet — this session builds it.
  Test data small: no more than ~20 rows/columns per side.

Facts base: FINDINGS.md (four comprehension reports consolidated).

---

## R2 frontier

❓ **Q1 - Canonical bridge & fate of the two impls**: The settled input policy
means our canonical driver is col-major-only + f-prefer zero-copy + clone-else —
LAPACKE's row-major branch (its `dge_trans` temps, its `matrix_layout` arg)
disappears from our design. What happens to `driver_impl/lapacke/` (the thin
C-LAPACKE passthrough, openblas+mkl, behind `--features lapacke`)?
(a) retire now — delete the feature in the retrofit PR; (b) deprecate — keep
compiling untouched, delete after the full retrofit checklist completes;
(c) keep both indefinitely as a user-selectable backend.

➡️ Recommend (b): one canonical impl is the deliverable, but deleting a shipped
feature mid-campaign widens the blast radius of the SVD PR; removal is its own
small PR at the end.

❓ **Q2 - Workspace ownership tier**: Today the vendor `driver_impl/lapack/`
files do query+alloc themselves. LAPACKE's own split puts workspace in the
high level and leaves `_work` a pure caller-of-LAPACK. Mirror that?
(a) move query+alloc into the generic builder layer (`internal_run`); vendor
`driver_xxx` shrinks to a thin FFI thunk taking `&mut [T]` work slices
(query = thunk call with lwork=-1, real = thunk call); (b) keep workspace in
the vendor impls (status quo).

➡️ Recommend (a): it kills the two-impl divergence at the root (one workspace
policy, in dtype-generic rust, testable once), vendors shrink to ~5-line
thunks, and `blas_int` stays confined to the thunk boundary.

❓ **Q3 - Error channel & info mapping**: Keep `unsafe fn driver_xxx(..) ->
blas_int` at the thunk (it IS the LAPACK info), and mandate at the builder
layer: `info < 0` → Error "illegal argument in <routine> arg <-info>
(internal bug, please report)"; `info > 0` → per-routine semantic message
(e.g. "SVD did not converge", "matrix is singular"); allocation failures →
normal rstsr Error (drop the −1010/−1011 sentinels); fix `info as i32`
truncation (ilp64); replace `sygv/sygvd`'s `rstsr_assert_eq!` and the raw
drivers' `.unwrap()` panics with the same mapping. Confirm this mapping, and
that we do NOT adopt LAPACKE's `info−1` remap (it exists only because
LAPACKE inserts `matrix_layout` as arg 1 — we don't have that arg)?

➡️ Recommend yes to all (the remap is meaningless without `matrix_layout`;
our thunk returns raw Fortran info).

❓ **Q4 - Output order policy**: With col-major-only drivers, LAPACK writes
F-ordered results. What order do builder-returned tensors get?
(a) always ColMajor (zero transpose-out ever; LAPACK writes straight into the
output tensor; row-major-device users receive F tensors); (b) device
`default_order()` (LAPACK-side buffers F; one transpose-out copy when the
device default is RowMajor — the price scipy/numpy also pay for their users);
(c) follow the input tensor's natural order (closest to current behavior,
keeps the 4-arm idiom).

➡️ Recommend (b): one mandated rule replacing the three coexisting idioms,
predictable for users (outputs match the device's convention), and col-major
devices (faer-style) get the zero-copy corner automatically. (a) is cheaper
per-call but leaks F tensors into row-major workflows.

❓ **Q5 - ipiv base unification**: Unify on 0-based rust-facing index vectors:
getrf/getri already pair (−1/+1) correctly end-to-end; sysv leaks 1-based
ipiv. Mandate "all index vectors (ipiv, jpvt, ...) are 0-based at the builder
boundary; ±1 conversion lives only inside the thunk" and fix sysv?

➡️ Recommend yes.

❓ **Q6 - gesvd superb**: Keep the LAPACKE-faithful 4-tuple (s, u, vt, superb)
at the driver/builder tier with the higher linalg layer discarding it
(status quo upstream), or drop superb entirely (3-tuple, matching gesdd)?
(KML's local override of f64/gesvd.rs references the current shape.)

➡️ Recommend keep at driver tier (LAPACKE-faithful; superb is the documented
diagnostic for info>0 convergence failures), re-check the KML override when
implementing.

❓ **Q7 - Zero-size matrices**: Current short-circuit returns zero-filled
U/VT; LAPACK returns identity sub-blocks. (a) synthesize identity sub-blocks
in the builder without calling the driver (LAPACK-faithful result, no backend
edge-case exposure); (b) keep zeros; (c) always call the driver (most
faithful, exercises m==0 paths, but trusts every vendor on empty input).

➡️ Recommend (a).

❓ **Q8 - NaN/finite input check**: LAPACKE default-ON full-pass nancheck
(and scipy's check_finite) exist to keep uninitialized/NaN input from
reaching memory-unsafe C. Adopt anything at our tier?
(a) none — document as deliberate divergence (user-level concern; rust
doesn't share the uninitialized-read hazard LAPACKE guards against);
(b) debug_assert-only finite check; (c) runtime check behind a flag.

➡️ Recommend (a).

❓ **Q9 - Workspace hardening set**: Mandate in the recipe: query with the
real (col-major) lds; `MAX(1, ·)` on every formula; `lwork==0 → 1` clamp;
checked int conversion of the query value. Additionally adopt scipy's
`nextafter` guard for f32/c64 queries (work(1) passes through the f32
mantissa; ≥2^24 rounds down → undersize)? And explicitly REJECT the 1.01
fudge factor (OpenBLAS getri-specific workaround)?

➡️ Recommend: all of the above; nextafter yes (one line, kills the class),
fudge no.

❓ **Q10 - Validation matrix for the retrofit branch**: Which vendors must
the branch prove locally before PR — openblas only, or openblas + mkl (both
installed per AGENTS.local.md), more? And is extending driver-tier tests to
c64 (complex) in scope for THIS PR (touched rwork paths say yes for
gesvd/gesdd; other routines go to the checklist)? All new tests small
(≤ 20 per side) per R1-Q5/Q6.

➡️ Recommend: openblas + mkl locally (the two with the `lapacke` feature at
stake), blis/aocl/kml via CI; c64 tests for gesvd/gesdd yes, rest deferred
to the checklist.

❓ **Q11 - Execution plan confirm**: After R2 closes: (1) draft the skill
`lapacke-translation` (recipe + rationale + gap-notice rule) and the ADR in
rstsr-book/dev; (2) branch `261011/lapacke-svd-retrofit` in rstsr; (3)
implement the settled policies (input taxonomy, workspace tier, error
mapping, output order, ipiv, zero-size, hardening) with gesvd/gesdd as the
vehicle; (4) add the 3-case layout tests (f-prefer ld>dim / c-prefer /
fully-non-contig, small sizes, f64+c64); (5) run the blas test matrix
locally, iterate; PR only on your explicit go. Confirm or amend?

➡️ Recommend confirm.

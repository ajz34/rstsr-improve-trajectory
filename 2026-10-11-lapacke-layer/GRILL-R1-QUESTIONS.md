# Grill R1 — LAPACKE translation layer (session 2026-10-11)

Scope being grilled: the LAPACKE→rust translation layer in `rstsr-blas-traits`
(`driver_impl/lapacke/`), its guideline-to-be, and the retrofit of deviating
routines. Four read-only comprehension agents (rstsr inventory, LAPACKE
patterns, scipy patterns, SVD cross-comparison) are running; their findings
feed R2.

---

❓ **Q1 - Guideline placement**: Where does the translation guideline live?
Candidates: (a) a skill in `rstsr-agents/skills/` (like `update-ffi-blas`),
auto-loaded by subagents doing translation tasks; (b) an ADR in
`rstsr-book/dev/`; (c) `rstsr-agents/rules/*.md` reference material (like
`code-concepts.md`); (d) dev-doc prose in `rstsr-book/dev/` outside ADRs.

➡️ Recommend (a)+(b): a skill `lapacke-translation` as the operational recipe
(checklists, decision tables, reference shapes) — skills are the mechanism that
actually reaches an unattended subagent; plus one short ADR recording the design
decisions (the f-prefer/c-prefer taxonomy, LAPACKE-fidelity policy) so the "why"
survives. The ecosystem map already assigns this split: skills = how, ADRs = why.

❓ **Q2 - Guideline form**: Prescriptive recipe (numbered mandatory steps,
per-routine decision tables, reference implementations pinned by file:line,
ready-to-fill code skeleton) vs design prose (principles + rationale, agent
improvises the application)?

➡️ Recommend the recipe. The stated goal is "different subagents behave the
same" — that is a checklist property, not a prose property. Rationale goes to
the ADR; the skill carries only what a translating agent must do, in order.

❓ **Q3 - Edit scope of this effort**: (a) write guideline only; (b) guideline +
retrofit the SVD workspace allocation to strict LAPACKE protocol; (c) guideline
+ retrofit of *all* deviations the inventory finds. And where do edits land:
prototype in this trajectory repo first, or directly on a branch in the rstsr
repo (PR gated, per the "partial AI usage → manual review" policy for
rstsr-blas-traits code)?

➡️ Recommend (b) as the committed scope of this session's follow-up (SVD is the
user-named pain point), with (c) turned into a checklist the guideline ships
with — retrofit executed routine-by-routine on a rstsr branch, not bulk-mutated;
experiments/benchmarks may use this trajectory dir, but the deliverable edits
belong in rstsr behind review.

❓ **Q4 - Fidelity principle**: Is the layer's contract "LAPACKE semantics
exactly, including its row-major bridging (dge_trans copies)" or "LAPACKE
semantics + sanctioned zero-copy stride extensions" (the f-prefer/c-prefer
taxonomy: pass strided buffers directly with derived ld; exploit
call-on-transpose where LAPACKE/scipy do)? When the two conflict, which wins?

➡️ Recommend: LAPACKE is the authority for *protocol* (workspace queries,
argument order, info mapping, what gets transposed); the stride extensions are
an additive generalization of LAPACKE's two layouts (f-contig/c-contig become
special cases of f-prefer/c-prefer). When a zero-copy trick is ambiguous or
routine-specific, fall back to the LAPACKE copy behavior and document the
exception.

❓ **Q5 - Determinism scaffolding**: To make unattended subagents converge,
should the guideline ship a mandatory per-routine checklist (input-case matrix
per tensor: f-prefer/c-prefer/non-contig; workspace policy; output policy; test
entries) and a code skeleton template (module layout, naming, _f twins,
dtype-twin macros) that the agent fills in?

➡️ Recommend yes to both; the checklist is the artifact a reviewer (human or
agent) diffs against, which is also how "evident random behavior" gets caught.

❓ **Q6 - Test policy in the guideline**: New/retrofitted drivers must extend
`blas_tests_impl` (f64/c64, python-generated validation) — should the guideline
additionally *require* the three layout cases as explicit test inputs (f-prefer
with ld>nrows via padding/striding, c-prefer likewise, fully-non-contiguous
e.g. broadcast-ish strides), since that is exactly where translations go wrong?

➡️ Recommend yes — the 3-case layout matrix per tensor input becomes a required
test dimension alongside dtype, mirroring how the entry-binary test matrix
(ADR-0002) pinned row/col-major.

---

*Pending for R2 (blocked on comprehension agents): exact f-prefer/c-prefer
classification predicate (what the codebase does today vs what the guideline
should mandate), per-routine workspace policy table (query vs formula),
c-prefer bridging mechanics (transpose-copy vs call-on-transpose per routine),
shared-helper design (central layout classifier / LapackMatrixView type),
output-order conventions, retrofit deviation list, error/NaN-check policy,
complex rwork/iwork handling, blas_int conventions.*

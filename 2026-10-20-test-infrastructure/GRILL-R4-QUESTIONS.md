# Grill R4 — test infrastructure redesign (final naming/destination round)

## R3 answers (agent transcription of the user's GRILL-R3-ANSWERS.md, 2026-10-20)

- **Q25 (okay)**: absorbed mechanics land in `core-test` as a conventions
  section; rule stays policy + routing.
- **Q26 (okay + additions)**: skill list accepted; test relocations/detailed
  changes within those skills' scope are also in-scope for the refactor; the
  test-related skills get a **`test-` prefix** (exact names = Q30).
- **Q27 (merged)**: one merged testing glossary instead of a scope split
  (destination = Q31).
- **Q28 (okay)**: device/order new-entry checklists accepted as the guideline
  backbone.
- **Q29 (yes)**: one branch for B1–B4, per-task review stops.

## Q30 — Renamed skill names under the `test-` prefix

Applies to the core test family + the new device skill; `cargo-inst`,
`rstsr-faer-py-tests`, `api-cheatsheet-conventions`, `api-doc-conventions` keep
their names (not primarily test-authoring skills, or already suffixed `*-tests`).

- (a) `test-core` (ex-`core-test`, + conventions section), `test-core-issues`
  (ex-`core-issue-regression`), `test-core-numpy-sync` (ex-`core-numpy-sync`),
  `test-device` (new).
- (b) Shorter: `test-core`, `test-issues`, `test-numpy-sync`, `test-device`.

➡️ (a). The `core-` stem keeps the rstsr-core vs device scoping that the deleted
name was criticized for losing; (b) reads as if `test-issues` were
ecosystem-wide.

## Q31 — Destination of the merged testing glossary

- (a) **`rstsr-core/tests/CONTEXT.md` becomes the single testing glossary**:
  ecosystem-level terms (tiers, entry kinds, sharing mechanisms, conformance
  entry, dedicated-file exception) are added next to the existing core terms
  (parity test, provenance, `specify_test!`, ...); `rstsr-agents/CONTEXT.md`
  keeps exactly its existing pointer line ("testing-domain terms live in
  rstsr-core/tests/CONTEXT.md").
- (b) `rstsr-agents/CONTEXT.md` absorbs all testing terms; the core file is
  deleted after migration.

➡️ (a). The pointer already exists today; the glossary stays next to the tests
it describes and loads with the authoring flow; (b) moves a large core-specific
glossary away from the test tree to sit beside the rule.

---

## Answering

Edit `GRILL-R4-ANSWERS.md` (user-owned) or answer on screen. This is the last
round — after it the frontier is empty and closure writes `DECISIONS.md` here
(no rstsr / rstsr-agents edits until you confirm and give the go).

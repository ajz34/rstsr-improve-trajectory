# Round-2 answers — dlpack-ffi repo redesign

- **Date**: 2026-10-03 (answered in session)
- **Questions**: [round-2-questions.md](./round-2-questions.md)

## Verbatim answers

- **R2-Q1**: Use 1.3.0, follow the crate's own policy (follow the dlpack's own version). If updates needed for tensions, use patch to update. This repo does not follow semver.
- **R2-Q2**: Keep.
- **R2-Q3**: Okay. You can try that.
- **R2-Q4**: Well rstsr-agents can notice this package (and skills added are directly related to this package). rstsr-improve-trajectory can notice this package. dlpack-ffi should be stay as pure, and not mention "rstsr" or "rest" in dlpack-ffi for any useful purpose.

## Decisions as resolved

| Q | Decision |
|---|----------|
| R2-Q1 | Release as **1.3.0**; crate version tracks the DLPack version; subsequent adjustments get patch bumps (1.3.1, …); the repo **does not follow semver** — the one-time API break ships inside 1.3.0, documented in the changelog. |
| R2-Q2 | Package name stays `dlpack-ffi`. |
| R2-Q3 | The 7-item upgrade checklist is approved (NewType regeneration, header → v1.3 + provenance, script fix, tests, LICENSE, readme rewrite, small cleanups). |
| R2-Q4 | Split policy: **rstsr-agents and rstsr-improve-trajectory may reference the package** (ecosystem-map line; skill `update-ffi-dlpack` is directly related to it; campaign docs); **dlpack-ffi itself stays pure** — no "rstsr"/"rest" references in its content, docs, code, or metadata. |

## Interpretation notes (drive the implementation)

- **Purity constraint in dlpack-ffi**:
  - The stale `cargo fmt -p rstsr-dlpack-ffi` in `scripts/bindgen.py` is the only known in-repo "rstsr" string and must be fixed (already on the checklist). Before finishing, verify with a repo-wide grep that no other such references remain (vendored header, generated code, readme, Cargo.toml, workflows stay clean).
  - The readme rewrite must not link to rstsr projects; keep it a self-contained minimal DLPack binding.
  - Agent access: the update skill lives in rstsr-agents; sessions rooted in the pack (or with optional, untracked local symlinks) can run it. **No tracked files are added to dlpack-ffi for agent wiring.**
- **Version policy to be written into the readme**: crate version = DLPack version (1.3.0 now), patch bumps for anything else, no semver promise.
- **Frontier closed**: no open design questions remain (rounds 1–2 fully answered).

## Next steps (not yet started)

- `dlpack-ffi`: the 7-item checklist on a `YYMMDD/…` branch, pushed fork-first (`ajz34` → PR to `RESTGroup/dlpack-ffi`) per pack convention; release via the existing release-plz `workflow_dispatch`.
- `rstsr-agents`: new skill `update-ffi-dlpack` (check + update modes), skills-tree registration, ecosystem-map line.
- This directory: a `PLAN.md` capturing the above before execution.

---
name: rstsr-arrayapi-nan-compliance
description: 2026-10-02 owner directive executed — array API NaN compliance lands for min/max family; supersedes patch-4 A2 closures and #100 plain-arg NaN decision; nanmin/nanmax added.
metadata:
  type: project
---

2026-10-02 session (rstsr master ca325a7): owner ruled array API compliance
beats efficiency ("no matter efficiency", formalizing the 260921 ruling,
[[rstsr-reductions-integration]]) and scoped the min/max family fix now;
systematic whole-standard audit + array-api test mechanism explicitly
postponed.

Decisions + facts (full trail in `2026-10-02-arrayapi-nan-compliance/README.md`):
- min/max reductions + elementwise minimum/maximum → NaN-propagating
  (spec MUST; empty-error and rstsr dtype behavior already conform).
- plain argmin/argmax → NumPy first-NaN-wins anyway (spec UNSPECIFIED;
  owner chose plain-op family consistency). SUPERSEDES the measured
  rejection in [[rstsr-nanarg-semantics]] — that rejection was
  efficiency-gated and the gate is gone; nanarg* (#100) unchanged.
- NEW nanmin/nanmax (NumPy value semantics: skip, all-NaN → NaN, no
  warning analogue, empty → error). Standard has NO nan_* at all.
- f64::minimum/maximum still unstable (issue #91079, nightly 1.100) →
  hand-rolled is_nan selects.
- Patch-4 A2 strict-compare closures SUPERSEDED (propagating forms land
  instead); A band-walk sum kernel still queued, post-compliance.
- Local spec source rule: ~/Git-Others/array-api (RST/docstring sources),
  NOT network fetches.

Cost numbers (arg variants): [[rstsr-nanarg-semantics]] sweeps stand as the
reference; paired A/B for the landed forms to be appended in the dir README.

---
name: rstsr-faer-py-wrapper-only
description: Owner directive — rstsr-faer-py is a pure wrapper (no algorithms in Rust OR Python); missing capabilities get registered and wait for rust-side fixes; shim-side algorithms need explicit per-case permission
metadata:
  type: feedback
---

Owner directive (2026-10-04, mid-`where` implementation): rstsr-faer-py
(`crates-interop/rstsr-faer-py`) must remain a simple wrapper — marshalling,
validation, rstsr calls — with **no algorithms or true implementations in
either its Rust or its Python layer**, unless the owner explicitly permits a
specific algorithm. "Honestly let it be not implemented in rstsr" — a
missing capability is a gap-register entry and a rust-side fix request, not
a shim workaround.

**Why:** the shim is a validation instrument for the Python array API
standard; algorithms written in the shim would mask rstsr's real gaps (the
thing the campaign measures) and duplicate rust-side work.

**How to apply:** before writing any non-trivial logic in the shim, ask:
is this marshalling (allowed) or an algorithm (needs permission)? Register
the gap either way. The mask/fancy
`__getitem__` gathers were briefly shim-side (written under the explicit
"fix __getitem__" instruction) and were REVERTED on the owner's follow-up
ruling — they are registered as rust-side gaps G-038/G-039. Also: no rstsr commits without owner review (reaffirmed same
day). Related: [[rstsr-faer-py-grill]], [[array-api-tests-explain-phase]].

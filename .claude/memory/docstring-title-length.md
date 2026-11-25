---
name: docstring-title-length
description: "Owner preference on rustdoc (2026-11-25): a long module/function docstring is tolerated, but its title (the first paragraph before the first blank `///` line) must stay short — one sentence, one or two lines; move qualifiers, axis rules and exceptions into the following paragraphs."
metadata:
  type: feedback
---

Given while reviewing the BLAS eig module docs: "long module docstring
is not fit to other files, but this is allowed now; just note not let
the title of docstring be too long."

- Title = the opening paragraph up to the first blank `///` / `//!`
  line. Keep it to one sentence that says *what* the item is; no
  parentheticals listing axis rules, orders, shapes or exceptions.
- Put those in named follow-up paragraphs ("Matrix slices: ...",
  "The inputs are consumed as scratch: ...").
- Precedent after the fix: `rstsr-linalg-traits/src/blas_impl/{eig,
  eigvals}.rs` — two-line titles, details in the body (43 / 28 doc
  lines, while the sibling `blas_impl/*` files carry no module doc).
- `wrap_comments = true`, `comment_width = 100` in `rustfmt.toml`
  re-wraps over-long comment lines; run `cargo fmt` after editing docs.

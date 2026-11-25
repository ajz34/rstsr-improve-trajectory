---
name: duplicate-item-identifier-traps
description: "duplicate_item substitutes every identifier token named like a column item (a loop variable `name` becomes the test name) and never touches identifiers inside string literals (format captures `{case}` are not substituted); data-function items belong in the table as full paths to avoid unused-import warnings."
metadata:
  type: reference
---

Seen in `blas_tests_impl` while merging the eig layout tests (2026-11-25):

- A `#[duplicate_item(name, case, …)]` body is token-substituted: any
  identifier equal to an item name is replaced, including a local binding
  (`for (name, …) in …` became `for (test_dggev_layout_f_contig, …)` → "cannot
  find value `name`" at the format capture). Do not name locals like items.
- `format!("{case} …")` is a string literal: implicit captures are not
  substituted. Write `format!("{} …", case, …)`.
- Substituting bare function names (`[data_f64]`) into only some rows leaves
  the others with unused imports; put the full path in the table
  (`[super::geev::data_f64]`) and drop the `use`.
- A table row may carry function items and driver aliases
  (`[DGEEV]`, `[ZGEEV]`): `driver::default().a(…)` keeps inference from the
  operands.

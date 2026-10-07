---
name: numpy-unique-aliases
description: NumPy array-API unique_* aliases are unordered and use equal_nan=False (numpy >=2.3); differs from np.unique
metadata:
  type: reference
---

Verified with numpy 2.5.1 (pinned reference v2.5.2), for rstsr `unique_*`
parity work:

- `np.unique([3,1,3,2,1])` → `[1 2 3]` (ascending), but
  `np.unique_values([3,1,3,2,1])` → `[2 1 3]` — since NumPy 2.3 the array-API
  aliases (`unique_values`/`unique_counts`/`unique_inverse`/`unique_all`) do
  **not** guarantee any order.
- `np.unique([nan,1,nan])` → `[1. nan]` (collapses NaNs), but
  `np.unique_all([nan,1,nan]).values` → `[1. nan nan]`, counts `[1 1 1]`
  (the aliases pass `equal_nan=False`, so NaNs stay distinct).
- `np.unique([-0.0,1.0,0.0])` → `[-0. 1.]` (first-seen signed zero is kept).

So a claim "NumPy always sorts / collapses NaNs" is true only for `np.unique`,
not for the aliases rstsr's `unique_*` actually mirror. Related test file:
`rstsr-core/tests/core_func/set/test_set.rs`, registry:
`rstsr-core/tests/tracking/numpy_differences.md`.

Also: `np.nonzero(np.array(0))` **raises** ValueError (it is not an empty
result) — 0-d nonzero is parity, not an rstsr deviation.

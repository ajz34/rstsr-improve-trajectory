# Wheel build + conformance-suite notes (2026-10-07, tensor-tier API revision round)

Context: end-to-end verification of the tensor-tier API revision (positional
`rt::func(x, args)` free functions + inherent methods, `XAPI`-like traits removed)
through the `rstsr-faer-py` bindings.

## Finding 1 — "opt-0 wheel" means the RELEASE profile at `opt-level = 0`, not the dev profile

The suite numbers depend on **overflow checks**:

- `maturin build` with no flags (dev profile) → `debug-assertions = true` → integer
  overflow panics. Result: **1049 / 251 / 82**, with exactly 10 nodes flipping
  `passed → failed`, all on overflowing integer arithmetic:
  `test_bitwise_left_shift[__lshift__(x, s)]` / `[__ilshift__(x, s)]`,
  `test_bitwise_right_shift[__rshift__(x, s)]` / `[__irshift__(x, s)]`,
  `test_subtract[__sub__(x, s)]`, `test_sum`, `test_prod`, `test_cumulative_prod`,
  `test_diff`, `test_diff_append_prepend`. Those 10 nodes passed in **all 6 earlier
  suite stamps** (`…002122` … `…092159`), so this is not hypothesis replay wobble.
- `maturin build --release` with `CARGO_PROFILE_RELEASE_OPT_LEVEL=0` → opt-0 **with
  `debug-assertions = false`** → the baseline numbers came back byte-for-byte
  (1059 / 241 / 82, 0 differing nodes).

Recipe that is correct *and* memory-lean:

```bash
cd <rstsr-checkout>/crates-interop/rstsr-faer-py
CARGO_BUILD_JOBS=2 CARGO_PROFILE_RELEASE_OPT_LEVEL=0 CARGO_PROFILE_RELEASE_DEBUG=0 \
  maturin build --release -i "$TEST_PY" -o /tmp/wheels
"$TEST_PY" -m pip install --force-reinstall --no-deps /tmp/wheels/rstsr_faer_py-*.whl
```

(The `--force-reinstall --no-deps` step is mandatory: maturin does not replace an
installed same-version extension, so a suite run can silently grade the old wheel.)

## Finding 2 — default `--release` (opt-level 3) is OOM-killed on this machine

`maturin build --release` (the literal skill recipe) died as:

```
rustc --crate-name rstsr_faer … --crate-type cdylib -C opt-level=3 … (signal: 9, SIGKILL: kill)
```

— the kernel's OOM killer, with no rustc diagnostic anywhere in the log (so not a code
error). At the time: 61 GiB RAM, 34 GiB used, **swap 8.0 / 8.0 GiB fully consumed**,
rust-analyzer (VS Code extension) alone at 9.1 GiB RSS, and cargo building with
`nproc` = 16 parallel jobs. The crates themselves are fine (debug-profile tests, doctests,
clippy, rustdoc and `cargo check --workspace --all-targets` all pass); only the optimized
cdylib codegen exceeded the remaining budget.

Cures, in increasing order of disruption: `CARGO_BUILD_JOBS=2`,
`CARGO_PROFILE_RELEASE_DEBUG=0`, `CARGO_PROFILE_RELEASE_OPT_LEVEL=0`, or free memory
(reload/disable rust-analyzer) before building.

## Finding 3 — the API revision is node-identical

Chunked, derandomized suite (`NO_EXPLAIN=1 MODULE=rstsr_faer.api CHUNKED=1`):

| stamp | passed | failed | skipped | total |
|--|--|--|--|--|
| `20261007-092159` (committed baseline, `2aacd5e`) | 1059 | 241 | 82 | 1382 |
| `20261007-114324` (API revision, working tree) | 1059 | 241 | 82 | 1382 |

**0 differing nodes** of 1382. The signature-level refactor (traits removed, positional
free functions, inherent `TensorAny` methods, `AxisIndex::None`, `RepeatArgs` / `RollArgs`)
is behavior-preserving through the Python layer.

## Round 2 (same day): method-parameter loosening + audit fixes re-verified

Follow-up changes: `isin` / `searchsorted` / `take_along_axis` methods now take
`impl TensorViewAPI<...>` for their non-self tensor argument; `matmul_from_f` lost a
redundant `TC: Zero` bound (the device impls carry that requirement themselves), so
`matmul_from`/`matmul_from_f` are now bound-identical; `allclose`, `allclose_all`,
`reduce_all`, `reduce_axes`, `reduce_with_args` and `into_atleast_1d/2d/3d` gained the
associated methods the audit had flagged as missing.

Re-verified with the same wheels recipe (release profile, opt-level 0, `-j 2`, no
debuginfo), stamp **`20261007-115926`**: **1059 / 241 / 82 of 1382**, **0 differing
nodes** vs `20261007-092159`. Rust-level gates at the same revision: entry tests
520/0, doctests 247/0, clippy `-D warnings` 0, rustdoc 0, `cargo check --workspace
--all-targets` and `col_major` clean.

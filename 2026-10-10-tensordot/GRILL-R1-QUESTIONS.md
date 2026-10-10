# GRILL-R1 — tensordot (API design)

Agent-authored. Answers go in `GRILL-R1-ANSWERS.md` (user-owned).

This round is the **frontier**: decisions that do not depend on each other. The
kernel-signature / axis-threading round comes after Q1 is settled.

---

## Facts gathered (basis for the recommendations)

Explorations of rstsr `main` @ `d7056ba`, 2026-10-10:

1. **No core `tensordot` exists.** The name exists only in the separate plugin
   `crates-plugin/rstsr-tblis/src/tensordot_impl.rs` (TBLIS-backed, `TblisFloatAPI`,
   real-float only) — `tensordot(a, b, axes: impl TryInto<AxesPairIndex<isize>>)`.
   Its semantics are defined by `rstsr-common/src/tensordot_to_einsum.rs::tensordot_to_einsum_str`,
   which already encodes the array-API rule: `None → 2`; `Val(n) → last n axes of a`
   vs `first n axes of b`; `Pair(seq_a, seq_b)` via `normalize_axes_index`.
2. **`vecdot` is the template**: it is a *device op* — `DeviceVecdotAPI`
   (`operators/linalg.rs`) + serial impl (`device_cpu_serial/linalg/vecdot.rs`,
   kernel in `rstsr-native-impl/src/cpu_serial/vecdot.rs`) + a **hand-written**
   rayon twin (`rstsr-native-impl/src/cpu_rayon/vecdot.rs`) exposed through
   `feature_rayon/auto_impl/vecdot.rs`, which is **symlinked** into `device_faer/`
   and all five `crates-device/*/src/rayon_auto_impl/`. There is **no macro** that
   derives rayon from serial — "rayon-auto-impl" *is* the per-consumer alias
   `DeviceRayonAutoImpl` + those symlinks.
3. **`AxesPairIndex<isize>`** (`rstsr-common/src/axes_pair_index.rs`) is the shared
   axes-pair type: `None | Val(T) | Pair(AxesIndex<T>, AxesIndex<T>)`. It has a
   deliberate **bare-collection shorthand**: `vec![0,1]`/`[0,1]`/`&[0,1]` →
   `Pair(same, same)` ("same axes on both sides"), while `(0,1)` → `Pair(Val(0),
   Val(1))`. vecdot documents this. **The convention of `Val(n)` is call-site
   specific**: vecdot reads it as "same axis `n` both sides"; the tblis tensordot
   reads it as "last `n` of a vs first `n` of b".
4. **`matmul` is NOT an auto_impl op** — it is BLAS/faer-specialized
   (`DeviceMatMulAPI` + `gemm`, per-device `matmul.rs`). A tensor-layer
   `tensordot` built from `permute → reshape → matmul → reshape` would need **zero**
   new device code and would be BLAS/faer-accelerated; a bespoke naive device
   kernel would need the 6-file + 6-symlink surface vecdot has.
5. **`vecdot` conjugates** (`x.ext_conj() * y`); **`tensordot` must not** (array
   API: "must not complex-conjugate or transpose either argument").
6. **Acceptance is already written down**: `2026-10-10-arrayapi-linalg/CHECKLIST.md`
   §R3 — core `tensordot` gates exactly:
   `test_has_names[linear_algebra-tensordot]`,
   `test_signatures.py::test_func_signature[tensordot]`,
   `test_linalg.py::test_tensordot`, and the `linalg-*` name node.
7. Trajectory convention: this dir; deliverable = patch/diff + porting notes;
   no commits into main rstsr repos from here.

---

## Round 1 questions

### Q1 — Architecture: bespoke device kernel, or tensor-layer composition?

- **(a) Device op mirroring `vecdot`** — new `DeviceTensordotAPI`; serial kernel
  `rstsr-native-impl/src/cpu_serial/tensordot.rs`; hand-written rayon twin
  `cpu_rayon/tensordot.rs`; `feature_rayon/auto_impl/tensordot.rs` + 6 symlinks;
  tensor-layer `tensor/linalg/tensordot.rs`. Every device (incl. faer/BLAS, via
  `DeviceRayonAutoImpl`) runs the *naive* kernel.
- **(b) Tensor-layer composition** `permute → reshape → matmul → reshape` — no new
  device trait, no kernels, no symlinks; BLAS/faer-fast on every device; built from
  existing `into_shape`/`to_permute`/`matmul`.
- **(c) Hybrid** — (b) now, (a)'s kernel later as an optimization.

➡️ **(a)**, per your directive ("CPU-serial and rayon-auto-impl", "naive is okay").
But flag: (b) is strictly less code and faster, and is essentially what the tblis
plugin already does via einsum; the naive *general-contraction* kernel is
meaningfully harder to get right than vecdot's single-axis-pair kernel. Confirm you
want the kernel as the deliverable rather than the composition.

### Q2 — Which `axes` call-shapes are accepted?

Array API allows only `int` or `(Sequence[int], Sequence[int])`; NumPy additionally
allows an int in either slot / a bare `(int, int)` pair; negatives allowed.

- **(a) Strict array-API only**: `int`, `(seq, seq)`.
- **(b) NumPy superset**: `int`; `(A, B)` where each of A, B is an int *or* a seq;
  negative axes allowed; bare flat even-length sequence rejected.

➡️ **(b)** — the parity tests are NumPy-based, so `(int, int)` (`np.tensordot(a,b,(1,0))`)
must work anyway; supporting it is free once each side goes through `AxesIndex`.

### Q3 — The bare-collection shorthand conflict (the crux)

`AxesPairIndex` maps `[0,1]`/`vec![0,1]` to **same axes on both sides**, but NumPy's
`tensordot(a, b, [0, 1])` means **a-axis 0 vs b-axis 1**. The tblis plugin inherits
the rstsr shorthand, so it currently diverges from NumPy on that exact call.

- **(a) Reuse `AxesPairIndex` verbatim** — inherit the shorthand; accept the NumPy
  divergence as a documented rstsr convention (consistent with vecdot + plugin).
- **(b) Reuse `AxesPairIndex`, but don't accept bare collections for tensordot** —
  not enforceable through the existing `TryFrom`, so really needs (c).
- **(c) New dedicated `TensordotArgs` (house pattern, like `RollArgs`/`SortArgs`)** —
  faithful to NumPy/array-API: `()`/`None`→2, `int`, `(A,B)`; the bare-collection
  shorthand is simply not part of tensordot's surface.

➡️ **(c)**, with the normalization factored into a shared `rstsr-common` helper so
the plugin can adopt it later and the two `tensordot`s never drift. This keeps the
rstsr-specific vecdot convenience out of the array-API-facing function. (If you
value a single axes type above NumPy fidelity here, (a) is the alternative — say so.)

### Q4 — Function / method / variant surface

➡️ Mirror `vecdot` + the plugin: free `rt::tensordot(a, b, axes)` and
`rt::tensordot_f`, associated methods `a.tensordot(b, axes)` / `a.tensordot_f`;
plus `tensordot_from(c, a, b, axes)` / `_from_f` (vecdot parity). Skip
`_with_output` unless the plugin's spelling is preferred. Parameter named `axes`
(array API / NumPy), not `axes_pair`. Add the names to the `rstsr_funcs` linalg
block in `rstsr-core/src/prelude.rs`; export the `DeviceTensordotAPI` trait through
`operators/exports` only (device-implementor surface, as vecdot does).

Confirm the variant set (minimal `tensordot`+`_f`+method, or the full `_from` family).

### Q5 — Semantics, bounds, edge cases

➡️ All array-API faithful:
- fully-contracted → **0-d tensor** (never a Rust scalar); `axes=0` (outer) and
  `axes=1` share the same code path;
- **no broadcasting** of contracted axes (sizes must match) and none of the
  non-contracted axes either (they form an independent outer product);
- **no conjugation** (diverges from `vecdot` — document loudly);
- mixed dtypes via `TA: Mul<TB, Output=TC>` (same bound shape as vecdot); complex
  works without special-casing;
- errors: mismatched contracted sizes, duplicate axis, out-of-range axis, `n > ndim`,
  device mismatch; tensor layer asserts, kernel re-checks via `normalize_axes_index`.

Anything to add/trim (keepdims? an `axes=None` meaning "contract all" — array API
says default is `2`, *not* all — confirm we follow the spec, not a convenience).

### Q6 — Scope and deliverable

- Rust core only, or also expose in the `rstsr-faer-py` shim (the S2 one-liner that
  makes the conformance nodes reachable)?
- Also migrate the tblis plugin onto the shared axes helper (Q3), or leave it?

➡️ Rust core now; wire the shim too, since the acceptance bar (array API) lives
there and the `261010/faer-py-linalg-init` branch already exposes the namespace —
but that is a *separate* commit in a *separate* crate (and the shim work is owned by
the arrayapi-linalg task). Leave the plugin alone unless you want the drift closed
in this task.

### Q7 — Where the code is developed, and the acceptance bar

➡️ Develop on branch `261010/tensordot` in `rstsr-local-workspace` (a worktree of
rstsr); record the patch + porting notes in this dir; **no commits into rstsr** from
here. Acceptance: the four nodes in arrayapi-linalg §R3 go green, plus rstsr's own
new parity tests (`tests/core_func/linalg/test_tensordot.rs`), a `doc_draft` test,
and the docstring (Overloads Table + row/col-major notice per the api-doc skill).
Confirm the bar, or name a lighter one.

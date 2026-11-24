# Grill R3 — eig / eigvals (complex) on the BLAS devices (closing round)

## Answers to R2 (agent transcription; answers were given on screen 2026-11-25)

- **Q17** — amended on three points:
  1. "it should be called `EigSide`, not `EigVectors`, which I comprehend as
     input not output." → renamed.
  2. "we need to think of how to properly cast `&str` and `FlagSide` to
     `EigSide`, and introduce them in the overloading table." → Q25.
  3. "probably we simply return the three-field result, not `EigResult`, in
     BLAS devices — a lack of consideration of the previous faer
     implementation. Rename `EigResult` to `<W, VL, VR>`; let
     `EigRightResult<W, VR>` be the previous one-sided form, or even discard
     it. API breaking change on faer is allowed; make `EigResult` uniformly
     describe three possible outputs." → Q24.
- **Q20** — amended: "faer can make the refactor on the output type; no other
  change on the faer implementation." → Q26.
- **Q23** — "given the previous discussion, you just evaluate this; you can
  close if you are confident." → the owner delegated the remaining evaluation;
  Q24–Q26 below are **agent-evaluated** and close the grill unless vetoed.
- **Q18, Q19, Q21, Q22** — recommendation accepted.

## §C Fact check for Q24/Q25

- `(Tr, EigSide)` is **one argument type**: `Out` cannot depend on the enum
  *variant*, only on the type. So a runtime side selector forces one `Out`
  for every side — the `()`-for-absent encoding of R2 is impossible for the
  side-marked forms, and `QrMode` was never a precedent for arity (it does
  not change `Out`). The owner's "uniformly describe three possible outputs"
  is the correct fix.
- `FlagSide` (`rstsr-common/src/flags.rs`): `enum FlagSide { L, R }` with
  `pub use FlagSide::L as Left; pub use FlagSide::R as Right;` re-exported
  through `rstsr_structs` → `eig((&a, Left))` resolves through
  `From<FlagSide> for EigSide`.
- Blast radius of the `EigResult` refactor: `faer_impl/eig.rs`,
  `rstsr-linalg-traits/tests/test_faer_func/func_{f64,c64}.rs`
  (`.eigenvectors`), and the py shim
  `crates-interop/rstsr-faer-py/linalg/src/linalg.rs:135` (`.eigenvectors`);
  `eigh` (`EighResult.eigenvectors`) is untouched.

---

## Round 3 — evaluation (delegaged by Q23; veto-able)

❓ **Q24 - Absent-side encoding = `Option`; `EigRightResult` discarded**:
`pub struct EigResult<W, VL, VR> { pub eigenvalues: W, pub left: VL, pub
right: VR }` with generic `From<(W, VL, VR)>` / `Into<(W, VL, VR)>`; the
2-tuple conversion is removed (breaking). **Every** BLAS `eig` form — the
bare `(Tr)` included — returns `EigResult<W, Option<Tensor<c, D>>,
Option<Tensor<c, D>>>` (`left`/`right` are `Some` exactly for the
requested sides; the bare form = `Right`). One result type, one `Out`.

➡️ Accepted by delegation. Alternative rejected: `EigRightResult<W, VR>` for
the bare form + `EigResult` for the marked forms (two result types again).

---

❓ **Q25 - `EigSide` surface and casts**:
`#[non_exhaustive] #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EigSide { None, Left, #[default] Right, Both }` with
`TryFrom<&str>`/`FromStr` accepting `"none" | "left" | "right" | "both"`
(case-sensitive, `QrMode` style) and `From<FlagSide>` (`L`/`Left` →
`Left`, `R`/`Right` → `Right`). `EigSide::None` → `EigResult<W, None,
None>` (SciPy `left=False, right=False`; `eigvals` stays the idiomatic
spectrum entry).

Overloading table (`W1` = `(..., M)`, `W2` = `(..., 2, M)`, `c =
Complex<T::Real>`; every `eig` row → `EigResult<W, Option<VL>,
Option<VR>>`):

| `eig` form | generalized (`b` after `a`) | `W` |
|--|--|--|
| `(Tr)` | `(Tr, &B)` | `W1` |
| `(Tr, EigSide)` | `(Tr, &B, EigSide)` | `W1` |
| `(Tr, &str)` | `(Tr, &B, &str)` | `W1` |
| `(Tr, FlagSide)` | `(Tr, &B, FlagSide)` | `W1` |
| `(Tr, Homogeneous)` | `(Tr, &B, Homogeneous)` | `W2` |
| `(Tr, EigSide, Homogeneous)` | `(Tr, &B, EigSide, Homogeneous)` | `W2` |
| `(Tr, &str, Homogeneous)` | `(Tr, &B, &str, Homogeneous)` | `W2` |
| `(Tr, FlagSide, Homogeneous)` | `(Tr, &B, FlagSide, Homogeneous)` | `W2` |

| `eigvals` form | `Out` |
|--|--|
| `(Tr)` · `(Tr, &B)` | `W1` |
| `(Tr, Homogeneous)` · `(Tr, &B, Homogeneous)` | `W2` |

The `&str`/`FlagSide` rows are one-line delegations to the `EigSide` rows
(parse/convert, then call); each matrix slot keeps the four input kinds of
R1 Q7 (`&TensorAny`, `TensorView`, `Tensor`, `TensorMut`).

➡️ Accepted by delegation.

---

❓ **Q26 - faer output-type refactor (and nothing else)**: faer `eig(&a)`
returns `EigResult<W1, Option<VL>, Option<VR>>` with `left: None`,
`right: Some(vr)` — the same shape as the BLAS bare form, so the two devices
stay drop-in for each other; `eigvals` unchanged. Mechanical compile-keeping
edits: the two faer test files and the py shim line read `.right.unwrap()`
instead of `.eigenvectors`.

➡️ Accepted by delegation.

---

The frontier is empty. Decisions consolidated in `DECISIONS.md`.
Implementation starts only on the owner's "go" (or after a veto of
Q24–Q26).

## Answer to R3 (agent transcription; given on screen 2026-11-25)

- **Q24** — amended: "`EigResult` can cast to `(W, V)` when one and only one
  of `VL`/`VR` is not `None`, otherwise panic; also cast to `(W, VL, VR)` if
  both are not `None`. So the previous faer `Into` to `(W, V)` should still
  work." → D4 updated; the generic type-preserving `(W, VL, VR)` conversion
  is kept beside the two runtime-checked ones (coherence probed: all three
  coexist).
- **Q25, Q26** — agreed with the Q24 comment.
- **Close** — "close grill, plan and commit files in
  rstsr-improve-trajectory." → `DECISIONS.md` final, `PLAN.md` written,
  memory recorded, this directory committed.

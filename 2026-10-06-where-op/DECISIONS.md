# DECISIONS — rt::where grill (closed 2026-10-06)

User answered on screen ("all agreed", with one Q8 contingency); transcribed
by the agent per convention (answers files are user-owned; this is the
agent-authored record of screen answers). Questions: `GRILL-R1-QUESTIONS.md`;
supporting facts: `FACTS-numpy-where.md`, `FACTS-rstsr-where.md`.

| Q | Decision |
| --- | --- |
| Q1 naming | **`r#where`** (raw identifier; first in the codebase); fallible twin `where_f` (plain ident). |
| Q2 scope | **3-arg form only**; 1-arg = `nonzero`, its own future proposal (G-038). |
| Q3 faer-py | **Rust only now** (core + devices + tests + docs); faer-py wiring is a follow-up in `2026-10-04-rstsr-faer-py` (W3 searching wave). |
| Q4 cond | **Strict bool dtype** (`TensorAny<R, bool, B, D>`); NumPy int-cond sub-case translated via `a != 0` + `numpy_differences.md` row. |
| Q5 x/y dtypes | **Common-family promotion** (`TA: DTypePromoteAPI<TB>`, output `TA::Res`, cond excluded — NumPy `result_type(x, y)`), **no arithmetic bound** (Clone-level; bool/ints/floats/complex/halfs). |
| Q6 semantics | **Three-way NumPy broadcast** (chained pairwise `broadcast_layout`, associative; `DimMaxAPI` chaining), **always-fresh output**, fallible `_f` + panicking pair. |
| Q7 kernel | **First-class 4-layout family** (`dispatch_4`/`_par_4`, `op_mutc_refa_refb_refc_func` serial+rayon, bridge trait, hand-written `OpWhereAPI`); `clip` reuses it later. V1 on the generic closure kernel; vectorized blend deferred. |
| Q8 scalars | **Scalar x/y from day 1, house-strong promotion; cond tensor-only; no both-scalar.** *User contingency:* scalar trait implementations need care — **if they become too complicated, fall back to (a) tensor-only v1** (scalars as 0-d tensors; overloads later). |
| Q9 tests | Transfer `test_basic`, `test_ndim`, `test_error`, `test_dtype_mix` (tensor-cond), `test_exotic` (0-d/zero-size) + custom bool×strided×broadcast (+scalar if Q8 holds); skip-with-documentation string/foreign/kwargs/1-arg/weak-scalar parts; `numpy_coverage.csv` + `numpy_differences.md` + `doc_draft/operators/`. |
| Q10 process | This dir auto-commits; rstsr work on branch **`261006/rt-where`**, **no rstsr commits without explicit user instruction**; implementation starts on user go. |

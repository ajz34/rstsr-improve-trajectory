---
name: rstsr-blas-eig
description: "eig/eigvals (complex) on the BLAS devices (2026-11-25, implemented): GEEV/GGEV builders, EigResult<W, VL, VR> with Option sides, runtime EigSide, Homogeneous marker, shared n-dim walk ref_impl_eig_nd_f; GGEV beta is COMPLEX for C/ZGGEV (plan was wrong); in-place per input; OpenBLAS/MKL/AOCL green, uncommitted on 261125/blas-eig"
metadata:
  type: project
---

2026-11-25. Grill (D1–D23) in `2026-11-24-blas-eig/DECISIONS.md`, plan B0–B5
all done, outcome and reviewer notes in `README.md`. Code on rstsr branch
`261125/blas-eig` (off `b01b6e0`), **uncommitted** (owner rule), PR not opened;
changelog/version untouched by instruction.

- Layers: `rstsr-blas-traits/src/lapack_eig/{geev,ggev}.rs` (builders +
  `GEEVDriverAPI<T>`/`GGEVDriverAPI<T>`), real→complex assembly
  `util/util_eig.rs`, thunks `driver_impl/{lapack,lapacke}/eig/`;
  `rstsr-linalg-traits/src/ref_impl_blas.rs` (`ref_impl_eig_nd_f` shared by
  `eig` and `eigvals`), `blas_impl/{eig,eigvals}.rs` symlinked into the five
  device crates (16 + 4 overload forms). faer `eig` refactored onto the same
  `EigResult` (breaking: output type only), `rstsr-faer-py` shim adapted.
- **GGEV `beta` is complex for C/ZGGEV** (`beta: *mut T` end to end). The plan
  assumed real β; the first draft allocated `n` reals → heap overflow. The
  simple impl returns `(α, β)` complex; linalg gives `w = α/β` with SciPy's
  inf/nan rules (β=0, α≠0 → `inf+0j`; α=β=0 → `nan+0j`, or `nan+nan j` when
  some α has an imaginary part).
- In-place rule (sharpened by the nd tests): each input is destroyed
  independently when its batch equals the output batch; a broadcast input's
  slices are copied out of a broadcast view. First draft needed both in place.
- `eig((a, Homogeneous))` keeps the default right side; `EigSide::None` gives
  `None/None`; homogeneous spectrum `(..., 2, M)` = `(α, β)`, β ≡ 1 for the
  standard problem; generalized vectors rescaled to unit 2-norm (SciPy).
- Verified: OpenBLAS driver 80 + linalg 74 (full), conda OpenBLAS `lapacke`
  eig subset 32, MKL 80 + eig subset 28 (lapacke subset 32), AOCL 80 + 28;
  faer suite 70, facade cheatsheet 11, clippy clean. BLIS/KML only `cargo
  check` (no local library). MKL needs `compiler/<ver>/lib` on
  `LD_LIBRARY_PATH` (libiomp5) for threaded layers.
- Fingerprint/test lessons moved to `rstsr-eig-fingerprint-robustness.md`.

# Grill R4 — BLAS-2 GEMV dispatch for matmul (`openblas` / `faer`)

Session 2026-11-23 (round 4). rstsr `d5c9b86`, clean; this repo clean. No rstsr
file edited. Recommended answers marked ➡️. This round should close the design
tree; Q3 is the go/no-go for running P0.

---

## R3 answers — agent-authored transcript (chat; no answers file exists)

- **Q1 = (a)** `gemv_blas_no_conj_*` in each `matmul_impl.rs`, calling
  `B::driver_gemv`.
- **Q2 = (b)** copy the vector rather than guard-and-fall-back.
- **Q3** *"But well, if stride positive, would you also choose to make copy?
  Anyway, I now think that make copy to vector is generally acceptable."*
- **Q4 = okay**; **Q5 = `dim_eliminate` first**; **Q6 = okay**;
  **Q7 = okay**, with *"Note this only for local test, not make test results into
  rstsr main repo code."* (restates D2).

### Settled

- D20 Device-side gemv wrapper: new `gemv_blas_no_conj_*` in each device
  `matmul_impl.rs`, routed through `B::driver_gemv` (five copies; cblas lives
  once in the template).
- D21 Vectors are copied rather than forwarded with increments (details in Q1).
- D22 `GEMV_` contract as Q4 of R3 (order-free, `flip_trans(ColMajor, …, false)`,
  `testify_ld`, `k == 0` fill, device/shape checks).
- D23 Size-1 axis removal uses `Layout::dim_eliminate`.
- D24 P0 cell table as Q6 of R3 (contiguous `256 × 452676` and its transpose,
  no `.t()`-view cell); criterion protocol as stated.
- D25 P0 runs on the system OpenBLAS 0.3.32 as its baseline; any results stay in
  this repo (D2).

---

## Q1 — Final vector policy

Your Q3 note makes "copy the vector" acceptable in general, which removes the
increment logic entirely:

- (a) unit-stride: direct; positive stride: pass `inc = stride`; stride ≤ 0:
  copy — the R2/Q4 rule, now with the device path also copying (D21);
- (b) **unit-stride: direct; anything else: copy to a contiguous buffer** (gather
  in, scatter out for `y`) — no increment arithmetic and no negative-inc pointer
  convention anywhere; `incx = incy = 1` at the driver;
- (c) always copy, even unit-stride.

➡️ (b). The copy is O(m) or O(n) against O(m·k) gemv work, F7 shows increment
forwarding is where the bugs live, and it makes the device path and the
container share one rule. Unit-stride vectors still avoid the copy, which is the
common case for `%` on contiguous operands.

---

## Q2 — Do `incx`/`incy` stay in `driver_gemv`

- (a) keep them (`usize`), both callers pass `1` in this campaign;
- (b) drop them, hard-coding unit stride, and add them back if a future caller
  needs increments.

➡️ (a). The trait models `cblas_?gemv`; the parameters cost nothing at the call
site and keep the abstraction honest if a later linalg path passes a strided
1-D view. (If you prefer a minimal surface, (b) is the alternative — say so.)

---

## Q3 — Go/no-go for executing P0

P0 = the standalone criterion crate in
`2026-11-18-gemv-dispatch/` (`Cargo.toml`, `src/lib.rs` devices + thread
assertions, `benches/matmul_vec.rs`, `examples/correctness.rs` gate,
`numpy_ref/`, `results/`, `reproduce.sh`), the cell table of R3/Q6, raw-cblas and
numpy anchors, and a README with the baseline table + environment record.
**No rstsr file is touched**; P1/P2 wait for your review of the numbers.

➡️ Yes. If you also want the plan written up first (`PLAN.md` with the P1/P2
file-by-file delta and the two-patch split), I can produce that before running
P0 — say which order you prefer.

---

## Remaining tree

After Q1–Q3 the design tree is empty: phasing (D1), measurement (D2–D6, D24–D25),
API surface (D8, D16–D17, D22), dispatch mechanics (D20–D23), propagation (D7,
D10), threading (D11, D18), verification and packaging (D14, D19, Q9/Q10 of R2)
are all settled. The next artefacts are P0's numbers and then the two patches.

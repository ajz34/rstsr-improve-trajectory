
## Column-major placement rule (2026-10-09, `5a42771`)

Decision (maintainer): the array-indexing placement rule is measured in the
device's **access order** — "access-contiguity-first axes layout". A run of
advanced indexers that other indexers *displace* is placed at the **back** of the
result under a column-major device instead of the front, so the broadcast block
keeps its contiguity role (the most-strided axis) in both orders; a run that
stays together keeps its subscript position. Row-major is byte-for-byte
unchanged; 1-D results are identical; the *shape* therefore differs between the
orders exactly when the advanced indexers are apart, and the arrangement
divergence stays for every rank >= 2 result.

The two candidate rules were separated by the maintainer's K2 case
(`a[:, [0,1], [2,0]]` -> `(3,2)` in both orders): "mirror the insertion count"
would have moved it to `(2,3)`, so the rule flips displaced runs only.
`mask_select` resolves to no change — its count axis replaces the leading axes
(the leading-together configuration). Design record:
`PLAN-col-major-placement.md`; response record: `REVIEW-R1-RESPONSE.md`.

Re-verified: 547 row-major suite tests, 253 doctests, lib 146 (default) / 145
(col-major), fmt/clippy/rustdoc clean.

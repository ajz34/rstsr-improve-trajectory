# Follow-up: parallel rayon kernel for `array_index` (the gather)

Status (2026-10-09): **not started.** `DeviceRayonAutoImpl::array_index`
(`rstsr-core/src/feature_rayon/auto_impl/array_indexing.rs`, also reached by
`DeviceFaer` and the five BLAS crates through the `rayon_auto_impl` symlink)
delegates straight to `array_index_cpu_serial`
(`rstsr-native-impl/src/cpu_serial/array_indexing.rs`). The delegation comment
already names the "perf pass" this note specifies. `mask_select` / `mask_fill`
are in the same state (serial) and share the problem shape.

## The constraint that makes it easy

The trait obligation `DeviceArrayIndexAPI::array_index` documents — **write
every output element exactly once** — means distinct `(base_flat, bulk_flat)`
pairs map to disjoint output offsets. So a parallel version needs no locking and
no reduction; it can use the house scattered-write pattern: hoist
`c.as_mut_ptr()` through an `AtomicPtr`, `into_par_iter()` the work, and
`unsafe { ptr.add(off).write(...) }`, wrapped in `pool.install(task)`.

Model files (copy their shape):
- `rstsr-native-impl/src/cpu_rayon/searching.rs` — `searchsorted_cpu_rayon`
  (pool check, size switch, `AtomicPtr` hoist, `pool.install`).
- `rstsr-native-impl/src/cpu_rayon/adv_indexing.rs` — `index_select_cpu_rayon`
  (contiguous-run tasks; `translate_to_col_major` + `TensorIterOrder::K` for
  cache-friendly traversal).

## Recommended design: flatten the (base × bulk) loop

Serial kernel today: per-bulk tables `src_bulk[n_bulk]` / `out_bulk[n_bulk]`,
then base (outer) × bulk (inner): `c[out_base + out_bulk[b]] =
a[src_base + src_bulk[b]].clone()`.

Parallel version:
1. Precompute **four** tables: the existing `src_bulk[n_bulk]`,
   `out_bulk[n_bulk]`, **plus** `src_base[n_base]`, `out_base[n_base]`.
   `O(n_base + n_bulk)` memory, not `O(n_out)`.
2. `(0..n_base * n_bulk).into_par_iter().for_each(|k| { let (b, q) = …;
   unsafe { c_ptr.add((out_base[b] + out_bulk[q]) as usize)
   .write(a[(src_base[b] + src_bulk[q]) as usize].clone()) } })`.
3. Layout-generic (offsets come from the tables, so any strides work — do not
   assume `lc` is contiguous, even though the tensor tier currently builds it
   with `new_contig(None, order)`).
4. Pick the nesting (`base`-major vs `bulk`-major) so the inner coordinate steps
   through the output in its own order — the analog of `index_select`'s
   col-major retranslation.

**Why flatten, not split one axis:** `n_base == 1` is common and important — it
is the pure zipped gather `x[i1, …, iN]` (every axis consumed), the array-API
reduced form. A base-only split has zero parallelism there; a bulk-only split
dies when the broadcast count is small. Flattening covers both and keeps the
per-element cost equal to today's serial inner loop (two table loads + two adds).

## Wiring and guards

- `DeviceRayonAutoImpl::array_index`: `let pool = self.get_current_pool();` →
  `array_index_cpu_rayon(..., pool)`; the serial device keeps its own call
  (exactly how `index_select` is wired in
  `feature_rayon/auto_impl/adv_indexing.rs`).
- Fall back to `array_index_cpu_serial` when `pool.is_none()` **or**
  `n_out * size_of::<T>()` is below a switch (~`searchsorted`'s 8192-element
  ballpark), so small gathers don't pay rayon overhead.
- Keep the `n_bulk == 0 || n_base == 0` early return.
- Add `pub mod array_indexing;` to `rstsr-native-impl/src/cpu_rayon/mod.rs`.
  No per-device symlink needed — the kernel lives in `rstsr-native-impl`, not in
  a `rayon_auto_impl` tree (the BLAS crates' `rayon_auto_impl/mod.rs` symlinks
  are only for modules under that tree).

## Alternatives, and when they win

- **`par_chunks_mut` over `c` with an odometer** — writes sequentially (the
  current inner loop writes strided along the bulk dims), no `unsafe`. Cost:
  assumes `c` is dense in `order` (true today, not part of the layout-generic
  contract) and needs an odometer mixing constant-stride base steps with
  table-indexed bulk steps. Only if profiling shows scattered writes dominate.
- **Serial `src_off[n_out]` table, then `c.par_iter_mut().zip(...)`** — trivial
  parallel copy, but `8·N` scratch and the serial index pass remains; moderate
  `N` only.

## Honest caveat

A gather is memory-bound (one load + one store + a `Clone` per element), so the
speedup is sublinear and bandwidth-limited. Land it with a small benchmark
before adding the complexity. The same table-based split later unblocks
`mask_select` / `mask_fill`, which additionally need a prefix-sum for the write
offsets.

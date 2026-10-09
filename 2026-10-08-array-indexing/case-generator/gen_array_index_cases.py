#!/usr/bin/env python3
"""Generate randomized array-indexing cases with NumPy reference results.

Task-record scratch tool (rstsr-improve-trajectory): rstsr-core ships no Python
regeneration step, so the generated rows are inlined as a Rust table in
`rstsr-core/tests/core_func/indexing/test_array_index.rs` (`NUMPY_CASES`); rerun
this script and refresh that table when the NumPy pin moves.

One case per line:

    shape ; index-spec ; out-shape ; out-values

with tokens of the index spec:

    I<n>          integer index (may be negative)
    S<a>:<b>:<c>  slice (a/b/c may be empty)
    A<d0>x..:<v>  index array, shape `d0 x ..` and C-order values `v`
    N             new axis (None)
    E             ellipsis

Usage: python3 gen_array_index_cases.py [n_cases] [seed] > array_index_numpy_cases.txt
"""

import sys

import numpy as np


def rand_shape(rng, ndim_max=4, dim_max=5):
    ndim = int(rng.integers(1, ndim_max + 1))
    return tuple(int(rng.integers(1, dim_max + 1)) for _ in range(ndim))


def rand_axis_indexer(rng, dim, allow_array=True):
    """One indexer for an axis of length `dim`; returns (token, consumes)."""
    choice = rng.integers(0, 3 if allow_array else 2)
    if choice == 0:  # integer
        value = int(rng.integers(-dim, dim))
        return f"I{value}", 1
    if choice == 1:  # slice
        step = int(rng.choice([1, 1, 1, 2, -1, -2]))
        if step < 0:
            start = int(rng.integers(-dim, dim))
            stop = int(rng.integers(-dim - 1, dim + 1))
        else:
            start = int(rng.integers(0, dim + 1))
            stop = int(rng.integers(0, dim + 1))
        return f"S{start}:{stop}:{step}", 1
    # index array, with a random (broadcastable-ish) shape
    ndim_idx = int(rng.integers(1, 3))
    shape_idx = tuple(int(rng.integers(1, 4)) for _ in range(ndim_idx))
    size = int(np.prod(shape_idx))
    values = rng.integers(0, dim, size=size)
    if rng.random() < 0.3:
        values = values - dim  # exercise negative indices
    shape_txt = "x".join(str(d) for d in shape_idx)
    return f"A{shape_txt}:" + ",".join(str(int(v)) for v in values), 1


def build_case(rng):
    shape = rand_shape(rng)
    a = np.arange(int(np.prod(shape))).reshape(shape)
    tokens = []
    axes_left = len(shape)
    # optionally let an ellipsis cover a trailing run of axes
    use_ellipsis = rng.random() < 0.3 and len(shape) > 1
    # a cover of 0 is a *zero-width* ellipsis (every axis consumed explicitly),
    # which still separates the advanced indexers around it
    ellipsis_cover = int(rng.integers(0, len(shape))) if use_ellipsis else 0
    n_explicit = len(shape) - ellipsis_cover
    for axis in range(n_explicit):
        token, _ = rand_axis_indexer(rng, shape[axis])
        tokens.append(token)
        if rng.random() < 0.15:
            tokens.append("N")
    # note: guarded by `use_ellipsis`, not `ellipsis_cover` — a cover of 0 is
    # still an ellipsis (a zero-width one, which separates advanced indexers)
    if use_ellipsis:
        pos = int(rng.integers(0, len(tokens) + 1))
        tokens.insert(pos, "E")
    return a, shape, tokens


def parse_token(token, axis, dim):
    """Turn a token into the Python index object (relative to `axis`)."""
    kind = token[0]
    if kind == "I":
        return int(token[1:]), 1
    if kind == "S":
        parts = token[1:].split(":")
        start = int(parts[0]) if parts[0] else None
        stop = int(parts[1]) if parts[1] else None
        step = int(parts[2]) if len(parts) > 2 and parts[2] else None
        return slice(start, stop, step), 1
    if kind == "A":
        shape_txt, values_txt = token[1:].split(":")
        shape_idx = tuple(int(d) for d in shape_txt.split("x"))
        values = np.array([int(v) for v in values_txt.split(",")], dtype=np.intp)
        return values.reshape(shape_idx), 1
    if kind == "N":
        return None, 0
    if kind == "E":
        return Ellipsis, None
    raise ValueError(token)


def main():
    n_cases = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 20261008
    rng = np.random.default_rng(seed)
    out_lines = []
    generated = 0
    while generated < n_cases:
        a, shape, tokens = build_case(rng)
        # translate tokens to a python index tuple
        idx = []
        axis = 0
        for token in tokens:
            item, consumes = parse_token(token, axis, shape[axis] if axis < len(shape) else 1)
            idx.append(item)
            if consumes:
                axis += consumes
        try:
            result = a[tuple(idx)]
        except Exception:
            # out-of-bound / non-broadcastable cases are errors, not cases
            continue
        if result.ndim > 6 or result.size > 4096:
            continue
        out_lines.append(
            ";".join(
                [
                    ",".join(str(d) for d in shape),
                    "|".join(tokens),
                    ",".join(str(d) for d in result.shape),
                    ",".join(str(int(v)) for v in result.reshape(-1)),
                ]
            )
        )
        generated += 1
    print("\n".join(out_lines))


if __name__ == "__main__":
    main()

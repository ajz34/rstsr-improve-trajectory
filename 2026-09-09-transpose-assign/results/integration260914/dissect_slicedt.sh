#!/usr/bin/env bash
# Targeted dissection of the one borderline canary cell (integration260914):
# assign_gates/sliced_t_large_2048x2048-faer16-f64/assign, native config.
# 3 alternating refA/cand single-bench runs (criterion substring filter).
set -uo pipefail
TR=/home/a/rstsr_pack/rstsr-improve-trajectory/2026-09-09-transpose-assign
RSTSR=/home/a/rstsr_pack/rstsr
PATCH="$TR/proposed.patch"
OUT="$TR/results/integration260914"
export RAYON_NUM_THREADS=16
export RUSTFLAGS="-C target-cpu=native"
is_cand () { git -C "$RSTSR" apply --check -R -q "$PATCH" 2>/dev/null; }
for i in 1 2 3; do
    for tag in refA cand; do
        L="$OUT/dissect_${tag}_run${i}.txt"
        [ -f "$L" ] && continue
        if [ "$tag" = refA ]; then
            is_cand && git -C "$RSTSR" apply -R -q "$PATCH"
        else
            is_cand || git -C "$RSTSR" apply -q "$PATCH"
        fi
        (cd "$TR" && cargo bench --bench transpose -- "sliced_t_large_2048x2048-faer16") > "$L" 2>&1
    done
done
is_cand || git -C "$RSTSR" apply -q "$PATCH"
git -C "$RSTSR" diff --quiet && echo "ERROR: tree ended clean" || echo "DISSECTION DONE (patch applied)"

#!/usr/bin/env bash
# Integration re-verification benches for the elementwise patch (patch 2) on
# merged master (e835173), 2026-09-14.
#
# Protocol (matches the patch-1 integration cycle): same-session paired
# criterion passes, refA = clean origin/master (patch stashed) vs cand =
# patched tree, alternating within each pass, 3 passes, portable + native.
# Idempotent: a stage is skipped when its log already carries the final
# benchmark id (suite completion marker).
#
# The ../rstsr tree is flipped via git stash (patch is NOT committed there);
# the script must end with the patch applied (verified by the final check).
set -uo pipefail
EW=/home/a/rstsr_pack/rstsr-improve-trajectory/2026-09-09-elementwise
RSTSR=/home/a/rstsr_pack/rstsr
OUT="$EW/results/integration260914"
mkdir -p "$OUT"
export RAYON_NUM_THREADS=16
MARKER='scale_contig_2048x2048-faer16_f64/B'

done_stage () { [ -f "$1" ] && grep -q "$MARKER" "$1"; }

run_bench () { # $1 = log path
    if [ "${2:-native}" = native ]; then export RUSTFLAGS="-C target-cpu=native"; else unset RUSTFLAGS; fi
    (cd "$EW" && cargo bench --bench elementwise) > "$1" 2>&1
    grep -q "$MARKER" "$1" || { echo "STAGE INCOMPLETE: $1"; return 1; }
}

for pass in 1 2 3; do
    for cfg in portable native; do
        L="refA_${cfg}_pass${pass}.log"
        if ! done_stage "$OUT/$L"; then
            git -C "$RSTSR" stash push -q -m "elementwise-patch-during-bench"
            echo "=== $L (refA = origin/master) ==="
            run_bench "$OUT/$L" "$cfg" || exit 1
        fi
        L="cand_${cfg}_pass${pass}.log"
        if ! done_stage "$OUT/$L"; then
            git -C "$RSTSR" stash pop -q
            echo "=== $L (cand = patched) ==="
            run_bench "$OUT/$L" "$cfg" || exit 1
        fi
    done
done

# invariant: patched tree restored
if git -C "$RSTSR" diff --quiet; then
    echo "ERROR: rstsr tree ended CLEAN - patch was lost"
    exit 1
fi
echo "ALL PASSES DONE (patch still applied)"

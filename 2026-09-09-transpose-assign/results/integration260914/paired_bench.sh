#!/usr/bin/env bash
# Integration re-verification benches for the transpose-assign patch (patch 3)
# on merged master (c08e44a, post PR #101 elementwise), 2026-09-14.
#
# Protocol (matches the patch-1/2 integration cycles): same-session paired
# criterion passes, refA = clean origin/master (patch stashed) vs cand =
# patched tree, alternating within each pass, 3 passes, portable + native.
# Full `--bench transpose` suite (56 benchmarks) per stage;
# RAYON_NUM_THREADS=16; no other load on the machine.
#
# The ../rstsr tree is flipped by apply / reverse-apply of the patch file
# (patch is NOT committed there; the git stash approach was rejected because
# the repo carries a pre-existing user stash from an unrelated branch). The
# script must end with the patch applied (verified by the final check).
#
# RUSTFLAGS caching guard (T8 lesson): the md5 of the newest bench binary is
# logged after every stage (binary_md5.txt) — identical md5 across two stages
# with different RUSTFLAGS would mean a stale binary was benched.
set -uo pipefail
TR=/home/a/rstsr_pack/rstsr-improve-trajectory/2026-09-09-transpose-assign
RSTSR=/home/a/rstsr_pack/rstsr
PATCH="$TR/proposed.patch"
OUT="$TR/results/integration260914"
mkdir -p "$OUT"
export RAYON_NUM_THREADS=16
MARKER='orderchange_extra/bcast_t_odd_1000x777-faer16-f64/B'

# "currently patched" == the patch reverse-applies cleanly
is_cand () { git -C "$RSTSR" apply --check -R -q "$PATCH" 2>/dev/null; }
flip_refA () { if is_cand; then git -C "$RSTSR" apply -R -q "$PATCH"; fi; git -C "$RSTSR" diff --quiet; }
flip_cand () { if ! is_cand; then git -C "$RSTSR" apply -q "$PATCH"; fi; ! git -C "$RSTSR" diff --quiet; }

# pre-flight: patch must be applied (dirty tree)
if git -C "$RSTSR" diff --quiet; then
    echo "ERROR: rstsr tree is CLEAN before start - patch not applied?"; exit 1
fi

done_stage () { [ -f "$1" ] && grep -q "$MARKER" "$1"; }

log_bin_md5 () { # $1 = tag
    local bin
    bin=$(ls -t "$TR"/target/release/deps/transpose-* 2>/dev/null | grep -vE '\.(d|pdb)$' | head -1)
    if [ -n "$bin" ]; then
        echo "$1 RUSTFLAGS='${RUSTFLAGS:-<unset>}' md5=$(md5sum "$bin" | cut -d' ' -f1)" >> "$OUT/binary_md5.txt"
    fi
}

run_bench () { # $1 = log path, $2 = config
    if [ "${2:-native}" = native ]; then export RUSTFLAGS="-C target-cpu=native"; else unset RUSTFLAGS; fi
    (cd "$TR" && cargo bench --bench transpose) > "$1" 2>&1
    log_bin_md5 "$(basename "$1")"
    grep -q "$MARKER" "$1" || { echo "STAGE INCOMPLETE: $1"; return 1; }
}

for pass in 1 2 3; do
    for cfg in portable native; do
        L="refA_${cfg}_pass${pass}.log"
        if ! done_stage "$OUT/$L"; then
            flip_refA || { echo "ERROR: flip to refA failed"; exit 1; }
            echo "=== $L (refA = origin/master) ==="
            run_bench "$OUT/$L" "$cfg" || { flip_cand; exit 1; }
        fi
        L="cand_${cfg}_pass${pass}.log"
        if ! done_stage "$OUT/$L"; then
            flip_cand || { echo "ERROR: flip to cand failed"; exit 1; }
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

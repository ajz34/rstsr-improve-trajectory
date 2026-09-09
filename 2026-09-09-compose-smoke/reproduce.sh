#!/usr/bin/env bash
# T8 compose-smoke: exact reproduction sequence.
# Run from /home/a/rstsr_pack/rstsr-improve-trajectory/2026-09-09-compose-smoke
# Requirements: rstsr checkout at 386948be with a CLEAN worktree,
# RAYON_NUM_THREADS=16. Toolchain: resolved per cwd (rstsr's rust-toolchain.toml
# pins nightly for ITS suites; this crate builds with the rustup default —
# stable 1.97.1 at gate time; see results/toolchain.txt).
set -euo pipefail

cd /home/a/rstsr_pack/rstsr
echo "== 0. verify clean at 386948be"
test -z "$(git status --porcelain)"
test "$(git rev-parse HEAD)" = 386948be819baa334b8da02232f3a1944e5447d5

SMOKE=/home/a/rstsr_pack/rstsr-improve-trajectory/2026-09-09-compose-smoke
CAMPAIGN=/home/a/rstsr_pack/rstsr-improve-trajectory

echo "== 1. apply five patches in campaign order (patch 3 = AMENDED, guard included)"
git apply "$CAMPAIGN"/2026-09-09-argmax-argmin/proposed.patch
git apply "$CAMPAIGN"/2026-09-09-elementwise/proposed.patch
git apply "$CAMPAIGN"/2026-09-09-transpose-assign/proposed.patch
git apply "$CAMPAIGN"/2026-09-09-reductions/proposed.patch
git apply "$CAMPAIGN"/2026-09-09-vecdot/proposed.patch
git diff --stat
# NOTE: no separate guard fix needed anymore — the shape-identity guard is
# inside the amended patch 3 (compose_guard_fix.SUPERSEDED.patch is the
# round-1 discovery record only; do not apply it).

echo "== 2. rstsr test suites (portable)"
cargo test -p rstsr-core --lib
cargo test -p rstsr-core --test entry_row_cpu --no-default-features --features "backtrace row_major"
cargo test -p rstsr-native-impl   # has zero test targets; kept for parity

echo "== 2b. rstsr test suites (native)"
RUSTFLAGS="-C target-cpu=native" cargo test -p rstsr-core --lib
RUSTFLAGS="-C target-cpu=native" cargo test -p rstsr-core --test entry_row_cpu --no-default-features --features "backtrace row_major"
RUSTFLAGS="-C target-cpu=native" cargo test -p rstsr-native-impl

echo "== 3. union correctness gate + spots (portable build)"
cd "$SMOKE"
cargo build --release --example correctness --example spot
RAYON_NUM_THREADS=16 ./target/release/examples/correctness
RAYON_NUM_THREADS=16 ./target/release/examples/spot

echo "== 3b. gate + spots (native build)"
touch src/lib.rs   # force rebuild: RUSTFLAGS change must take effect
RUSTFLAGS="-C target-cpu=native" cargo build --release --example correctness --example spot
RAYON_NUM_THREADS=16 ./target/release/examples/correctness
RAYON_NUM_THREADS=16 ./target/release/examples/spot

echo "== 3c. optional criterion cross-check of strided-B on the combined tree"
cd "$CAMPAIGN"/2026-09-09-elementwise
RUSTFLAGS="-C target-cpu=native" cargo bench --bench elementwise -- 'add/add_strided_2048x2048-serial_f64/B'
cd "$SMOKE"

echo "== 4. restore rstsr to clean 386948be"
cd /home/a/rstsr_pack/rstsr
git checkout -- .
test -z "$(git status --porcelain --untracked-files=all)"
test "$(git rev-parse HEAD)" = 386948be819baa334b8da02232f3a1944e5447d5
echo "OK: rstsr restored clean at 386948be"

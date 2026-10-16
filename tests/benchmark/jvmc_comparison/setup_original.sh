#!/bin/bash
# Extracts the original jVMC (the package jVMC of github.com/markusschmitt/vmc_jax) at the
# last upstream commit contained in this branch (the commit this fork started from, as long
# as upstream has not been merged in since), into a folder that the scripts take as --jvmc-path:
#
#     bash setup_original.sh [target folder, default: $SCRATCH/env/jVMC_original]
#
# Run it on a login node: it needs git and the remote "upstream" of this repository.
# The original jVMC also imports mpi4py (see send_job.sh).
set -euo pipefail

TARGET="${1:-$SCRATCH/env/jVMC_original}"
REPO="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

git -C "$REPO" fetch upstream
COMMIT="$(git -C "$REPO" merge-base HEAD upstream/master)"

mkdir -p "$TARGET"
rm -rf "$TARGET/jVMC"
git -C "$REPO" archive "$COMMIT" jVMC | tar -x -C "$TARGET"
# Read by the scripts, which save it with the results
echo "$COMMIT $(git -C "$REPO" log -1 --format=%cs "$COMMIT")" > "$TARGET/ORIGINAL_COMMIT"

echo "jVMC $(sed -n 's/__version__ = "\(.*\)"/\1/p' "$TARGET/jVMC/version.py") (commit $COMMIT) extracted to $TARGET"

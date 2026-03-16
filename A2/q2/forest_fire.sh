#!/usr/bin/env bash
set -e

GRAPH_PATH="$1"
SEED_PATH="$2"
OUT_PATH="$3"
K="$4"
R="$5"
HOPS="$6"

python3 "$(dirname "$0")/fireblock.py" \
  --graph "$GRAPH_PATH" \
  --seeds "$SEED_PATH" \
  --out "$OUT_PATH" \
  --k "$K" \
  --r "$R" \
  --hops "$HOPS"

#!/bin/bash
# ============================================================
# Q2: Frequent Subgraph Mining
#
# Usage:
# bash new_q2.sh <gspan> <fsg> <gaston> <raw_dataset> <outdir>
# ============================================================

set -e

GSPAN="$1"
FSG="$2"
GASTON="$3"
RAW_DATASET="$4"
OUTDIR="$5"

SUPPORTS=(90 50 25 10 5)

BASE_DIR=$(pwd)
mkdir -p "$OUTDIR"

echo "Running Q2: Frequent Subgraph Mining"
echo "-----------------------------------"

# ============================================================
# Step 1: Convert RAW → gSpan / FSG / Gaston
# ============================================================

DATASET_GSPAN="$OUTDIR/yeast.gspan"
DATASET_FSG="$OUTDIR/yeast.fsg"
DATASET_GASTON="$OUTDIR/yeast.gaston"

python3 "$BASE_DIR/convert2gspan.py"  "$RAW_DATASET" "$DATASET_GSPAN"
python3 "$BASE_DIR/convert2fsg.py"    "$RAW_DATASET" "$DATASET_FSG"
python3 "$BASE_DIR/convert2gaston.py" "$RAW_DATASET" "$DATASET_GASTON"

# ============================================================
# Count graphs (for Gaston absolute support)
# ============================================================

TOTAL_GRAPHS=$(grep -c "^t" "$DATASET_GSPAN")
echo "Total graphs detected: $TOTAL_GRAPHS"

cd "$OUTDIR"

# ============================================================
# PHASE 1: GASTON (ABSOLUTE SUPPORT)
# ============================================================
echo "========== PHASE 1: GASTON =========="

for s in "${SUPPORTS[@]}"; do
    DIR="gaston$s"
    mkdir -p "$DIR"

    ABS_SUP=$(( (TOTAL_GRAPHS * s + 99) / 100 ))

    START=$(date +%s.%N)
    "$GASTON" "$ABS_SUP" "$DATASET_GASTON" "$DIR/output.gaston" \
        > "$DIR/log.txt" 2>&1
    END=$(date +%s.%N)

    echo "$(echo "$END - $START" | bc)" > "$DIR/time.txt"
done

# ============================================================
# PHASE 2: FSG (PERCENT SUPPORT)
# ============================================================
echo "========== PHASE 2: FSG =========="

for s in "${SUPPORTS[@]}"; do
    DIR="fsg$s"
    mkdir -p "$DIR"

    START=$(date +%s.%N)
    "$FSG" -s "$s" -pt "$DATASET_FSG" \
        > "$DIR/log.txt" 2>&1
    END=$(date +%s.%N)

    echo "$(echo "$END - $START" | bc)" > "$DIR/time.txt"
done

# ============================================================
# PHASE 3: gSpan (RELATIVE SUPPORT)
# ============================================================
echo "========== PHASE 3: gSpan =========="

for s in "${SUPPORTS[@]}"; do
    DIR="gspan$s"
    mkdir -p "$DIR"

    REL_SUP=$(echo "$s / 100" | bc -l)

    START=$(date +%s.%N)
    "$GSPAN" -f "$DATASET_GSPAN" -s "$REL_SUP" -o -i \
        > "$DIR/log.txt" 2>&1
    END=$(date +%s.%N)

    mv "$DATASET_GSPAN.fp" "$DIR/output.fp"
    echo "$(echo "$END - $START" | bc)" > "$DIR/time.txt"
done

echo "All phases completed successfully."

python3 "$BASE_DIR/plot_runtime.py"



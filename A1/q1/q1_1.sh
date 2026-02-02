#!/bin/bash

APRIORI_EXEC=$1
FPGROWTH_EXEC=$2
DATASET=$3
OUTDIR=$4

mkdir -p $OUTDIR

SUPPORTS=(5 10 25 50 90)

for S in "${SUPPORTS[@]}"; do
    echo "Running Apriori at ${S}% support"
    /usr/bin/time -v $APRIORI_EXEC -s $S $DATASET > $OUTDIR/ap${S} 2>&1

    echo "Running FP-Growth at ${S}% support"
    /usr/bin/time -v $FPGROWTH_EXEC -s $S $DATASET > $OUTDIR/fp${S} 2>&1
done

python3 - <<EOF
import matplotlib.pyplot as plt

supports = [5,10,25,50,90]
ap_times = []
fp_times = []

for s in supports:
    with open(f"$OUTDIR/ap{s}") as f:
        for line in f:
            if "Elapsed (wall clock) time" in line:
                t = line.strip().split()[-1]
                if ":" in t:
                    m, sec = t.split(":")
                    ap_times.append(float(m)*60 + float(sec))
                else:
                    ap_times.append(float(t))

    with open(f"$OUTDIR/fp{s}") as f:
        for line in f:
            if "Elapsed (wall clock) time" in line:
                t = line.strip().split()[-1]
                if ":" in t:
                    m, sec = t.split(":")
                    fp_times.append(float(m)*60 + float(sec))
                else:
                    fp_times.append(float(t))

plt.plot(supports, ap_times, marker='o', label='Apriori')
plt.plot(supports, fp_times, marker='o', label='FP-Growth')
plt.xlabel("Support Threshold (%)")
plt.ylabel("Runtime (seconds)")
plt.legend()
plt.savefig(f"$OUTDIR/plot.png")
EOF

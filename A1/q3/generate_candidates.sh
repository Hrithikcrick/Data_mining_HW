#!/bin/bash
# Usage: bash generate_candidates.sh <db_features> <query_features> <out_file>

DBF="$1"
QF="$2"
OUT="$3"

python3 - "$DBF" "$QF" "$OUT" << 'EOF'
import sys

db_file = sys.argv[1]
q_file  = sys.argv[2]
out_file = sys.argv[3]

# Load binary feature vectors
db = [list(map(int, l.split())) for l in open(db_file)]
queries = [list(map(int, l.split())) for l in open(q_file)]

num_db = len(db)

with open(out_file, "w") as f:
    for qi, q in enumerate(queries, start=1):

        # 🔴 FALLBACK: if query has no discriminative subgraphs
        if sum(q) == 0:
            candidates = list(range(1, num_db + 1))
            f.write(f"q # {qi}\n")
            f.write("c # " + " ".join(map(str, candidates)) + "\n")
            continue

        candidates = []
        for di, d in enumerate(db, start=1):
            ok = True
            for k in range(len(q)):
                if q[k] == 1 and d[k] == 0:
                    ok = False
                    break
            if ok:
                candidates.append(di)

        f.write(f"q # {qi}\n")
        f.write("c # " + " ".join(map(str, candidates)) + "\n")
EOF



#!/bin/bash
# Usage: bash convert.sh <path_graphs> <path_discriminative_subgraphs> <path_features>

GRAPHS="$1"
SUBS="$2"
OUT="$3"

python3 - "$GRAPHS" "$SUBS" "$OUT" << 'EOF'
import sys

graphs = sys.argv[1]
subs_file = sys.argv[2]
out_file = sys.argv[3]

subs = [tuple(line.strip().split()) for line in open(subs_file)]

features = []
current_edges = set()
node_labels = {}

with open(graphs) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        if line.startswith('#'):
            if current_edges:
                row = [1 if s in current_edges else 0 for s in subs]
                features.append(row)
            current_edges = set()
            node_labels = {}

        elif line.startswith('v'):
            _, nid, label = line.split()
            node_labels[nid] = label

        elif line.startswith('e'):
            _, u, v, el = line.split()
            if u in node_labels and v in node_labels:
                current_edges.add((node_labels[u], el, node_labels[v]))

# last graph
if current_edges:
    row = [1 if s in current_edges else 0 for s in subs]
    features.append(row)

with open(out_file, 'w') as f:
    for row in features:
        f.write(" ".join(map(str, row)) + "\n")
EOF


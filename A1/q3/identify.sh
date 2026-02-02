#!/bin/bash
# Usage: bash identify.sh <path_graph_dataset> <path_discriminative_subgraphs>

DATASET="$1"
OUTFILE="$2"

python3 - "$DATASET" "$OUTFILE" << 'EOF'
import sys
from collections import defaultdict
from itertools import combinations

dataset = sys.argv[1]
outfile = sys.argv[2]

edge_features = defaultdict(int)
path2_features = defaultdict(int)
triangle_features = defaultdict(int)

node_labels = {}
adj = defaultdict(list)

def flush_graph():
    for u in adj:
        nbrs = adj[u]
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                v1, el1 = nbrs[i]
                v2, el2 = nbrs[j]
                path = (node_labels[v1], el1, node_labels[u], el2, node_labels[v2])
                path2_features[path] += 1

    nodes = list(adj.keys())
    for a, b, c in combinations(nodes, 3):
        if any(x[0] == b for x in adj[a]) and \
           any(x[0] == c for x in adj[b]) and \
           any(x[0] == a for x in adj[c]):
            tri = tuple(sorted([node_labels[a], node_labels[b], node_labels[c]]))
            triangle_features[tri] += 1

with open(dataset) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        if line.startswith('#'):
            flush_graph()
            node_labels = {}
            adj = defaultdict(list)

        elif line.startswith('v'):
            _, nid, lbl = line.split()
            node_labels[nid] = lbl

        elif line.startswith('e'):
            _, u, v, el = line.split()
            adj[u].append((v, el))
            adj[v].append((u, el))
            edge_features[(node_labels[u], el, node_labels[v])] += 1

flush_graph()

MIN_SUP = 3
features = []

for k, v in edge_features.items():
    if v >= MIN_SUP:
        features.append(("EDGE", k, v))

for k, v in path2_features.items():
    if v >= MIN_SUP:
        features.append(("PATH2", k, v))

for k, v in triangle_features.items():
    if v >= MIN_SUP:
        features.append(("TRI", k, v))

K = 50
features.sort(key=lambda x: x[2], reverse=True)
features = features[:K]

with open(outfile, 'w') as f:
    for t, feat, _ in features:
        f.write(f"{t} {' '.join(feat)}\n")
EOF

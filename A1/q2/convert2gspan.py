import sys

inp = sys.argv[1]
out = sys.argv[2]

# -------------------------------------------------
# Pass 1: collect all node & edge labels
# -------------------------------------------------
node_labels = set()
edge_labels = set()

with open(inp) as f:
    lines = [l.strip() for l in f if l.strip()]

i = 0
while i < len(lines):
    # graph id line
    i += 1

    n = int(lines[i])
    i += 1

    # node labels
    for _ in range(n):
        node_labels.add(lines[i])
        i += 1

    m = int(lines[i])
    i += 1

    # edge labels
    for _ in range(m):
        _, _, lab = lines[i].replace(",", " ").split()
        edge_labels.add(lab)
        i += 1

# create integer mappings
node_labels = sorted(node_labels)
edge_labels = sorted(edge_labels, key=int)

vmap = {lab: idx for idx, lab in enumerate(node_labels)}
emap = {lab: idx for idx, lab in enumerate(edge_labels)}

print("Node label mapping:", vmap)
print("Edge label mapping:", emap)

# -------------------------------------------------
# Pass 2: write gSpan format
# -------------------------------------------------
with open(out, "w") as g:
    i = 0
    graph_id = 0

    while i < len(lines):
        # graph id (ignore actual value)
        i += 1

        g.write(f"t # {graph_id}\n")
        graph_id += 1

        # number of nodes
        n = int(lines[i])
        i += 1

        # nodes
        for vid in range(n):
            lab = lines[i]
            g.write(f"v {vid} {vmap[lab]}\n")
            i += 1

        # number of edges
        m = int(lines[i])
        i += 1

        # edges
        for _ in range(m):
            u, v, lab = lines[i].replace(",", " ").split()
            g.write(f"e {u} {v} {emap[lab]}\n")
            i += 1


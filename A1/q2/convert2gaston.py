import sys

inp = sys.argv[1]
out = sys.argv[2]

# -------------------------------------------------
# Pass 1: collect ALL node + edge labels
# -------------------------------------------------
node_labels = set()
edge_labels = set()

with open(inp) as f:
    lines = [l.strip() for l in f if l.strip()]

i = 0
while i < len(lines):
    # graph id (may start with # or be numeric)
    i += 1

    n = int(lines[i])
    i += 1

    # node labels
    for _ in range(n):
        node_labels.add(lines[i])
        i += 1

    m = int(lines[i])
    i += 1

    # edges
    for _ in range(m):
        _, _, elab = lines[i].split()
        edge_labels.add(elab)
        i += 1

# create mappings
node_labels = sorted(node_labels)
edge_labels = sorted(edge_labels, key=int)

vmap = {lab: idx for idx, lab in enumerate(node_labels)}
emap = {lab: idx for idx, lab in enumerate(edge_labels)}

print("Node label mapping:", vmap)
print("Edge label mapping:", emap)

# -------------------------------------------------
# Pass 2: write GASTON format
# -------------------------------------------------
with open(inp) as f, open(out, "w") as fout:
    lines = [l.strip() for l in f if l.strip()]
    i = 0
    tid = 0

    while i < len(lines):
        graph_id = lines[i]
        i += 1
        tid += 1

        fout.write(f"t # {tid}\n")

        n = int(lines[i])
        i += 1

        # nodes
        for vid in range(n):
            lab = lines[i]
            fout.write(f"v {vid} {vmap[lab]}\n")
            i += 1

        m = int(lines[i])
        i += 1

        # edges
        for _ in range(m):
            u, v, elab = lines[i].split()
            fout.write(f"e {u} {v} {emap[elab]}\n")
            i += 1


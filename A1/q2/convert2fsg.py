import sys

inp = sys.argv[1]
out = sys.argv[2]

# -------------------------------------------------
# Pass 1: collect all vertex & edge labels
# -------------------------------------------------
vlabels = set()
elabels = set()

with open(inp) as f:
    lines = [l.strip() for l in f if l.strip()]

i = 0
while i < len(lines):
    # graph id line (ignore actual value)
    i += 1

    n = int(lines[i])
    i += 1

    for _ in range(n):
        vlabels.add(lines[i])
        i += 1

    m = int(lines[i])
    i += 1

    for _ in range(m):
        _, _, el = lines[i].replace(",", " ").split()
        elabels.add(el)
        i += 1

# global mappings
vlabels = sorted(vlabels)
elabels = sorted(elabels, key=int)

vmap = {lab: idx for idx, lab in enumerate(vlabels)}
emap = {lab: idx for idx, lab in enumerate(elabels)}

print("Vertex label map:", vmap)
print("Edge label map:", emap)

# -------------------------------------------------
# Pass 2: write FSG format
# -------------------------------------------------
with open(out, "w") as g:
    i = 0

    while i < len(lines):
        # graph id line
        i += 1

        g.write("t\n")

        n = int(lines[i])
        i += 1

        # vertices
        for vid in range(n):
            lab = lines[i]
            g.write(f"v {vid} {vmap[lab]}\n")
            i += 1

        m = int(lines[i])
        i += 1

        edges = set()
        for _ in range(m):
            u, v, lab = lines[i].replace(",", " ").split()
            u, v = int(u), int(v)
            if u > v:
                u, v = v, u
            edges.add((u, v, emap[lab]))
            i += 1

        # edges must be sorted
        for u, v, lab in sorted(edges):
            g.write(f"u {u} {v} {lab}\n")


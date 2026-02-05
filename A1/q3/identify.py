#!/usr/bin/env python3
import sys, math
from collections import defaultdict

# Graph parser for gSpan-like format:
# v <id> <node_label>
# e <src> <dst> <edge_label>
# # ends graph
def parse_graphs(path):
    graphs=[]
    node_labels={}
    edges=[]  # list of (u,v,el)
    with open(path,'r') as f:
        for line in f:
            s=line.strip()
            if not s:
                continue
            if s=="#":
                # finalize
                if node_labels:
                    graphs.append((node_labels, edges))
                node_labels={}
                edges=[]
                continue
            parts=s.split()
            if parts[0]=='v':
                node_labels[int(parts[1])] = int(parts[2])
            elif parts[0]=='e':
                edges.append((int(parts[1]), int(parts[2]), int(parts[3])))
            else:
                raise ValueError(f"Unknown line: {s}")
    return graphs

def canonical_edge(lu, el, lv):
    # undirected canonical (by endpoint labels, then edge label)
    if lu < lv:
        return (lu, el, lv)
    if lu > lv:
        return (lv, el, lu)
    return (lu, el, lv)


def graph_signature(node_labels, edges):
    """Create an order-invariant signature for exact-duplicate detection.

    We consider two graphs duplicates if they have the same set of labeled vertices
    (by vertex id -> label) and the same multiset of labeled edges (undirected).
    This is sufficient for removing exact duplicates in the provided dataset while
    preserving the original ordering of first occurrences.
    """
    # vertex labels: order-invariant
    v_sig = tuple(sorted(node_labels.items()))
    # edges: order-invariant, treat as undirected; keep multiplicity if any
    e_norm = []
    for u, v, el in edges:
        a, b = (u, v) if u <= v else (v, u)
        e_norm.append((a, b, el))
    e_sig = tuple(sorted(e_norm))
    return (v_sig, e_sig)

def dedup_graphs_preserve_order(graphs):
    """Remove exact duplicate graphs while preserving the first occurrence order."""
    seen = set()
    out = []
    for g in graphs:
        sig = graph_signature(g[0], g[1])
        if sig in seen:
            continue
        seen.add(sig)
        out.append(g)
    return out

def mine_patterns(graphs, min_sup_cnt, max_patterns=200000):
    edge_sup=defaultdict(int)
    path2_sup=defaultdict(int)
    tri_sup=defaultdict(int)

    for node_labels, edges in graphs:
        # build undirected adjacency: node -> list[(nbr, edge_label)]
        adj=defaultdict(list)
        for u,v,el in edges:
            adj[u].append((v,el))
        # EDGE patterns per-graph
        seen_edge=set()
        for u,v,el in edges:
            if u not in node_labels or v not in node_labels:
                continue
            pat = canonical_edge(node_labels[u], el, node_labels[v])
            seen_edge.add(pat)
        for pat in seen_edge:
            edge_sup[pat]+=1

        # PATH2 patterns per-graph: a -e1- b -e2- c, b is center
        seen_path=set()
        for b, neighs in adj.items():
            lb=node_labels.get(b)
            if lb is None:
                continue
            # consider unordered pairs of incident edges (including same neighbor via different labels)
            m=len(neighs)
            for i in range(m):
                a, e1 = neighs[i]
                la=node_labels.get(a)
                if la is None: 
                    continue
                for j in range(i+1, m):
                    c, e2 = neighs[j]
                    lc=node_labels.get(c)
                    if lc is None:
                        continue
                    # canonicalize ends to remove symmetry: order (la,e1) vs (lc,e2) by tuple
                    left=(la,e1)
                    right=(lc,e2)
                    if left <= right:
                        pat=('PATH2', la, e1, lb, e2, lc)
                    else:
                        pat=('PATH2', lc, e2, lb, e1, la)
                    seen_path.add(pat)
        for pat in seen_path:
            path2_sup[pat]+=1

        # TRI patterns per-graph: triangle among nodes (x,y,z) with edge labels
        # Build neighbor sets with edge labels map for fast lookups
        nbrs=defaultdict(set)
        el_map=defaultdict(dict)  # u -> {v: el} (take first if multi-edge)
        for u,v,el in edges:
            nbrs[u].add(v)
            if v not in el_map[u]:
                el_map[u][v]=el
        seen_tri=set()
        nodes=list(node_labels.keys())
        # iterate u, v>u, w>v triangles using neighbor intersections
        for u in nodes:
            for v in nbrs.get(u, ()):
                if v <= u: 
                    continue
                inter = nbrs.get(u,set()).intersection(nbrs.get(v,set()))
                for w in inter:
                    if w <= v:
                        continue
                    # edges (u,v),(u,w),(v,w) must exist both directions already in nbrs
                    lu,lv,lw = node_labels[u], node_labels[v], node_labels[w]
                    el_uv = el_map[u].get(v)
                    el_uw = el_map[u].get(w)
                    el_vw = el_map[v].get(w)
                    if el_uv is None or el_uw is None or el_vw is None:
                        continue
                    # canonicalize by sorting nodes by labels then tie-break by ids
                    trip = sorted([(lu,u),(lv,v),(lw,w)])
                    (l1,n1),(l2,n2),(l3,n3)=trip
                    # retrieve edge labels between n1-n2, n1-n3, n2-n3 in undirected way
                    def get_el(a,b):
                        return el_map[a].get(b, el_map[b].get(a))
                    e12=get_el(n1,n2); e13=get_el(n1,n3); e23=get_el(n2,n3)
                    if e12 is None or e13 is None or e23 is None:
                        continue
                    pat=('TRI', l1, e12, l2, e13, l3, e23)
                    seen_tri.add(pat)
        for pat in seen_tri:
            tri_sup[pat]+=1

    # filter by min support
    def filt(d):
        return {k:v for k,v in d.items() if v>=min_sup_cnt}

    edge_sup=filt(edge_sup)
    path2_sup=filt(path2_sup)
    tri_sup=filt(tri_sup)

    # score patterns by support*(1-support) to favor mid-frequency
    n=len(graphs)
    scored=[]
    for pat,sup in edge_sup.items():
        frac=sup/n
        scored.append( (frac*(1-frac), ('EDGE', pat[0], pat[1], pat[2]), sup) )
    for pat,sup in path2_sup.items():
        frac=sup/n
        # pat includes 'PATH2' already, but store without duplication
        scored.append( (frac*(1-frac), pat, sup) )
    for pat,sup in tri_sup.items():
        frac=sup/n
        scored.append( (frac*(1-frac), pat, sup) )

    scored.sort(key=lambda x:(-x[0], x[2]))  # high score, then smaller support
    return scored

def main():
    if len(sys.argv)!=3:
        print("Usage: identify_fixed.py <path_graph_dataset> <path_discriminative_subgraphs>", file=sys.stderr)
        sys.exit(2)
    in_path=sys.argv[1]
    out_path=sys.argv[2]

    graphs=parse_graphs(in_path)
    graphs = dedup_graphs_preserve_order(graphs)
    if not graphs:
        raise RuntimeError("No graphs parsed")

    min_sup_cnt=1  # allow rare patterns to reach k
    scored=mine_patterns(graphs, min_sup_cnt)

    k=50
    top=scored[:k]
    with open(out_path,'w') as f:
        for _, pat, sup in top:
            # pat is tuple like ('EDGE', lu, el, lv) or ('PATH2', la,e1,lb,e2,lc) or ('TRI', ...)
            f.write(" ".join(map(str, pat))+"\n")

if __name__=="__main__":
    main()

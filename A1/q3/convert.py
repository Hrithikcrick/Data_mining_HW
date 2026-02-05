#!/usr/bin/env python3
import sys, numpy as np
from collections import defaultdict

def parse_graphs(path):
    graphs=[]
    node_labels={}
    edges=[]
    with open(path,'r') as f:
        for line in f:
            s=line.strip()
            if not s:
                continue
            if s=="#":
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
    if lu < lv:
        return (lu, el, lv)
    if lu > lv:
        return (lv, el, lu)
    return (lu, el, lv)

def load_features(path):
    feats=[]
    with open(path,'r') as f:
        for line in f:
            s=line.strip()
            if not s:
                continue
            parts=s.split()
            feats.append(tuple(parts))
    return feats

def graph_feature_presence(node_labels, edges, feature):
    # feature is tuple of strings
    ftype=feature[0]
    adj=defaultdict(list)
    el_map=defaultdict(dict)
    for u,v,el in edges:
        adj[u].append((v,el))
        adj[v].append((u,el))  # make undirected for matching
        if v not in el_map[u]: el_map[u][v]=el
        if u not in el_map[v]: el_map[v][u]=el

    if ftype=='EDGE':
        lu=int(feature[1]); el=int(feature[2]); lv=int(feature[3])
        # check any edge with endpoint labels and edge label
        for u,v,el0 in edges:
            if el0!=el: 
                continue
            a=node_labels.get(u); b=node_labels.get(v)
            if a is None or b is None: 
                continue
            if canonical_edge(a, el0, b)==(min(lu,lv), el, max(lu,lv)) and ((lu<=lv and a==lu and b==lv) or True):
                # canonical match already ignores direction
                return True
        return False

    if ftype=='PATH2':
        la=int(feature[1]); e1=int(feature[2]); lb=int(feature[3]); e2=int(feature[4]); lc=int(feature[5])
        for b, neighs in adj.items():
            if node_labels.get(b)!=lb:
                continue
            # find neighbor a with label la and edge e1, neighbor c with label lc and edge e2
            # Because we canonicalized ends, either ordering is allowed; but feature stored already canonical,
            # so match exact (la,e1) on one side and (lc,e2) on other, in any assignment.
            has_left=[]
            has_right=[]
            for nbr, el in neighs:
                ln=node_labels.get(nbr)
                if ln is None: 
                    continue
                if ln==la and el==e1:
                    has_left.append(nbr)
                if ln==lc and el==e2:
                    has_right.append(nbr)
            if has_left and has_right:
                # ensure distinct endpoints (not required by spec but sensible)
                for a in has_left:
                    for c in has_right:
                        if a!=c:
                            return True
        return False

    if ftype=='TRI':
        l1=int(feature[1]); e12=int(feature[2]); l2=int(feature[3]); e13=int(feature[4]); l3=int(feature[5]); e23=int(feature[6])
        # find any triple with labels and edge labels
        nodes=[n for n,l in node_labels.items()]
        # build candidate nodes by label
        by_label=defaultdict(list)
        for n,l in node_labels.items():
            by_label[l].append(n)
        for n1 in by_label.get(l1, []):
            for n2 in by_label.get(l2, []):
                if n2==n1: 
                    continue
                if el_map[n1].get(n2)!=e12:
                    continue
                for n3 in by_label.get(l3, []):
                    if n3==n1 or n3==n2:
                        continue
                    if el_map[n1].get(n3)!=e13:
                        continue
                    if el_map[n2].get(n3)!=e23:
                        continue
                    return True
        return False

    raise ValueError(f"Unknown feature type: {ftype}")

def main():
    if len(sys.argv)!=4:
        print("Usage: convert_fixed.py <path_graphs> <path_discriminative_subgraphs> <path_features>", file=sys.stderr)
        sys.exit(2)
    graphs_path=sys.argv[1]
    feat_path=sys.argv[2]
    out_path=sys.argv[3]

    graphs=parse_graphs(graphs_path)
    feats=load_features(feat_path)
    k=len(feats)
    X=np.zeros((len(graphs), k), dtype=np.uint8)

    for i,(node_labels, edges) in enumerate(graphs):
        for j,feat in enumerate(feats):
            if graph_feature_presence(node_labels, edges, feat):
                X[i,j]=1
    np.save(out_path, X)

if __name__=="__main__":
    main()

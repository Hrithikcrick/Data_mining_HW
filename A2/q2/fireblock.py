import argparse
import os
import sys
import time
import random
from collections import deque
import heapq


def write_atomic(out_path: str, edges):
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for (u, v) in edges:
            f.write(f"{u} {v}\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out_path)


def read_seeds(seed_path: str):
    seeds = []
    with open(seed_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            seeds.append(int(s.split()[0]))
    return seeds


def read_graph(graph_path: str):
    lines = []
    with open(graph_path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            lines.append(s)

    # optional header skip: "n m" on first line
    header_skipped = False
    if lines:
        p0 = lines[0].split()
        if len(p0) == 2 and all(x.lstrip("-").isdigit() for x in p0):
            # if many later lines have >=3 cols, treat first as header
            has_three = 0
            for t in lines[1: min(len(lines), 200)]:
                if len(t.split()) >= 3:
                    has_three += 1
            if has_three >= 5:
                header_skipped = True

    start = 1 if header_skipped else 0
    edges = []
    max_node = 0
    for s in lines[start:]:
        parts = s.split()
        if len(parts) < 2:
            continue
        u = int(parts[0]); v = int(parts[1])
        p = float(parts[2]) if len(parts) >= 3 else 1.0
        edges.append((u, v, p))
        if u > max_node: max_node = u
        if v > max_node: max_node = v
    return max_node, edges


def bfs_dist(n, out_adj, seeds, hops):
    dist = [-1] * (n + 1)
    q = deque()
    for s in seeds:
        if 1 <= s <= n and dist[s] == -1:
            dist[s] = 0
            q.append(s)
    while q:
        u = q.popleft()
        du = dist[u]
        if hops != -1 and du >= hops:
            continue
        for v, _p in out_adj[u]:
            if 1 <= v <= n and dist[v] == -1:
                dist[v] = du + 1
                q.append(v)
    return dist


# ---------- MODE A (hops = -1): your strong layered heuristic ----------

def layered_prob(n, out_adj, dist, seeds, hops):
    maxd = 0
    for i in range(1, n + 1):
        if dist[i] > maxd:
            maxd = dist[i]
    layers = [[] for _ in range(maxd + 1)]
    for i in range(1, n + 1):
        d = dist[i]
        if d >= 0:
            layers[d].append(i)

    prob = [0.0] * (n + 1)
    prod_not = [1.0] * (n + 1)

    for s in seeds:
        if 1 <= s <= n and dist[s] == 0:
            prob[s] = 1.0

    for d in range(0, maxd):
        for u in layers[d]:
            pu = prob[u]
            if pu <= 0.0:
                continue
            for v, p in out_adj[u]:
                if dist[v] == d + 1:
                    x = 1.0 - pu * p
                    if x < 0.0: x = 0.0
                    prod_not[v] *= x
        for v in layers[d + 1]:
            prob[v] = 1.0 - prod_not[v]
    return prob


def select_edges_layered(n, edges, out_adj, dist, prob, k, out_path, checkpoint_sec=5.0):
    t0 = time.time()
    last_write = 0.0

    # immediate valid baseline
    baseline = []
    seen = set()
    for (u, v, _p) in edges:
        if (u, v) not in seen:
            seen.add((u, v))
            baseline.append((u, v))
            if len(baseline) == k:
                break
    if len(baseline) < k:
        baseline += [(1, 1)] * (k - len(baseline))
    write_atomic(out_path, baseline)

    heap = []  # min-heap of (score,u,v)
    in_heap = set()

    def checkpoint(force=False):
        nonlocal last_write
        now = time.time()
        if force or (now - last_write >= checkpoint_sec):
            best = sorted(heap, reverse=True)
            chosen = [(u, v) for (sc, u, v) in best[:k]]
            if len(chosen) < k:
                used = set(chosen)
                for (u, v) in baseline:
                    if len(chosen) == k:
                        break
                    if (u, v) not in used:
                        chosen.append((u, v))
                        used.add((u, v))
            write_atomic(out_path, chosen[:k])
            last_write = now

    for (u, v, p) in edges:
        du = dist[u]; dv = dist[v]
        if du < 0 or dv < 0:
            continue
        if dv != du + 1:
            continue
        score = prob[u] * p * (1.0 - prob[v]) * (1.0 + 0.05 * len(out_adj[v]))
        if score <= 0.0:
            continue
        key = (u, v)
        if key in in_heap:
            continue
        if len(heap) < k:
            heapq.heappush(heap, (score, u, v))
            in_heap.add(key)
        else:
            if score > heap[0][0]:
                sc0, u0, v0 = heapq.heapreplace(heap, (score, u, v))
                in_heap.discard((u0, v0))
                in_heap.add(key)
        checkpoint(False)

    checkpoint(True)
    best = sorted(heap, reverse=True)
    chosen = [(u, v) for (sc, u, v) in best[:k]]
    if len(chosen) < k:
        used = set(chosen)
        for (u, v) in baseline:
            if len(chosen) == k:
                break
            if (u, v) not in used:
                chosen.append((u, v))
                used.add((u, v))
    chosen = chosen[:k]
    write_atomic(out_path, chosen)
    return chosen, time.time() - t0


# ---------- MODE B (hops > 0): Monte Carlo edge-usage (better for hops=3) ----------

def select_edges_mc_hops(n, out_adj, seeds, hops, k, r, out_path, checkpoint_sec=5.0):
    t0 = time.time()
    last_write = 0.0

    # nodes within hop-limit for fast sim
    dist = bfs_dist(n, out_adj, seeds, hops)
    allowed = [False] * (n + 1)
    for i in range(1, n + 1):
        if dist[i] != -1 and dist[i] <= hops:
            allowed[i] = True

    # baseline: just take first k allowed outgoing edges
    baseline = []
    seen = set()
    for u in range(1, n + 1):
        if not allowed[u]:
            continue
        for v, _p in out_adj[u]:
            if allowed[v] and (u, v) not in seen:
                seen.add((u, v))
                baseline.append((u, v))
                if len(baseline) == k:
                    break
        if len(baseline) == k:
            break
    if len(baseline) < k:
        baseline += [(1, 1)] * (k - len(baseline))
    write_atomic(out_path, baseline)

    edge_count = {}  # (u,v) -> times it caused activation

    rnd = random.Random(42)  # deterministic

    def checkpoint(force=False):
        nonlocal last_write
        now = time.time()
        if force or (now - last_write >= checkpoint_sec):
            top = sorted(edge_count.items(), key=lambda x: x[1], reverse=True)
            chosen = [e for (e, c) in top[:k]]
            if len(chosen) < k:
                used = set(chosen)
                for e in baseline:
                    if len(chosen) == k:
                        break
                    if e not in used:
                        chosen.append(e)
                        used.add(e)
            write_atomic(out_path, chosen[:k])
            last_write = now

    # run r simulations (IC), but hop-limited by BFS levels from seeds
    for sim in range(max(1, r)):
        active = set()
        q = deque()

        for s in seeds:
            if 1 <= s <= n and allowed[s]:
                if s not in active:
                    active.add(s)
                    q.append((s, 0))

        while q:
            u, d = q.popleft()
            if d >= hops:
                continue
            for v, p in out_adj[u]:
                if not (1 <= v <= n and allowed[v]):
                    continue
                if v in active:
                    continue
                # count attempt weight (better for choosing edges to block)
                edge_count[(u, v)] = edge_count.get((u, v), 0.0) + float(p)

                # still simulate successful activation
                if rnd.random() < p:
                    active.add(v)
                    q.append((v, d + 1))

        checkpoint(False)

    checkpoint(True)

    top = sorted(edge_count.items(), key=lambda x: x[1], reverse=True)
    chosen = [e for (e, c) in top[:k]]
    if len(chosen) < k:
        used = set(chosen)
        for e in baseline:
            if len(chosen) == k:
                break
            if e not in used:
                chosen.append(e)
                used.add(e)
    chosen = chosen[:k]
    write_atomic(out_path, chosen)
    return chosen, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--r", type=int, required=True)
    ap.add_argument("--hops", type=int, required=True)
    args = ap.parse_args()

    n, edges = read_graph(args.graph)
    seeds = read_seeds(args.seeds)

    out_adj = [[] for _ in range(n + 1)]
    for (u, v, p) in edges:
        if 1 <= u <= n:
            out_adj[u].append((v, p))

    # Always do an immediate write (partial credit / crash safety)
    write_atomic(args.out, [])

    if args.hops == -1:
        dist = bfs_dist(n, out_adj, seeds, -1)
        prob = layered_prob(n, out_adj, dist, seeds, -1)
        chosen, dt = select_edges_layered(n, edges, out_adj, dist, prob, args.k, args.out, checkpoint_sec=5.0)
        print(f"[info] layered seeds={len(seeds)} edges={len(edges)} chose={len(chosen)} time={dt:.2f}s", file=sys.stderr)
    else:
        # Monte Carlo mode for hop-limited
        chosen, dt = select_edges_mc_hops(n, out_adj, seeds, args.hops, args.k, args.r, args.out, checkpoint_sec=5.0)
        print(f"[info] mc_hops seeds={len(seeds)} edges={len(edges)} chose={len(chosen)} r={args.r} time={dt:.2f}s", file=sys.stderr)


if __name__ == "__main__":
    main()

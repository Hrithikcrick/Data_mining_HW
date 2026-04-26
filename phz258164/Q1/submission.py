import os
import numpy as np
import faiss

# ------------------------------------------------------------
# Budget-aware Q1 solver
#
# D2 (<=25s):
#   cosine + IVFPQ + evenly spaced query sampling
# D1 (>25s):
#   exact L2 if cheap, else HNSW-L2 with full voting
#
# Designed to preserve the required solve(...) interface while
# reducing D2 runtime substantially.
# ------------------------------------------------------------

MAX_THREADS = 8
SEED = 42

# ---------- D2 / fast ----------
D2_SAMPLE_Q1 = 32768
D2_SAMPLE_Q2 = 16384
D2_W1 = 0.70
D2_W2 = 0.30
D2_NPROBE = 12
D2_BATCH_SIZE = 8192
D2_EXACT_PAIR_THRESHOLD = 12_000_000
D2_MIN_TRAIN = 25000
D2_TRAIN_MULT = 20

# ---------- D1 / slow ----------
D1_BATCH_SIZE = 16384
D1_HNSW_M = 24
D1_EF_CONSTRUCTION = 48
D1_EF_SEARCH = 256
D1_EXACT_COST_THRESHOLD = 600_000_000

# ---------- shared ----------
MIN_TRAIN = 50000
TRAIN_MULT = 40


def _to_float32_contiguous(x):
    x = np.asarray(x, dtype=np.float32, order="C")
    if not x.flags.c_contiguous:
        x = np.ascontiguousarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        x = np.nan_to_num(x, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
    return x


def _l2_normalize_inplace(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    x /= norms
    return x


def _set_num_threads():
    try:
        nthreads = max(1, min(MAX_THREADS, os.cpu_count() or 1))
        faiss.omp_set_num_threads(nthreads)
    except Exception:
        pass


def _sample_even(n, s, offset=0.0):
    if s >= n:
        return np.arange(n, dtype=np.int64)
    pos = (np.arange(s, dtype=np.float64) + offset) * (n / float(s))
    ids = np.floor(pos).astype(np.int64)
    np.clip(ids, 0, n - 1, out=ids)
    return ids


def _pick_nlist(n):
    if n <= 50000:
        return 256
    elif n <= 150000:
        return 512
    elif n <= 400000:
        return 1024
    elif n <= 1000000:
        return 2048
    else:
        return 4096


def _pick_pq_m(d):
    preferred = [64, 56, 48, 40, 32, 28, 24, 20, 16, 14, 12, 10, 8, 7, 6, 5, 4, 3, 2, 1]
    for m in preferred:
        if m <= d and d % m == 0:
            return m
    for m in range(min(64, d), 0, -1):
        if d % m == 0:
            return m
    return 1


def _exact_search_l2(base, queries, k, K):
    index = faiss.IndexFlatL2(base.shape[1])
    index.add(base)
    _, neigh = index.search(queries, k)

    counts = np.bincount(neigh.ravel(), minlength=base.shape[0]).astype(np.float64)
    rank_bonus = np.arange(k, 0, -1, dtype=np.float64)
    bonus = np.bincount(
        neigh.ravel(),
        weights=np.broadcast_to(rank_bonus, neigh.shape).ravel(),
        minlength=base.shape[0],
    )

    order = np.lexsort((np.arange(base.shape[0]), -bonus, -counts))
    return order[:K]


def _exact_search_cosine(base, queries, k, K):
    index = faiss.IndexFlatIP(base.shape[1])
    index.add(base)
    _, neigh = index.search(queries, k)

    counts = np.bincount(neigh.ravel(), minlength=base.shape[0]).astype(np.float64)
    rank_bonus = np.arange(k, 0, -1, dtype=np.float64)
    bonus = np.bincount(
        neigh.ravel(),
        weights=np.broadcast_to(rank_bonus, neigh.shape).ravel(),
        minlength=base.shape[0],
    )

    order = np.lexsort((np.arange(base.shape[0]), -bonus, -counts))
    return order[:K]


def _build_ivfpq_cosine(base):
    n, d = base.shape
    nlist = _pick_nlist(n)
    m = _pick_pq_m(d)

    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8, faiss.METRIC_INNER_PRODUCT)

    train_size = min(n, max(D2_MIN_TRAIN, nlist * D2_TRAIN_MULT))
    train_idx = _sample_even(n, train_size)
    index.train(base[train_idx])
    index.add(base)
    index.nprobe = min(nlist, D2_NPROBE)
    return index


def _build_hnsw_l2(base):
    d = base.shape[1]
    index = faiss.IndexHNSWFlat(d, D1_HNSW_M)
    index.hnsw.efConstruction = D1_EF_CONSTRUCTION
    index.add(base)
    index.hnsw.efSearch = D1_EF_SEARCH
    return index


def _weighted_vote_from_index(index, queries, k, n_base, batch_size, weight_per_query):
    counts = np.zeros(n_base, dtype=np.float64)
    bonus = np.zeros(n_base, dtype=np.float64)
    rank_bonus = np.arange(k, 0, -1, dtype=np.float64)

    for st in range(0, queries.shape[0], batch_size):
        en = min(st + batch_size, queries.shape[0])
        _, neigh = index.search(queries[st:en], k)

        valid = neigh >= 0
        if not np.any(valid):
            continue

        flat_ids = neigh[valid]
        counts += np.bincount(flat_ids, weights=np.full(flat_ids.shape[0], weight_per_query), minlength=n_base)

        rb = np.broadcast_to(rank_bonus, neigh.shape)
        bonus += np.bincount(flat_ids, weights=rb[valid] * weight_per_query, minlength=n_base)

    return counts, bonus


def _combine_scores(counts, bonus):
    return counts * 1000.0 + bonus


def _rank_from_scores(scores, K):
    order = np.lexsort((np.arange(scores.shape[0]), -scores))
    return order[:K]


def _solve_d2(base, queries, k, K):
    _l2_normalize_inplace(base)
    _l2_normalize_inplace(queries)

    n_base = base.shape[0]
    n_queries = queries.shape[0]

    if n_base * n_queries <= D2_EXACT_PAIR_THRESHOLD:
        return _exact_search_cosine(base, queries, k, K)

    index = _build_ivfpq_cosine(base)

    s1 = min(n_queries, D2_SAMPLE_Q1)
    ids1 = _sample_even(n_queries, s1, offset=0.0)
    q1 = queries[ids1]
    w1 = D2_W1 * n_queries / float(s1)

    counts1, bonus1 = _weighted_vote_from_index(
        index=index,
        queries=q1,
        k=k,
        n_base=n_base,
        batch_size=D2_BATCH_SIZE,
        weight_per_query=w1,
    )

    s2 = min(n_queries, D2_SAMPLE_Q2)
    ids2 = _sample_even(n_queries, s2, offset=0.5)
    q2 = queries[ids2]
    w2 = D2_W2 * n_queries / float(s2)

    counts2, bonus2 = _weighted_vote_from_index(
        index=index,
        queries=q2,
        k=k,
        n_base=n_base,
        batch_size=D2_BATCH_SIZE,
        weight_per_query=w2,
    )

    scores = _combine_scores(counts1 + counts2, bonus1 + bonus2)
    return _rank_from_scores(scores, K)


def _solve_d1(base, queries, k, K):
    n_base = base.shape[0]
    n_queries = queries.shape[0]

    exact_cost = n_base * n_queries * base.shape[1]
    if exact_cost <= D1_EXACT_COST_THRESHOLD:
        return _exact_search_l2(base, queries, k, K)

    index = _build_hnsw_l2(base)

    counts, bonus = _weighted_vote_from_index(
        index=index,
        queries=queries,
        k=k,
        n_base=n_base,
        batch_size=D1_BATCH_SIZE,
        weight_per_query=1.0,
    )

    scores = _combine_scores(counts, bonus)
    return _rank_from_scores(scores, K)


def solve(base_vectors, query_vectors, k, K, time_budget):
    """
    base_vectors: np.ndarray of shape (N, d)
    query_vectors: np.ndarray of shape (Q, d)
    k: int, fixed number of nearest neighbors per query
    K: int, number of output items to return
    time_budget: float, maximum allowed time in seconds (use as internal hint)
    Returns:
        np.ndarray of shape (K,), containing selected base indices
    """
    if base_vectors is None or query_vectors is None:
        return np.empty((0,), dtype=np.int64)

    base = _to_float32_contiguous(base_vectors)
    queries = _to_float32_contiguous(query_vectors)

    n_base = base.shape[0]
    n_queries = queries.shape[0]
    if n_base == 0 or n_queries == 0:
        return np.empty((0,), dtype=np.int64)

    k = int(min(k, n_base))
    K = int(min(K, n_base))

    _set_num_threads()

    if time_budget <= 25:
        ranking = _solve_d2(base, queries, k, K)
    else:
        ranking = _solve_d1(base, queries, k, K)

    return ranking.astype(np.int64)

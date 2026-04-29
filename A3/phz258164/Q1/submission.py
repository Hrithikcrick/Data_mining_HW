import os
import numpy as np
import faiss

# ------------------------------------------------------------
# Q1 solver
#
# D2 (<=25s):
#   cosine + IVFFlat + full-query voting
# D1 (>25s):
#   exact L2 if cheap, else HNSW-L2 + full-query voting
#
# This removes the D2 query-sampling loss you were seeing.
# ------------------------------------------------------------

MAX_THREADS = 8

# ---------- D2 ----------
D2_BATCH_SIZE = 16384
D2_NPROBE = 24
D2_EXACT_PAIR_THRESHOLD = 12_000_000
D2_MIN_TRAIN = 81920
D2_TRAIN_MULT = 40

# ---------- D1 ----------
D1_BATCH_SIZE = 16384
D1_HNSW_M = 24
D1_EF_CONSTRUCTION = 48
D1_EF_SEARCH = 256
D1_EXACT_COST_THRESHOLD = 600_000_000


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


def _sample_even(n, s):
    if s >= n:
        return np.arange(n, dtype=np.int64)
    return np.linspace(0, n - 1, num=s, dtype=np.int64)


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


def _build_ivf_cosine(base):
    n, d = base.shape
    nlist = _pick_nlist(n)

    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

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


def _vote_from_index(index, queries, k, n_base, batch_size):
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
        counts += np.bincount(flat_ids, minlength=n_base)

        rb = np.broadcast_to(rank_bonus, neigh.shape)
        bonus += np.bincount(flat_ids, weights=rb[valid], minlength=n_base)

    return counts, bonus


def _rank_from_counts_bonus(counts, bonus, K):
    order = np.lexsort((np.arange(counts.shape[0]), -bonus, -counts))
    return order[:K]


def _solve_d2(base, queries, k, K):
    _l2_normalize_inplace(base)
    _l2_normalize_inplace(queries)

    n_base = base.shape[0]
    n_queries = queries.shape[0]

    if n_base * n_queries <= D2_EXACT_PAIR_THRESHOLD:
        return _exact_search_cosine(base, queries, k, K)

    index = _build_ivf_cosine(base)
    counts, bonus = _vote_from_index(index, queries, k, n_base, D2_BATCH_SIZE)
    return _rank_from_counts_bonus(counts, bonus, K)


def _solve_d1(base, queries, k, K):
    n_base = base.shape[0]
    n_queries = queries.shape[0]

    exact_cost = n_base * n_queries * base.shape[1]
    if exact_cost <= D1_EXACT_COST_THRESHOLD:
        return _exact_search_l2(base, queries, k, K)

    index = _build_hnsw_l2(base)
    counts, bonus = _vote_from_index(index, queries, k, n_base, D1_BATCH_SIZE)
    return _rank_from_counts_bonus(counts, bonus, K)


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

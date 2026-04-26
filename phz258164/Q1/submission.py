import os
import numpy as np
import faiss

# ----------------------------------------------------------------------
# High‑accuracy solver for "most representative database items" via k‑NN voting
#
# - D2 (time_budget ≤ 20s): cosine similarity + IVF (fast, good recall)
# - D1 (time_budget = 70s): exact L2 search (perfect recall) if feasible,
#   otherwise high‑recall HNSW with L2.
# ----------------------------------------------------------------------

MAX_THREADS = 8
SEED = 42

def _to_float32_contiguous(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32, order="C")
    if not x.flags.c_contiguous:
        x = np.ascontiguousarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        x = np.nan_to_num(x, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
    return x

def _l2_normalize_inplace(x: np.ndarray) -> np.ndarray:
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

def _exact_search_l2(base, queries, k, K):
    """Exact brute‑force L2 search – perfect recall."""
    index = faiss.IndexFlatL2(base.shape[1])
    index.add(base)
    _, neigh = index.search(queries, k)
    counts = np.bincount(neigh.ravel(), minlength=base.shape[0])
    order = np.lexsort((np.arange(base.shape[0]), -counts))
    return order[:K]

def _exact_search_cosine(base, queries, k, K):
    """Exact brute‑force inner product (cosine after normalisation)."""
    index = faiss.IndexFlatIP(base.shape[1])
    index.add(base)
    _, neigh = index.search(queries, k)
    counts = np.bincount(neigh.ravel(), minlength=base.shape[0])
    order = np.lexsort((np.arange(base.shape[0]), -counts))
    return order[:K]

def _build_ivf_cosine(base, n_queries):
    """IVF flat index with cosine similarity (inner product after norm)."""
    n, d = base.shape
    if n <= 50000:
        nlist = 256
    elif n <= 150000:
        nlist = 512
    elif n <= 400000:
        nlist = 1024
    elif n <= 1000000:
        nlist = 2048
    else:
        nlist = 4096

    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

    train_size = min(n, max(50000, nlist * 40))
    rng = np.random.RandomState(SEED)
    train_idx = rng.choice(n, train_size, replace=False)
    index.train(base[train_idx])
    index.add(base)
    index.nprobe = min(nlist, 48)   # good recall within 20s
    return index

def _build_hnsw_l2(base, time_budget):
    """HNSW index with L2 distance – high recall for D1."""
    d = base.shape[1]
    # For D1 (70s) we can afford high efSearch
    M = 32
    ef_construction = 80
    ef_search = 500   # near‑exact recall
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = ef_construction
    index.add(base)
    index.hnsw.efSearch = ef_search
    return index

def _voting_from_index(index, queries, k, n_base):
    """Perform k‑NN search for all queries and return vote counts."""
    q = queries.shape[0]
    batch_size = 8192
    counts = np.zeros(n_base, dtype=np.int64)
    for st in range(0, q, batch_size):
        en = min(st + batch_size, q)
        _, neigh = index.search(queries[st:en], k)
        counts += np.bincount(neigh.ravel(), minlength=n_base)
    return counts

def solve(base_vectors, query_vectors, k, K, time_budget):
    """
    Returns the K most representative database indices according to
    the k‑NN voting rule.
    """
    if base_vectors is None or query_vectors is None:
        return np.empty((0,), dtype=np.int64)

    base = _to_float32_contiguous(base_vectors)
    queries = _to_float32_contiguous(query_vectors)

    n_base = base.shape[0]
    n_queries = queries.shape[0]
    k = min(k, n_base)
    K = min(K, n_base)

    if n_base == 0 or n_queries == 0:
        return np.empty((0,), dtype=np.int64)

    _set_num_threads()

    # --------------------------------------------------------------
    # D2 (tight budget): cosine similarity + IVF (fast)
    # --------------------------------------------------------------
    if time_budget <= 25:
        _l2_normalize_inplace(base)
        _l2_normalize_inplace(queries)
        # Exact search if dataset is small enough
        if n_base * n_queries <= 15_000_000:
            ranking = _exact_search_cosine(base, queries, k, K)
        else:
            index = _build_ivf_cosine(base, n_queries)
            counts = _voting_from_index(index, queries, k, n_base)
            order = np.lexsort((np.arange(n_base), -counts))
            ranking = order[:K]
        return ranking.astype(np.int64)

    # --------------------------------------------------------------
    # D1 (70s budget): try exact L2 search first (perfect recall)
    # If too large, fallback to high‑recall HNSW with L2.
    # --------------------------------------------------------------
    # Estimate cost: n_base * n_queries * d operations
    # If less than ~2e9, exact L2 should finish within 70s on modern CPU.
    # Use a conservative threshold.
    exact_cost = n_base * n_queries * base.shape[1]
    if exact_cost <= 2_000_000_000:   # 2e9 flops ~ 2-3 seconds? Actually safe bound
        ranking = _exact_search_l2(base, queries, k, K)
        return ranking.astype(np.int64)

    # Fallback: HNSW with L2 and very high ef_search
    index = _build_hnsw_l2(base, time_budget)
    counts = _voting_from_index(index, queries, k, n_base)
    order = np.lexsort((np.arange(n_base), -counts))
    ranking = order[:K]
    return ranking.astype(np.int64)
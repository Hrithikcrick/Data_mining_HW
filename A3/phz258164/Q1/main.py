from __future__ import annotations

import argparse
import importlib
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np


def configure_parallelism(parallel: bool, num_threads: Optional[int]) -> None:
    import faiss

    if parallel:
        n = num_threads if num_threads is not None else faiss.omp_get_max_threads()
        n = max(1, int(n))
    else:
        n = 1
    faiss.omp_set_num_threads(n)


def load_vector_matrix(path: Path, mmap: bool) -> np.ndarray:
    arr = np.load(path, mmap_mode="r" if mmap else None)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array in {path}, got shape {arr.shape}")
    if arr.dtype != np.float32:
        arr = np.ascontiguousarray(arr, dtype=np.float32)
    else:
        arr = np.ascontiguousarray(arr)
    return arr


def load_ground_truth(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        gt = np.load(path)
    else:
        gt = np.loadtxt(path, dtype=np.int64)
    gt = np.asarray(gt, dtype=np.int64).ravel()
    if gt.ndim != 1:
        raise ValueError(f"Ground truth must be 1D, got shape {gt.shape}")
    return gt


def ndcg_at_k(student_ranking: np.ndarray, ground_truth: np.ndarray) -> float:
    K = int(ground_truth.shape[0])
    if K == 0:
        return 1.0

    rel_by_index = {int(ground_truth[i]): float(K - i) for i in range(K)}

    idcg = 0.0
    for i in range(K):
        rel = float(K - i)
        idcg += rel / np.log2(i + 2.0)

    dcg = 0.0
    seen = set()
    for j in range(min(K, int(student_ranking.shape[0]))):
        idx = int(student_ranking[j])
        if idx in seen:
            continue
        seen.add(idx)
        rel = rel_by_index.get(idx, 0.0)
        dcg += rel / np.log2(j + 2.0)

    if idcg <= 0.0:
        return 0.0
    return float(dcg / idcg)


def normalize_student_output(result: Any, K: int) -> np.ndarray:
    if result is None:
        return np.full(K, -1, dtype=np.int64)
    out = np.asarray(result, dtype=np.int64).ravel()
    if out.shape[0] > K:
        out = out[:K]
    elif out.shape[0] < K:
        pad = np.full(K - out.shape[0], -1, dtype=np.int64)
        out = np.concatenate([out, pad])
    return out


def _solve_entry(
    q: "mp.Queue[Tuple[str, Any]]",
    module_name: str,
    base_vectors: np.ndarray,
    query_vectors: np.ndarray,
    k: int,
    K: int,
    time_budget: float,
) -> None:
    try:
        mod = importlib.import_module(module_name)
        solve = getattr(mod, "solve")
        out = solve(base_vectors, query_vectors, k, K, time_budget)
        q.put(("ok", out))
    except Exception as e:
        q.put(("err", e))


def run_solve_with_time_limit(
    module_name: str,
    base_vectors: np.ndarray,
    query_vectors: np.ndarray,
    k: int,
    K: int,
    time_budget: float,
    time_limit_sec: float,
    use_fork: bool,
) -> Tuple[np.ndarray, float, bool]:
    ctx = mp.get_context("fork" if use_fork else "spawn")
    q: mp.Queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=_solve_entry,
        args=(q, module_name, base_vectors, query_vectors, k, K, time_budget),
    )
    t0 = time.perf_counter()
    proc.start()
    proc.join(timeout=time_limit_sec)
    elapsed = time.perf_counter() - t0

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5.0)
        print(
            f"WARNING: time limit ({time_limit_sec}s) exceeded; no valid result returned.",
            file=sys.stderr,
        )
        return normalize_student_output(None, K), elapsed, True

    if not q.empty():
        status, payload = q.get()
        if status == "err":
            raise payload
        return normalize_student_output(payload, K), elapsed, False

    print(
        "WARNING: worker exited without result (crashed or killed).",
        file=sys.stderr,
    )
    return normalize_student_output(None, K), elapsed, True


def write_indices(path: Path, indices: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, indices.reshape(-1, 1), fmt="%d")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="COL761 similarity search assignment runner")
    p.set_defaults(parallel=True)
    p.add_argument("--base_vectors", type=Path, required=True)
    p.add_argument("--query_vectors", type=Path, required=True)
    p.add_argument("--ground_truth", type=Path, required=True)
    p.add_argument("--k", type=int, required=True, help="Neighbors per query (passed to solve)")
    p.add_argument("--K", type=int, required=True, help="Size of ranked list / nDCG cutoff")
    p.add_argument("--time_limit", type=float, required=True, help="Wall-clock limit in seconds")
    p.add_argument(
        "--time_budget",
        type=float,
        default=None,
        help="Budget passed to solve() (defaults to --time_limit)",
    )
    p.add_argument("--output", type=Path, default=None, help="Where to save selected indices")
    p.add_argument("--submission_module", type=str, default="submission")
    p.add_argument("--mmap", action="store_true", help="Memory-map large .npy inputs")
    p.add_argument(
        "--parallel",
        action="store_true",
        help="Allow FAISS/OpenMP multi-threading (default)",
    )
    p.add_argument(
        "--no-parallel",
        action="store_false",
        dest="parallel",
        help="Use a single thread for FAISS/OpenMP",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=None,
        help="OMP/FAISS thread count when --parallel (default: all)",
    )
    p.add_argument(
        "--no_subprocess",
        action="store_true",
        help="Run solve in-process (no hard wall-clock kill; faster for huge arrays)",
    )
    p.add_argument(
        "--transpose",
        action="store_true",
        help="Transpose each matrix after load: use when files are (d, N) and (d, Q) instead of (N, d) and (Q, d)",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    time_budget = float(args.time_budget) if args.time_budget is not None else float(args.time_limit)

    if args.parallel:
        if args.threads is not None:
            os.environ["OMP_NUM_THREADS"] = str(max(1, int(args.threads)))
    else:
        os.environ["OMP_NUM_THREADS"] = "1"

    import faiss

    configure_parallelism(args.parallel, args.threads)

    base = load_vector_matrix(args.base_vectors, mmap=args.mmap)
    query = load_vector_matrix(args.query_vectors, mmap=args.mmap)
    if args.transpose:
        base = np.ascontiguousarray(base.T)
        query = np.ascontiguousarray(query.T)
    db, dq = int(base.shape[1]), int(query.shape[1])
    if db != dq:
        raise ValueError(
            f"Embedding dimension mismatch: base_vectors has d={db}, "
            f"query_vectors has d={dq}. Use the same d for both (FAISS otherwise raises AssertionError)."
        )
    gt = load_ground_truth(args.ground_truth)

    K = int(args.K)
    if gt.shape[0] != K:
        raise ValueError(f"Ground truth length {gt.shape[0]} does not match K={K}")

    importlib.invalidate_caches()
    mod_name = args.submission_module

    use_fork = sys.platform != "win32" and not args.no_subprocess
    if args.no_subprocess:
        t0 = time.perf_counter()
        mod = importlib.import_module(mod_name)
        result = mod.solve(base, query, int(args.k), K, time_budget)
        elapsed = time.perf_counter() - t0
        timed_out = elapsed > float(args.time_limit)
        if timed_out:
            print(
                f"WARNING: wall time {elapsed:.4f}s exceeded limit {args.time_limit}s.",
                file=sys.stderr,
            )
        out = normalize_student_output(result, K)
    else:
        out, elapsed, timed_out = run_solve_with_time_limit(
            mod_name,
            base,
            query,
            int(args.k),
            K,
            time_budget,
            float(args.time_limit),
            use_fork=use_fork,
        )

    score = ndcg_at_k(out, gt)

    out_path = args.output
    if out_path is None:
        out_path = Path("output_indices.txt")
    write_indices(out_path, out)

    print(f"runtime: {elapsed:.6f}")
    print(f"score (nDCG@{K}): {score:.6f}")
    if timed_out:
        print("note: time limit was exceeded during grading run.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
